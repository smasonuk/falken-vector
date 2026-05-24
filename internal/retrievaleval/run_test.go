package retrievaleval

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/smasonuk/falken-vector/internal/vectorstore"
)

func TestRunCallsRetrieveAndAggregatesMetrics(t *testing.T) {
	ctx := context.Background()
	paths := testEvalPaths(t)
	store := &fakeEvalManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "internal/rag/retrieve.go", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"chunk-1": {ID: "chunk-1", DocumentID: "doc", Active: true, StartLine: 1, EndLine: 3},
		},
	}
	summary, err := Run(ctx, store, []Case{{
		ID:               "case",
		Question:         "question",
		ExpectedChunkIDs: []string{"chunk-1"},
	}}, Options{
		Paths:    paths,
		TopK:     1,
		Embedder: fakeEvalEmbedder{},
		Vector:   fakeEvalVector{hits: []vectorstore.Hit{{ChunkID: "chunk-1", Score: 0.8}}},
		Details:  true,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.Cases != 1 || summary.HitAtK != 1 || summary.RecallAtK != 1 || summary.PrecisionAtK != 1 || summary.MRRAtK != 1 {
		t.Fatalf("summary = %+v, want perfect metrics", summary)
	}
	if len(summary.CaseResults) != 1 || !summary.CaseResults[0].Hit {
		t.Fatalf("case results = %+v, want included details", summary.CaseResults)
	}
}

func TestRunCapturesPerCaseRetrievalErrors(t *testing.T) {
	ctx := context.Background()
	paths := testEvalPaths(t)
	summary, err := Run(ctx, &fakeEvalManifest{}, []Case{{
		ID:               "case",
		Question:         "question",
		ExpectedChunkIDs: []string{"chunk-1"},
	}}, Options{
		Paths:    paths,
		Embedder: fakeEvalEmbedder{},
		Vector:   fakeEvalVector{err: errors.New("search failed")},
		Details:  true,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.FailedCases != 1 || len(summary.CaseResults) != 1 || summary.CaseResults[0].Error == "" {
		t.Fatalf("summary = %+v, want captured case error", summary)
	}
}

func TestRunSupportsLexicalMode(t *testing.T) {
	ctx := context.Background()
	paths := testEvalPaths(t)
	store := &fakeEvalManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "internal/rag/retrieve.go", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"chunk-1": {ID: "chunk-1", DocumentID: "doc", Active: true},
		},
		lexicalHits: []manifest.LexicalHit{{ChunkID: "chunk-1", Score: -1}},
	}
	summary, err := Run(ctx, store, []Case{{
		Question:             "retrieve",
		ExpectedPathSuffixes: []string{"internal/rag/retrieve.go"},
	}}, Options{
		Paths: paths,
		Mode:  rag.RetrievalModeLexical,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.HitAtK != 1 {
		t.Fatalf("summary = %+v, want lexical hit", summary)
	}
}

func TestRunPassesRerankerModeToRetrieve(t *testing.T) {
	ctx := context.Background()
	paths := testEvalPaths(t)
	store := &fakeEvalManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"low":  {ID: "low", DocumentID: "doc", ChunkText: "unrelated", Active: true},
			"high": {ID: "high", DocumentID: "doc", ChunkText: "write lock acquired", Active: true},
		},
	}
	summary, err := Run(ctx, store, []Case{{
		Question:         "write lock acquired",
		ExpectedChunkIDs: []string{"high"},
	}}, Options{
		Paths:        paths,
		TopK:         1,
		Embedder:     fakeEvalEmbedder{},
		RerankerMode: rag.RerankerModeHeuristic,
		Vector: fakeEvalVector{hits: []vectorstore.Hit{
			{ChunkID: "low", Score: 0.5},
			{ChunkID: "high", Score: 0.5},
		}},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.HitAtK != 1 {
		t.Fatalf("summary = %+v, want reranked hit", summary)
	}
}

func TestRunPassesQueryPlannerModeAndIncludesPlanInDetails(t *testing.T) {
	ctx := context.Background()
	paths := testEvalPaths(t)
	store := &fakeEvalManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"a": {ID: "a", DocumentID: "doc", Active: true},
			"b": {ID: "b", DocumentID: "doc", Active: true},
		},
		lexicalByQuery: map[string][]manifest.LexicalHit{
			"original": {{ChunkID: "a", Score: -1}},
			"sub":      {{ChunkID: "b", Score: -1}},
		},
	}
	summary, err := Run(ctx, store, []Case{{
		Question:         "original",
		ExpectedChunkIDs: []string{"b"},
	}}, Options{
		Paths:            paths,
		TopK:             2,
		Mode:             rag.RetrievalModeLexical,
		QueryPlannerMode: rag.QueryPlannerModeHeuristic,
		QueryPlanner:     staticEvalQueryPlanner{queries: []string{"sub"}},
		MaxSubqueries:    2,
		Details:          true,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.HitAtK != 1 || len(summary.CaseResults) != 1 {
		t.Fatalf("summary = %+v, want query-planned hit with details", summary)
	}
	if !sameEvalStrings(summary.CaseResults[0].QueryPlan, []string{"original", "sub"}) {
		t.Fatalf("query plan = %+v", summary.CaseResults[0].QueryPlan)
	}
}

func TestRunDefaultsTopKToEight(t *testing.T) {
	ctx := context.Background()
	paths := testEvalPaths(t)
	vector := &recordingEvalVector{}
	_, err := Run(ctx, &fakeEvalManifest{}, []Case{{
		Question:         "question",
		ExpectedChunkIDs: []string{"chunk-1"},
	}}, Options{
		Paths:    paths,
		Embedder: fakeEvalEmbedder{},
		Vector:   vector,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if vector.limit != 50 {
		t.Fatalf("vector limit = %d, want default candidate-k 50", vector.limit)
	}
}

func TestRunReturnsMissingIndexError(t *testing.T) {
	paths, err := config.ResolvePaths(filepath.Join(t.TempDir(), ".falkengo"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = Run(context.Background(), &fakeEvalManifest{}, []Case{{
		Question:         "question",
		ExpectedChunkIDs: []string{"chunk-1"},
	}}, Options{
		Paths:    paths,
		Embedder: fakeEvalEmbedder{},
	})
	if err == nil || !errors.Is(err, rag.ErrNoIndex) {
		t.Fatalf("Run error = %v, want ErrNoIndex", err)
	}
}

type fakeEvalEmbedder struct{}

func (fakeEvalEmbedder) EmbedText(context.Context, string) (llm.Embedding, error) {
	return llm.Embedding{Model: "fake", Vector: []float32{1, 0}}, nil
}

type fakeEvalVector struct {
	hits []vectorstore.Hit
	err  error
}

func (f fakeEvalVector) Close() error                                             { return nil }
func (f fakeEvalVector) Insert(context.Context, []float32, string) (int64, error) { return 0, nil }
func (f fakeEvalVector) Search(context.Context, []float32, int) ([]vectorstore.Hit, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.hits, nil
}
func (f fakeEvalVector) Commit(context.Context) error { return nil }

type recordingEvalVector struct {
	limit int
}

func (f *recordingEvalVector) Close() error { return nil }
func (f *recordingEvalVector) Insert(context.Context, []float32, string) (int64, error) {
	return 0, nil
}
func (f *recordingEvalVector) Search(_ context.Context, _ []float32, limit int) ([]vectorstore.Hit, error) {
	f.limit = limit
	return nil, nil
}
func (f *recordingEvalVector) Commit(context.Context) error { return nil }

type fakeEvalManifest struct {
	manifest.Store
	docs           map[string]*manifest.Document
	chunks         map[string]manifest.Chunk
	lexicalHits    []manifest.LexicalHit
	lexicalByQuery map[string][]manifest.LexicalHit
}

func (f *fakeEvalManifest) GetDocumentByID(_ context.Context, id string) (*manifest.Document, error) {
	doc, ok := f.docs[id]
	if !ok {
		return nil, manifest.ErrNotFound
	}
	return doc, nil
}

func (f *fakeEvalManifest) GetActiveChunksByIDs(_ context.Context, ids []string) ([]manifest.Chunk, error) {
	var chunks []manifest.Chunk
	for _, id := range ids {
		chunk, ok := f.chunks[id]
		if ok && chunk.Active {
			chunks = append(chunks, chunk)
		}
	}
	return chunks, nil
}

func (f *fakeEvalManifest) SearchLexicalChunks(_ context.Context, query string, _ int) ([]manifest.LexicalHit, error) {
	if f.lexicalByQuery != nil {
		return f.lexicalByQuery[query], nil
	}
	return f.lexicalHits, nil
}

func (f *fakeEvalManifest) RebuildLexicalIndex(context.Context) error {
	return nil
}

func (f *fakeEvalManifest) EnsureLexicalIndex(context.Context) (bool, error) {
	return false, nil
}

type staticEvalQueryPlanner struct {
	queries []string
}

func (p staticEvalQueryPlanner) Plan(_ context.Context, question string, _ rag.QueryPlanOptions) (rag.QueryPlan, error) {
	return rag.QueryPlan{
		OriginalQuestion: question,
		Queries:          append([]string(nil), p.queries...),
		Mode:             string(rag.QueryPlannerModeHeuristic),
	}, nil
}

func sameEvalStrings(got []string, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func testEvalPaths(t *testing.T) config.Paths {
	t.Helper()
	paths, err := config.ResolvePaths(filepath.Join(t.TempDir(), ".falkengo"))
	if err != nil {
		t.Fatal(err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.ManifestPath, []byte("manifest"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(paths.VecgoPath, 0o755); err != nil {
		t.Fatal(err)
	}
	return paths
}
