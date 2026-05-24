package compact

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/smasonuk/falken-vector/internal/vectorstore"
)

func TestDryRunDoesNotEmbedOrWriteVectors(t *testing.T) {
	paths := testPaths(t)
	store := &fakeManifestStore{
		stats:  manifest.Stats{InactiveChunks: 3},
		chunks: testChunks(2),
	}
	calledVector := false
	summary, err := Run(context.Background(), store, Options{
		Paths:  paths,
		DryRun: true,
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			calledVector = true
			return nil, errors.New("should not be called")
		},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.ActiveChunks != 2 || summary.InactiveChunks != 3 {
		t.Fatalf("summary = %+v, want active/inactive counts", summary)
	}
	if store.updateCalls != 0 || calledVector {
		t.Fatalf("dry run wrote manifest or vector store: updates=%d vector=%v", store.updateCalls, calledVector)
	}
	if _, err := os.Stat(paths.VecgoPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("vecgo path stat = %v, want missing", err)
	}
}

func TestNoActiveChunksDoesNotRequireEmbedder(t *testing.T) {
	paths := testPaths(t)
	summary, err := Run(context.Background(), &fakeManifestStore{}, Options{Paths: paths})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.ActiveChunks != 0 || summary.ReembeddedChunks != 0 {
		t.Fatalf("summary = %+v, want no active chunks", summary)
	}
}

func TestSuccessfulCompactionReplacesVectorDBAndUpdatesRefs(t *testing.T) {
	paths := testPaths(t)
	if err := os.MkdirAll(paths.VecgoPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(paths.VecgoPath, "old"), []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	store := &fakeManifestStore{chunks: testChunks(2)}
	inserted := make([]string, 0)
	summary, err := Run(context.Background(), store, Options{
		Paths:      paths,
		Embedder:   &sequenceEmbedder{model: "new-model", vectors: [][]float32{{1, 0, 0}, {0, 1, 0}}},
		OpenVector: fakeVectorOpener(&inserted, nil, nil),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.ReembeddedChunks != 2 || summary.UpdatedChunks != 2 || summary.EmbeddingModel != "new-model" {
		t.Fatalf("summary = %+v, want reembedded chunks with model", summary)
	}
	if !sameStrings(inserted, []string{"chunk-0", "chunk-1"}) {
		t.Fatalf("inserted = %+v, want active chunk IDs", inserted)
	}
	if len(store.updates) != 2 || store.updates[0].VectorID != 1 || store.updates[1].VectorID != 2 {
		t.Fatalf("updates = %+v, want new vector IDs", store.updates)
	}
	if _, err := os.Stat(filepath.Join(paths.VecgoPath, "committed")); err != nil {
		t.Fatalf("new vector db missing committed marker: %v", err)
	}
	if _, err := os.Stat(filepath.Join(paths.VecgoPath, "old")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("old marker stat = %v, want removed with backup", err)
	}
	if summary.BackupPath != "" {
		t.Fatalf("backup path = %q, want removed by default", summary.BackupPath)
	}
}

func TestCompactEmbedsIndexedTextWhenPresent(t *testing.T) {
	paths := testPaths(t)
	chunks := testChunks(1)
	chunks[0].ChunkText = "raw body"
	chunks[0].IndexedText = "Document: README.md\nChunk:\nraw body"
	embedder := &sequenceEmbedder{model: "model", vectors: [][]float32{{1, 0, 0}}}
	inserted := make([]string, 0)
	_, err := Run(context.Background(), &fakeManifestStore{chunks: chunks}, Options{
		Paths:      paths,
		Embedder:   embedder,
		OpenVector: fakeVectorOpener(&inserted, nil, nil),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(embedder.inputs) != 1 || embedder.inputs[0] != chunks[0].IndexedText {
		t.Fatalf("embed inputs = %+v, want indexed text", embedder.inputs)
	}
}

func TestCompactFallsBackToChunkTextWhenIndexedTextEmpty(t *testing.T) {
	paths := testPaths(t)
	chunks := testChunks(1)
	chunks[0].ChunkText = "raw body"
	chunks[0].IndexedText = ""
	embedder := &sequenceEmbedder{model: "model", vectors: [][]float32{{1, 0, 0}}}
	inserted := make([]string, 0)
	_, err := Run(context.Background(), &fakeManifestStore{chunks: chunks}, Options{
		Paths:      paths,
		Embedder:   embedder,
		OpenVector: fakeVectorOpener(&inserted, nil, nil),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(embedder.inputs) != 1 || embedder.inputs[0] != chunks[0].ChunkText {
		t.Fatalf("embed inputs = %+v, want raw chunk text fallback", embedder.inputs)
	}
}

func TestPendingRunCausesError(t *testing.T) {
	paths := testPaths(t)
	store := &fakeManifestStore{
		chunks:  testChunks(1),
		pending: []manifest.PendingRun{{ID: "run-1", DocumentCount: 1, ChunkCount: 1, CreatedAt: time.Now().UTC()}},
	}
	_, err := Run(context.Background(), store, Options{
		Paths:    paths,
		Embedder: &sequenceEmbedder{model: "model", vectors: [][]float32{{1, 0, 0}}},
	})
	if !errors.Is(err, ErrPendingRunDetected) {
		t.Fatalf("Run error = %v, want ErrPendingRunDetected", err)
	}
}

func TestDryRunWithPendingRunReportsWarningSummary(t *testing.T) {
	paths := testPaths(t)
	store := &fakeManifestStore{
		chunks:  testChunks(1),
		pending: []manifest.PendingRun{{ID: "run-1", DocumentCount: 1, ChunkCount: 1, CreatedAt: time.Now().UTC()}},
	}
	summary, err := Run(context.Background(), store, Options{
		Paths:  paths,
		DryRun: true,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.PendingRuns != 1 || summary.ActiveChunks != 1 {
		t.Fatalf("summary = %+v, want pending run and active chunk counts", summary)
	}
	var out strings.Builder
	PrintSummary(&out, summary, true)
	if !strings.Contains(out.String(), "warning: pending ingest run detected") {
		t.Fatalf("summary output = %q, want pending-run warning", out.String())
	}
}

func TestCompactReadsChunksInBatches(t *testing.T) {
	paths := testPaths(t)
	store := &fakeManifestStore{chunks: testChunks(5)}
	inserted := make([]string, 0)
	_, err := Run(context.Background(), store, Options{
		Paths:      paths,
		BatchSize:  2,
		Embedder:   &sequenceEmbedder{model: "model", vectors: [][]float32{{1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}}},
		OpenVector: fakeVectorOpener(&inserted, nil, nil),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !sameInts(store.offsets, []int{0, 2, 4, 6}) {
		t.Fatalf("offsets = %+v, want page offsets 0,2,4,6", store.offsets)
	}
	if !sameStrings(inserted, []string{"chunk-0", "chunk-1", "chunk-2", "chunk-3", "chunk-4"}) {
		t.Fatalf("inserted = %+v, want all chunks in order", inserted)
	}
}

func TestDimensionMismatchLeavesOldVectorDBUntouched(t *testing.T) {
	paths := testPaths(t)
	if err := os.MkdirAll(paths.VecgoPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(paths.VecgoPath, "old"), []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	inserted := make([]string, 0)
	_, err := Run(context.Background(), &fakeManifestStore{chunks: testChunks(2)}, Options{
		Paths:      paths,
		Embedder:   &sequenceEmbedder{model: "model", vectors: [][]float32{{1, 0, 0}, {1, 0}}},
		OpenVector: fakeVectorOpener(&inserted, nil, nil),
	})
	if err == nil || !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("Run error = %v, want dimension mismatch", err)
	}
	if _, err := os.Stat(filepath.Join(paths.VecgoPath, "old")); err != nil {
		t.Fatalf("old vector db was touched: %v", err)
	}
	if len(inserted) != 1 {
		t.Fatalf("inserted = %+v, want first chunk inserted before mismatch", inserted)
	}
}

func TestVectorInsertFailureLeavesOldVectorDBUntouched(t *testing.T) {
	paths := testPaths(t)
	if err := os.MkdirAll(paths.VecgoPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(paths.VecgoPath, "old"), []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	insertErr := errors.New("insert failed")
	inserted := make([]string, 0)
	_, err := Run(context.Background(), &fakeManifestStore{chunks: testChunks(1)}, Options{
		Paths:      paths,
		Embedder:   &sequenceEmbedder{model: "model", vectors: [][]float32{{1, 0, 0}}},
		OpenVector: fakeVectorOpener(&inserted, insertErr, nil),
	})
	if !errors.Is(err, insertErr) {
		t.Fatalf("Run error = %v, want insert error", err)
	}
	if _, err := os.Stat(filepath.Join(paths.VecgoPath, "old")); err != nil {
		t.Fatalf("old vector db was touched: %v", err)
	}
}

func TestManifestUpdateFailureKeepsBackup(t *testing.T) {
	paths := testPaths(t)
	if err := os.MkdirAll(paths.VecgoPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(paths.VecgoPath, "old"), []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	updateErr := errors.New("update refs failed")
	store := &fakeManifestStore{chunks: testChunks(1), updateErr: updateErr}
	inserted := make([]string, 0)
	summary, err := Run(context.Background(), store, Options{
		Paths:      paths,
		Embedder:   &sequenceEmbedder{model: "model", vectors: [][]float32{{1, 0, 0}}},
		OpenVector: fakeVectorOpener(&inserted, nil, nil),
	})
	if !errors.Is(err, updateErr) {
		t.Fatalf("Run error = %v, want manifest update error", err)
	}
	if summary.UpdatedChunks != 0 {
		t.Fatalf("UpdatedChunks = %d, want 0 after manifest update failure", summary.UpdatedChunks)
	}
	if summary.BackupPath == "" {
		t.Fatal("backup path is empty, want retained backup")
	}
	if _, err := os.Stat(filepath.Join(summary.BackupPath, "old")); err != nil {
		t.Fatalf("backup missing old vector db: %v", err)
	}
	if _, err := os.Stat(filepath.Join(paths.VecgoPath, "committed")); err != nil {
		t.Fatalf("new vector db missing committed marker: %v", err)
	}
}

func TestSuccessfulCompactionKeepsBackupWhenRequested(t *testing.T) {
	paths := testPaths(t)
	if err := os.MkdirAll(paths.VecgoPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(paths.VecgoPath, "old"), []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	store := &fakeManifestStore{chunks: testChunks(1)}
	inserted := make([]string, 0)
	summary, err := Run(context.Background(), store, Options{
		Paths:      paths,
		Embedder:   &sequenceEmbedder{model: "model", vectors: [][]float32{{1, 0, 0}}},
		OpenVector: fakeVectorOpener(&inserted, nil, nil),
		KeepBackup: true,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.BackupPath == "" {
		t.Fatal("BackupPath is empty, want retained backup")
	}
	if _, err := os.Stat(filepath.Join(summary.BackupPath, "old")); err != nil {
		t.Fatalf("backup missing old marker: %v", err)
	}
	if _, err := os.Stat(filepath.Join(paths.VecgoPath, "committed")); err != nil {
		t.Fatalf("new vector db missing committed marker: %v", err)
	}
	if len(store.updates) != 1 || summary.UpdatedChunks != 1 {
		t.Fatalf("updates = %+v summary=%+v, want one manifest update", store.updates, summary)
	}
}

func TestCompactWithRealVectorStoreSupportsRetrieval(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t)
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatal(err)
	}
	store, err := manifest.Open(paths.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := store.Init(ctx); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	docID := manifest.DocumentID("/tmp/readme.md")
	oldActiveVector := int64(10)
	oldInactiveVector := int64(11)
	doc := manifest.Document{
		ID:          docID,
		Path:        "/tmp/readme.md",
		ContentHash: "doc",
		SizeBytes:   10,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      manifest.DocumentStatusIndexed,
	}
	active := manifest.Chunk{
		ID:             manifest.ChunkID(docID, 0, "active"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "active",
		ChunkText:      "active context",
		VectorID:       &oldActiveVector,
		EmbeddingModel: "old",
		Active:         true,
		CreatedAt:      now,
	}
	inactive := manifest.Chunk{
		ID:             manifest.ChunkID(docID, 1, "inactive"),
		DocumentID:     docID,
		ChunkIndex:     1,
		ContentHash:    "inactive",
		ChunkText:      "inactive context",
		VectorID:       &oldInactiveVector,
		EmbeddingModel: "old",
		Active:         false,
		CreatedAt:      now,
	}
	if err := store.ReplaceDocumentChunks(ctx, doc, []manifest.Chunk{active, inactive}); err != nil {
		t.Fatalf("ReplaceDocumentChunks: %v", err)
	}
	embedder := constantEmbedder{model: "compact-model", vector: []float32{1, 0, 0}}
	if _, err := Run(ctx, store, Options{
		Paths:    paths,
		Embedder: embedder,
	}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	results, err := rag.Retrieve(ctx, store, rag.RetrieveOptions{
		Question: "active",
		TopK:     5,
		Paths:    paths,
		Embedder: embedder,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(results) != 1 || results[0].Chunk.ID != active.ID {
		t.Fatalf("results = %+v, want active chunk only", results)
	}
	updated, err := store.ListActiveIndexedChunks(ctx, 10, 0)
	if err != nil {
		t.Fatalf("ListActiveIndexedChunks: %v", err)
	}
	if len(updated) != 1 || updated[0].VectorID == nil || *updated[0].VectorID == oldActiveVector || updated[0].EmbeddingModel != "compact-model" {
		t.Fatalf("updated active chunk = %+v, want new vector ref/model", updated)
	}
}

type fakeManifestStore struct {
	manifest.EmptyStore
	stats       manifest.Stats
	chunks      []manifest.Chunk
	pending     []manifest.PendingRun
	updateErr   error
	updateCalls int
	updates     []manifest.ChunkVectorUpdate
	offsets     []int
	limits      []int
}

func (s *fakeManifestStore) Stats(context.Context) (manifest.Stats, error) {
	return s.stats, nil
}

func (s *fakeManifestStore) ListActiveIndexedChunks(_ context.Context, limit int, offset int) ([]manifest.Chunk, error) {
	s.limits = append(s.limits, limit)
	s.offsets = append(s.offsets, offset)
	if offset >= len(s.chunks) {
		return nil, nil
	}
	end := offset + limit
	if end > len(s.chunks) {
		end = len(s.chunks)
	}
	return append([]manifest.Chunk(nil), s.chunks[offset:end]...), nil
}

func (s *fakeManifestStore) UpdateChunkVectorRefs(_ context.Context, updates []manifest.ChunkVectorUpdate) error {
	s.updateCalls++
	if s.updateErr != nil {
		return s.updateErr
	}
	s.updates = append(s.updates, updates...)
	return nil
}

func (s *fakeManifestStore) ListPendingRuns(context.Context) ([]manifest.PendingRun, error) {
	return s.pending, nil
}

func (s *fakeManifestStore) ActivatePendingRun(context.Context, string) error {
	return nil
}

func (s *fakeManifestStore) ClearPendingRun(context.Context, string) error {
	return nil
}

type sequenceEmbedder struct {
	model   string
	vectors [][]float32
	calls   int
	inputs  []string
}

func (e *sequenceEmbedder) EmbedText(_ context.Context, input string) (llm.Embedding, error) {
	if e.calls >= len(e.vectors) {
		return llm.Embedding{}, errors.New("unexpected embed call")
	}
	e.inputs = append(e.inputs, input)
	vector := append([]float32(nil), e.vectors[e.calls]...)
	e.calls++
	return llm.Embedding{Model: e.model, Vector: vector}, nil
}

type constantEmbedder struct {
	model  string
	vector []float32
}

func (e constantEmbedder) EmbedText(context.Context, string) (llm.Embedding, error) {
	return llm.Embedding{Model: e.model, Vector: append([]float32(nil), e.vector...)}, nil
}

type fakeVectorStore struct {
	path      string
	inserted  *[]string
	insertErr error
	commitErr error
	nextID    int64
}

func fakeVectorOpener(inserted *[]string, insertErr error, commitErr error) func(context.Context, string, int) (vectorstore.Store, error) {
	return func(_ context.Context, path string, dimensions int) (vectorstore.Store, error) {
		if dimensions <= 0 {
			return nil, errors.New("dimensions must be positive")
		}
		if err := os.MkdirAll(path, 0o755); err != nil {
			return nil, err
		}
		if err := os.WriteFile(filepath.Join(path, "CURRENT"), []byte("fake"), 0o644); err != nil {
			return nil, err
		}
		return &fakeVectorStore{path: path, inserted: inserted, insertErr: insertErr, commitErr: commitErr}, nil
	}
}

func (s *fakeVectorStore) Close() error {
	return nil
}

func (s *fakeVectorStore) Insert(_ context.Context, _ []float32, chunkID string) (int64, error) {
	if s.insertErr != nil {
		return 0, s.insertErr
	}
	*s.inserted = append(*s.inserted, chunkID)
	s.nextID++
	return s.nextID, nil
}

func (s *fakeVectorStore) Search(context.Context, []float32, int) ([]vectorstore.Hit, error) {
	return nil, nil
}

func (s *fakeVectorStore) Commit(context.Context) error {
	if s.commitErr != nil {
		return s.commitErr
	}
	return os.WriteFile(filepath.Join(s.path, "committed"), []byte(strings.Join(*s.inserted, "\n")), 0o644)
}

func testPaths(t *testing.T) config.Paths {
	t.Helper()
	paths, err := config.ResolvePaths(filepath.Join(t.TempDir(), ".falkengo"))
	if err != nil {
		t.Fatal(err)
	}
	return paths
}

func testChunks(n int) []manifest.Chunk {
	now := time.Now().UTC()
	chunks := make([]manifest.Chunk, 0, n)
	for i := 0; i < n; i++ {
		chunks = append(chunks, manifest.Chunk{
			ID:             "chunk-" + string(rune('0'+i)),
			DocumentID:     "doc",
			ChunkIndex:     i,
			ContentHash:    "hash",
			ChunkText:      "text",
			StartLine:      1,
			EndLine:        1,
			EmbeddingModel: "old",
			Active:         true,
			CreatedAt:      now,
		})
	}
	return chunks
}

func sameStrings(got []string, want []string) bool {
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

func sameInts(got []int, want []int) bool {
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
