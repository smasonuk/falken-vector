package cli

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/smasonuk/falken-vector/internal/retrievaleval"
	_ "modernc.org/sqlite"
)

func TestEvalRetrievalRequiresDataset(t *testing.T) {
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"eval", "retrieval"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--dataset is required") {
		t.Fatalf("Execute error = %v, want dataset required", err)
	}
}

func TestEvalRetrievalRejectsInvalidFormat(t *testing.T) {
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"eval", "retrieval", "--dataset", "dataset.jsonl", "--format", "xml"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--format must be text or json") {
		t.Fatalf("Execute error = %v, want format validation", err)
	}
}

func TestEvalRetrievalRejectsInvalidThreshold(t *testing.T) {
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"eval", "retrieval", "--dataset", "dataset.jsonl", "--fail-under-hit", "1.2"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--fail-under-hit must be between 0 and 1") {
		t.Fatalf("Execute error = %v, want threshold validation", err)
	}
}

func TestEvalRetrievalTextOutput(t *testing.T) {
	state, dataset := setupEvalCLITest(t)
	restore := stubEvalCLI(t, retrievaleval.Summary{
		Cases:          1,
		EvaluatedCases: 1,
		TopK:           8,
		HitAtK:         1,
		RecallAtK:      1,
		PrecisionAtK:   0.125,
		MRRAtK:         1,
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "eval", "retrieval", "--dataset", dataset})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	for _, want := range []string{"Retrieval evaluation", "hit@8:", "recall@8:", "precision@8:", "mrr@8:"} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, missing %q", output, want)
		}
	}
}

func TestEvalRetrievalJSONOutput(t *testing.T) {
	state, dataset := setupEvalCLITest(t)
	restore := stubEvalCLI(t, retrievaleval.Summary{
		Cases:          1,
		EvaluatedCases: 1,
		TopK:           8,
		HitAtK:         1,
		RecallAtK:      0.5,
		PrecisionAtK:   0.25,
		MRRAtK:         1,
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "eval", "retrieval", "--dataset", dataset, "--format", "json"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	var summary retrievaleval.Summary
	if err := json.Unmarshal(out.Bytes(), &summary); err != nil {
		t.Fatalf("decode JSON output: %v\n%s", err, out.String())
	}
	if summary.HitAtK != 1 || summary.RecallAtK != 0.5 || summary.TopK != 8 {
		t.Fatalf("summary = %+v, want JSON metrics", summary)
	}
}

func TestEvalRetrievalDetailsOutput(t *testing.T) {
	state, dataset := setupEvalCLITest(t)
	restore := stubEvalCLI(t, retrievaleval.Summary{
		Cases:          1,
		EvaluatedCases: 1,
		TopK:           8,
		HitAtK:         1,
		RecallAtK:      1,
		PrecisionAtK:   1,
		MRRAtK:         1,
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "eval", "retrieval", "--dataset", dataset, "--details"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "[PASS] case") {
		t.Fatalf("output = %q, want per-case details", out.String())
	}
}

func TestEvalRetrievalShowQueryPlanEnablesDetails(t *testing.T) {
	state, dataset := setupEvalCLITest(t)
	restore := stubEvalCLI(t, retrievaleval.Summary{
		Cases:          1,
		EvaluatedCases: 1,
		TopK:           8,
		HitAtK:         1,
		RecallAtK:      1,
		PrecisionAtK:   1,
		MRRAtK:         1,
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "eval", "retrieval", "--dataset", dataset, "--show-query-plan"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	for _, want := range []string{"[PASS] case", "query plan:", "expanded question"} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, missing %q", output, want)
		}
	}
}

func TestEvalRetrievalThresholdFailurePrintsReport(t *testing.T) {
	state, dataset := setupEvalCLITest(t)
	restore := stubEvalCLI(t, retrievaleval.Summary{
		Cases:          1,
		EvaluatedCases: 1,
		TopK:           8,
		HitAtK:         0.7,
		RecallAtK:      0.8,
		PrecisionAtK:   0.1,
		MRRAtK:         0.7,
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "eval", "retrieval", "--dataset", dataset, "--fail-under-hit", "0.8"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "hit@8 0.7000 below threshold 0.8000") {
		t.Fatalf("Execute error = %v, want threshold failure", err)
	}
	if !strings.Contains(out.String(), "hit@8:") {
		t.Fatalf("output = %q, want report before threshold error", out.String())
	}
}

func TestEvalRetrievalPassesHybridMode(t *testing.T) {
	state, dataset := setupEvalCLITest(t)
	restore := stubEvalCLI(t, retrievaleval.Summary{Cases: 1, EvaluatedCases: 1, TopK: 8})
	defer restore()
	seenMode := rag.RetrievalMode("")
	seenReranker := rag.RerankerMode("")
	oldRunner := runRetrievalEval
	runRetrievalEval = func(ctx context.Context, store manifest.Store, cases []retrievaleval.Case, opts retrievaleval.Options) (retrievaleval.Summary, error) {
		seenMode = opts.Mode
		seenReranker = opts.RerankerMode
		if opts.CandidateK != 20 || opts.VectorCandidateK != 10 || opts.LexicalCandidateK != 12 {
			t.Fatalf("candidate opts = %+v, want propagated values", opts)
		}
		return oldRunner(ctx, store, cases, opts)
	}
	defer func() { runRetrievalEval = oldRunner }()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "eval", "retrieval", "--dataset", dataset, "--retrieval", "hybrid", "--reranker", "heuristic", "--candidate-k", "20", "--vector-candidate-k", "10", "--lexical-candidate-k", "12"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if seenMode != rag.RetrievalModeHybrid {
		t.Fatalf("mode = %q, want hybrid", seenMode)
	}
	if seenReranker != rag.RerankerModeHeuristic {
		t.Fatalf("reranker = %q, want heuristic", seenReranker)
	}
}

func TestEvalRetrievalPassesQueryPlannerMode(t *testing.T) {
	state, dataset := setupEvalCLITest(t)
	restore := stubEvalCLI(t, retrievaleval.Summary{Cases: 1, EvaluatedCases: 1, TopK: 8})
	defer restore()
	seenPlanner := rag.QueryPlannerMode("")
	seenMaxSubqueries := 0
	oldRunner := runRetrievalEval
	runRetrievalEval = func(ctx context.Context, store manifest.Store, cases []retrievaleval.Case, opts retrievaleval.Options) (retrievaleval.Summary, error) {
		seenPlanner = opts.QueryPlannerMode
		seenMaxSubqueries = opts.MaxSubqueries
		return oldRunner(ctx, store, cases, opts)
	}
	defer func() { runRetrievalEval = oldRunner }()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "eval", "retrieval", "--dataset", dataset, "--query-planner", "heuristic", "--max-subqueries", "3"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if seenPlanner != rag.QueryPlannerModeHeuristic || seenMaxSubqueries != 3 {
		t.Fatalf("planner = %q max = %d, want heuristic max 3", seenPlanner, seenMaxSubqueries)
	}
}

func TestQueryRetrievalFlagsPassThrough(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubQueryCLI(t)
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "query", "hello", "--retrieval", "hybrid", "--reranker", "heuristic", "--query-planner", "heuristic", "--max-subqueries", "3", "--show-query-plan", "--candidate-k", "25"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "Query plan:") || !strings.Contains(out.String(), "expanded hello") {
		t.Fatalf("output = %q, want query plan", out.String())
	}
}

func TestAskRetrievalLexicalDoesNotRequireEmbedder(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAskCLI(t)
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--retrieval", "lexical", "--reranker", "heuristic", "--query-planner", "heuristic", "--max-subqueries", "2", "--show-query-plan"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "Query plan:") || !strings.Contains(out.String(), "expanded hello") {
		t.Fatalf("output = %q, want query plan", out.String())
	}
}

func TestQueryJSONOutputsValidJSONAndOmitsIndexedText(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	oldRetrieve := retrieveWithPlan
	defer func() { retrieveWithPlan = oldRetrieve }()
	retrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		if opts.SourceFilter.IsEmpty() || opts.CandidateK != 100 {
			t.Fatalf("retrieve opts = %+v, want source filter and expanded candidate default", opts)
		}
		return rag.RetrieveResult{
			Plan: rag.QueryPlan{Mode: string(rag.QueryPlannerModeHeuristic), Queries: []string{opts.Question, "source citation validator"}},
			Chunks: []rag.RetrievedChunk{{
				Path:        "internal/rag/ask.go",
				Score:       0.8123,
				RankScore:   0.0317,
				Sources:     []string{"lexical"},
				Reranker:    string(rag.RerankerModeHeuristic),
				RerankScore: 0.9,
				Chunk: manifest.Chunk{
					ID:          "chunk",
					DocumentID:  "doc",
					ChunkIndex:  1,
					ChunkText:   "raw chunk text",
					IndexedText: "Document: hidden\nChunk:\nraw chunk text",
					StartLine:   10,
					EndLine:     42,
					Chunker:     "code",
					Language:    "go",
					SymbolName:  "Ask",
					SymbolKind:  "function",
				},
			}},
		}, nil
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "query", "citation validation", "--retrieval", "lexical", "--reranker", "heuristic", "--query-planner", "heuristic", "--include", "internal/**", "--exclude", "internal/vendor/**", "--source-root", ".", "--json"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if strings.Contains(out.String(), "Query:") || strings.Contains(out.String(), "Document: hidden") {
		t.Fatalf("JSON output leaked plain text or indexed text: %q", out.String())
	}
	var decoded queryJSONOutput
	if err := json.Unmarshal(out.Bytes(), &decoded); err != nil {
		t.Fatalf("decode JSON: %v\n%s", err, out.String())
	}
	if decoded.Question != "citation validation" || len(decoded.Chunks) != 1 || decoded.Chunks[0].Text != "raw chunk text" {
		t.Fatalf("decoded = %+v, want query JSON chunk", decoded)
	}
	if decoded.QueryPlan == nil || len(decoded.QueryPlan.Queries) != 2 {
		t.Fatalf("query plan = %+v, want included plan", decoded.QueryPlan)
	}
	if decoded.SourceFilter == nil || len(decoded.SourceFilter.IncludeGlobs) != 1 || len(decoded.SourceFilter.SourceRoots) != 1 {
		t.Fatalf("source filter = %+v, want included filter", decoded.SourceFilter)
	}
}

func TestQueryShowRetrievalDebugPrintsModeCountsFilterAndPlan(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	oldRetrieve := retrieveWithPlan
	defer func() { retrieveWithPlan = oldRetrieve }()
	retrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		return rag.RetrieveResult{
			Plan: rag.QueryPlan{Mode: string(rag.QueryPlannerModeHeuristic), Queries: []string{opts.Question, "expanded"}},
			Chunks: []rag.RetrievedChunk{{
				Path:  "internal/rag/retrieve.go",
				Score: 0.5,
				Chunk: manifest.Chunk{
					ID:        "chunk",
					ChunkText: "text",
					StartLine: 3,
					EndLine:   9,
				},
			}},
		}, nil
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "query", "hello", "--retrieval", "lexical", "--query-planner", "heuristic", "--include", "internal/**", "--show-retrieval-debug"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	for _, want := range []string{"Retrieval debug:", "mode: lexical", "candidate-k: 100", "source filter:", "include: internal/**", "Query plan:", "expanded"} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, missing %q", output, want)
		}
	}
}

func TestAskPrintsCitationWarningsToStderr(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAskForCitationTest(t, func(opts rag.AskOptions) rag.AskResult {
		return rag.AskResult{
			Answer:           "answer",
			Sources:          []rag.SourceChunk{{SourceNumber: 1, Path: "README.md", StartLine: 1, EndLine: 2}},
			CitationWarnings: []string{"answer did not cite any source"},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var errOut bytes.Buffer
	cmd.SetErr(&errOut)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(errOut.String(), "warning: answer did not cite any source") {
		t.Fatalf("stderr = %q, want citation warning", errOut.String())
	}
}

func TestAskCitationFlagsSetPolicy(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	var seen []rag.CitationPolicy
	restore := stubAskForCitationTest(t, func(opts rag.AskOptions) rag.AskResult {
		seen = append(seen, opts.CitationPolicy)
		return rag.AskResult{Answer: "answer", Sources: []rag.SourceChunk{{SourceNumber: 1, Path: "README.md"}}}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--retrieval", "lexical", "--no-citation-validation"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute no validation: %v", err)
	}
	cmd = NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--retrieval", "lexical", "--no-citation-retry"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute no retry: %v", err)
	}
	if len(seen) != 2 || seen[0] != rag.CitationPolicyOff || seen[1] != rag.CitationPolicyValidateOnly {
		t.Fatalf("policies = %+v, want off then validate-only", seen)
	}
}

func TestAskOpenSourceUsesRetrievedSource(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restoreAsk := stubAskForCitationTest(t, func(opts rag.AskOptions) rag.AskResult {
		return rag.AskResult{Answer: "answer", Sources: []rag.SourceChunk{{SourceNumber: 1, Path: "README.md"}}}
	})
	defer restoreAsk()
	oldOpener := newSourceOpener
	opener := &recordingSourceOpener{}
	newSourceOpener = func(func(string) string) SourceOpener { return opener }
	defer func() { newSourceOpener = oldOpener }()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--retrieval", "lexical", "--open-source", "1"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if opener.path != "README.md" || opener.line != 1 {
		t.Fatalf("opener = %+v, want README start line", opener)
	}
}

func TestRetrievalFlagValidation(t *testing.T) {
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"query", "hello", "--retrieval", "bogus"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--retrieval must be vector, lexical, or hybrid") {
		t.Fatalf("Execute error = %v, want invalid retrieval mode", err)
	}

	cmd = NewRootCommand()
	cmd.SetArgs([]string{"query", "hello", "--candidate-k", "0"})
	err = cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--candidate-k must be > 0") {
		t.Fatalf("Execute error = %v, want invalid candidate-k", err)
	}

	cmd = NewRootCommand()
	cmd.SetArgs([]string{"query", "hello", "--reranker", "bogus"})
	err = cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--reranker must be none or heuristic") {
		t.Fatalf("Execute error = %v, want invalid reranker mode", err)
	}

	cmd = NewRootCommand()
	cmd.SetArgs([]string{"query", "hello", "--query-planner", "bogus"})
	err = cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--query-planner must be none, heuristic, or llm") {
		t.Fatalf("Execute error = %v, want invalid query planner mode", err)
	}

	cmd = NewRootCommand()
	cmd.SetArgs([]string{"query", "hello", "--max-subqueries", "0"})
	err = cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--max-subqueries must be between 1 and 8") {
		t.Fatalf("Execute error = %v, want invalid max-subqueries", err)
	}
}

func TestPrintQueryResultsIncludesRerankerMetadata(t *testing.T) {
	var out bytes.Buffer
	printQueryResults(&out, "hello", []rag.RetrievedChunk{{
		Chunk:       manifest.Chunk{ID: "chunk", ChunkText: "text", StartLine: 1, EndLine: 2},
		Path:        "README.md",
		Score:       0.75,
		RankScore:   0.25,
		Reranker:    string(rag.RerankerModeHeuristic),
		RerankScore: 0.75,
	}})
	output := out.String()
	for _, want := range []string{"Reranker: heuristic", "Rerank score: 0.7500", "Rank score: 0.2500", "Source: [source 1] README.md:1-2"} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, missing %q", output, want)
		}
	}
}

func TestSourceReferenceFormatsLineRange(t *testing.T) {
	got := SourceReference(rag.SourceChunk{SourceNumber: 2, Path: "internal/rag/retrieve.go", StartLine: 35, EndLine: 73})
	if got != "[source 2] internal/rag/retrieve.go:35-73" {
		t.Fatalf("SourceReference = %q", got)
	}
}

func TestAskPrintsEditorFriendlySources(t *testing.T) {
	var out bytes.Buffer
	printAnswer(&out, rag.AskResult{
		Answer:  "answer [source 1].",
		Sources: []rag.SourceChunk{{SourceNumber: 1, Path: "internal/rag/retrieve.go", StartLine: 35, EndLine: 73}},
	})
	if !strings.Contains(out.String(), "Sources:\n[source 1] internal/rag/retrieve.go:35-73") {
		t.Fatalf("output = %q, want editor-friendly source", out.String())
	}
}

func TestPrepareLexicalIndexEnsuresInsteadOfAlwaysRebuilding(t *testing.T) {
	store := &lexicalPrepStore{}
	if err := prepareLexicalIndex(context.Background(), store, rag.RetrievalModeHybrid); err != nil {
		t.Fatalf("prepareLexicalIndex: %v", err)
	}
	if store.initCalls != 1 || store.ensureCalls != 1 || store.rebuildCalls != 0 {
		t.Fatalf("store calls init=%d ensure=%d rebuild=%d, want init+ensure only", store.initCalls, store.ensureCalls, store.rebuildCalls)
	}
}

func TestRepairRebuildLexical(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	paths, err := config.ResolvePaths(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatal(err)
	}
	store, err := manifest.Open(paths.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Init(context.Background()); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	docID := manifest.DocumentID("README.md")
	if err := store.ReplaceDocumentChunks(context.Background(), manifest.Document{
		ID:          docID,
		Path:        "README.md",
		ContentHash: "hash",
		SizeBytes:   1,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      manifest.DocumentStatusIndexed,
	}, []manifest.Chunk{{
		ID:             manifest.ChunkID(docID, 0, "chunk"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunk",
		ChunkText:      "repairlexical",
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      now,
	}}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.SearchLexicalChunks(context.Background(), "repairlexical", 10); err != nil {
		t.Fatal(err)
	}
	rawDB, err := sql.Open("sqlite", paths.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rawDB.ExecContext(context.Background(), `delete from chunk_fts`); err != nil {
		rawDB.Close()
		t.Fatal(err)
	}
	if err := rawDB.Close(); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "repair", "--rebuild-lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "rebuilt lexical index") {
		t.Fatalf("repair output = %q, want lexical rebuild message", out.String())
	}
	store, err = manifest.Open(paths.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	hits, err := store.SearchLexicalChunks(context.Background(), "repairlexical", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v, want rebuilt lexical hit", hits)
	}
}

func setupEvalCLITest(t *testing.T) (string, string) {
	t.Helper()
	root := t.TempDir()
	state := filepath.Join(root, ".falkengo")
	paths, err := config.ResolvePaths(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatal(err)
	}
	store, err := manifest.Open(paths.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Init(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	touchDirForEvalCLI(t, paths.VecgoPath)

	dataset := filepath.Join(root, "retrieval.jsonl")
	writeFileForEvalCLI(t, dataset, `{"id":"case","question":"question","expected_chunk_ids":["chunk-1"]}`+"\n")
	return state, dataset
}

func stubEvalCLI(t *testing.T, summary retrievaleval.Summary) func() {
	t.Helper()
	oldEmbedder := newEvalEmbedder
	oldRunner := runRetrievalEval
	newEvalEmbedder = func() (llm.Embedder, error) {
		return fakeCLIEmbedder{}, nil
	}
	runRetrievalEval = func(_ context.Context, _ manifest.Store, cases []retrievaleval.Case, opts retrievaleval.Options) (retrievaleval.Summary, error) {
		if len(cases) != 1 || cases[0].Question != "question" {
			t.Fatalf("cases = %+v, want loaded dataset", cases)
		}
		if opts.TopK != 8 {
			t.Fatalf("TopK = %d, want 8", opts.TopK)
		}
		summary.TopK = opts.TopK
		if opts.Details {
			summary.CaseResults = []retrievaleval.CaseResult{{CaseID: "case", Question: "question", Hit: true, QueryPlan: []string{"question", "expanded question"}}}
		}
		return summary, nil
	}
	return func() {
		newEvalEmbedder = oldEmbedder
		runRetrievalEval = oldRunner
	}
}

func stubQueryCLI(t *testing.T) func() {
	t.Helper()
	oldEmbedder := newCLIEmbedder
	oldRetrieve := retrieveWithPlan
	newCLIEmbedder = func() (llm.Embedder, error) { return fakeCLIEmbedder{}, nil }
	retrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		if opts.Mode != rag.RetrievalModeHybrid || opts.RerankerMode != rag.RerankerModeHeuristic || opts.QueryPlannerMode != rag.QueryPlannerModeHeuristic || opts.MaxSubqueries != 3 || opts.CandidateK != 25 {
			t.Fatalf("retrieve opts = %+v, want hybrid heuristic query planner candidate-k 25", opts)
		}
		return rag.RetrieveResult{Plan: rag.QueryPlan{Queries: []string{opts.Question, "expanded " + opts.Question}}}, nil
	}
	return func() {
		newCLIEmbedder = oldEmbedder
		retrieveWithPlan = oldRetrieve
	}
}

func stubAskCLI(t *testing.T) func() {
	t.Helper()
	oldEmbedder := newCLIEmbedder
	oldRetrieve := retrieveWithPlan
	oldClient := newCLIChatClient
	oldAsk := askWithLLM
	newCLIEmbedder = func() (llm.Embedder, error) {
		t.Fatal("embedder should not be configured for lexical retrieval")
		return nil, nil
	}
	retrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		if opts.Mode != rag.RetrievalModeLexical || opts.RerankerMode != rag.RerankerModeHeuristic || opts.QueryPlannerMode != rag.QueryPlannerModeHeuristic || opts.MaxSubqueries != 2 {
			t.Fatalf("retrieve opts = %+v, want lexical heuristic query planner", opts)
		}
		return rag.RetrieveResult{
			Chunks: []rag.RetrievedChunk{{Chunk: manifest.Chunk{ID: "chunk", ChunkText: "text", Active: true}, Path: "README.md", Sources: []string{"lexical"}}},
			Plan:   rag.QueryPlan{Queries: []string{opts.Question, "expanded " + opts.Question}},
		}, nil
	}
	newCLIChatClient = func() (llm.Client, error) { return fakeCLIClient{}, nil }
	askWithLLM = func(context.Context, rag.AskOptions) (rag.AskResult, error) {
		return rag.AskResult{Answer: "answer", Sources: []rag.SourceChunk{{SourceNumber: 1, Path: "README.md"}}}, nil
	}
	return func() {
		newCLIEmbedder = oldEmbedder
		retrieveWithPlan = oldRetrieve
		newCLIChatClient = oldClient
		askWithLLM = oldAsk
	}
}

func stubAskForCitationTest(t *testing.T, ask func(rag.AskOptions) rag.AskResult) func() {
	t.Helper()
	oldEmbedder := newCLIEmbedder
	oldRetrieve := retrieveWithPlan
	oldClient := newCLIChatClient
	oldAsk := askWithLLM
	newCLIEmbedder = func() (llm.Embedder, error) {
		t.Fatal("embedder should not be configured for lexical retrieval")
		return nil, nil
	}
	retrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		if opts.Mode != rag.RetrievalModeLexical {
			t.Fatalf("retrieve mode = %q, want lexical", opts.Mode)
		}
		return rag.RetrieveResult{
			Chunks: []rag.RetrievedChunk{{
				Chunk: manifest.Chunk{ID: "chunk", ChunkText: "text", Active: true, StartLine: 1, EndLine: 2},
				Path:  "README.md",
			}},
			Plan: rag.QueryPlan{Queries: []string{opts.Question}},
		}, nil
	}
	newCLIChatClient = func() (llm.Client, error) { return fakeCLIClient{}, nil }
	askWithLLM = func(_ context.Context, opts rag.AskOptions) (rag.AskResult, error) {
		return ask(opts), nil
	}
	return func() {
		newCLIEmbedder = oldEmbedder
		retrieveWithPlan = oldRetrieve
		newCLIChatClient = oldClient
		askWithLLM = oldAsk
	}
}

type fakeCLIEmbedder struct{}

func (fakeCLIEmbedder) EmbedText(context.Context, string) (llm.Embedding, error) {
	return llm.Embedding{Model: "fake", Vector: []float32{1, 0}}, nil
}

type fakeCLIClient struct{}

func (fakeCLIClient) Complete(context.Context, llm.CompletionRequest) (llm.CompletionResponse, error) {
	return llm.CompletionResponse{Text: "answer"}, nil
}

type lexicalPrepStore struct {
	manifest.EmptyStore
	initCalls    int
	ensureCalls  int
	rebuildCalls int
}

func (s *lexicalPrepStore) Init(context.Context) error {
	s.initCalls++
	return nil
}

func (s *lexicalPrepStore) EnsureLexicalIndex(context.Context) (bool, error) {
	s.ensureCalls++
	return false, nil
}

func (s *lexicalPrepStore) RebuildLexicalIndex(context.Context) error {
	s.rebuildCalls++
	return nil
}

func touchDirForEvalCLI(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
}

func writeFileForEvalCLI(t *testing.T, path string, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
