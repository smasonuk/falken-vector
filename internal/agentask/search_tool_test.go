package agentask

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
)

func TestSearchIndexToolDescriptorName(t *testing.T) {
	tool := NewSearchIndexTool(SearchToolOptions{})
	if got := tool.Descriptor().Name; got != SearchIndexToolName {
		t.Fatalf("tool name = %q, want %q", got, SearchIndexToolName)
	}
}

func TestSearchIndexToolRejectsEmptyQuery(t *testing.T) {
	tool := NewSearchIndexTool(SearchToolOptions{})
	result := executeSearchTool(t, tool, `{"query":"   "}`)
	if result.Success || result.Status != "invalid_arguments" {
		t.Fatalf("result = %+v, want invalid arguments failure", result)
	}
}

func TestSearchIndexToolRejectsInvalidRetrievalMode(t *testing.T) {
	tool := NewSearchIndexTool(SearchToolOptions{})
	result := executeSearchTool(t, tool, `{"query":"hello","retrieval":"bogus"}`)
	if result.Success || result.Status != "invalid_retrieval" {
		t.Fatalf("result = %+v, want invalid retrieval failure", result)
	}
}

func TestSearchIndexToolRejectsInvalidStrategy(t *testing.T) {
	tool := NewSearchIndexTool(SearchToolOptions{})
	result := executeSearchTool(t, tool, `{"query":"hello","strategy":"wandering"}`)
	if result.Success || result.Status != "invalid_strategy" {
		t.Fatalf("result = %+v, want invalid strategy failure", result)
	}
}

func TestSearchIndexToolLexicalDoesNotConstructEmbedder(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	calledEmbedder := false
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 4,
		},
		EmbedderFactory: func() (llm.Embedder, error) {
			calledEmbedder = true
			return fakeAgentAskEmbedder{}, nil
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			if opts.Mode != rag.RetrievalModeLexical {
				t.Fatalf("mode = %q, want lexical", opts.Mode)
			}
			return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"hello"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if calledEmbedder {
		t.Fatal("embedder factory was called for lexical retrieval")
	}
}

func TestSearchIndexToolVectorConstructsEmbedder(t *testing.T) {
	paths := searchToolTestPaths(t, true)
	calledEmbedder := false
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeVector,
			TopK: 4,
		},
		EmbedderFactory: func() (llm.Embedder, error) {
			calledEmbedder = true
			return fakeAgentAskEmbedder{}, nil
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			if opts.Embedder == nil {
				t.Fatal("embedder was not assigned")
			}
			return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"hello"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if !calledEmbedder {
		t.Fatal("embedder factory was not called for vector retrieval")
	}
}

func TestSearchIndexToolTopKDefaultsToRetrievalDefault(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	var seenTopK int
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 5,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			seenTopK = opts.TopK
			return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"hello"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if seenTopK != 5 {
		t.Fatalf("topK = %d, want 5", seenTopK)
	}
}

func TestSearchIndexToolTopKIsCapped(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	var seenTopK int
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 5,
		},
		MaxTopK: 7,
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			seenTopK = opts.TopK
			return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"hello","top_k":99}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if seenTopK != 7 {
		t.Fatalf("topK = %d, want capped 7", seenTopK)
	}
	payload := decodeSearchPayload(t, result.Payload)
	if len(payload.Warnings) != 1 || !strings.Contains(payload.Warnings[0], "capped") {
		t.Fatalf("warnings = %+v, want cap warning", payload.Warnings)
	}
	if !strings.Contains(result.Content, "Warnings:") || !strings.Contains(result.Content, "top_k capped at 7") {
		t.Fatalf("content = %q, want cap warning visible to model", result.Content)
	}
}

func TestSearchIndexToolTopKUsesRetrievalDefaultAsFloor(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	var seenTopK int
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 12,
		},
		MaxTopK: 20,
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			seenTopK = opts.TopK
			return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"hello","top_k":5}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if seenTopK != 12 {
		t.Fatalf("topK = %d, want floor 12", seenTopK)
	}
	payload := decodeSearchPayload(t, result.Payload)
	if len(payload.Warnings) != 1 || !strings.Contains(payload.Warnings[0], "floor 12") {
		t.Fatalf("warnings = %+v, want floor warning", payload.Warnings)
	}
}

func TestSearchIndexToolTopKAboveFloorStillHonored(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	var seenTopK int
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 12,
		},
		MaxTopK: 20,
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			seenTopK = opts.TopK
			return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"hello","top_k":18}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if seenTopK != 18 {
		t.Fatalf("topK = %d, want agent value above floor", seenTopK)
	}
}

func TestSearchIndexToolInvalidArgumentsDoNotConsumeSearchBudget(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths:          paths,
		Store:          manifest.EmptyStore{},
		MaxSearchCalls: 1,
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
		},
	})
	invalid := executeSearchTool(t, tool, `{"query":"hello","unexpected":true}`)
	if invalid.Success || invalid.Status != "invalid_arguments" {
		t.Fatalf("invalid result = %+v, want invalid arguments", invalid)
	}
	valid := executeSearchTool(t, tool, `{"query":"hello"}`)
	if !valid.Success {
		t.Fatalf("valid result = %+v, want invalid call not to consume budget", valid)
	}
	limited := executeSearchTool(t, tool, `{"query":"hello again"}`)
	if limited.Success || limited.Status != "search_call_limit" {
		t.Fatalf("limited result = %+v, want search call limit after one valid search", limited)
	}
}

func TestSearchIndexToolRegistersRetrievedChunks(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	registry := NewCitationRegistry()
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths:    paths,
		Store:    manifest.EmptyStore{},
		Registry: registry,
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			return rag.RetrieveResult{
				Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{testRetrievedChunk("chunk-1", "README.md", "raw text", "indexed text")},
			}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"hello"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if len(registry.Sources()) != 1 || registry.Sources()[0].SourceNumber != 1 {
		t.Fatalf("sources = %+v, want source 1", registry.Sources())
	}
	if !strings.Contains(result.Content, "[source 1] README.md:10-12") {
		t.Fatalf("content = %q, want source reference", result.Content)
	}
}

func TestSearchIndexToolDuplicateAcrossSearchesKeepsSourceNumber(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	registry := NewCitationRegistry()
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths:    paths,
		Store:    manifest.EmptyStore{},
		Registry: registry,
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			return rag.RetrieveResult{
				Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{testRetrievedChunk("chunk-1", "README.md", "raw text", "indexed text")},
			}, nil
		},
	})
	first := executeSearchTool(t, tool, `{"query":"hello"}`)
	second := executeSearchTool(t, tool, `{"query":"hello again"}`)
	if !first.Success || !second.Success {
		t.Fatalf("results = %+v %+v, want success", first, second)
	}
	if registry.SourceCount() != 1 {
		t.Fatalf("source count = %d, want duplicate kept at 1", registry.SourceCount())
	}
	secondPayload := decodeSearchPayload(t, second.Payload)
	if secondPayload.NewSources != 0 || secondPayload.DuplicateSources != 1 {
		t.Fatalf("second metrics = new %d dup %d, want duplicate source reuse", secondPayload.NewSources, secondPayload.DuplicateSources)
	}
	if !strings.Contains(second.Content, "added no new sources") {
		t.Fatalf("second content = %q, want duplicate stop hint", second.Content)
	}
	if !strings.Contains(second.Content, "[source 1]") {
		t.Fatalf("second content = %q, want reused source 1", second.Content)
	}
}

func TestSearchIndexToolFocusedStrategyReturnsSuggestedQueries(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	retrievalCalls := 0
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 5,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			retrievalCalls++
			return rag.RetrieveResult{
				Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{
					testRetrievedChunk("alphafold-outputs", "alphafold/meetings/sdb.md", "AlphaFold outputs include FASTA, A3M, PDB, error JSON, UniProt metadata, monomer, dimer, and ligand workflow notes.", "indexed"),
				},
			}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"AlphaFold","strategy":"focused"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if retrievalCalls != 1 {
		t.Fatalf("retrieval calls = %d, want suggestions not to execute searches", retrievalCalls)
	}
	payload := decodeSearchPayload(t, result.Payload)
	got := strings.Join(payload.SuggestedQueries, "\n")
	for _, want := range []string{"AlphaFold A3M PDB", "UniProt", "monomer"} {
		if !strings.Contains(got, want) {
			t.Fatalf("suggested queries = %+v, want %q", payload.SuggestedQueries, want)
		}
	}
	if !strings.Contains(result.Content, "Suggested follow-up queries:") {
		t.Fatalf("content = %q, want suggested query section", result.Content)
	}
}

func TestSearchIndexToolBroadStrategyExpandsFromFirstPassSources(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	registry := NewCitationRegistry()
	seenQueries := []string{}
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths:    paths,
		Store:    manifest.EmptyStore{},
		Registry: registry,
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 5,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			seenQueries = append(seenQueries, opts.Question)
			chunk := testRetrievedChunk("alphafold-outputs", "alphafold/meetings/sdb.md", "AlphaFold outputs include FASTA, A3M, PDB, and error JSON. UniProt appears in metadata.", "indexed")
			if strings.Contains(opts.Question, "A3M") || strings.Contains(opts.Question, "PDB") {
				chunk = testRetrievedChunk("alphafold-workflow", "alphafold/meetings/sdb.md", "The folding service has monomer, dimer, and ligand workflow notes.", "indexed")
			}
			return rag.RetrieveResult{
				Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{chunk},
			}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"AlphaFold folding","strategy":"broad"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if len(seenQueries) < 2 {
		t.Fatalf("seen queries = %+v, want expansion query", seenQueries)
	}
	if registry.SourceCount() != 2 {
		t.Fatalf("source count = %d, want deduped seed plus expansion sources", registry.SourceCount())
	}
	payload := decodeSearchPayload(t, result.Payload)
	if payload.Strategy != "broad" || len(payload.ExpansionQueries) == 0 {
		t.Fatalf("payload = %+v, want broad strategy with expansion queries", payload)
	}
	if payload.NewSources != 2 || payload.DuplicateSources != 0 || payload.UniqueDocuments != 1 || payload.RetrievalCalls < 2 {
		t.Fatalf("payload metrics = new %d dup %d docs %d retrievals %d, want useful broad metrics", payload.NewSources, payload.DuplicateSources, payload.UniqueDocuments, payload.RetrievalCalls)
	}
	if len(payload.Sources) != 2 {
		t.Fatalf("sources = %+v, want two fused sources", payload.Sources)
	}
	if !strings.Contains(result.Content, "Broad search expansion queries:") {
		t.Fatalf("content = %q, want broad strategy debug output", result.Content)
	}
}

func TestSearchIndexToolMaxExpansionQueriesDisablesBroadExpansion(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	retrievalCalls := 0
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths:               paths,
		Store:               manifest.EmptyStore{},
		MaxExpansionQueries: -1,
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 5,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			retrievalCalls++
			return rag.RetrieveResult{
				Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{testRetrievedChunk("alphafold-outputs", "alphafold/meetings/sdb.md", "AlphaFold outputs include A3M, PDB, error JSON, UniProt, monomer, dimer, and ligand.", "indexed")},
			}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"AlphaFold","strategy":"broad"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	payload := decodeSearchPayload(t, result.Payload)
	if retrievalCalls != 1 || payload.RetrievalCalls != 1 {
		t.Fatalf("retrieval calls = %d payload=%d, want seed search only", retrievalCalls, payload.RetrievalCalls)
	}
	if len(payload.ExpansionQueries) != 0 {
		t.Fatalf("expansion queries = %+v, want disabled expansion", payload.ExpansionQueries)
	}
}

func TestSearchIndexToolStopsBroadExpansionAtRetrievalBudget(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	retrievalCalls := 0
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths:               paths,
		Store:               manifest.EmptyStore{},
		MaxExpansionQueries: 2,
		MaxRetrievalCalls:   2,
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 5,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			retrievalCalls++
			return rag.RetrieveResult{
				Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{
					testRetrievedChunk("alphafold-outputs", "alphafold/meetings/sdb.md", "AlphaFold outputs include A3M, PDB, error JSON, UniProt, monomer, dimer, and ligand.", "indexed"),
				},
			}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"AlphaFold","strategy":"broad"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	payload := decodeSearchPayload(t, result.Payload)
	if retrievalCalls != 2 || payload.RetrievalCalls != 2 {
		t.Fatalf("retrieval calls = %d payload=%d, want seed plus one expansion", retrievalCalls, payload.RetrievalCalls)
	}
	if len(payload.ExpansionQueries) != 1 {
		t.Fatalf("expansion queries = %+v, want one executed expansion", payload.ExpansionQueries)
	}
	if len(payload.Warnings) != 1 || !strings.Contains(payload.Warnings[0], "max retrieval calls") {
		t.Fatalf("warnings = %+v, want retrieval budget warning", payload.Warnings)
	}
}

func TestSearchIndexToolBroadStrategyRejectsTranscriptFragmentsFromAlphaFoldTrace(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths:               paths,
		Store:               manifest.EmptyStore{},
		MaxExpansionQueries: 3,
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 5,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			chunk := testRetrievedChunk("alphafold-noisy", "alphafold/meetings/sdb.md", `So these two folders here which are shared completed and in progress.
For example amaranthus is species. Outputs include FASTA, A3M, PDB, error JSON and NCBI tax IDs.
UniProt metadata tracks monomer, dimer, and ligand workflow notes.`, "indexed")
			return rag.RetrieveResult{
				Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{chunk},
			}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"AlphaFold","strategy":"broad"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	payload := decodeSearchPayload(t, result.Payload)
	got := strings.ToLower(strings.Join(payload.ExpansionQueries, "\n"))
	for _, bad := range []string{"these two folders here", "for example", "amaranthus", "species"} {
		if strings.Contains(got, bad) {
			t.Fatalf("expansion queries = %+v, leaked transcript fragment %q", payload.ExpansionQueries, bad)
		}
	}
	for _, want := range []string{"a3m", "pdb", "json", "ncbi"} {
		if !strings.Contains(got, want) {
			t.Fatalf("expansion queries = %+v, want %q", payload.ExpansionQueries, want)
		}
	}
}

func TestSearchIndexToolDefaultsBroadStrategyForBroadUserQuestion(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	seenQueries := []string{}
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths:        paths,
		Store:        manifest.EmptyStore{},
		UserQuestion: "summarize anything related to AlphaFold folding",
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 5,
		},
		MaxExpansionQueries: 1,
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			seenQueries = append(seenQueries, opts.Question)
			return rag.RetrieveResult{
				Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{testRetrievedChunk("alphafold-outputs", "alphafold/meetings/sdb.md", "AlphaFold outputs include A3M, PDB, error JSON, UniProt, monomer, dimer, and ligand.", "indexed")},
			}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"AlphaFold"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	payload := decodeSearchPayload(t, result.Payload)
	if payload.Strategy != "broad" {
		t.Fatalf("strategy = %q, want broad", payload.Strategy)
	}
	if len(payload.ExpansionQueries) != 1 {
		t.Fatalf("expansion queries = %+v, want one configured expansion", payload.ExpansionQueries)
	}
	if len(seenQueries) != 2 {
		t.Fatalf("seen queries = %+v, want seed plus one expansion", seenQueries)
	}
}

func TestBroadExpansionDoesNotRunHeuristicPlannerOnExpansionQueries(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	configureCalls := 0
	var expansionPlannerModes []rag.QueryPlannerMode
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode:             rag.RetrievalModeLexical,
			TopK:             5,
			QueryPlannerMode: rag.QueryPlannerModeHeuristic,
		},
		ConfigureQueryPlanner: func(opts *rag.RetrieveOptions) error {
			configureCalls++
			opts.QueryPlannerMode = rag.QueryPlannerModeHeuristic
			return nil
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			if opts.Question != "AlphaFold folding" {
				expansionPlannerModes = append(expansionPlannerModes, opts.QueryPlannerMode)
			}
			chunk := testRetrievedChunk("seed", "alphafold/meetings/sdb.md", "AlphaFold outputs include A3M, PDB, error JSON, UniProt, monomer, dimer, and ligand.", "indexed")
			return rag.RetrieveResult{
				Plan:   rag.QueryPlan{Mode: string(opts.QueryPlannerMode), Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{chunk},
			}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"AlphaFold folding","strategy":"broad"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if configureCalls != 1 {
		t.Fatalf("ConfigureQueryPlanner calls = %d, want seed query only", configureCalls)
	}
	if len(expansionPlannerModes) == 0 {
		t.Fatal("expansion planner modes empty, want expansion retrievals")
	}
	for _, mode := range expansionPlannerModes {
		if mode != rag.QueryPlannerModeNone {
			t.Fatalf("expansion planner mode = %q, want none", mode)
		}
	}
}

func TestSearchIndexToolNormalizesDuplicateQueryTerms(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	var seenQuery string
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			seenQuery = opts.Question
			return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"protein folding folding"}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if seenQuery != "protein folding" {
		t.Fatalf("query = %q, want duplicate term removed", seenQuery)
	}
}

func TestMergeBroadSearchResultsInterleavesExpansionEvidence(t *testing.T) {
	merged := mergeBroadSearchResults([]rag.RetrieveResult{
		{
			Plan: rag.QueryPlan{Mode: "none", Queries: []string{"seed"}},
			Chunks: []rag.RetrievedChunk{
				testRetrievedChunk("seed-1", "seed.md", "seed one", "indexed"),
				testRetrievedChunk("seed-2", "seed.md", "seed two", "indexed"),
			},
		},
		{
			Plan: rag.QueryPlan{Mode: "none", Queries: []string{"expansion"}},
			Chunks: []rag.RetrievedChunk{
				testRetrievedChunk("expansion-1", "expansion.md", "expansion one", "indexed"),
			},
		},
	}, 2)
	if len(merged.Chunks) != 2 {
		t.Fatalf("chunks = %+v, want two", merged.Chunks)
	}
	if merged.Chunks[0].Chunk.ID != "seed-1" || merged.Chunks[1].Chunk.ID != "expansion-1" {
		t.Fatalf("chunks = %+v, want interleaved seed and expansion", merged.Chunks)
	}
}

func TestSearchIndexToolContentUsesChunkTextNotIndexedText(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			return rag.RetrieveResult{
				Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{testRetrievedChunk("chunk-1", "README.md", "raw chunk text", "Document: hidden")},
			}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"hello"}`)
	if !strings.Contains(result.Content, "raw chunk text") {
		t.Fatalf("content = %q, want raw chunk text", result.Content)
	}
	if strings.Contains(result.Content, "Document: hidden") || strings.Contains(string(result.Payload), "Document: hidden") {
		t.Fatalf("tool output leaked IndexedText: content=%q payload=%s", result.Content, string(result.Payload))
	}
}

func TestSearchIndexToolPayloadIsValidJSON(t *testing.T) {
	paths := searchToolTestPaths(t, false)
	tool := NewSearchIndexTool(SearchToolOptions{
		Paths: paths,
		Store: manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
		},
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			return rag.RetrieveResult{
				Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{testRetrievedChunk("chunk-1", "README.md", "raw chunk text", "indexed text")},
			}, nil
		},
	})
	result := executeSearchTool(t, tool, `{"query":"hello"}`)
	if !json.Valid(result.Payload) {
		t.Fatalf("payload is invalid JSON: %s", string(result.Payload))
	}
	payload := decodeSearchPayload(t, result.Payload)
	if !payload.Success || len(payload.Sources) != 1 || payload.Sources[0].Text != "raw chunk text" {
		t.Fatalf("payload = %+v, want source payload", payload)
	}
}

func executeSearchTool(t *testing.T, tool falken.Tool, args string) falken.ToolExecutionResult {
	t.Helper()
	result, err := tool.Execute(context.Background(), falken.ToolInvocation{
		CallID:    "call-1",
		Name:      SearchIndexToolName,
		Arguments: json.RawMessage(args),
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	return result
}

func searchToolTestPaths(t *testing.T, withVector bool) config.Paths {
	t.Helper()
	state := filepath.Join(t.TempDir(), ".falkengo")
	paths, err := config.ResolvePaths(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.ManifestPath, []byte("manifest"), 0o644); err != nil {
		t.Fatal(err)
	}
	if withVector {
		if err := os.MkdirAll(paths.VecgoPath, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	return paths
}

func decodeSearchPayload(t *testing.T, raw json.RawMessage) searchToolPayload {
	t.Helper()
	var payload searchToolPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("decode payload: %v\n%s", err, string(raw))
	}
	return payload
}

type fakeAgentAskEmbedder struct{}

func (fakeAgentAskEmbedder) EmbedText(context.Context, string) (llm.Embedding, error) {
	return llm.Embedding{Model: "fake", Vector: []float32{1, 0}}, nil
}
