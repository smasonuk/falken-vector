package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/agentask"
	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/smasonuk/falken-vector/internal/retrievaleval"
)

func TestAskWithoutAgentUsesLegacyLLMPath(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAskForCitationTest(t, func(opts rag.AskOptions) rag.AskResult {
		return rag.AskResult{Answer: "legacy answer", Sources: []rag.SourceChunk{{SourceNumber: 1, Path: "README.md"}}}
	})
	defer restore()
	oldRunAgentAsk := runAgentAsk
	oldNewAgentLLM := newCLIAgentLLM
	runAgentAsk = func(context.Context, agentask.Options) (agentask.Result, error) {
		t.Fatal("runAgentAsk should not be called for legacy ask")
		return agentask.Result{}, nil
	}
	newCLIAgentLLM = func() (falken.LLM, error) {
		t.Fatal("newCLIAgentLLM should not be called for legacy ask")
		return noopAgentLLM{}, nil
	}
	defer func() {
		runAgentAsk = oldRunAgentAsk
		newCLIAgentLLM = oldNewAgentLLM
	}()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "legacy answer") {
		t.Fatalf("output = %q, want legacy answer", out.String())
	}
}

func TestAskAgentCallsRunnerWithoutPreRetrieval(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.RetrieveWithPlan == nil {
			t.Fatal("RetrieveWithPlan seam was not passed to agentask")
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()
	oldRetrieve := retrieveWithPlan
	retrieveWithPlan = func(context.Context, manifest.Store, rag.RetrieveOptions) (rag.RetrieveResult, error) {
		t.Fatal("retrieveWithPlan should not be called before runAgentAsk in agent mode")
		return rag.RetrieveResult{}, nil
	}
	defer func() { retrieveWithPlan = oldRetrieve }()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "agent answer") {
		t.Fatalf("output = %q, want agent answer", out.String())
	}
}

func TestAskAgentPassesRetrievalModeAndSourceFilters(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.RetrievalDefaults.Mode != rag.RetrievalModeLexical {
			t.Fatalf("mode = %q, want lexical", opts.RetrievalDefaults.Mode)
		}
		if opts.RetrievalDefaults.SourceFilter.IsEmpty() || len(opts.RetrievalDefaults.SourceFilter.IncludeGlobs) != 1 || opts.RetrievalDefaults.SourceFilter.IncludeGlobs[0] != "internal/**" {
			t.Fatalf("source filter = %+v, want include restriction", opts.RetrievalDefaults.SourceFilter)
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--include", "internal/**"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentCitationFlagsMapToPolicy(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	var seen []rag.CitationPolicy
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		seen = append(seen, opts.CitationPolicy)
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	for _, args := range [][]string{
		{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical"},
		{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--no-citation-retry"},
		{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--no-citation-validation"},
	} {
		cmd := NewRootCommand()
		cmd.SetArgs(args)
		if err := cmd.Execute(); err != nil {
			t.Fatalf("Execute %v: %v", args, err)
		}
	}
	want := []rag.CitationPolicy{rag.CitationPolicyValidateAndRetry, rag.CitationPolicyValidateOnly, rag.CitationPolicyOff}
	if len(seen) != len(want) {
		t.Fatalf("policies = %+v, want %+v", seen, want)
	}
	for i := range want {
		if seen[i] != want[i] {
			t.Fatalf("policies = %+v, want %+v", seen, want)
		}
	}
}

func TestAskAgentOpenSourceUsesAgentSources(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer:  "agent answer",
			Sources: []rag.SourceChunk{{SourceNumber: 2, Path: "internal/agentask/run.go", StartLine: 44, EndLine: 55}},
		}
	})
	defer restore()
	oldOpener := newSourceOpener
	opener := &recordingSourceOpener{}
	newSourceOpener = func(func(string) string) SourceOpener { return opener }
	defer func() { newSourceOpener = oldOpener }()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--open-source", "2"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if opener.path != "internal/agentask/run.go" || opener.line != 44 {
		t.Fatalf("opener = %+v, want agent source", opener)
	}
}

func TestAskAgentOpenSourceOutOfRangeReportsAvailableSources(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "one.go"},
				{SourceNumber: 2, Path: "two.go"},
			},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--open-source", "9"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "available 2 sources") {
		t.Fatalf("Execute error = %v, want available source count", err)
	}
}

func TestAskAgentShowToolsAndWarnings(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.Events == nil {
			t.Fatal("Events sink was not configured")
		}
		opts.Events(falken.Event{ToolCall: &falken.ToolCall{
			ID:        "call-1",
			Name:      "search_index",
			Arguments: json.RawMessage(`{"query":" hello ","top_k":8}`),
		}})
		opts.Events(falken.Event{ToolResult: &falken.ToolResult{
			CallID: "call-1",
			Name:   "search_index",
			Payload: json.RawMessage(`{
				"success": true,
				"status": "ok",
				"query": "hello",
				"top_k": 8,
				"query_plan": {"queries": ["hello", "hello source"]},
				"sources": [{"text": "hidden source text"}]
			}`),
		}})
		return agentask.Result{
			Answer:           "agent answer",
			ToolCalls:        []string{"search_index"},
			CitationWarnings: []string{"answer did not cite any source"},
			CoverageWarnings: []string{"coverage nudge skipped: search call limit reached"},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var errOut bytes.Buffer
	cmd.SetErr(&errOut)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--show-agent-tools"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	stderr := errOut.String()
	if !strings.Contains(stderr, `agent tool call: search_index {"query":" hello ","top_k":8}`) {
		t.Fatalf("stderr = %q, want live tool call with arguments", stderr)
	}
	if !strings.Contains(stderr, `agent tool result: search_index ok, query="hello", top_k=8, sources=1`) || !strings.Contains(stderr, "agent search query plan:") {
		t.Fatalf("stderr = %q, want compact tool result", stderr)
	}
	if strings.Contains(stderr, "hidden source text") {
		t.Fatalf("stderr = %q, leaked source text", stderr)
	}
	if !strings.Contains(stderr, "agent coverage nudge skipped: search call limit reached") {
		t.Fatalf("stderr = %q, want coverage warning in debug output", stderr)
	}
	if !strings.Contains(stderr, "warning: answer did not cite any source") {
		t.Fatalf("stderr = %q, want warning", stderr)
	}
	if strings.Count(stderr, "agent tool call: search_index") != 1 {
		t.Fatalf("stderr = %q, want one live tool call line", stderr)
	}
}

func TestAskAgentPassesCoverageAndReadSourceFlags(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.CoverageNudge == nil || !*opts.CoverageNudge {
			t.Fatalf("CoverageNudge = %v, want explicit true", opts.CoverageNudge)
		}
		if opts.MinBroadSearchCalls != 3 {
			t.Fatalf("MinBroadSearchCalls = %d, want 3", opts.MinBroadSearchCalls)
		}
		if opts.MaxCoverageRetries != 2 {
			t.Fatalf("MaxCoverageRetries = %d, want 2", opts.MaxCoverageRetries)
		}
		if !opts.EnableReadSourceTool {
			t.Fatal("EnableReadSourceTool = false, want true")
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--agent-coverage-nudge", "--min-agent-searches", "3", "--max-agent-coverage-retries", "2", "--agent-read-source-tool"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentNoCoverageNudgeFlagDisablesCoverage(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.CoverageNudge == nil || *opts.CoverageNudge {
			t.Fatalf("CoverageNudge = %v, want explicit false", opts.CoverageNudge)
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--no-agent-coverage-nudge"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentCoverageFlagsConflict(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		t.Fatal("runAgentAsk should not be called when coverage flags conflict")
		return agentask.Result{}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--agent-coverage-nudge", "--no-agent-coverage-nudge"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "cannot both be set") {
		t.Fatalf("Execute error = %v, want coverage flag conflict", err)
	}
}

func TestAskAgentResultWithSourcePrintsReference(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer:  "agent answer [source 1].",
			Sources: []rag.SourceChunk{{SourceNumber: 1, Path: "internal/rag/retrieve.go", StartLine: 35, EndLine: 73}},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "Sources:\n[source 1] internal/rag/retrieve.go:35-73") {
		t.Fatalf("output = %q, want source reference", out.String())
	}
}

func TestAskAgentPrintsOnlyCitedSourcesByDefault(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 2].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "uncited.go", StartLine: 1, EndLine: 2},
				{SourceNumber: 2, Path: "cited.go", StartLine: 3, EndLine: 4},
			},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	if !strings.Contains(output, "[source 2] cited.go:3-4") {
		t.Fatalf("output = %q, want cited source", output)
	}
	if strings.Contains(output, "uncited.go") {
		t.Fatalf("output = %q, want uncited source hidden by default", output)
	}
}

func TestAskAgentSourcesAllPrintsRetrievedSources(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 2].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "uncited.go", StartLine: 1, EndLine: 2},
				{SourceNumber: 2, Path: "cited.go", StartLine: 3, EndLine: 4},
			},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "all"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	if !strings.Contains(output, "uncited.go") || !strings.Contains(output, "cited.go") {
		t.Fatalf("output = %q, want all sources", output)
	}
}

func TestAskAgentResultWithoutSourcesOmitsSourcesSection(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if strings.Contains(out.String(), "Sources:") {
		t.Fatalf("output = %q, want no empty Sources section", out.String())
	}
}

func TestAskAgentModelOverrideUsesTestSeam(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--model", "local-test-model", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "agent answer") {
		t.Fatalf("output = %q, want agent answer without real LLM env", out.String())
	}
}

func TestAskAgentDoesNotPreCheckVectorWhenToolCanUseLexical(t *testing.T) {
	state := setupAgentManifestOnly(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.RetrievalDefaults.Mode != rag.RetrievalModeVector {
			t.Fatalf("mode = %q, want vector default passed through", opts.RetrievalDefaults.Mode)
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "agent answer") {
		t.Fatalf("output = %q, want agent answer despite missing vector DB", out.String())
	}
}

func TestQueryAndEvalDoNotUseAgentSeams(t *testing.T) {
	state, dataset := setupEvalCLITest(t)
	restoreQuery := stubQueryCLI(t)
	defer restoreQuery()
	restoreEval := stubEvalCLI(t, retrievaleval.Summary{Cases: 1, EvaluatedCases: 1, TopK: 8})
	defer restoreEval()
	oldRunAgentAsk := runAgentAsk
	oldNewAgentLLM := newCLIAgentLLM
	runAgentAsk = func(context.Context, agentask.Options) (agentask.Result, error) {
		t.Fatal("runAgentAsk should not be called by query or eval retrieval")
		return agentask.Result{}, nil
	}
	newCLIAgentLLM = func() (falken.LLM, error) {
		t.Fatal("newCLIAgentLLM should not be called by query or eval retrieval")
		return noopAgentLLM{}, nil
	}
	defer func() {
		runAgentAsk = oldRunAgentAsk
		newCLIAgentLLM = oldNewAgentLLM
	}()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "query", "hello", "--retrieval", "hybrid", "--reranker", "heuristic", "--query-planner", "heuristic", "--max-subqueries", "3", "--candidate-k", "25"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("query Execute: %v", err)
	}

	cmd = NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "eval", "retrieval", "--dataset", dataset})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("eval Execute: %v", err)
	}
}

func TestAskAgentDoesNotCallLegacyAskWithLLM(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()
	oldAsk := askWithLLM
	askWithLLM = func(context.Context, rag.AskOptions) (rag.AskResult, error) {
		t.Fatal("askWithLLM should not be called in agent mode")
		return rag.AskResult{}, nil
	}
	defer func() { askWithLLM = oldAsk }()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestLegacyAskOpenSourceIgnoresAgentSources(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restoreAsk := stubAskForCitationTest(t, func(opts rag.AskOptions) rag.AskResult {
		return rag.AskResult{Answer: "answer", Sources: []rag.SourceChunk{{SourceNumber: 99, Path: "agent.go", StartLine: 99}}}
	})
	defer restoreAsk()
	oldRunAgentAsk := runAgentAsk
	runAgentAsk = func(context.Context, agentask.Options) (agentask.Result, error) {
		return agentask.Result{Sources: []rag.SourceChunk{{SourceNumber: 1, Path: "agent.go", StartLine: 99}}}, nil
	}
	defer func() { runAgentAsk = oldRunAgentAsk }()
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
		t.Fatalf("opener = %+v, want legacy retrieved source", opener)
	}
}

func stubAgentAskCLI(t *testing.T, run func(agentask.Options) agentask.Result) func() {
	t.Helper()
	oldNewAgentLLM := newCLIAgentLLM
	oldNewAgentLLMWithModel := newCLIAgentLLMWithModel
	oldRunAgentAsk := runAgentAsk
	newCLIAgentLLM = func() (falken.LLM, error) {
		return noopAgentLLM{}, nil
	}
	newCLIAgentLLMWithModel = func(string) (falken.LLM, error) {
		return noopAgentLLM{}, nil
	}
	runAgentAsk = func(_ context.Context, opts agentask.Options) (agentask.Result, error) {
		return run(opts), nil
	}
	return func() {
		newCLIAgentLLM = oldNewAgentLLM
		newCLIAgentLLMWithModel = oldNewAgentLLMWithModel
		runAgentAsk = oldRunAgentAsk
	}
}

func setupAgentManifestOnly(t *testing.T) string {
	t.Helper()
	state, _ := setupEvalCLITest(t)
	paths, err := config.ResolvePaths(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(paths.VecgoPath); err != nil {
		t.Fatal(err)
	}
	return state
}

type noopAgentLLM struct{}

func (noopAgentLLM) Complete(context.Context, falken.CompletionRequest) (falken.CompletionResponse, error) {
	return falken.CompletionResponse{FinishReason: falken.FinishReasonStop}, nil
}

var _ llm.Client = fakeCLIClient{}
