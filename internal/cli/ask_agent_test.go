package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
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

func TestAskWithoutAgentDefaultsToCitedSources(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAskForCitationTest(t, func(opts rag.AskOptions) rag.AskResult {
		return rag.AskResult{
			Answer: "legacy answer [source 2].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "available.go", StartLine: 1, EndLine: 2},
				{SourceNumber: 2, Path: "cited.go", StartLine: 3, EndLine: 4},
			},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	if !strings.Contains(output, "Sources:\n[source 2] cited.go:3-4") {
		t.Fatalf("output = %q, want cited source", output)
	}
	if strings.Contains(output, "available.go") || strings.Contains(output, "Sources available to the agent:") {
		t.Fatalf("output = %q, want non-agent default to stay cited", output)
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
				"sources": [{"text": "hidden source text"}],
				"warnings": ["top_k raised to configured floor 12"]
			}`),
		}})
		opts.Events(falken.Event{ToolResult: &falken.ToolResult{
			CallID: "call-2",
			Name:   "search_index",
			Payload: json.RawMessage(`{
				"success": true,
				"status": "ok",
				"query": "hello again",
				"top_k": 8,
				"sources": [],
				"warnings": ["top_k raised to configured floor 12", "different warning"]
			}`),
		}})
		opts.Events(falken.Event{Type: falken.EventThought, Text: "agent thin-source nudge: expanding [source 1]"})
		opts.Events(falken.Event{ToolCall: &falken.ToolCall{
			ID:        "call-read",
			Name:      "read_index_source",
			Arguments: json.RawMessage(`{"source_number":1,"context_lines":20}`),
		}})
		opts.Events(falken.Event{ToolResult: &falken.ToolResult{
			CallID: "call-read",
			Name:   "read_index_source",
			Payload: json.RawMessage(`{
				"success": true,
				"status": "ok",
				"source_number": 1,
				"path": "alphafold.md",
				"start_line": 1,
				"end_line": 20
			}`),
		}})
		return agentask.Result{
			Answer:             "agent answer",
			ToolCalls:          []string{"search_index"},
			CitationWarnings:   []string{"answer did not cite any source"},
			CoverageWarnings:   []string{"coverage nudge skipped: search call limit reached"},
			ThinSourceWarnings: []string{"thin-source nudge: expanding [source 1]"},
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
	if !strings.Contains(stderr, "agent thin-source nudge: expanding [source 1]") {
		t.Fatalf("stderr = %q, want thin-source debug output", stderr)
	}
	nudgePos := strings.Index(stderr, "agent thin-source nudge: expanding [source 1]")
	readPos := strings.Index(stderr, "agent tool call: read_index_source")
	if nudgePos < 0 || readPos < 0 || nudgePos > readPos {
		t.Fatalf("stderr = %q, want thin-source nudge before read_index_source call", stderr)
	}
	if strings.Count(stderr, "agent thin-source nudge: expanding [source 1]") != 1 {
		t.Fatalf("stderr = %q, want thin-source nudge printed once", stderr)
	}
	if !strings.Contains(stderr, "warning: answer did not cite any source") {
		t.Fatalf("stderr = %q, want warning", stderr)
	}
	if strings.Count(stderr, "agent tool call: search_index") != 1 {
		t.Fatalf("stderr = %q, want one live tool call line", stderr)
	}
	if strings.Count(stderr, "top_k raised to configured floor 12") != 1 {
		t.Fatalf("stderr = %q, want duplicate warning suppressed", stderr)
	}
	if !strings.Contains(stderr, "different warning") {
		t.Fatalf("stderr = %q, want distinct warning printed", stderr)
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
		if opts.ReadSourceOverlapPolicy != agentask.ReadSourceOverlapMerge {
			t.Fatalf("ReadSourceOverlapPolicy = %q, want merge", opts.ReadSourceOverlapPolicy)
		}
		if opts.MaxMergedReadSourceLines != 120 {
			t.Fatalf("MaxMergedReadSourceLines = %d, want 120", opts.MaxMergedReadSourceLines)
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--agent-coverage-nudge", "--min-agent-searches", "3", "--max-agent-coverage-retries", "2", "--agent-read-source-tool", "--read-source-overlap-policy", "merge", "--max-merged-read-source-lines", "120"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentPassesDefaultMaxMergedReadSourceLines(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.MaxMergedReadSourceLines != agentask.DefaultMaxMergedReadSourceLines {
			t.Fatalf("MaxMergedReadSourceLines = %d, want CLI default %d", opts.MaxMergedReadSourceLines, agentask.DefaultMaxMergedReadSourceLines)
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentReadSourceOverlapPolicyInvalid(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		t.Fatal("runAgentAsk should not be called when overlap policy is invalid")
		return agentask.Result{}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--read-source-overlap-policy", "bogus"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--read-source-overlap-policy must be skip, merge, or allow") {
		t.Fatalf("Execute error = %v, want overlap policy validation", err)
	}
}

func TestAskAgentMaxMergedReadSourceLinesInvalid(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		t.Fatal("runAgentAsk should not be called when max merged read-source lines is invalid")
		return agentask.Result{}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--max-merged-read-source-lines", "-1"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--max-merged-read-source-lines must be >= 0") {
		t.Fatalf("Execute error = %v, want max merged read-source validation", err)
	}
}

func TestAskAgentPassesDocumentPromotionControls(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if !opts.DisableDocumentPromotion {
			t.Fatal("DisableDocumentPromotion = false, want true from --no-agent-document-promotion")
		}
		if opts.MaxDocumentReadLines != 123 || opts.MaxDocumentReadTokens != 456 || opts.MaxDocumentReads != 1 {
			t.Fatalf("document caps = lines %d tokens %d reads %d, want 123/456/1", opts.MaxDocumentReadLines, opts.MaxDocumentReadTokens, opts.MaxDocumentReads)
		}
		return agentask.Result{Answer: "ok"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{
		"--state-dir", state,
		"ask", "hello",
		"--agent",
		"--retrieval", "lexical",
		"--no-agent-document-promotion",
		"--max-agent-document-lines", "123",
		"--max-agent-document-tokens", "456",
		"--max-agent-document-reads", "1",
	})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestReadSourceOverlapPolicyFromFlag(t *testing.T) {
	tests := []struct {
		name     string
		value    string
		changed  bool
		question string
		want     agentask.ReadSourceOverlapPolicy
		wantErr  bool
	}{
		{name: "default narrow skip", value: "skip", question: "Where is citation validation implemented?", want: agentask.ReadSourceOverlapSkip},
		{name: "default broad merge", value: "skip", question: "summarize anything related to AlphaFold", want: agentask.ReadSourceOverlapMerge},
		{name: "explicit skip broad", value: "skip", changed: true, question: "summarize anything related to AlphaFold", want: agentask.ReadSourceOverlapSkip},
		{name: "explicit merge", value: "merge", changed: true, want: agentask.ReadSourceOverlapMerge},
		{name: "explicit allow", value: "allow", changed: true, want: agentask.ReadSourceOverlapAllow},
		{name: "invalid", value: "bogus", changed: true, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := readSourceOverlapPolicyFromFlag(tt.value, tt.changed, tt.question)
			if tt.wantErr {
				if err == nil {
					t.Fatal("readSourceOverlapPolicyFromFlag succeeded, want error")
				}
				return
			}
			if err != nil || got != tt.want {
				t.Fatalf("readSourceOverlapPolicyFromFlag = %q, %v; want %q, nil", got, err, tt.want)
			}
		})
	}
}

func TestAskAgentPassesExpansionQueryBudget(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.MaxBroadExpansionQueries != 4 {
			t.Fatalf("MaxBroadExpansionQueries = %d, want 4", opts.MaxBroadExpansionQueries)
		}
		if opts.MaxRetrievalCalls != 16 {
			t.Fatalf("MaxRetrievalCalls = %d, want 16", opts.MaxRetrievalCalls)
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--max-agent-expansion-queries", "4", "--max-agent-retrievals", "16"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentExpansionQueryBudgetZeroDisablesExpansion(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.MaxBroadExpansionQueries != -1 {
			t.Fatalf("MaxBroadExpansionQueries = %d, want internal disable marker", opts.MaxBroadExpansionQueries)
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--max-agent-expansion-queries", "0"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentAutoEnablesReadSourceForBroadQuestion(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if !opts.EnableReadSourceTool {
			t.Fatal("EnableReadSourceTool = false, want broad question auto-enable")
		}
		if opts.ReadSourceOverlapPolicy != agentask.ReadSourceOverlapMerge {
			t.Fatalf("ReadSourceOverlapPolicy = %q, want broad default merge", opts.ReadSourceOverlapPolicy)
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "summarize anything related to AlphaFold folding", "--agent", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentExplicitReadSourceOverlapSkipOverridesBroadDefault(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.ReadSourceOverlapPolicy != agentask.ReadSourceOverlapSkip {
			t.Fatalf("ReadSourceOverlapPolicy = %q, want explicit skip", opts.ReadSourceOverlapPolicy)
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "summarize anything related to AlphaFold folding", "--agent", "--retrieval", "lexical", "--read-source-overlap-policy", "skip"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentNarrowQuestionDefaultsReadSourceOverlapSkip(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.ReadSourceOverlapPolicy != agentask.ReadSourceOverlapSkip {
			t.Fatalf("ReadSourceOverlapPolicy = %q, want narrow default skip", opts.ReadSourceOverlapPolicy)
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "Where is citation validation implemented?", "--agent", "--retrieval", "lexical"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentNoReadSourceFlagDisablesAutoReadSource(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.EnableReadSourceTool {
			t.Fatal("EnableReadSourceTool = true, want explicit no flag to disable")
		}
		return agentask.Result{Answer: "agent answer"}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "summarize anything related to AlphaFold folding", "--agent", "--retrieval", "lexical", "--no-agent-read-source-tool"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestAskAgentReadSourceFlagsConflict(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		t.Fatal("runAgentAsk should not be called when read-source flags conflict")
		return agentask.Result{}
	})
	defer restore()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--agent-read-source-tool", "--no-agent-read-source-tool"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "cannot both be set") {
		t.Fatalf("Execute error = %v, want read-source flag conflict", err)
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
	output := out.String()
	for _, want := range []string{
		"Sources cited:",
		"[source 1] internal/rag/retrieve.go:35-73",
		"Source audit:",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, want %q", output, want)
		}
	}
	if strings.Contains(output, "Other sources available to the agent:") || strings.Contains(output, "(cited)") {
		t.Fatalf("output = %q, want no repeated cited source in both mode", output)
	}
}

func TestAskAgentDefaultsToBothSourceMode(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 2].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "available.go", StartLine: 1, EndLine: 2},
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
	for _, want := range []string{
		"Sources cited:",
		"[source 2] cited.go:3-4",
		"Other sources available to the agent:",
		"[source 1] available.go:1-2",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, want %q", output, want)
		}
	}
	if strings.Contains(output, "[source 2] cited.go:3-4  (cited)") {
		t.Fatalf("output = %q, want cited source only in cited section", output)
	}
}

func TestAskAgentSourcesCitedPrintsOnlyCitedSources(t *testing.T) {
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
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "cited"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	if !strings.Contains(output, "[source 2] cited.go:3-4") {
		t.Fatalf("output = %q, want source reference", out.String())
	}
	if strings.Contains(output, "uncited.go") || strings.Contains(output, "Source audit:") {
		t.Fatalf("output = %q, want only cited source", output)
	}
}

func TestAskAgentSourcesBothPrintsCitedAndAllSources(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 2].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "available.go", StartLine: 1, EndLine: 2},
				{SourceNumber: 2, Path: "cited.go", StartLine: 3, EndLine: 4},
			},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	for _, want := range []string{
		"Sources cited:",
		"[source 2] cited.go:3-4",
		"Other sources available to the agent:",
		"[source 1] available.go:1-2",
		"Source audit:",
		"- available to agent: 2",
		"- cited in answer: 1",
		"- uncited: 1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, want %q", output, want)
		}
	}
	if strings.Contains(output, "[source 2] cited.go:3-4  (cited)") {
		t.Fatalf("output = %q, want cited source only in cited section for both mode", output)
	}
	if strings.Contains(output, "search tool calls:") || strings.Contains(output, "retrieval calls:") {
		t.Fatalf("output = %q, want unavailable audit metrics omitted", output)
	}
}

func TestAskAgentSourcesBothMarksOtherSourcesCoveredByCitedExpansion(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 11].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 10, Path: "alphafold/meetings/sdb.md", StartLine: 21, EndLine: 26},
				{SourceNumber: 11, Path: "alphafold/meetings/sdb.md", StartLine: 1, EndLine: 33},
				{SourceNumber: 12, Path: "alphafold/meetings/sdb.md", StartLine: 30, EndLine: 40},
				{SourceNumber: 13, Path: "other.md", StartLine: 21, EndLine: 26},
			},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	if !strings.Contains(output, "[source 10] alphafold/meetings/sdb.md:21-26  (covered by cited source 11)") {
		t.Fatalf("output = %q, want covered-source marker", output)
	}
	if strings.Contains(output, "[source 12] alphafold/meetings/sdb.md:30-40  (covered") {
		t.Fatalf("output = %q, partial overlap should not be marked covered", output)
	}
	if strings.Contains(output, "[source 13] other.md:21-26  (covered") {
		t.Fatalf("output = %q, different document should not be marked covered", output)
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
	for _, want := range []string{
		"Sources available to the agent:",
		"[source 1] uncited.go:1-2",
		"[source 2] cited.go:3-4  (cited)",
		"Source audit:",
		"- available to agent: 2",
		"- cited in answer: 1",
		"- uncited: 1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, want %q", output, want)
		}
	}
	if strings.Contains(output, "Sources cited:") {
		t.Fatalf("output = %q, did not want cited section for --sources all", output)
	}
}

func TestAskAgentSourceAuditPrintsSearchAndRetrievalTotals(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 2] [source 3] [source 4] [source 5].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "one.go", StartLine: 1, EndLine: 2, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "AlphaFold"}},
				{SourceNumber: 2, Path: "two.go", StartLine: 1, EndLine: 2, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "AlphaFold"}},
				{SourceNumber: 3, Path: "three.go", StartLine: 1, EndLine: 2, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "AlphaFold"}},
				{SourceNumber: 4, Path: "four.go", StartLine: 1, EndLine: 2, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "protein folding"}},
				{SourceNumber: 5, Path: "five.go", StartLine: 1, EndLine: 2, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "protein folding"}},
			},
			SearchToolCalls: 2,
			RetrievalCalls:  5,
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "all", "--show-agent-tools"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	for _, want := range []string{
		"Source audit:",
		"- available to agent: 5",
		"- cited in answer: 4",
		"- uncited: 1",
		"- search tool calls: 2",
		"- retrieval calls: 5",
		"- introduced by query:",
		"  - AlphaFold: 3",
		"  - protein folding: 2",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, want %q", output, want)
		}
	}
}

func TestAskAgentSourceAuditPrintsReadSourceOverlapTotals(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 1].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "one.go", StartLine: 1, EndLine: 100},
				{SourceNumber: 2, Path: "one.go", StartLine: 80, EndLine: 90},
			},
			ReadSourceCalls:          3,
			ReadSourceOverlapPolicy:  "merge",
			ReadSourceAlreadyCovered: 1,
			ReadSourceMerges:         1,
			ReadSourceMergeTooLarge:  1,
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	for _, want := range []string{
		"- read source calls: 3",
		"- read source overlap policy: merge",
		"- read source already covered: 1",
		"- read source merges: 1",
		"- read source merge too large: 1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, want %q", output, want)
		}
	}
}

func TestAskAgentSourceAuditPrintsReadSourceCallsWithoutOverlapDecisions(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 1].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "one.go", StartLine: 1, EndLine: 100},
			},
			ReadSourceCalls:         2,
			ReadSourceOverlapPolicy: "merge",
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	for _, want := range []string{
		"- read source calls: 2",
		"- read source overlap policy: merge",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, want %q", output, want)
		}
	}
	for _, bad := range []string{
		"read source already covered",
		"read source merges:",
		"read source merge too large",
	} {
		if strings.Contains(output, bad) {
			t.Fatalf("output = %q, did not want %q", output, bad)
		}
	}
}

func TestAskAgentSourceAuditPrintsDocumentReadMetrics(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 1].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "meeting.md", StartLine: 1, EndLine: 206},
				{SourceNumber: 2, Path: "meeting.md", StartLine: 20, EndLine: 25},
			},
			DocumentReadCalls:                1,
			DocumentReadWholeCalls:           1,
			DocumentPromotionAutoReads:       1,
			DocumentPromotionSkippedTooLarge: 1,
			DocumentPromotionPolicy:          "auto",
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	for _, want := range []string{
		"- document read calls: 1",
		"- document read whole calls: 1",
		"- document promotion auto reads: 1",
		"- document promotion skipped too large: 1",
		"- document promotion policy: auto",
		"[source 2] meeting.md:20-25  (covered by cited source 1)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, want %q", output, want)
		}
	}
}

func TestAskAgentSourceAuditOmitsReadSourceLinesWithoutCalls(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer:  "agent answer [source 1].",
			Sources: []rag.SourceChunk{{SourceNumber: 1, Path: "one.go", StartLine: 1, EndLine: 2}},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	if strings.Contains(output, "read source calls") || strings.Contains(output, "read source overlap policy") {
		t.Fatalf("output = %q, want read-source audit lines omitted", output)
	}
}

func TestAskAgentSourcesAllCanPrintSourceProvenance(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 2].",
			Sources: []rag.SourceChunk{
				{
					SourceNumber: 1,
					Path:         "available.go",
					StartLine:    1,
					EndLine:      2,
					Provenance: &rag.SourceProvenance{
						ToolName: agentask.SearchIndexToolName,
						Query:    "AlphaFold",
						Strategy: "broad",
						Rank:     4,
					},
				},
				{SourceNumber: 2, Path: "cited.go", StartLine: 3, EndLine: 4},
			},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "all", "--show-source-provenance"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	if !strings.Contains(output, `introduced by: search_index query="AlphaFold", strategy="broad", rank=4`) {
		t.Fatalf("output = %q, want provenance line", output)
	}
}

func TestAskAgentSourceAuditIncludesQueryCountsWhenShowingTools(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 2].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 1, Path: "one.md", StartLine: 1, EndLine: 2, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "AlphaFold"}},
				{SourceNumber: 2, Path: "two.md", StartLine: 3, EndLine: 4, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "AlphaFold"}},
				{SourceNumber: 3, Path: "three.md", StartLine: 5, EndLine: 6, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "protein folding"}},
			},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both", "--show-agent-tools"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	for _, want := range []string{
		"- introduced by query:",
		"  - AlphaFold: 2",
		"  - protein folding: 1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, want %q", output, want)
		}
	}
}

func TestAskAgentSourceAuditOmitsQueryCountsWithoutDebug(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer: "agent answer [source 1].",
			Sources: []rag.SourceChunk{{
				SourceNumber: 1,
				Path:         "one.md",
				StartLine:    1,
				EndLine:      2,
				Provenance:   &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "AlphaFold"},
			}},
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if strings.Contains(out.String(), "introduced by query") {
		t.Fatalf("output = %q, want no provenance summary without --show-agent-tools", out.String())
	}
}

func TestAskAgentAlphaFoldFlowRegression(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(opts agentask.Options) agentask.Result {
		if opts.ReadSourceOverlapPolicy != agentask.ReadSourceOverlapMerge {
			t.Fatalf("ReadSourceOverlapPolicy = %q, want broad default merge", opts.ReadSourceOverlapPolicy)
		}
		if opts.MaxMergedReadSourceLines != agentask.DefaultMaxMergedReadSourceLines {
			t.Fatalf("MaxMergedReadSourceLines = %d, want CLI default %d", opts.MaxMergedReadSourceLines, agentask.DefaultMaxMergedReadSourceLines)
		}
		opts.Events(falken.Event{ToolCall: &falken.ToolCall{
			ID:        "call-search",
			Name:      "search_index",
			Arguments: json.RawMessage(`{"query":"AlphaFold","strategy":"broad","top_k":8}`),
		}})
		opts.Events(falken.Event{ToolResult: &falken.ToolResult{
			CallID: "call-search",
			Name:   "search_index",
			Payload: json.RawMessage(`{
				"success": true,
				"status": "ok",
				"query": "AlphaFold",
				"top_k": 12,
				"seed_query_plan": {"queries": ["AlphaFold"]},
				"expansion_queries": ["AlphaFold A3M PDB JSON NCBI", "AlphaFold species NCBI", "protein folding PDB error JSON"],
				"sources": [{"number":10},{"number":11},{"number":12}],
				"new_sources": 3,
				"duplicate_sources": 0,
				"unique_documents": 1,
				"retrieval_calls": 3
			}`),
		}})
		opts.Events(falken.Event{ToolCall: &falken.ToolCall{
			ID:        "call-search-2",
			Name:      "search_index",
			Arguments: json.RawMessage(`{"query":"protein folding folding","strategy":"broad","top_k":5}`),
		}})
		opts.Events(falken.Event{ToolResult: &falken.ToolResult{
			CallID: "call-search-2",
			Name:   "search_index",
			Payload: json.RawMessage(`{
				"success": true,
				"status": "ok",
				"query": "protein folding",
				"original_query": "protein folding folding",
				"query_normalized": true,
				"top_k": 12,
				"seed_query_plan": {"queries": ["protein folding"]},
				"expansion_queries": ["protein folding PDB error JSON"],
				"sources": [{"number":12}],
				"new_sources": 1,
				"duplicate_sources": 0,
				"unique_documents": 1,
				"retrieval_calls": 2
			}`),
		}})
		opts.Events(falken.Event{Type: falken.EventThought, Text: "agent thin-source nudge: expanding [source 11]"})
		opts.Events(falken.Event{ToolCall: &falken.ToolCall{
			ID:        "call-read",
			Name:      "read_index_source",
			Arguments: json.RawMessage(`{"source_number":11,"context_lines":20}`),
		}})
		opts.Events(falken.Event{ToolResult: &falken.ToolResult{
			CallID: "call-read",
			Name:   "read_index_source",
			Payload: json.RawMessage(`{
				"success": true,
				"status": "ok",
				"source_number": 11,
				"path": "alphafold/meetings/sdb.md",
				"start_line": 1,
				"end_line": 33
			}`),
		}})
		return agentask.Result{
			Answer: "AlphaFold outputs include A3M, PDB, JSON, NCBI, UniProt metadata, and folding workflow notes [source 11].",
			Sources: []rag.SourceChunk{
				{SourceNumber: 10, Path: "alphafold/meetings/sdb.md", StartLine: 21, EndLine: 26, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "AlphaFold", Strategy: "broad", Rank: 1}},
				{SourceNumber: 11, Path: "alphafold/meetings/sdb.md", StartLine: 1, EndLine: 33, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "AlphaFold", Strategy: "broad", Rank: 2}},
				{SourceNumber: 12, Path: "science_cloud/info/apis.md", StartLine: 103, EndLine: 103, Provenance: &rag.SourceProvenance{ToolName: agentask.SearchIndexToolName, Query: "protein folding", Strategy: "broad", Rank: 3}},
			},
			ThinSourceNudged: true,
			ThinSourceWarnings: []string{
				"thin-source nudge: expanding [source 11]",
			},
			SearchToolCalls:         2,
			RetrievalCalls:          5,
			ReadSourceCalls:         1,
			ReadSourceOverlapPolicy: "merge",
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	var errOut bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&errOut)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "summarize information on alphafold or anything related to folding", "--agent", "--retrieval", "lexical", "--show-agent-tools"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	stderr := errOut.String()
	output := out.String()
	for _, want := range []string{
		"agent broad expansion queries:",
		"  1. AlphaFold A3M PDB JSON NCBI",
		"  2. AlphaFold species NCBI",
		"  3. protein folding PDB error JSON",
		`query="protein folding" (normalized from "protein folding folding")`,
		"top_k=12",
		"agent thin-source nudge: expanding [source 11]",
		"agent tool call: read_index_source",
	} {
		if !strings.Contains(stderr, want) {
			t.Fatalf("stderr = %q, want %q", stderr, want)
		}
	}
	for _, bad := range []string{"these two folders here", "going to it probably", "DVI 000 150", "got any metadata what was used run", "[sources", "[source 2, 3]"} {
		if strings.Contains(stderr, bad) || strings.Contains(output, bad) {
			t.Fatalf("stderr = %q output = %q, leaked noisy fragment %q", stderr, output, bad)
		}
	}
	for _, want := range []string{
		"Sources cited:",
		"[source 11] alphafold/meetings/sdb.md:1-33",
		"Other sources available to the agent:",
		"[source 10] alphafold/meetings/sdb.md:21-26  (covered by cited source 11)",
		"Source audit:",
		"- search tool calls: 2",
		"- retrieval calls: 5",
		"- read source calls: 1",
		"- read source overlap policy: merge",
		"- introduced by query:",
		"  - AlphaFold: 2",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output = %q, want %q", output, want)
		}
	}
	if strings.Contains(output, "Other sources available to the agent:\n[source 11]") {
		t.Fatalf("output = %q, cited source repeated in other section", output)
	}
}

func TestAskAgentNormalizesGroupedCitationsForSourcesAndAudit(t *testing.T) {
	cases := []struct {
		name     string
		answer   string
		wantRefs []string
	}{
		{name: "plural comma", answer: "The artifacts are FASTA and PDB [sources 2, 3].", wantRefs: []string{"[source 2]", "[source 3]"}},
		{name: "singular comma", answer: "The artifacts are FASTA and PDB [source 2, 3].", wantRefs: []string{"[source 2]", "[source 3]"}},
		{name: "and", answer: "The artifacts are FASTA and PDB [source 2 and 3].", wantRefs: []string{"[source 2]", "[source 3]"}},
		{name: "mixed", answer: "The pipeline is described here [source 13; sources 4, 19].", wantRefs: []string{"[source 13]", "[source 4]", "[source 19]"}},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			state, _ := setupEvalCLITest(t)
			restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
				return agentask.Result{
					Answer:  tt.answer,
					Sources: numberedTestSources(20),
				}
			})
			defer restore()

			cmd := NewRootCommand()
			var out bytes.Buffer
			cmd.SetOut(&out)
			cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both"})
			if err := cmd.Execute(); err != nil {
				t.Fatalf("Execute: %v", err)
			}
			output := out.String()
			if strings.Contains(output, "[sources") || strings.Contains(output, "[source 2, 3]") || strings.Contains(output, "[source 2 and 3]") {
				t.Fatalf("output = %q, grouped citation was not normalized", output)
			}
			for _, want := range tt.wantRefs {
				if !strings.Contains(output, want) {
					t.Fatalf("output = %q, want normalized ref %q", output, want)
				}
			}
			if !strings.Contains(output, "Sources cited:") || !strings.Contains(output, "Other sources available to the agent:") || !strings.Contains(output, "Source audit:") {
				t.Fatalf("output = %q, want cited/other/audit sections", output)
			}
			if !strings.Contains(output, "- available to agent: 20") {
				t.Fatalf("output = %q, want available count", output)
			}
			if !strings.Contains(output, fmt.Sprintf("- cited in answer: %d", len(tt.wantRefs))) ||
				!strings.Contains(output, fmt.Sprintf("- uncited: %d", 20-len(tt.wantRefs))) {
				t.Fatalf("output = %q, want normalized citation audit counts", output)
			}
			other := output[strings.Index(output, "Other sources available to the agent:"):]
			for _, ref := range tt.wantRefs {
				number := strings.TrimSuffix(strings.TrimPrefix(ref, "[source "), "]")
				sourceNumber, err := strconv.Atoi(number)
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(other, fmt.Sprintf("[source %d] source-%02d.md", sourceNumber, sourceNumber)) {
					t.Fatalf("output = %q, cited source %s appeared in other section", output, number)
				}
			}
		})
	}
}

func TestAskAgentGroupedCitationNormalizationDebugNote(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer:  "The artifacts are FASTA and PDB [sources 2, 3].",
			Sources: numberedTestSources(3),
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	var errOut bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&errOut)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both", "--show-agent-tools"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(errOut.String(), "agent citation note: normalized [sources 2, 3] -> [source 2] [source 3]") {
		t.Fatalf("stderr = %q, want citation normalization note", errOut.String())
	}
}

func TestAskAgentGroupedCitationNormalizationHiddenWithoutDebug(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	restore := stubAgentAskCLI(t, func(agentask.Options) agentask.Result {
		return agentask.Result{
			Answer:  "The artifacts are FASTA and PDB [sources 2, 3].",
			Sources: numberedTestSources(3),
		}
	})
	defer restore()

	cmd := NewRootCommand()
	var out bytes.Buffer
	var errOut bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&errOut)
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--agent", "--retrieval", "lexical", "--sources", "both"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if strings.Contains(errOut.String(), "normalized [sources") {
		t.Fatalf("stderr = %q, want no citation normalization note without debug", errOut.String())
	}
}

func TestAskSourcesModeRejectsInvalidValue(t *testing.T) {
	state, _ := setupEvalCLITest(t)
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ask", "hello", "--retrieval", "lexical", "--sources", "nearby"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--sources must be cited, all, or both") {
		t.Fatalf("Execute error = %v, want invalid sources mode", err)
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

func numberedTestSources(count int) []rag.SourceChunk {
	sources := make([]rag.SourceChunk, 0, count)
	for i := 1; i <= count; i++ {
		sources = append(sources, rag.SourceChunk{
			SourceNumber: i,
			Path:         fmt.Sprintf("source-%02d.md", i),
			StartLine:    i,
			EndLine:      i,
		})
	}
	return sources
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
