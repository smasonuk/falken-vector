package agentask

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
)

func TestRunCompletesToolCallingAgentWithCitations(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"citation validation"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "Citation validation is implemented here [source 1].", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Answer != "Citation validation is implemented here [source 1]." {
		t.Fatalf("answer = %q", result.Answer)
	}
	if len(result.Sources) != 1 || result.Sources[0].SourceNumber != 1 {
		t.Fatalf("sources = %+v, want source 1", result.Sources)
	}
	if !result.CitationValid {
		t.Fatalf("CitationValid = false, warnings = %+v", result.CitationWarnings)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0] != SearchIndexToolName {
		t.Fatalf("tool calls = %+v, want search_index", result.ToolCalls)
	}
	if len(result.Trace.ToolCalls) != 1 || result.Trace.ToolCalls[0].Name != SearchIndexToolName || string(result.Trace.ToolCalls[0].Arguments) != `{"query":"citation validation"}` {
		t.Fatalf("trace tool calls = %+v, want structured search_index arguments", result.Trace.ToolCalls)
	}
	if len(result.Trace.ToolResults) != 1 || result.Trace.ToolResults[0].Name != SearchIndexToolName || !result.Trace.ToolResults[0].Success || result.Trace.ToolResults[0].Status != "ok" {
		t.Fatalf("trace tool results = %+v, want successful search_index result", result.Trace.ToolResults)
	}
	if len(llm.requests) == 0 || !requestHasTool(llm.requests[0], SearchIndexToolName) {
		t.Fatalf("first request tools = %+v, want search_index", llm.requests[0].Tools)
	}
}

func TestRunCoverageNudgesBroadValidAnswerAfterOneSearch(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search-1",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"AlphaFold folding"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "AlphaFold outputs include PDB files [source 1].", FinishReason: falken.FinishReasonStop},
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search-2",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"AlphaFold A3M PDB error JSON"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search-3",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"folding monomer dimer ligand"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "AlphaFold outputs include PDB files [source 1] and A3M inputs [source 2].", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.Question = "summarize information on alphafold or anything related to folding"
	opts.RetrieveWithPlan = alphafoldRetrieveWithPlan

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.Retried {
		t.Fatal("Retried = false, want coverage retry")
	}
	if !result.CoverageNudged {
		t.Fatal("CoverageNudged = false, want coverage nudge status")
	}
	if len(result.CoverageWarnings) == 0 || !strings.Contains(result.CoverageWarnings[0], "coverage nudge") {
		t.Fatalf("CoverageWarnings = %+v, want coverage nudge warning", result.CoverageWarnings)
	}
	if result.Answer != "AlphaFold outputs include PDB files [source 1] and A3M inputs [source 2]." {
		t.Fatalf("answer = %q, want post-nudge answer", result.Answer)
	}
	if !result.CitationValid {
		t.Fatalf("CitationValid = false, warnings = %+v", result.CitationWarnings)
	}
	if len(result.ToolCalls) != 3 {
		t.Fatalf("tool calls = %+v, want three search calls", result.ToolCalls)
	}
	if len(result.Trace.ToolCalls) != 3 || string(result.Trace.ToolCalls[1].Arguments) != `{"query":"AlphaFold A3M PDB error JSON"}` {
		t.Fatalf("trace tool calls = %+v, want coverage follow-up query", result.Trace.ToolCalls)
	}
	if len(llm.requests) < 3 {
		t.Fatalf("requests = %+v, want coverage retry request", llm.requests)
	}
	prompt := lastUserPrompt(llm.requests[2])
	if !strings.Contains(prompt, "additional search_index") || !strings.Contains(prompt, "materially different queries") || !strings.Contains(prompt, opts.Question) || !strings.Contains(prompt, "AlphaFold outputs include PDB files [source 1].") {
		t.Fatalf("coverage prompt = %q, want coverage nudge wording", prompt)
	}
}

func TestRunDoesNotCoverageNudgeNarrowValidAnswer(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"citation validation"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "Citation validation is implemented here [source 1].", FinishReason: falken.FinishReasonStop},
	}}
	result, err := Run(context.Background(), testRunOptions(t, llm))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Retried {
		t.Fatalf("Retried = true, want no coverage retry for narrow question")
	}
	if len(llm.requests) != 2 {
		t.Fatalf("request count = %d, want no retry request", len(llm.requests))
	}
}

func TestRunCoverageNudgeRespectsSearchBudget(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"AlphaFold folding"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "AlphaFold outputs include PDB files [source 1].", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.Question = "summarize information on alphafold or anything related to folding"
	opts.MaxSearchCalls = 1
	opts.RetrieveWithPlan = alphafoldRetrieveWithPlan

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Retried {
		t.Fatal("Retried = true, want search budget to prevent coverage retry")
	}
	if result.CoverageNudged {
		t.Fatal("CoverageNudged = true, want skipped coverage nudge")
	}
	if len(result.CoverageWarnings) != 1 || !strings.Contains(result.CoverageWarnings[0], "search call limit reached") {
		t.Fatalf("CoverageWarnings = %+v, want search budget warning", result.CoverageWarnings)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("request count = %d, want no retry request", len(llm.requests))
	}
}

func TestRunCoverageNudgeCanBeDisabled(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"AlphaFold folding"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "AlphaFold outputs include PDB files [source 1].", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.Question = "summarize information on alphafold or anything related to folding"
	opts.CoverageNudge = boolPtr(false)
	opts.RetrieveWithPlan = alphafoldRetrieveWithPlan

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Retried {
		t.Fatal("Retried = true, want disabled coverage nudge")
	}
	if len(llm.requests) != 2 {
		t.Fatalf("request count = %d, want no retry request", len(llm.requests))
	}
}

func TestRunCoverageNudgeUsesExplicitMinimumSearchCount(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search-1",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"AlphaFold folding"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "AlphaFold outputs include PDB files [source 1].", FinishReason: falken.FinishReasonStop},
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search-2",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"AlphaFold A3M PDB error JSON"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search-3",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"folding monomer dimer ligand"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "AlphaFold outputs include PDB files [source 1] and service notes [source 3].", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.Question = "summarize information on alphafold or anything related to folding"
	opts.MinBroadSearchCalls = 3
	opts.RetrieveWithPlan = alphafoldRetrieveWithPlan

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.Retried || !result.CitationValid {
		t.Fatalf("result = %+v, want valid coverage retry", result)
	}
	if len(llm.requests) < 3 || !strings.Contains(lastUserPrompt(llm.requests[2]), "at least 2 additional search_index call(s)") {
		t.Fatalf("coverage prompt = %q, want explicit remaining search count", lastUserPrompt(llm.requests[2]))
	}
}

func TestRunRetriesInvalidCitationAnswerOnce(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"citation validation"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "Citation validation is implemented here.", FinishReason: falken.FinishReasonStop},
		{AssistantText: "Citation validation is implemented here [source 1].", FinishReason: falken.FinishReasonStop},
	}}
	result, err := Run(context.Background(), testRunOptions(t, llm))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.Retried {
		t.Fatal("Retried = false, want retry")
	}
	if result.Answer != "Citation validation is implemented here [source 1]." || !result.CitationValid {
		t.Fatalf("result = %+v, want corrected citation answer", result)
	}
	if len(llm.requests) < 3 || !strings.Contains(lastUserPrompt(llm.requests[2]), "Your previous answer had citation issues") {
		t.Fatalf("retry request = %+v, want corrective prompt", llm.requests)
	}
}

func TestRunNoToolAnswerReturnsWarning(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{AssistantText: "I think it is in README.", FinishReason: falken.FinishReasonStop},
	}}
	result, err := Run(context.Background(), testRunOptions(t, llm))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Answer != "I think it is in README." {
		t.Fatalf("answer = %q", result.Answer)
	}
	if len(result.Sources) != 0 {
		t.Fatalf("sources = %+v, want none", result.Sources)
	}
	if len(result.CitationWarnings) != 1 || result.CitationWarnings[0] != "agent did not return cited indexed sources" {
		t.Fatalf("warnings = %+v, want no-tool warning", result.CitationWarnings)
	}
}

func TestRunNoSourcesButInventedCitationIsInvalidAndRetries(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"citation validation"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "It is in the code [source 1].", FinishReason: falken.FinishReasonStop},
		{AssistantText: "I do not know.", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.RetrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
	}

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.Retried {
		t.Fatal("Retried = false, want retry for invented citation")
	}
	if !result.CitationValid {
		t.Fatalf("CitationValid = false, warnings = %+v", result.CitationWarnings)
	}
	if len(result.CitationWarnings) == 0 || !strings.Contains(strings.Join(result.CitationWarnings, "\n"), "unavailable source [source 1]") {
		t.Fatalf("warnings = %+v, want unavailable source warning", result.CitationWarnings)
	}
	if result.Answer != "I do not know." {
		t.Fatalf("answer = %q, want retry answer", result.Answer)
	}
}

func TestRunNoSourcesButInventedCitationIsInvalidWithoutRetry(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"citation validation"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "It is in the code [source 1].", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.CitationPolicy = rag.CitationPolicyValidateOnly
	opts.RetrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
	}

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.CitationValid || result.Retried {
		t.Fatalf("result = %+v, want invalid without retry", result)
	}
	if len(result.CitationWarnings) == 0 || !strings.Contains(strings.Join(result.CitationWarnings, "\n"), "unavailable source [source 1]") {
		t.Fatalf("warnings = %+v, want unavailable source warning", result.CitationWarnings)
	}
}

func TestRunNoSourcesUncitedFactualAnswerIsInvalid(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"citation validation"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "It is implemented in internal/cli/ask.go.", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.CitationPolicy = rag.CitationPolicyValidateOnly
	opts.RetrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
	}

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.CitationValid {
		t.Fatalf("CitationValid = true, want invalid result")
	}
	if result.Retried {
		t.Fatal("Retried = true, want validate-only result without retry")
	}
	if len(result.Sources) != 0 {
		t.Fatalf("sources = %+v, want none", result.Sources)
	}
	if len(result.CitationWarnings) == 0 || !strings.Contains(strings.Join(result.CitationWarnings, "\n"), "unsupported answer") {
		t.Fatalf("warnings = %+v, want unsupported answer warning", result.CitationWarnings)
	}
}

func TestRunNoSourcesUncitedFactualAnswerRetries(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"citation validation"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "It is implemented in internal/cli/ask.go.", FinishReason: falken.FinishReasonStop},
		{AssistantText: "I do not know; the indexed corpus does not contain enough information.", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.RetrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
	}

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.Retried {
		t.Fatal("Retried = false, want retry for unsupported answer")
	}
	if !result.CitationValid {
		t.Fatalf("CitationValid = false, warnings = %+v", result.CitationWarnings)
	}
	if !strings.Contains(result.Answer, "I do not know") {
		t.Fatalf("answer = %q, want retry no-support answer", result.Answer)
	}
	if len(result.CitationWarnings) == 0 || !strings.Contains(strings.Join(result.CitationWarnings, "\n"), "unsupported answer") {
		t.Fatalf("warnings = %+v, want unsupported answer warning retained", result.CitationWarnings)
	}
	if len(llm.requests) < 3 || !strings.Contains(lastUserPrompt(llm.requests[2]), "Your previous answer was unsupported") {
		t.Fatalf("retry request = %+v, want unsupported-answer corrective prompt", llm.requests)
	}
}

func TestRunNoSourcesUncitedFactualAnswerAllowsCitationPolicyOff(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"citation validation"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "It is implemented in internal/cli/ask.go.", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.CitationPolicy = rag.CitationPolicyOff
	opts.RetrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
	}

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.CitationValid || result.Retried || len(result.CitationWarnings) != 0 {
		t.Fatalf("result = %+v, want citation policy off to skip validation", result)
	}
}

func TestRunNoSourcesIDontKnowIsValidAfterSearch(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"citation validation"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "I do not know.", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.RetrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		return rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}}}, nil
	}

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.CitationValid || result.Retried || len(result.CitationWarnings) != 0 {
		t.Fatalf("result = %+v, want valid no-source I-dont-know after search", result)
	}
}

func TestRunDoesNotExposeReadSourceToolByDefault(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{AssistantText: "I do not know.", FinishReason: falken.FinishReasonStop},
	}}
	_, err := Run(context.Background(), testRunOptions(t, llm))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if requestHasTool(llm.requests[0], ReadIndexSourceToolName) {
		t.Fatalf("tools = %+v, read_index_source should be opt-in", llm.requests[0].Tools)
	}
}

func TestRunExposesReadSourceToolWhenEnabled(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{AssistantText: "I do not know.", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.EnableReadSourceTool = true
	_, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !requestHasTool(llm.requests[0], ReadIndexSourceToolName) {
		t.Fatalf("tools = %+v, want read_index_source when enabled", llm.requests[0].Tools)
	}
	if !strings.Contains(llm.requests[0].Messages[0].Content, "read_index_source") {
		t.Fatalf("system prompt = %q, want read_index_source instructions", llm.requests[0].Messages[0].Content)
	}
}

func TestRunThinSourceNudgeExpandsShortCitedSource(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	sourcePath := filepath.Join(t.TempDir(), "alphafold.md")
	if err := os.WriteFile(sourcePath, []byte("one\ntwo\nAlphaFold thin line\nfour\nfive\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"AlphaFold","strategy":"focused"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "AlphaFold is mentioned in the notes [source 1].", FinishReason: falken.FinishReasonStop},
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-read",
				Name:      ReadIndexSourceToolName,
				Arguments: json.RawMessage(`{"source_number":1,"context_lines":20}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "AlphaFold is mentioned in the surrounding notes [source 1].", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.Question = "summarize anything related to AlphaFold folding"
	opts.EnableReadSourceTool = true
	opts.CoverageNudge = boolPtr(false)
	opts.RetrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		chunk := testRetrievedChunk("alphafold-thin", sourcePath, "AlphaFold thin line", "indexed")
		chunk.Chunk.StartLine = 3
		chunk.Chunk.EndLine = 3
		return rag.RetrieveResult{
			Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
			Chunks: []rag.RetrievedChunk{chunk},
		}, nil
	}

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !result.ThinSourceNudged || !result.Retried {
		t.Fatalf("result = %+v, want thin-source retry", result)
	}
	if !result.CitationValid {
		t.Fatalf("CitationValid = false warnings=%+v", result.CitationWarnings)
	}
	if !hasToolCall(result.ToolCalls, ReadIndexSourceToolName) {
		t.Fatalf("tool calls = %+v, want read_index_source", result.ToolCalls)
	}
	if len(result.Sources) != 1 || result.Sources[0].StartLine != 1 || result.Sources[0].EndLine != 5 {
		t.Fatalf("sources = %+v, want expanded source range", result.Sources)
	}
	if len(result.ThinSourceWarnings) == 0 || !strings.Contains(result.ThinSourceWarnings[0], "thin-source nudge: expanding [source 1]") {
		t.Fatalf("thin warnings = %+v, want nudge status", result.ThinSourceWarnings)
	}
	if len(llm.requests) < 3 || !strings.Contains(lastUserPrompt(llm.requests[2]), "very short source spans") {
		t.Fatalf("thin nudge prompt missing in requests = %+v", llm.requests)
	}
	prompt := lastUserPrompt(llm.requests[2])
	for _, want := range []string{
		"Previous answer:",
		"not restarting",
		"Preserve relevant points",
		"AlphaFold is mentioned in the notes [source 1].",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("thin nudge prompt = %q, want %q", prompt, want)
		}
	}
}

func TestRunThinSourceNudgePromptPreservesSupportedSections(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	sourcePath := filepath.Join(t.TempDir(), "alphafold.md")
	if err := os.WriteFile(sourcePath, []byte("one\ntwo\nAlphaFold thin line\nfour\nfive\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	previousAnswer := "Storage/artifacts: AlphaFold produces PDB files [source 1].\n\nDB lookup: check the structure DB first, then fold if needed [source 1]."
	finalAnswer := previousAnswer + "\n\nExpanded context confirms the workflow [source 1]."
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"AlphaFold","strategy":"focused"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: previousAnswer, FinishReason: falken.FinishReasonStop},
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-read",
				Name:      ReadIndexSourceToolName,
				Arguments: json.RawMessage(`{"source_number":1,"context_lines":20}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: finalAnswer, FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.Question = "summarize anything related to AlphaFold folding"
	opts.EnableReadSourceTool = true
	opts.CoverageNudge = boolPtr(false)
	opts.RetrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		chunk := testRetrievedChunk("alphafold-thin", sourcePath, "AlphaFold thin line", "indexed")
		chunk.Chunk.StartLine = 3
		chunk.Chunk.EndLine = 3
		return rag.RetrieveResult{
			Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
			Chunks: []rag.RetrievedChunk{chunk},
		}, nil
	}

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Answer != finalAnswer {
		t.Fatalf("answer = %q, want post-nudge answer", result.Answer)
	}
	if !result.ThinSourceNudged {
		t.Fatalf("ThinSourceNudged = false, want true")
	}
	prompt := lastUserPrompt(llm.requests[2])
	for _, want := range []string{
		"Previous answer:",
		"Storage/artifacts:",
		"DB lookup:",
		"not restarting",
		"Preserve relevant points",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("thin nudge prompt = %q, want %q", prompt, want)
		}
	}
}

func TestRunThinSourceNudgeSkippedWhenReadSourceDisabled(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	llm := &fakeAgentLLM{responses: []falken.CompletionResponse{
		{
			ToolCalls: []falken.ToolCall{{
				ID:        "call-search",
				Name:      SearchIndexToolName,
				Arguments: json.RawMessage(`{"query":"AlphaFold","strategy":"focused"}`),
			}},
			FinishReason: falken.FinishReasonToolCalls,
		},
		{AssistantText: "AlphaFold is mentioned in the notes [source 1].", FinishReason: falken.FinishReasonStop},
	}}
	opts := testRunOptions(t, llm)
	opts.Question = "summarize anything related to AlphaFold folding"
	opts.CoverageNudge = boolPtr(false)
	opts.RetrieveWithPlan = func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		chunk := testRetrievedChunk("alphafold-thin", "alphafold.md", "AlphaFold thin line", "indexed")
		chunk.Chunk.StartLine = 3
		chunk.Chunk.EndLine = 3
		return rag.RetrieveResult{
			Plan:   rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
			Chunks: []rag.RetrievedChunk{chunk},
		}, nil
	}

	result, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.ThinSourceNudged || result.Retried {
		t.Fatalf("result = %+v, want no thin-source retry", result)
	}
	if len(result.ThinSourceWarnings) != 1 || !strings.Contains(result.ThinSourceWarnings[0], "read_index_source disabled") {
		t.Fatalf("thin warnings = %+v, want disabled skip status", result.ThinSourceWarnings)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("requests = %d, want only initial search run", len(llm.requests))
	}
}

func testRunOptions(t *testing.T, model falken.LLM) Options {
	t.Helper()
	paths := searchToolTestPaths(t, false)
	return Options{
		Question: "Where is citation validation implemented?",
		Paths:    paths,
		Store:    manifest.EmptyStore{},
		RetrievalDefaults: rag.RetrieveOptions{
			Mode: rag.RetrievalModeLexical,
			TopK: 2,
		},
		AgentLLM: model,
		RetrieveWithPlan: func(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
			return rag.RetrieveResult{
				Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
				Chunks: []rag.RetrievedChunk{
					testRetrievedChunk("chunk-1", "internal/rag/citation.go", "ValidateAnswerCitations validates [source N] references.", "Document: hidden"),
				},
			}, nil
		},
		CitationPolicy: rag.CitationPolicyValidateAndRetry,
	}
}

type fakeAgentLLM struct {
	requests  []falken.CompletionRequest
	responses []falken.CompletionResponse
}

func (f *fakeAgentLLM) Complete(_ context.Context, request falken.CompletionRequest) (falken.CompletionResponse, error) {
	f.requests = append(f.requests, request)
	if len(f.responses) == 0 {
		return falken.CompletionResponse{FinishReason: falken.FinishReasonStop}, nil
	}
	response := f.responses[0]
	f.responses = f.responses[1:]
	return response, nil
}

func requestHasTool(request falken.CompletionRequest, name string) bool {
	for _, tool := range request.Tools {
		if tool.Name == name {
			return true
		}
	}
	return false
}

func lastUserPrompt(request falken.CompletionRequest) string {
	for i := len(request.Messages) - 1; i >= 0; i-- {
		if request.Messages[i].Role == falken.RoleUser {
			return request.Messages[i].Content
		}
	}
	return ""
}

func boolPtr(value bool) *bool {
	return &value
}

func alphafoldRetrieveWithPlan(_ context.Context, _ manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
	chunk := testRetrievedChunk("alphafold-outputs", "alphafold/meetings/sdb.md", "AlphaFold outputs include PDB files.", "Document: hidden")
	switch {
	case strings.Contains(opts.Question, "A3M"):
		chunk = testRetrievedChunk("alphafold-a3m", "alphafold/meetings/sdb.md", "AlphaFold jobs use A3M inputs and can produce error JSON.", "Document: hidden")
	case strings.Contains(opts.Question, "monomer"):
		chunk = testRetrievedChunk("alphafold-service", "alphafold/meetings/sdb.md", "The folding workflow mentions monomer, dimer, and ligand service notes.", "Document: hidden")
	}
	return rag.RetrieveResult{
		Plan: rag.QueryPlan{Mode: "none", Queries: []string{opts.Question}},
		Chunks: []rag.RetrievedChunk{
			chunk,
		},
	}, nil
}
