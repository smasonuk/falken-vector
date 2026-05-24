package agentask

import (
	"context"
	"encoding/json"
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
	if len(llm.requests) == 0 || !requestHasTool(llm.requests[0], SearchIndexToolName) {
		t.Fatalf("first request tools = %+v, want search_index", llm.requests[0].Tools)
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
