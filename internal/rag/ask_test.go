package rag

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
)

func TestAskReturnsAnswerAndSources(t *testing.T) {
	client := &fakeLLMClient{response: llm.CompletionResponse{Text: "Use the README [source 1]."}}
	result, err := Ask(context.Background(), AskOptions{
		Question: "Where is it documented?",
		Chunks: []RetrievedChunk{{
			Path:  "README.md",
			Score: 0.75,
			Chunk: manifest.Chunk{
				ChunkText: "Documentation lives in README.",
				StartLine: 1,
				EndLine:   2,
			},
		}},
		LLM: client,
	})
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if result.Answer != "Use the README [source 1]." {
		t.Fatalf("answer = %q", result.Answer)
	}
	if len(result.Sources) != 1 || result.Sources[0].Path != "README.md" {
		t.Fatalf("sources = %+v", result.Sources)
	}
	if !strings.Contains(client.request.User, "Documentation lives in README.") {
		t.Fatalf("LLM prompt did not include retrieved context: %s", client.request.User)
	}
}

func TestAskDoesNotCallLLMWithoutChunks(t *testing.T) {
	client := &fakeLLMClient{}
	_, err := Ask(context.Background(), AskOptions{
		Question: "Anything?",
		LLM:      client,
	})
	if err == nil {
		t.Fatal("Ask succeeded, want missing chunks error")
	}
	if client.called {
		t.Fatal("LLM was called without chunks")
	}
}

func TestValidateAnswerCitationsAcceptsValidSources(t *testing.T) {
	validation := ValidateAnswerCitations("Answer [source 1] and more [source 2].", 2)
	if !validation.Valid || len(validation.CitedSources) != 2 {
		t.Fatalf("validation = %+v, want valid citations", validation)
	}
}

func TestValidateAnswerCitationsDetectsMissingCitations(t *testing.T) {
	validation := ValidateAnswerCitations("Answer without citations.", 1)
	if validation.Valid || !validation.MissingCitation || len(validation.Warnings) == 0 {
		t.Fatalf("validation = %+v, want missing citation warning", validation)
	}
}

func TestValidateAnswerCitationsDetectsOutOfRangeSources(t *testing.T) {
	validation := ValidateAnswerCitations("Answer [source 9].", 2)
	if validation.Valid || len(validation.OutOfRangeSources) != 1 || validation.OutOfRangeSources[0] != 9 {
		t.Fatalf("validation = %+v, want out-of-range source", validation)
	}
}

func TestValidateAnswerCitationsIgnoresDuplicateCitations(t *testing.T) {
	validation := ValidateAnswerCitations("Answer [source 1] again [source 1].", 1)
	if !validation.Valid || len(validation.CitedSources) != 1 {
		t.Fatalf("validation = %+v, want duplicate citations accepted once", validation)
	}
}

func TestValidateAnswerCitationsDetectsMalformedGroupedCitations(t *testing.T) {
	tests := []string{
		"The files are kept in folders [sources 2, 3].",
		"The files are kept in folders [source 2, 3].",
		"The files are kept in folders [source 13; sources 4, 19].",
		"The files are kept in folders [source 2 and 3].",
		"The files are kept in folders [source 2 / source 3].",
		"The files are kept in folders [source two].",
		"The files are kept in folders [source A].",
		"The files are kept in folders [sources 1-3].",
	}
	for _, answer := range tests {
		t.Run(answer, func(t *testing.T) {
			validation := ValidateAnswerCitations(answer, 20)
			if validation.Valid || len(validation.MalformedCitations) != 1 {
				t.Fatalf("validation = %+v, want malformed citation", validation)
			}
			if !strings.Contains(strings.Join(validation.Warnings, "\n"), "one bracket per source") {
				t.Fatalf("warnings = %+v, want bracket guidance", validation.Warnings)
			}
		})
	}
}

func TestValidateAnswerCitationsAcceptsSeparateCitations(t *testing.T) {
	validation := ValidateAnswerCitations("The files are kept in folders [source 2] [source 3].", 3)
	if !validation.Valid || len(validation.MalformedCitations) != 0 {
		t.Fatalf("validation = %+v, want separate citations valid", validation)
	}
}

func TestNormalizeGroupedCitations(t *testing.T) {
	tests := []struct {
		name      string
		answer    string
		want      string
		wantCited []int
	}{
		{name: "plural comma", answer: "Artifacts [sources 2, 3].", want: "Artifacts [source 2] [source 3].", wantCited: []int{2, 3}},
		{name: "singular comma", answer: "Artifacts [source 2, 3].", want: "Artifacts [source 2] [source 3].", wantCited: []int{2, 3}},
		{name: "and", answer: "Artifacts [source 2 and 3].", want: "Artifacts [source 2] [source 3].", wantCited: []int{2, 3}},
		{name: "mixed semicolon", answer: "Pipeline [source 13; sources 4, 19].", want: "Pipeline [source 13] [source 4] [source 19].", wantCited: []int{4, 13, 19}},
		{name: "already canonical", answer: "Artifacts [source 2] [source 3].", want: "Artifacts [source 2] [source 3].", wantCited: []int{2, 3}},
		{name: "ambiguous text", answer: "Artifacts [source two].", want: "Artifacts [source two]."},
		{name: "range unsupported", answer: "Artifacts [source 1-3].", want: "Artifacts [source 1-3]."},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, warnings := NormalizeGroupedCitations(tt.answer)
			if got != tt.want {
				t.Fatalf("NormalizeGroupedCitations = %q, want %q", got, tt.want)
			}
			if tt.wantCited != nil {
				if cited := ExtractCitedSourceNumbers(got); !reflect.DeepEqual(cited, tt.wantCited) {
					t.Fatalf("cited = %+v, want %+v", cited, tt.wantCited)
				}
			}
			if tt.answer != tt.want && len(warnings) == 0 {
				t.Fatalf("warnings empty, want rewrite note")
			}
			if tt.answer == tt.want && len(warnings) != 0 {
				t.Fatalf("warnings = %+v, want none", warnings)
			}
		})
	}
}

func TestNormalizeGroupedCitationsKeepsOutOfRangeVisible(t *testing.T) {
	normalized, _ := NormalizeGroupedCitations("Bad [sources 2, 999].")
	validation := ValidateAnswerCitations(normalized, 3)
	if validation.Valid || len(validation.OutOfRangeSources) != 1 || validation.OutOfRangeSources[0] != 999 {
		t.Fatalf("validation = %+v, want out-of-range source after normalization", validation)
	}
}

func TestAskReturnsCitationWarningsForMissingCitation(t *testing.T) {
	client := &fakeLLMClient{responses: []llm.CompletionResponse{{Text: "Use the README."}}}
	result, err := Ask(context.Background(), AskOptions{
		Question:       "Where is it documented?",
		Chunks:         askTestChunks(),
		LLM:            client,
		CitationPolicy: CitationPolicyValidateOnly,
	})
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if result.CitationValid || len(result.CitationWarnings) == 0 {
		t.Fatalf("result = %+v, want citation warnings", result)
	}
	if client.calls != 1 {
		t.Fatalf("calls = %d, want one call without retry", client.calls)
	}
}

func TestAskRetriesOnMissingCitation(t *testing.T) {
	client := &fakeLLMClient{responses: []llm.CompletionResponse{
		{Text: "Use the README."},
		{Text: "Use the README [source 1]."},
	}}
	result, err := Ask(context.Background(), AskOptions{
		Question: "Where is it documented?",
		Chunks:   askTestChunks(),
		LLM:      client,
	})
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if !result.CitationValid || !result.Retried || result.Answer != "Use the README [source 1]." {
		t.Fatalf("result = %+v, want valid retried answer", result)
	}
	if client.calls != 2 {
		t.Fatalf("calls = %d, want retry", client.calls)
	}
	if !strings.Contains(client.requests[1].System, "Every factual claim must include a citation") {
		t.Fatalf("retry system prompt = %q, want strict citation prompt", client.requests[1].System)
	}
}

func TestAskRetriesOnOutOfRangeCitation(t *testing.T) {
	client := &fakeLLMClient{responses: []llm.CompletionResponse{
		{Text: "Use the README [source 9]."},
		{Text: "Use the README [source 1]."},
	}}
	result, err := Ask(context.Background(), AskOptions{
		Question: "Where is it documented?",
		Chunks:   askTestChunks(),
		LLM:      client,
	})
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if !result.CitationValid || !result.Retried {
		t.Fatalf("result = %+v, want valid retried answer", result)
	}
}

func TestAskDoesNotRetryWhenCitationValidationDisabled(t *testing.T) {
	client := &fakeLLMClient{responses: []llm.CompletionResponse{{Text: "Use the README."}}}
	result, err := Ask(context.Background(), AskOptions{
		Question:       "Where is it documented?",
		Chunks:         askTestChunks(),
		LLM:            client,
		CitationPolicy: CitationPolicyOff,
	})
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if !result.CitationValid || result.Retried || len(result.CitationWarnings) != 0 || client.calls != 1 {
		t.Fatalf("result = %+v calls=%d, want validation disabled", result, client.calls)
	}
}

func TestAskDoesNotRetryWhenRetryDisabled(t *testing.T) {
	client := &fakeLLMClient{responses: []llm.CompletionResponse{{Text: "Use the README."}}}
	result, err := Ask(context.Background(), AskOptions{
		Question:       "Where is it documented?",
		Chunks:         askTestChunks(),
		LLM:            client,
		CitationPolicy: CitationPolicyValidateOnly,
	})
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if result.Retried || client.calls != 1 {
		t.Fatalf("result = %+v calls=%d, want no retry", result, client.calls)
	}
}

func askTestChunks() []RetrievedChunk {
	return []RetrievedChunk{{
		Path:  "README.md",
		Score: 0.75,
		Chunk: manifest.Chunk{
			ChunkText: "Documentation lives in README.",
			StartLine: 1,
			EndLine:   2,
		},
	}}
}

type fakeLLMClient struct {
	called    bool
	calls     int
	request   llm.CompletionRequest
	requests  []llm.CompletionRequest
	response  llm.CompletionResponse
	responses []llm.CompletionResponse
	err       error
}

func (f *fakeLLMClient) Complete(_ context.Context, request llm.CompletionRequest) (llm.CompletionResponse, error) {
	f.called = true
	f.calls++
	f.request = request
	f.requests = append(f.requests, request)
	if len(f.responses) != 0 {
		response := f.responses[0]
		f.responses = f.responses[1:]
		return response, f.err
	}
	return f.response, f.err
}
