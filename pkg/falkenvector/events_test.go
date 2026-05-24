package falkenvector

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/llm"
)

func TestEventEmitterPopulatesRunIDTimestampAndSeq(t *testing.T) {
	var events []Event
	emitter := newEventEmitter(func(event Event) {
		events = append(events, event)
	})

	emitter.emit(Event{Type: EventRunStarted})
	emitter.emit(Event{Type: EventRunCompleted})

	if len(events) != 2 {
		t.Fatalf("events = %d, want 2", len(events))
	}
	if events[0].RunID == "" || events[1].RunID != events[0].RunID {
		t.Fatalf("run IDs = %q, %q; want shared non-empty run ID", events[0].RunID, events[1].RunID)
	}
	if events[0].At.IsZero() || events[1].At.IsZero() {
		t.Fatalf("timestamps = %v, %v; want populated", events[0].At, events[1].At)
	}
	if events[0].Seq != 1 || events[1].Seq != 2 {
		t.Fatalf("seq = %d, %d; want 1, 2", events[0].Seq, events[1].Seq)
	}
}

func TestEventEmitterNilSinkIsSafe(t *testing.T) {
	newEventEmitter(nil).emit(Event{Type: EventRunStarted})
}

func TestEventEmitterSinkPanicDoesNotCrashAndDisablesSink(t *testing.T) {
	calls := 0
	emitter := newEventEmitter(func(Event) {
		calls++
		panic("boom")
	})

	emitter.emit(Event{Type: EventRunStarted})
	emitter.emit(Event{Type: EventRunCompleted})

	if calls != 1 {
		t.Fatalf("sink calls = %d, want panic sink disabled after first call", calls)
	}
}

func TestObservedTextRedactsAndTruncatesByDefault(t *testing.T) {
	long := strings.Repeat("x", 100)
	config := ObservabilityConfig{EmitLLMRequests: true, MaxMessageBytes: 32}

	got := observedText(long, config, config.EmitLLMRequests)
	if got == "" || got == long {
		t.Fatalf("observedText = %q, want redacted/truncated content", got)
	}
	if !strings.Contains(got, "[truncated/redacted]") {
		t.Fatalf("observedText = %q, want redaction marker", got)
	}

	short := observedText("secret", config, config.EmitLLMRequests)
	if short != "[redacted 6 bytes]" {
		t.Fatalf("short observedText = %q, want redacted marker", short)
	}
}

func TestObservedTextRawPayloadsPreservesOnlyUpToLimit(t *testing.T) {
	config := ObservabilityConfig{EmitLLMRequests: true, RawPayloads: true, MaxMessageBytes: 5}
	got := observedText("abcdefgh", config, config.EmitLLMRequests)
	if got != "abcde" {
		t.Fatalf("observedText = %q, want byte-limited raw content", got)
	}
}

func TestConvertFalkenEventMapsToolCallAndResult(t *testing.T) {
	call := convertFalkenEvent(falken.Event{
		Type: falken.EventToolCall,
		ToolCall: &falken.ToolCall{
			ID:        "call-1",
			Name:      "search_index",
			Arguments: json.RawMessage(`{"query":"alpha"}`),
		},
	})
	if call.Type != EventToolCall || call.ToolCall == nil {
		t.Fatalf("converted tool call = %+v, want EventToolCall payload", call)
	}
	if call.ToolCall.CallID != "call-1" || call.ToolCall.Name != "search_index" || call.ToolCall.Arguments != `{"query":"alpha"}` {
		t.Fatalf("tool call payload = %+v", call.ToolCall)
	}

	result := convertFalkenEvent(falken.Event{
		Type: falken.EventToolResult,
		ToolResult: &falken.ToolResult{
			CallID:  "call-1",
			Name:    "search_index",
			Content: "found",
			Payload: json.RawMessage(`{"success":true}`),
		},
	})
	if result.Type != EventToolResult || result.ToolResult == nil {
		t.Fatalf("converted tool result = %+v, want EventToolResult payload", result)
	}
	if !result.ToolResult.Success || string(result.ToolResult.Payload) != `{"success":true}` {
		t.Fatalf("tool result payload = %+v", result.ToolResult)
	}
}

func TestConvertFalkenEventMapsAssistantAndRunEvents(t *testing.T) {
	delta := convertFalkenEvent(falken.Event{Type: falken.EventAssistantText, Text: "hello"})
	if delta.Type != EventLLMDelta || delta.LLM == nil || delta.LLM.Text != "hello" {
		t.Fatalf("assistant event = %+v, want llm delta", delta)
	}

	completed := convertFalkenEvent(falken.Event{
		Type:      falken.EventRunCompleted,
		RunResult: &falken.RunResult{FinalOutput: "done", Completed: true},
	})
	if completed.Type != EventRunCompleted || completed.Answer == nil || completed.Answer.Text != "done" {
		t.Fatalf("completed event = %+v", completed)
	}

	failed := convertFalkenEvent(falken.Event{Type: falken.EventRunFailed, Error: "bad"})
	if failed.Type != EventRunFailed || failed.Error != "bad" {
		t.Fatalf("failed event = %+v", failed)
	}
}

func TestObservableInternalLLMEmitsSuccessEvents(t *testing.T) {
	var events []Event
	emitter := newEventEmitter(func(event Event) {
		events = append(events, event)
	})
	client := observableInternalLLM{
		next: fakeInternalLLM{
			response: llm.CompletionResponse{Model: "chat-model", Text: "answer"},
		},
		emit:   emitter,
		config: ObservabilityConfig{EmitLLMRequests: true, EmitLLMResponses: true},
		label:  "planner",
	}

	response, err := client.Complete(context.Background(), llm.CompletionRequest{
		Model:       "requested-model",
		System:      "system secret",
		User:        "user secret",
		Temperature: 0.2,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Text != "answer" {
		t.Fatalf("response = %+v", response)
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, []EventType{EventLLMRequest, EventLLMResponse}) {
		t.Fatalf("events = %v", got)
	}
	if events[0].LLM == nil || events[0].LLM.Messages[0].Content == "system secret" {
		t.Fatalf("request event = %+v, want redacted prompt content", events[0].LLM)
	}
	if events[1].LLM == nil || events[1].LLM.Text == "answer" {
		t.Fatalf("response event = %+v, want redacted response content", events[1].LLM)
	}
}

func TestObservableInternalLLMRawPayloads(t *testing.T) {
	var events []Event
	client := observableInternalLLM{
		next: fakeInternalLLM{
			response: llm.CompletionResponse{Model: "chat-model", Text: "abcdef"},
		},
		emit: newEventEmitter(func(event Event) {
			events = append(events, event)
		}),
		config: ObservabilityConfig{
			EmitLLMRequests:  true,
			EmitLLMResponses: true,
			RawPayloads:      true,
			MaxMessageBytes:  4,
		},
	}

	_, err := client.Complete(context.Background(), llm.CompletionRequest{System: "123456", User: "uvwxyz"})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if events[0].LLM.Messages[0].Content != "1234" {
		t.Fatalf("system content = %q, want raw truncated content", events[0].LLM.Messages[0].Content)
	}
	if events[1].LLM.Text != "abcd" {
		t.Fatalf("response content = %q, want raw truncated content", events[1].LLM.Text)
	}
}

func TestObservableInternalLLMEmitsFailure(t *testing.T) {
	var events []Event
	wantErr := errors.New("llm failed")
	client := observableInternalLLM{
		next: fakeInternalLLM{err: wantErr},
		emit: newEventEmitter(func(event Event) {
			events = append(events, event)
		}),
	}

	_, err := client.Complete(context.Background(), llm.CompletionRequest{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("Complete error = %v, want %v", err, wantErr)
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, []EventType{EventLLMRequest, EventLLMFailed}) {
		t.Fatalf("events = %v", got)
	}
}

func TestObservableFalkenLLMEmitsToolCallMetadata(t *testing.T) {
	var events []Event
	client := observableFalkenLLM{
		next: fakeFalkenLLM{
			response: falken.CompletionResponse{
				AssistantText: "calling",
				ToolCalls: []falken.ToolCall{{
					ID:        "call-1",
					Name:      "search_index",
					Arguments: json.RawMessage(`{"query":"secret"}`),
				}},
				FinishReason: falken.FinishReasonToolCalls,
			},
		},
		emit: newEventEmitter(func(event Event) {
			events = append(events, event)
		}),
		config: ObservabilityConfig{EmitLLMRequests: true, EmitLLMResponses: true},
	}

	_, err := client.Complete(context.Background(), falken.CompletionRequest{
		Messages: []falken.Message{{Role: falken.RoleUser, Content: "hello"}},
		Tools:    []falken.ToolDefinition{{Name: "search_index"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, []EventType{EventLLMRequest, EventLLMResponse}) {
		t.Fatalf("events = %v", got)
	}
	if len(events[1].LLM.ToolCalls) != 1 || events[1].LLM.ToolCalls[0].Name != "search_index" {
		t.Fatalf("tool call metadata = %+v", events[1].LLM.ToolCalls)
	}
	if strings.Contains(events[1].LLM.ToolCalls[0].Arguments, "secret") {
		t.Fatalf("tool call arguments = %q, want redacted", events[1].LLM.ToolCalls[0].Arguments)
	}
}

func TestObservableEmbedderEmitsSuccessAndFailure(t *testing.T) {
	var successEvents []Event
	embedder := observableEmbedder{
		next: fakeEmbedder{embedding: llm.Embedding{Model: "embed-model", Vector: []float32{1, 2, 3}}},
		emit: newEventEmitter(func(event Event) {
			successEvents = append(successEvents, event)
		}),
		config: ObservabilityConfig{EmitEmbeddings: true},
	}
	embedding, err := embedder.EmbedText(context.Background(), "embedding secret")
	if err != nil {
		t.Fatalf("EmbedText: %v", err)
	}
	if len(embedding.Vector) != 3 {
		t.Fatalf("embedding = %+v", embedding)
	}
	if got := eventTypes(successEvents); !reflect.DeepEqual(got, []EventType{EventEmbeddingRequest, EventEmbeddingResponse}) {
		t.Fatalf("success events = %v", got)
	}
	if successEvents[0].Embedding.InputPreview == "embedding secret" {
		t.Fatalf("embedding request = %+v, want redacted input", successEvents[0].Embedding)
	}
	if successEvents[1].Embedding.Dimensions != 3 {
		t.Fatalf("embedding response = %+v, want dimensions", successEvents[1].Embedding)
	}

	var failureEvents []Event
	wantErr := errors.New("embed failed")
	failing := observableEmbedder{
		next: fakeEmbedder{err: wantErr},
		emit: newEventEmitter(func(event Event) {
			failureEvents = append(failureEvents, event)
		}),
	}
	_, err = failing.EmbedText(context.Background(), "input")
	if !errors.Is(err, wantErr) {
		t.Fatalf("EmbedText error = %v, want %v", err, wantErr)
	}
	if got := eventTypes(failureEvents); !reflect.DeepEqual(got, []EventType{EventEmbeddingRequest, EventEmbeddingFailed}) {
		t.Fatalf("failure events = %v", got)
	}
}

type fakeInternalLLM struct {
	response llm.CompletionResponse
	err      error
}

func (f fakeInternalLLM) Complete(context.Context, llm.CompletionRequest) (llm.CompletionResponse, error) {
	if f.err != nil {
		return llm.CompletionResponse{}, f.err
	}
	return f.response, nil
}

type fakeFalkenLLM struct {
	response falken.CompletionResponse
	err      error
}

func (f fakeFalkenLLM) Complete(context.Context, falken.CompletionRequest) (falken.CompletionResponse, error) {
	if f.err != nil {
		return falken.CompletionResponse{}, f.err
	}
	return f.response, nil
}

type fakeEmbedder struct {
	embedding llm.Embedding
	err       error
}

func (f fakeEmbedder) EmbedText(context.Context, string) (llm.Embedding, error) {
	if f.err != nil {
		return llm.Embedding{}, f.err
	}
	return f.embedding, nil
}

func eventTypes(events []Event) []EventType {
	types := make([]EventType, 0, len(events))
	for _, event := range events {
		types = append(types, event.Type)
	}
	return types
}
