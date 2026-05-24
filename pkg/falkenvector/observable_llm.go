package falkenvector

import (
	"context"
	"errors"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/llm"
)

type observableInternalLLM struct {
	next   llm.Client
	emit   *eventEmitter
	config ObservabilityConfig
	label  string
}

func (o observableInternalLLM) Complete(
	ctx context.Context,
	req llm.CompletionRequest,
) (llm.CompletionResponse, error) {
	if o.next == nil {
		err := errors.New("llm client is required")
		o.emit.emitError(EventLLMFailed, err)
		return llm.CompletionResponse{}, err
	}
	config := normalizeObservabilityConfig(o.config)
	o.emit.emit(Event{
		Type: EventLLMRequest,
		LLM: &LLMEvent{
			Label:       o.label,
			Model:       req.Model,
			Temperature: req.Temperature,
			Messages: []LLMMessage{
				{
					Role:          "system",
					Content:       observedText(req.System, config, config.EmitLLMRequests),
					ContentLength: len(req.System),
				},
				{
					Role:          "user",
					Content:       observedText(req.User, config, config.EmitLLMRequests),
					ContentLength: len(req.User),
				},
			},
		},
	})
	response, err := o.next.Complete(ctx, req)
	if err != nil {
		o.emit.emit(Event{
			Type:  EventLLMFailed,
			Error: err.Error(),
			LLM: &LLMEvent{
				Label: o.label,
				Model: req.Model,
			},
		})
		return llm.CompletionResponse{}, err
	}
	o.emit.emit(Event{
		Type: EventLLMResponse,
		LLM: &LLMEvent{
			Label: o.label,
			Model: response.Model,
			Text:  observedText(response.Text, config, config.EmitLLMResponses),
		},
	})
	return response, nil
}

type observableFalkenLLM struct {
	next   falken.LLM
	emit   *eventEmitter
	config ObservabilityConfig
}

func (o observableFalkenLLM) Complete(
	ctx context.Context,
	req falken.CompletionRequest,
) (falken.CompletionResponse, error) {
	return o.complete(ctx, req, nil)
}

func (o observableFalkenLLM) StreamComplete(
	ctx context.Context,
	req falken.CompletionRequest,
	sink falken.AssistantTextSink,
) (falken.CompletionResponse, error) {
	streaming, ok := o.next.(falken.StreamingLLM)
	if !ok {
		return o.Complete(ctx, req)
	}
	return o.complete(ctx, req, func(ctx context.Context, req falken.CompletionRequest) (falken.CompletionResponse, error) {
		config := normalizeObservabilityConfig(o.config)
		return streaming.StreamComplete(ctx, req, func(delta string) {
			o.emit.emit(Event{
				Type: EventLLMDelta,
				LLM: &LLMEvent{
					Text: observedText(delta, config, config.EmitLLMResponses),
				},
			})
			if sink != nil {
				sink(delta)
			}
		})
	})
}

func (o observableFalkenLLM) complete(
	ctx context.Context,
	req falken.CompletionRequest,
	call func(context.Context, falken.CompletionRequest) (falken.CompletionResponse, error),
) (falken.CompletionResponse, error) {
	if o.next == nil {
		err := errors.New("falken llm is required")
		o.emit.emitError(EventLLMFailed, err)
		return falken.CompletionResponse{}, err
	}
	config := normalizeObservabilityConfig(o.config)
	o.emit.emit(Event{
		Type: EventLLMRequest,
		LLM: &LLMEvent{
			Messages: falkenMessages(req.Messages, config),
			Tools:    falkenTools(req.Tools),
		},
	})
	if call == nil {
		call = o.next.Complete
	}
	response, err := call(ctx, req)
	if err != nil {
		o.emit.emit(Event{
			Type:  EventLLMFailed,
			Error: err.Error(),
			LLM: &LLMEvent{
				Tools: falkenTools(req.Tools),
			},
		})
		return falken.CompletionResponse{}, err
	}
	o.emit.emit(Event{
		Type: EventLLMResponse,
		LLM: &LLMEvent{
			Text:         observedText(response.AssistantText, config, config.EmitLLMResponses),
			FinishReason: string(response.FinishReason),
			ToolCalls:    falkenToolCallEvents(response.ToolCalls, config, config.EmitLLMResponses),
		},
	})
	return response, nil
}

func falkenMessages(messages []falken.Message, config ObservabilityConfig) []LLMMessage {
	out := make([]LLMMessage, 0, len(messages))
	for _, message := range messages {
		out = append(out, LLMMessage{
			Role:          string(message.Role),
			Content:       observedText(message.Content, config, config.EmitLLMRequests),
			ContentLength: len(message.Content),
		})
	}
	return out
}

func falkenTools(tools []falken.ToolDefinition) []LLMTool {
	out := make([]LLMTool, 0, len(tools))
	for _, tool := range tools {
		out = append(out, LLMTool{Name: tool.Name})
	}
	return out
}

func falkenToolCallEvents(calls []falken.ToolCall, config ObservabilityConfig, include bool) []ToolCallEvent {
	out := make([]ToolCallEvent, 0, len(calls))
	for _, call := range calls {
		out = append(out, ToolCallEvent{
			CallID:    call.ID,
			Name:      call.Name,
			Arguments: observedText(string(call.Arguments), config, include),
		})
	}
	return out
}
