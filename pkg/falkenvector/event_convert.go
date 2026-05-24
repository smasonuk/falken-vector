package falkenvector

import "github.com/smasonuk/falken-core/pkg/falken"

func convertFalkenEvent(event falken.Event) Event {
	switch event.Type {
	case falken.EventAssistantText:
		return Event{
			Type: EventLLMDelta,
			LLM:  &LLMEvent{Text: event.Text},
		}
	case falken.EventToolCall:
		converted := Event{Type: EventToolCall}
		if event.ToolCall != nil {
			converted.ToolCall = &ToolCallEvent{
				CallID:    event.ToolCall.ID,
				Name:      event.ToolCall.Name,
				Arguments: string(event.ToolCall.Arguments),
			}
		}
		return converted
	case falken.EventToolResult:
		converted := Event{Type: EventToolResult}
		if event.ToolResult != nil {
			converted.ToolResult = &ToolResultEvent{
				CallID:  event.ToolResult.CallID,
				Name:    event.ToolResult.Name,
				Content: event.ToolResult.Content,
				Payload: append([]byte(nil), event.ToolResult.Payload...),
				Error:   event.ToolResult.Error,
				Success: event.ToolResult.Error == "",
			}
		}
		return converted
	case falken.EventRunCompleted:
		converted := Event{Type: EventRunCompleted}
		if event.RunResult != nil {
			converted.Answer = &AnswerEvent{
				Text: event.RunResult.FinalOutput,
			}
			if event.RunResult.Error != "" {
				converted.Error = event.RunResult.Error
			}
		}
		return converted
	case falken.EventRunFailed:
		return Event{Type: EventRunFailed, Error: event.Error}
	case falken.EventThought:
		return Event{Type: EventWarning, Message: event.Text}
	default:
		if event.Error != "" {
			return Event{Type: EventRunFailed, Error: event.Error}
		}
		return Event{Type: EventWarning, Message: string(event.Type)}
	}
}
