package agentask

import (
	"encoding/json"
	"strings"

	"github.com/smasonuk/falken-core/pkg/falken"
)

func toolCallRecord(call falken.ToolCall) ToolCallRecord {
	return ToolCallRecord{
		Name:      call.Name,
		Arguments: append(json.RawMessage(nil), call.Arguments...),
	}
}

func toolResultRecord(result falken.ToolResult) ToolResultRecord {
	status, success := toolResultStatus(result)
	return ToolResultRecord{
		Name:    result.Name,
		Status:  status,
		Success: success,
		Payload: append(json.RawMessage(nil), result.Payload...),
	}
}

func toolResultStatus(result falken.ToolResult) (string, bool) {
	success := strings.TrimSpace(result.Error) == ""
	status := ""
	var payload struct {
		Success *bool  `json:"success"`
		Status  string `json:"status"`
	}
	if len(result.Payload) != 0 && json.Unmarshal(result.Payload, &payload) == nil {
		if payload.Success != nil {
			success = *payload.Success
		}
		status = payload.Status
	}
	if status == "" && result.Error != "" {
		status = "error"
	}
	return status, success
}

func cloneAgentTrace(trace AgentTrace) AgentTrace {
	return AgentTrace{
		ToolCalls:   cloneToolCallRecords(trace.ToolCalls),
		ToolResults: cloneToolResultRecords(trace.ToolResults),
	}
}

func cloneToolCallRecords(records []ToolCallRecord) []ToolCallRecord {
	out := make([]ToolCallRecord, 0, len(records))
	for _, record := range records {
		out = append(out, ToolCallRecord{
			Name:      record.Name,
			Arguments: append(json.RawMessage(nil), record.Arguments...),
		})
	}
	return out
}

func cloneToolResultRecords(records []ToolResultRecord) []ToolResultRecord {
	out := make([]ToolResultRecord, 0, len(records))
	for _, record := range records {
		out = append(out, ToolResultRecord{
			Name:    record.Name,
			Status:  record.Status,
			Success: record.Success,
			Payload: append(json.RawMessage(nil), record.Payload...),
		})
	}
	return out
}
