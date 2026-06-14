package agentask

import (
	"reflect"
	"testing"

	"github.com/smasonuk/falken-core/pkg/falken"
)

func TestToolResultRecord(t *testing.T) {
	tests := []struct {
		name     string
		result   falken.ToolResult
		expected ToolResultRecord
	}{
		{
			name: "success payload",
			result: falken.ToolResult{
				Name:    "my_tool",
				Payload: []byte(`{"success": true, "status": "ok"}`),
			},
			expected: ToolResultRecord{
				Name:    "my_tool",
				Status:  "ok",
				Success: true,
				Payload: []byte(`{"success": true, "status": "ok"}`),
			},
		},
		{
			name: "error payload",
			result: falken.ToolResult{
				Name:    "my_tool",
				Payload: []byte(`{"success": false, "status": "failed"}`),
			},
			expected: ToolResultRecord{
				Name:    "my_tool",
				Status:  "failed",
				Success: false,
				Payload: []byte(`{"success": false, "status": "failed"}`),
			},
		},
		{
			name: "error field with empty payload",
			result: falken.ToolResult{
				Name:  "my_tool",
				Error: "some error occurred",
			},
			expected: ToolResultRecord{
				Name:    "my_tool",
				Status:  "error",
				Success: false,
				Payload: nil, // Note: append(json.RawMessage(nil), nil...) == nil
			},
		},
		{
			name: "empty result",
			result: falken.ToolResult{
				Name: "my_tool",
			},
			expected: ToolResultRecord{
				Name:    "my_tool",
				Status:  "",
				Success: true,
				Payload: nil,
			},
		},
		{
			name: "malformed payload uses default success based on error string",
			result: falken.ToolResult{
				Name:    "my_tool",
				Payload: []byte(`not json`),
			},
			expected: ToolResultRecord{
				Name:    "my_tool",
				Status:  "",
				Success: true,
				Payload: []byte(`not json`),
			},
		},
		{
			name: "malformed payload with error",
			result: falken.ToolResult{
				Name:    "my_tool",
				Error:   "bad things",
				Payload: []byte(`not json`),
			},
			expected: ToolResultRecord{
				Name:    "my_tool",
				Status:  "error",
				Success: false,
				Payload: []byte(`not json`),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := toolResultRecord(tt.result)

			if got.Name != tt.expected.Name {
				t.Errorf("Name = %q, want %q", got.Name, tt.expected.Name)
			}
			if got.Status != tt.expected.Status {
				t.Errorf("Status = %q, want %q", got.Status, tt.expected.Status)
			}
			if got.Success != tt.expected.Success {
				t.Errorf("Success = %v, want %v", got.Success, tt.expected.Success)
			}

			// Special case to handle the case where both are conceptually empty/nil
			// append(nil, nil...) gives nil, append(nil, []byte{}...) might give []byte{} depending on capacity.
			if len(got.Payload) == 0 && len(tt.expected.Payload) == 0 {
				return
			}
			if !reflect.DeepEqual(got.Payload, tt.expected.Payload) {
				t.Errorf("Payload = %q, want %q", got.Payload, tt.expected.Payload)
			}
		})
	}
}
