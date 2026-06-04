package agentask

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/smasonuk/falken-core/pkg/falken"
)

func TestToolCallRecord(t *testing.T) {
	tests := []struct {
		name     string
		call     falken.ToolCall
		expected ToolCallRecord
	}{
		{
			name: "basic conversion",
			call: falken.ToolCall{
				Name:      "test_tool",
				Arguments: json.RawMessage(`{"key": "value"}`),
			},
			expected: ToolCallRecord{
				Name:      "test_tool",
				Arguments: json.RawMessage(`{"key": "value"}`),
			},
		},
		{
			name: "nil arguments",
			call: falken.ToolCall{
				Name:      "test_tool_nil",
				Arguments: nil,
			},
			expected: ToolCallRecord{
				Name:      "test_tool_nil",
				Arguments: nil,
			},
		},
		{
			name: "empty arguments",
			call: falken.ToolCall{
				Name:      "test_tool_empty",
				Arguments: json.RawMessage(""),
			},
			expected: ToolCallRecord{
				Name:      "test_tool_empty",
				Arguments: json.RawMessage(""),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := toolCallRecord(tt.call)

			if result.Name != tt.expected.Name {
				t.Errorf("Name mismatch: got %v, want %v", result.Name, tt.expected.Name)
			}

			if !bytes.Equal(result.Arguments, tt.expected.Arguments) {
				t.Errorf("Arguments mismatch: got %v, want %v", string(result.Arguments), string(tt.expected.Arguments))
			}
		})
	}

	t.Run("arguments copy", func(t *testing.T) {
		call := falken.ToolCall{
			Name:      "test_copy",
			Arguments: json.RawMessage(`{"key": "value"}`),
		}

		result := toolCallRecord(call)

		// Modify the original slice
		call.Arguments[0] = '['

		// Check if the result was modified
		if bytes.Equal(result.Arguments, call.Arguments) {
			t.Errorf("Arguments were not copied: modifying original affected the result")
		}
		expectedArgs := json.RawMessage(`{"key": "value"}`)
		if !bytes.Equal(result.Arguments, expectedArgs) {
			t.Errorf("Result arguments changed unexpectedly: got %v, want %v", string(result.Arguments), string(expectedArgs))
		}

		// Ensure different underlying array
		if len(call.Arguments) > 0 && len(result.Arguments) > 0 {
			origPtr := reflect.ValueOf(call.Arguments).Pointer()
			resPtr := reflect.ValueOf(result.Arguments).Pointer()
			if origPtr == resPtr {
				t.Errorf("Arguments share the same underlying array")
			}
		}
	})
}
