package agentask

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/rag"
)

const ReadIndexSourceToolName = "read_index_source"

type ReadSourceToolOptions struct {
	Registry *CitationRegistry
	ReadFile func(string) ([]byte, error)
}

type readIndexSourceArgs struct {
	SourceNumber int  `json:"source_number"`
	ContextLines *int `json:"context_lines"`
}

func NewReadIndexSourceTool(opts ReadSourceToolOptions) falken.Tool {
	if opts.ReadFile == nil {
		opts.ReadFile = os.ReadFile
	}
	return falken.ToolFunc(readIndexSourceDescriptor(), func(ctx context.Context, invocation falken.ToolInvocation) (falken.ToolExecutionResult, error) {
		select {
		case <-ctx.Done():
			return failedReadSourceToolResult("cancelled", ctx.Err().Error()), nil
		default:
		}
		args, err := decodeReadIndexSourceArgs(invocation.Arguments)
		if err != nil {
			return failedReadSourceToolResult("invalid_arguments", err.Error()), nil
		}
		if args.SourceNumber <= 0 {
			return failedReadSourceToolResult("invalid_arguments", "source_number must be >= 1"), nil
		}
		if opts.Registry == nil {
			return failedReadSourceToolResult("unknown_source", fmt.Sprintf("unknown source [source %d]", args.SourceNumber)), nil
		}
		source, ok := opts.Registry.SourceByNumber(args.SourceNumber)
		if !ok {
			return failedReadSourceToolResult("unknown_source", fmt.Sprintf("unknown source [source %d]", args.SourceNumber)), nil
		}
		contextLines, warnings := normalizeReadContextLines(args.ContextLines)
		data, err := opts.ReadFile(source.Path)
		if err != nil {
			return failedReadSourceToolResult("read_source_failed", fmt.Sprintf("read source %q: %v", source.Path, err)), nil
		}
		text, startLine, endLine := sourceContextText(string(data), source.StartLine, source.EndLine, contextLines)
		if err := opts.Registry.ExpandSource(source.SourceNumber, startLine, endLine, text); err != nil {
			return failedReadSourceToolResult("expand_source_failed", err.Error()), nil
		}
		payload := readSourcePayload{
			Success:      true,
			Status:       "ok",
			SourceNumber: source.SourceNumber,
			Path:         rag.DisplayPath(source.Path, source.SourceRoot),
			StartLine:    startLine,
			EndLine:      endLine,
			Text:         text,
			Warnings:     warnings,
		}
		content := fmt.Sprintf("[source %d] %s:%d-%d expanded context\nText:\n%s\n\nContinue citing [source %d].", source.SourceNumber, rag.DisplayPath(source.Path, source.SourceRoot), startLine, endLine, text, source.SourceNumber)
		return falken.ToolExecutionResult{
			Success: true,
			Status:  "ok",
			Content: content,
			Payload: marshalReadSourcePayload(payload),
		}, nil
	})
}

func readIndexSourceDescriptor() falken.ToolDescriptor {
	return falken.ToolDescriptor{
		Name:        ReadIndexSourceToolName,
		Description: "Read additional nearby lines for a source previously returned by search_index. This tool can only read registered source numbers; it cannot read arbitrary paths.",
		Parameters: json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "required": ["source_number"],
  "properties": {
    "source_number": {
      "type": "integer",
      "description": "Source number previously returned by search_index, for example 1 for [source 1]."
    },
    "context_lines": {
      "type": "integer",
      "description": "Number of lines before and after the source range to include. Defaults to 20 and caps at 100."
    }
  }
}`),
		Safety: falken.ToolSafety{
			ReadsWorkspace: true,
		},
	}
}

func decodeReadIndexSourceArgs(raw json.RawMessage) (readIndexSourceArgs, error) {
	var args readIndexSourceArgs
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&args); err != nil {
		return args, fmt.Errorf("decode read_index_source arguments: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return args, fmt.Errorf("decode read_index_source arguments: multiple JSON values")
	}
	return args, nil
}

func normalizeReadContextLines(value *int) (int, []string) {
	if value == nil {
		return 20, nil
	}
	if *value < 0 {
		return 0, nil
	}
	if *value > 100 {
		return 100, []string{"context_lines capped at 100"}
	}
	return *value, nil
}

func sourceContextText(content string, sourceStart, sourceEnd, contextLines int) (string, int, int) {
	lines := splitLines(content)
	if len(lines) == 0 {
		return "", 1, 1
	}
	if sourceStart <= 0 {
		sourceStart = 1
	}
	if sourceStart > len(lines) {
		sourceStart = len(lines)
	}
	if sourceEnd < sourceStart {
		sourceEnd = sourceStart
	}
	if sourceEnd > len(lines) {
		sourceEnd = len(lines)
	}
	start := sourceStart - contextLines
	if start < 1 {
		start = 1
	}
	end := sourceEnd + contextLines
	if end > len(lines) {
		end = len(lines)
	}
	return strings.Join(lines[start-1:end], "\n"), start, end
}

func splitLines(content string) []string {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	content = strings.TrimSuffix(content, "\n")
	if content == "" {
		return nil
	}
	return strings.Split(content, "\n")
}

func failedReadSourceToolResult(status, message string) falken.ToolExecutionResult {
	payload := marshalReadSourcePayload(readSourcePayload{
		Success:  false,
		Status:   status,
		Error:    message,
		Warnings: []string{},
	})
	return falken.ToolExecutionResult{
		Success: false,
		Status:  status,
		Content: message,
		Payload: payload,
		Error:   message,
	}
}

func marshalReadSourcePayload(payload readSourcePayload) json.RawMessage {
	if payload.Warnings == nil {
		payload.Warnings = []string{}
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return json.RawMessage(`{"success":false,"status":"internal_error","error":"encode read_index_source payload"}`)
	}
	return data
}

type readSourcePayload struct {
	Success      bool     `json:"success"`
	Status       string   `json:"status"`
	SourceNumber int      `json:"source_number,omitempty"`
	Path         string   `json:"path,omitempty"`
	StartLine    int      `json:"start_line,omitempty"`
	EndLine      int      `json:"end_line,omitempty"`
	Text         string   `json:"text,omitempty"`
	Warnings     []string `json:"warnings"`
	Error        string   `json:"error,omitempty"`
}
