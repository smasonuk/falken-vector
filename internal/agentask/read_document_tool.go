package agentask

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/rag"
)

type ReadDocumentToolOptions struct {
	Registry  *CitationRegistry
	ReadFile  func(string) ([]byte, error)
	MaxLines  int
	MaxTokens int
}

type readIndexDocumentArgs struct {
	SourceNumber int    `json:"source_number"`
	Mode         string `json:"mode"`
	StartLine    *int   `json:"start_line"`
	EndLine      *int   `json:"end_line"`
	MaxLines     *int   `json:"max_lines"`
	MaxTokens    *int   `json:"max_tokens"`
}

type readDocumentRequest struct {
	SourceNumber int
	Mode         string
	StartLine    int
	EndLine      int
	MaxLines     int
	MaxTokens    int
}

func NewReadIndexDocumentTool(opts ReadDocumentToolOptions) falken.Tool {
	return falken.ToolFunc(readIndexDocumentDescriptor(), func(ctx context.Context, invocation falken.ToolInvocation) (falken.ToolExecutionResult, error) {
		select {
		case <-ctx.Done():
			return failedReadDocumentToolResult("cancelled", ctx.Err().Error()), nil
		default:
		}
		args, err := decodeReadIndexDocumentArgs(invocation.Arguments)
		if err != nil {
			return failedReadDocumentToolResult("invalid_arguments", err.Error()), nil
		}
		request, err := normalizeReadDocumentRequest(args, opts)
		if err != nil {
			return failedReadDocumentToolResult("invalid_arguments", err.Error()), nil
		}
		return ExecuteReadIndexDocument(request, opts), nil
	})
}

func readIndexDocumentDescriptor() falken.ToolDescriptor {
	return falken.ToolDescriptor{
		Name:        ReadIndexDocumentToolName,
		Description: "Read a whole registered document, a large line range, or parent context for a source previously returned by search_index. This tool only accepts source numbers; it cannot read arbitrary paths.",
		Parameters: json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "required": ["source_number"],
  "properties": {
    "source_number": {
      "type": "integer",
      "description": "Source number previously returned by search_index, for example 12 for [source 12]."
    },
    "mode": {
      "type": "string",
      "enum": ["whole", "range", "parent"],
      "description": "Read mode. whole reads the whole registered file, range reads start_line..end_line, and parent reads the enclosing section/symbol/window. Defaults to whole."
    },
    "start_line": {
      "type": "integer",
      "description": "First line to read when mode is range."
    },
    "end_line": {
      "type": "integer",
      "description": "Last line to read when mode is range."
    },
    "max_lines": {
      "type": "integer",
      "description": "Maximum lines to return. Defaults to the configured agent cap."
    },
    "max_tokens": {
      "type": "integer",
      "description": "Maximum estimated tokens to return. Defaults to the configured agent cap."
    }
  }
}`),
		Safety: falken.ToolSafety{
			ReadsWorkspace: true,
		},
	}
}

func decodeReadIndexDocumentArgs(raw json.RawMessage) (readIndexDocumentArgs, error) {
	var args readIndexDocumentArgs
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&args); err != nil {
		return args, fmt.Errorf("decode read_index_document arguments: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return args, fmt.Errorf("decode read_index_document arguments: multiple JSON values")
	}
	return args, nil
}

func normalizeReadDocumentRequest(args readIndexDocumentArgs, opts ReadDocumentToolOptions) (readDocumentRequest, error) {
	if args.SourceNumber <= 0 {
		return readDocumentRequest{}, fmt.Errorf("source_number must be >= 1")
	}
	mode := strings.TrimSpace(args.Mode)
	if mode == "" {
		mode = "whole"
	}
	switch mode {
	case "whole", "range", "parent":
	default:
		return readDocumentRequest{}, fmt.Errorf("mode must be whole, range, or parent")
	}
	maxLines := opts.MaxLines
	if maxLines <= 0 {
		maxLines = DefaultMaxDocumentReadLines
	}
	if args.MaxLines != nil {
		maxLines = *args.MaxLines
	}
	if maxLines <= 0 {
		return readDocumentRequest{}, fmt.Errorf("max_lines must be >= 1")
	}
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxDocumentReadTokens
	}
	if args.MaxTokens != nil {
		maxTokens = *args.MaxTokens
	}
	if maxTokens <= 0 {
		return readDocumentRequest{}, fmt.Errorf("max_tokens must be >= 1")
	}
	startLine := 0
	endLine := 0
	if args.StartLine != nil {
		startLine = *args.StartLine
	}
	if args.EndLine != nil {
		endLine = *args.EndLine
	}
	return readDocumentRequest{
		SourceNumber: args.SourceNumber,
		Mode:         mode,
		StartLine:    startLine,
		EndLine:      endLine,
		MaxLines:     maxLines,
		MaxTokens:    maxTokens,
	}, nil
}

func ExecuteReadIndexDocument(request readDocumentRequest, opts ReadDocumentToolOptions) falken.ToolExecutionResult {
	if opts.ReadFile == nil {
		opts.ReadFile = os.ReadFile
	}
	if opts.Registry == nil {
		return failedReadDocumentToolResult("unknown_source", fmt.Sprintf("unknown source [source %d]", request.SourceNumber))
	}
	source, ok := opts.Registry.SourceByNumber(request.SourceNumber)
	if !ok {
		return failedReadDocumentToolResult("unknown_source", fmt.Sprintf("unknown source [source %d]", request.SourceNumber))
	}
	data, err := opts.ReadFile(source.Path)
	if err != nil {
		return failedReadDocumentToolResult("read_document_failed", fmt.Sprintf("read document %q: %v", source.Path, err))
	}
	content := string(data)
	stats := documentTextStats(data)
	readRange, kind := requestedDocumentRange(content, source, request)
	text, startLine, endLine := sourceRangeText(content, readRange.StartLine, readRange.EndLine)
	lineCount := endLine - startLine + 1
	tokenCount := estimateDocumentTokens(text)
	displayPath := rag.DisplayPath(source.Path, source.SourceRoot)
	if lineCount > request.MaxLines || tokenCount > request.MaxTokens {
		payload := readDocumentPayload{
			Success:             true,
			Status:              "too_large",
			SourceNumber:        source.SourceNumber,
			Path:                displayPath,
			Mode:                request.Mode,
			ParentKind:          string(kind),
			StartLine:           startLine,
			EndLine:             endLine,
			Lines:               lineCount,
			FileLineCount:       stats.LineCount,
			FileByteCount:       stats.ByteCount,
			EstimatedTokens:     tokenCount,
			EstimatedFileTokens: stats.EstimatedTokens,
			MaxLines:            request.MaxLines,
			MaxTokens:           request.MaxTokens,
			Warnings:            []string{},
		}
		content := fmt.Sprintf("[source %d] %s is too large to read %s.\nFile stats: %d lines, estimated %d tokens.\nTry read_index_document with mode=\"range\" or read_index_source around specific sources.",
			source.SourceNumber,
			displayPath,
			request.Mode,
			stats.LineCount,
			stats.EstimatedTokens,
		)
		return falken.ToolExecutionResult{
			Success: true,
			Status:  "too_large",
			Content: content,
			Payload: marshalReadDocumentPayload(payload),
		}
	}
	if err := opts.Registry.ExpandSource(source.SourceNumber, startLine, endLine, text); err != nil {
		return failedReadDocumentToolResult("expand_source_failed", err.Error())
	}
	payload := readDocumentPayload{
		Success:             true,
		Status:              "ok",
		SourceNumber:        source.SourceNumber,
		Path:                displayPath,
		Mode:                request.Mode,
		ParentKind:          string(kind),
		StartLine:           startLine,
		EndLine:             endLine,
		Lines:               lineCount,
		FileLineCount:       stats.LineCount,
		FileByteCount:       stats.ByteCount,
		EstimatedTokens:     tokenCount,
		EstimatedFileTokens: stats.EstimatedTokens,
		Text:                text,
		Warnings:            []string{},
	}
	resultContent := fmt.Sprintf("[source %d] %s:%d-%d %s context\nText:\n%s\n\nContinue citing [source %d].",
		source.SourceNumber,
		displayPath,
		startLine,
		endLine,
		documentContextLabel(request.Mode, kind),
		text,
		source.SourceNumber,
	)
	return falken.ToolExecutionResult{
		Success: true,
		Status:  "ok",
		Content: resultContent,
		Payload: marshalReadDocumentPayload(payload),
	}
}

type ParentContextKind string

const (
	ParentContextMarkdownHeadingSection ParentContextKind = "markdown_heading_section"
	ParentContextCodeSymbolSection      ParentContextKind = "code_symbol_section"
	ParentContextPlainTextWindow        ParentContextKind = "plain_text_window"
	ParentContextWholeDocument          ParentContextKind = "whole_document"
)

func requestedDocumentRange(content string, source rag.SourceChunk, request readDocumentRequest) (Range, ParentContextKind) {
	lines := splitLines(content)
	lineCount := len(lines)
	if lineCount == 0 {
		return Range{StartLine: 1, EndLine: 1}, ParentContextWholeDocument
	}
	switch request.Mode {
	case "range":
		start := request.StartLine
		end := request.EndLine
		if start <= 0 {
			start = source.StartLine
		}
		if end <= 0 {
			end = source.EndLine
		}
		return clampRange(Range{StartLine: start, EndLine: end}, lineCount), ParentContextWholeDocument
	case "parent":
		return resolveParentContext(content, source)
	default:
		return Range{StartLine: 1, EndLine: lineCount}, ParentContextWholeDocument
	}
}

func resolveParentContext(content string, source rag.SourceChunk) (Range, ParentContextKind) {
	if isMarkdownPath(source.Path) || strings.EqualFold(source.Chunker, "markdown") {
		if r, ok := markdownHeadingSectionRange(content, source); ok {
			return r, ParentContextMarkdownHeadingSection
		}
	}
	if strings.EqualFold(source.Chunker, "code") || source.SymbolName != "" || source.Language != "" {
		if r, ok := codeSymbolSectionRange(content, source); ok {
			return r, ParentContextCodeSymbolSection
		}
	}
	lines := splitLines(content)
	if len(lines) == 0 {
		return Range{StartLine: 1, EndLine: 1}, ParentContextPlainTextWindow
	}
	return clampRange(Range{StartLine: source.StartLine - 60, EndLine: source.EndLine + 60}, len(lines)), ParentContextPlainTextWindow
}

func isMarkdownPath(path string) bool {
	lower := strings.ToLower(path)
	return strings.HasSuffix(lower, ".md") || strings.HasSuffix(lower, ".markdown") || strings.HasSuffix(lower, ".mdx")
}

var markdownHeadingLinePattern = regexp.MustCompile(`^(#{1,6})\s+(.+?)\s*#*\s*$`)

func markdownHeadingSectionRange(content string, source rag.SourceChunk) (Range, bool) {
	lines := splitLines(content)
	if len(lines) == 0 {
		return Range{}, false
	}
	targetTitle := ""
	if len(source.HeadingPath) != 0 {
		targetTitle = source.HeadingPath[len(source.HeadingPath)-1]
	}
	start := 0
	level := 0
	for i := 0; i < len(lines) && i+1 <= maxInt(source.StartLine, 1); i++ {
		match := markdownHeadingLinePattern.FindStringSubmatch(strings.TrimSpace(lines[i]))
		if match == nil {
			continue
		}
		title := strings.TrimSpace(match[2])
		if targetTitle != "" && title != targetTitle && i+1 == source.StartLine {
			continue
		}
		start = i + 1
		level = len(match[1])
	}
	if start == 0 || level == 0 {
		return Range{}, false
	}
	end := len(lines)
	for i := start; i < len(lines); i++ {
		match := markdownHeadingLinePattern.FindStringSubmatch(strings.TrimSpace(lines[i]))
		if match == nil {
			continue
		}
		if len(match[1]) <= level {
			end = i
			break
		}
	}
	return Range{StartLine: start, EndLine: end}, true
}

var codeTopLevelSymbolPattern = regexp.MustCompile(`^(func\s+(\([^)]+\)\s*)?[A-Za-z_][A-Za-z0-9_]*\s*\(|type\s+[A-Za-z_][A-Za-z0-9_]*\s+|def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(|class\s+[A-Za-z_][A-Za-z0-9_]*|function\s+[A-Za-z_$][A-Za-z0-9_$]*\s*\(|(export\s+)?class\s+[A-Za-z_$][A-Za-z0-9_$]*)`)

func codeSymbolSectionRange(content string, source rag.SourceChunk) (Range, bool) {
	lines := splitLines(content)
	if len(lines) == 0 {
		return Range{}, false
	}
	start := 0
	for i := 0; i < len(lines) && i+1 <= maxInt(source.StartLine, 1); i++ {
		line := lines[i]
		if strings.TrimLeft(line, " \t") != line {
			continue
		}
		if codeTopLevelSymbolPattern.MatchString(strings.TrimSpace(line)) {
			start = i + 1
		}
	}
	if start == 0 {
		return Range{}, false
	}
	end := len(lines)
	for i := start; i < len(lines); i++ {
		line := lines[i]
		if strings.TrimLeft(line, " \t") != line {
			continue
		}
		if codeTopLevelSymbolPattern.MatchString(strings.TrimSpace(line)) {
			end = i
			break
		}
	}
	return Range{StartLine: start, EndLine: end}, true
}

func clampRange(r Range, lineCount int) Range {
	if lineCount <= 0 {
		return Range{StartLine: 1, EndLine: 1}
	}
	if r.StartLine <= 0 {
		r.StartLine = 1
	}
	if r.StartLine > lineCount {
		r.StartLine = lineCount
	}
	if r.EndLine < r.StartLine {
		r.EndLine = r.StartLine
	}
	if r.EndLine > lineCount {
		r.EndLine = lineCount
	}
	return r
}

func documentContextLabel(mode string, kind ParentContextKind) string {
	switch mode {
	case "whole":
		return "whole document"
	case "parent":
		return string(kind)
	default:
		return "document range"
	}
}

func failedReadDocumentToolResult(status, message string) falken.ToolExecutionResult {
	payload := marshalReadDocumentPayload(readDocumentPayload{
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

func marshalReadDocumentPayload(payload readDocumentPayload) json.RawMessage {
	if payload.Warnings == nil {
		payload.Warnings = []string{}
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return json.RawMessage(`{"success":false,"status":"internal_error","error":"encode read_index_document payload"}`)
	}
	return data
}

type readDocumentPayload struct {
	Success             bool     `json:"success"`
	Status              string   `json:"status"`
	SourceNumber        int      `json:"source_number,omitempty"`
	Path                string   `json:"path,omitempty"`
	Mode                string   `json:"mode,omitempty"`
	ParentKind          string   `json:"parent_kind,omitempty"`
	StartLine           int      `json:"start_line,omitempty"`
	EndLine             int      `json:"end_line,omitempty"`
	Lines               int      `json:"lines,omitempty"`
	FileLineCount       int      `json:"file_line_count,omitempty"`
	FileByteCount       int64    `json:"file_byte_count,omitempty"`
	EstimatedTokens     int      `json:"estimated_tokens,omitempty"`
	EstimatedFileTokens int      `json:"estimated_file_tokens,omitempty"`
	MaxLines            int      `json:"max_lines,omitempty"`
	MaxTokens           int      `json:"max_tokens,omitempty"`
	Text                string   `json:"text,omitempty"`
	Warnings            []string `json:"warnings"`
	Error               string   `json:"error,omitempty"`
}
