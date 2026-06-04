package agentask

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/rag"
)

const ReadIndexSourceToolName = "read_index_source"

type ReadSourceToolOptions struct {
	Registry                 *CitationRegistry
	ReadFile                 func(string) ([]byte, error)
	OverlapPolicy            ReadSourceOverlapPolicy
	MaxMergedReadSourceLines int
}

type readIndexSourceArgs struct {
	SourceNumber int  `json:"source_number"`
	ContextLines *int `json:"context_lines"`
}

type ReadSourceOverlapPolicy string

const (
	ReadSourceOverlapDefault ReadSourceOverlapPolicy = ""
	ReadSourceOverlapSkip    ReadSourceOverlapPolicy = "skip"
	ReadSourceOverlapMerge   ReadSourceOverlapPolicy = "merge"
	ReadSourceOverlapAllow   ReadSourceOverlapPolicy = "allow"
)

type readIndexSourceTool struct {
	opts            ReadSourceToolOptions
	overlapPolicy   ReadSourceOverlapPolicy
	maxMergedLines  int
	expandedMu      sync.Mutex
	expandedSources map[int]struct{}
}

func (t *readIndexSourceTool) execute(ctx context.Context, invocation falken.ToolInvocation) (falken.ToolExecutionResult, error) {
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
	if t.opts.Registry == nil {
		return failedReadSourceToolResult("unknown_source", fmt.Sprintf("unknown source [source %d]", args.SourceNumber)), nil
	}
	source, ok := t.opts.Registry.SourceByNumber(args.SourceNumber)
	if !ok {
		return failedReadSourceToolResult("unknown_source", fmt.Sprintf("unknown source [source %d]", args.SourceNumber)), nil
	}
	contextLines, warnings := normalizeReadContextLines(args.ContextLines)
	projectedRange := projectedReadSourceRange(source.StartLine, source.EndLine, contextLines)
	t.expandedMu.Lock()
	expandedCandidates := expandedReadSourceCandidates(t.opts.Registry.Sources(), t.expandedSources)
	t.expandedMu.Unlock()
	decision := decideReadSourceOverlap(source, projectedRange, expandedCandidates, t.overlapPolicy)
	switch decision.Kind {
	case readSourceOverlapAlreadyCovered:
		return t.handleAlreadyCovered(source, projectedRange, decision.CoveringSource, warnings)
	case readSourceOverlapMergeExisting:
		return t.handleMergeExisting(source, projectedRange, decision.CoveringSource, warnings)
	}
	return t.handleNewSource(source, contextLines, warnings)
}

func (t *readIndexSourceTool) handleAlreadyCovered(source rag.SourceChunk, projectedRange lineRange, coveringSource rag.SourceChunk, warnings []string) (falken.ToolExecutionResult, error) {
	payload := readSourcePayload{
		Success:               true,
		Status:                "already_covered",
		SourceNumber:          source.SourceNumber,
		CoveredBySourceNumber: coveringSource.SourceNumber,
		Path:                  rag.DisplayPath(source.Path, source.SourceRoot),
		StartLine:             projectedRange.Start,
		EndLine:               projectedRange.End,
		CoveredByPath:         rag.DisplayPath(coveringSource.Path, coveringSource.SourceRoot),
		CoveredByStartLine:    coveringSource.StartLine,
		CoveredByEndLine:      coveringSource.EndLine,
		Warnings:              warnings,
	}
	content := fmt.Sprintf("[source %d] already covered by expanded [source %d] %s:%d-%d.\nContinue citing [source %d] for this expanded context.",
		source.SourceNumber,
		coveringSource.SourceNumber,
		rag.DisplayPath(coveringSource.Path, coveringSource.SourceRoot),
		coveringSource.StartLine,
		coveringSource.EndLine,
		coveringSource.SourceNumber,
	)
	return falken.ToolExecutionResult{
		Success: true,
		Status:  "already_covered",
		Content: content,
		Payload: marshalReadSourcePayload(payload),
	}, nil
}

func (t *readIndexSourceTool) handleMergeExisting(source rag.SourceChunk, projectedRange lineRange, coveringSource rag.SourceChunk, warnings []string) (falken.ToolExecutionResult, error) {
	mergedRange := lineRange{
		Start: minInt(coveringSource.StartLine, projectedRange.Start),
		End:   maxInt(coveringSource.EndLine, projectedRange.End),
	}
	if mergedRange.len() > t.maxMergedLines {
		payload := readSourcePayload{
			Success:                  true,
			Status:                   "merge_too_large",
			SourceNumber:             source.SourceNumber,
			CoveredBySourceNumber:    coveringSource.SourceNumber,
			Path:                     rag.DisplayPath(source.Path, source.SourceRoot),
			StartLine:                projectedRange.Start,
			EndLine:                  projectedRange.End,
			CoveredByPath:            rag.DisplayPath(coveringSource.Path, coveringSource.SourceRoot),
			CoveredByStartLine:       coveringSource.StartLine,
			CoveredByEndLine:         coveringSource.EndLine,
			MaxMergedReadSourceLines: t.maxMergedLines,
			Warnings:                 warnings,
		}
		content := fmt.Sprintf("requested [source %d] overlaps [source %d], but merging would exceed the max merged read range of %d lines.\nChoose a narrower source or fewer context lines.",
			source.SourceNumber,
			coveringSource.SourceNumber,
			t.maxMergedLines,
		)
		return falken.ToolExecutionResult{
			Success: true,
			Status:  "merge_too_large",
			Content: content,
			Payload: marshalReadSourcePayload(payload),
		}, nil
	}
	data, err := t.opts.ReadFile(coveringSource.Path)
	if err != nil {
		return failedReadSourceToolResult("read_source_failed", fmt.Sprintf("read source %q: %v", coveringSource.Path, err)), nil
	}
	text, startLine, endLine := sourceRangeText(string(data), mergedRange.Start, mergedRange.End)
	if err := t.opts.Registry.ExpandSource(coveringSource.SourceNumber, startLine, endLine, text); err != nil {
		return failedReadSourceToolResult("expand_source_failed", err.Error()), nil
	}
	t.expandedMu.Lock()
	t.expandedSources[coveringSource.SourceNumber] = struct{}{}
	t.expandedMu.Unlock()
	payload := readSourcePayload{
		Success:               true,
		Status:                "merge_existing",
		SourceNumber:          source.SourceNumber,
		CoveredBySourceNumber: coveringSource.SourceNumber,
		Path:                  rag.DisplayPath(source.Path, source.SourceRoot),
		StartLine:             projectedRange.Start,
		EndLine:               projectedRange.End,
		CoveredByPath:         rag.DisplayPath(coveringSource.Path, coveringSource.SourceRoot),
		CoveredByStartLine:    startLine,
		CoveredByEndLine:      endLine,
		Text:                  text,
		Warnings:              warnings,
	}
	content := fmt.Sprintf("requested [source %d] overlaps [source %d]; expanded [source %d] to %s:%d-%d.\nText:\n%s\n\nContinue citing [source %d] for this expanded context.",
		source.SourceNumber,
		coveringSource.SourceNumber,
		coveringSource.SourceNumber,
		rag.DisplayPath(coveringSource.Path, coveringSource.SourceRoot),
		startLine,
		endLine,
		text,
		coveringSource.SourceNumber,
	)
	return falken.ToolExecutionResult{
		Success: true,
		Status:  "merge_existing",
		Content: content,
		Payload: marshalReadSourcePayload(payload),
	}, nil
}

func (t *readIndexSourceTool) handleNewSource(source rag.SourceChunk, contextLines int, warnings []string) (falken.ToolExecutionResult, error) {
	data, err := t.opts.ReadFile(source.Path)
	if err != nil {
		return failedReadSourceToolResult("read_source_failed", fmt.Sprintf("read source %q: %v", source.Path, err)), nil
	}
	text, startLine, endLine := sourceContextText(string(data), source.StartLine, source.EndLine, contextLines)
	if err := t.opts.Registry.ExpandSource(source.SourceNumber, startLine, endLine, text); err != nil {
		return failedReadSourceToolResult("expand_source_failed", err.Error()), nil
	}
	t.expandedMu.Lock()
	t.expandedSources[source.SourceNumber] = struct{}{}
	t.expandedMu.Unlock()
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
}

func NewReadIndexSourceTool(opts ReadSourceToolOptions) falken.Tool {
	if opts.ReadFile == nil {
		opts.ReadFile = os.ReadFile
	}
	t := &readIndexSourceTool{
		opts:            opts,
		overlapPolicy:   normalizeReadSourceOverlapPolicy(opts.OverlapPolicy),
		maxMergedLines:  normalizeMaxMergedReadSourceLines(opts.MaxMergedReadSourceLines),
		expandedSources: map[int]struct{}{},
	}
	return falken.ToolFunc(readIndexSourceDescriptor(), t.execute)
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

func normalizeReadSourceOverlapPolicy(policy ReadSourceOverlapPolicy) ReadSourceOverlapPolicy {
	switch policy {
	case ReadSourceOverlapAllow, ReadSourceOverlapMerge, ReadSourceOverlapSkip:
		return policy
	case ReadSourceOverlapDefault:
		return ReadSourceOverlapSkip
	default:
		return ReadSourceOverlapSkip
	}
}

func normalizeMaxMergedReadSourceLines(value int) int {
	if value <= 0 {
		return 300
	}
	return value
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

func sourceRangeText(content string, startLine, endLine int) (string, int, int) {
	lines := splitLines(content)
	if len(lines) == 0 {
		return "", 1, 1
	}
	if startLine <= 0 {
		startLine = 1
	}
	if startLine > len(lines) {
		startLine = len(lines)
	}
	if endLine < startLine {
		endLine = startLine
	}
	if endLine > len(lines) {
		endLine = len(lines)
	}
	return strings.Join(lines[startLine-1:endLine], "\n"), startLine, endLine
}

func splitLines(content string) []string {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	content = strings.TrimSuffix(content, "\n")
	if content == "" {
		return nil
	}
	return strings.Split(content, "\n")
}

type lineRange struct {
	Start int
	End   int
}

func (r lineRange) len() int {
	if r.End < r.Start {
		return 0
	}
	return r.End - r.Start + 1
}

type readSourceOverlapKind string

const (
	readSourceOverlapNone           readSourceOverlapKind = "none"
	readSourceOverlapAlreadyCovered readSourceOverlapKind = "already_covered"
	readSourceOverlapMergeExisting  readSourceOverlapKind = "merge_existing"
)

type readSourceOverlapDecision struct {
	Kind           readSourceOverlapKind
	CoveringSource rag.SourceChunk
}

func projectedReadSourceRange(sourceStart, sourceEnd, contextLines int) lineRange {
	if sourceStart <= 0 {
		sourceStart = 1
	}
	if sourceEnd < sourceStart {
		sourceEnd = sourceStart
	}
	start := sourceStart - contextLines
	if start < 1 {
		start = 1
	}
	return lineRange{Start: start, End: sourceEnd + contextLines}
}

func decideReadSourceOverlap(source rag.SourceChunk, requested lineRange, existing []rag.SourceChunk, policy ReadSourceOverlapPolicy) readSourceOverlapDecision {
	if policy == ReadSourceOverlapAllow || requested.len() == 0 {
		return readSourceOverlapDecision{Kind: readSourceOverlapNone}
	}
	var best rag.SourceChunk
	bestOverlap := 0
	for _, candidate := range existing {
		if candidate.SourceNumber == source.SourceNumber {
			continue
		}
		if candidate.StartLine <= 0 || candidate.EndLine < candidate.StartLine {
			continue
		}
		if !sameReadSourceDocument(source, candidate) {
			continue
		}
		overlap := rangeOverlap(requested, lineRange{Start: candidate.StartLine, End: candidate.EndLine})
		if overlap > bestOverlap {
			bestOverlap = overlap
			best = candidate
		}
	}
	if bestOverlap == 0 {
		return readSourceOverlapDecision{Kind: readSourceOverlapNone}
	}
	existingRange := lineRange{Start: best.StartLine, End: best.EndLine}
	if requested.Start >= existingRange.Start && requested.End <= existingRange.End {
		return readSourceOverlapDecision{Kind: readSourceOverlapAlreadyCovered, CoveringSource: best}
	}
	smaller := minInt(requested.len(), existingRange.len())
	if smaller <= 0 || float64(bestOverlap)/float64(smaller) < 0.75 {
		return readSourceOverlapDecision{Kind: readSourceOverlapNone}
	}
	if policy == ReadSourceOverlapMerge && newLineCount(requested, existingRange) >= 10 {
		return readSourceOverlapDecision{Kind: readSourceOverlapMergeExisting, CoveringSource: best}
	}
	return readSourceOverlapDecision{Kind: readSourceOverlapAlreadyCovered, CoveringSource: best}
}

func expandedReadSourceCandidates(sources []rag.SourceChunk, expanded map[int]struct{}) []rag.SourceChunk {
	if len(sources) == 0 || len(expanded) == 0 {
		return nil
	}
	out := make([]rag.SourceChunk, 0, len(expanded))
	for _, source := range sources {
		if _, ok := expanded[source.SourceNumber]; ok {
			out = append(out, source)
		}
	}
	return out
}

func sameReadSourceDocument(a, b rag.SourceChunk) bool {
	if strings.TrimSpace(a.Path) != "" && a.Path == b.Path {
		return true
	}
	aDisplay := rag.DisplayPath(a.Path, a.SourceRoot)
	bDisplay := rag.DisplayPath(b.Path, b.SourceRoot)
	return aDisplay != "" && aDisplay == bDisplay
}

func rangeOverlap(a, b lineRange) int {
	start := maxInt(a.Start, b.Start)
	end := minInt(a.End, b.End)
	if end < start {
		return 0
	}
	return end - start + 1
}

func newLineCount(requested, existing lineRange) int {
	newLines := 0
	if requested.Start < existing.Start {
		newLines += existing.Start - requested.Start
	}
	if requested.End > existing.End {
		newLines += requested.End - existing.End
	}
	return newLines
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
	Success                  bool     `json:"success"`
	Status                   string   `json:"status"`
	SourceNumber             int      `json:"source_number,omitempty"`
	CoveredBySourceNumber    int      `json:"covered_by_source_number,omitempty"`
	Path                     string   `json:"path,omitempty"`
	StartLine                int      `json:"start_line,omitempty"`
	EndLine                  int      `json:"end_line,omitempty"`
	CoveredByPath            string   `json:"covered_by_path,omitempty"`
	CoveredByStartLine       int      `json:"covered_by_start_line,omitempty"`
	CoveredByEndLine         int      `json:"covered_by_end_line,omitempty"`
	MaxMergedReadSourceLines int      `json:"max_merged_read_source_lines,omitempty"`
	Text                     string   `json:"text,omitempty"`
	Warnings                 []string `json:"warnings"`
	Error                    string   `json:"error,omitempty"`
}
