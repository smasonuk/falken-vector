package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/agentask"
	"github.com/smasonuk/falken-vector/internal/rag"
)

const (
	maxAgentToolArgumentBytes = 2048
	agentToolArgumentSuffix   = "... <truncated>"
)

func formatAgentToolArguments(raw json.RawMessage) string {
	if len(bytes.TrimSpace(raw)) == 0 {
		return "{}"
	}

	var decoded any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return "<invalid arguments>"
	}
	redactSensitiveToolArguments(decoded)

	formatted, err := json.Marshal(decoded)
	if err != nil {
		return "<invalid arguments>"
	}
	return truncateAgentToolArguments(string(formatted))
}

func redactSensitiveToolArguments(value any) {
	switch typed := value.(type) {
	case map[string]any:
		for key, nested := range typed {
			if isSensitiveToolArgumentKey(key) {
				typed[key] = "[redacted]"
				continue
			}
			redactSensitiveToolArguments(nested)
		}
	case []any:
		for _, nested := range typed {
			redactSensitiveToolArguments(nested)
		}
	}
}

func isSensitiveToolArgumentKey(key string) bool {
	normalized := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(key, "_", ""), "-", ""), " ", ""))
	if normalized == "key" || strings.HasSuffix(normalized, "key") {
		return true
	}
	for _, marker := range []string{"token", "apikey", "authorization", "password", "secret", "credential"} {
		if strings.Contains(normalized, marker) {
			return true
		}
	}
	return false
}

func truncateAgentToolArguments(value string) string {
	if len(value) <= maxAgentToolArgumentBytes {
		return value
	}
	limit := maxAgentToolArgumentBytes - len(agentToolArgumentSuffix)
	if limit <= 0 {
		return agentToolArgumentSuffix
	}
	for limit > 0 && !utf8.ValidString(value[:limit]) {
		_, size := utf8.DecodeLastRuneInString(value[:limit])
		limit -= size
	}
	return value[:limit] + agentToolArgumentSuffix
}

func printAgentToolEvent(w io.Writer, event falken.Event) {
	newAgentToolPrinter().printEvent(w, event)
}

type agentToolPrinter struct {
	seenWarnings map[string]struct{}
}

func newAgentToolPrinter() *agentToolPrinter {
	return &agentToolPrinter{seenWarnings: map[string]struct{}{}}
}

func (p *agentToolPrinter) printEvent(w io.Writer, event falken.Event) {
	if event.Type == falken.EventThought && strings.HasPrefix(event.Text, "agent ") {
		fmt.Fprintln(w, event.Text)
	}
	if event.ToolCall != nil {
		fmt.Fprintf(w, "agent tool call: %s %s\n", event.ToolCall.Name, formatAgentToolArguments(event.ToolCall.Arguments))
	}
	if event.ToolResult != nil {
		for _, line := range formatAgentToolResult(*event.ToolResult) {
			if !p.shouldPrintLine(line) {
				continue
			}
			fmt.Fprintln(w, line)
		}
	}
}

func (p *agentToolPrinter) shouldPrintLine(line string) bool {
	if !strings.HasPrefix(line, "agent tool warning: ") {
		return true
	}
	if p.seenWarnings == nil {
		p.seenWarnings = map[string]struct{}{}
	}
	if _, ok := p.seenWarnings[line]; ok {
		return false
	}
	p.seenWarnings[line] = struct{}{}
	return true
}

func formatAgentToolResult(result falken.ToolResult) []string {
	switch result.Name {
	case agentask.SearchIndexToolName:
		if lines, ok := formatAgentSearchToolResult(result); ok {
			return lines
		}
	case agentask.ReadIndexSourceToolName:
		if lines, ok := formatAgentReadSourceToolResult(result); ok {
			return lines
		}
	case agentask.ReadIndexDocumentToolName:
		if lines, ok := formatAgentReadDocumentToolResult(result); ok {
			return lines
		}
	}
	status, success := agentToolResultStatus(result)
	out := []string{fmt.Sprintf("agent tool result: %s %s", result.Name, status)}
	if !success && result.Error != "" {
		out = append(out, fmt.Sprintf("agent tool error: %s: %s", result.Name, result.Error))
	}
	return out
}

type agentSearchDocumentPromotion struct {
	SourceNumber    int    `json:"source_number"`
	Path            string `json:"path"`
	Status          string `json:"status"`
	Reason          string `json:"reason"`
	StartLine       int    `json:"start_line"`
	EndLine         int    `json:"end_line"`
	Lines           int    `json:"lines"`
	EstimatedTokens int    `json:"estimated_tokens"`
	MaxLines        int    `json:"max_lines"`
	MaxTokens       int    `json:"max_tokens"`
}

type agentSearchPayload struct {
	Success         bool   `json:"success"`
	Status          string `json:"status"`
	Query           string `json:"query"`
	OriginalQuery   string `json:"original_query"`
	QueryNormalized bool   `json:"query_normalized"`
	TopK            int    `json:"top_k"`
	QueryPlan       struct {
		Mode    string   `json:"mode"`
		Queries []string `json:"queries"`
	} `json:"query_plan"`
	SeedQueryPlan struct {
		Mode    string   `json:"mode"`
		Queries []string `json:"queries"`
	} `json:"seed_query_plan"`
	Sources            []json.RawMessage              `json:"sources"`
	ExpansionQueries   []string                       `json:"expansion_queries"`
	SuggestedQueries   []string                       `json:"suggested_queries"`
	NewSources         int                            `json:"new_sources"`
	DuplicateSources   int                            `json:"duplicate_sources"`
	UniqueDocuments    int                            `json:"unique_documents"`
	RetrievalCalls     int                            `json:"retrieval_calls"`
	DocumentPromotions []agentSearchDocumentPromotion `json:"document_promotions"`
	Warnings           []string                       `json:"warnings"`
	Error              string                         `json:"error"`
}

func formatAgentSearchSummary(result falken.ToolResult, payload *agentSearchPayload) string {
	status := payload.Status
	if status == "" {
		status, _ = agentToolResultStatus(result)
	}
	summary := fmt.Sprintf("agent tool result: %s %s, query=%s, top_k=%d, sources=%d", result.Name, status, strconv.Quote(payload.Query), payload.TopK, len(payload.Sources))
	if payload.QueryNormalized && payload.OriginalQuery != "" {
		summary = fmt.Sprintf("agent tool result: %s %s, query=%s (normalized from %s), top_k=%d, sources=%d", result.Name, status, strconv.Quote(payload.Query), strconv.Quote(payload.OriginalQuery), payload.TopK, len(payload.Sources))
	}
	if payload.RetrievalCalls > 0 {
		summary += fmt.Sprintf(", new=%d, duplicates=%d, docs=%d, retrievals=%d", payload.NewSources, payload.DuplicateSources, payload.UniqueDocuments, payload.RetrievalCalls)
	}
	return summary
}

func formatAgentSearchQueries(payload *agentSearchPayload, out []string) []string {
	seedQueries := payload.SeedQueryPlan.Queries
	label := "agent search query plan:"
	if len(payload.ExpansionQueries) != 0 {
		label = "agent seed query plan:"
	}
	if len(seedQueries) == 0 {
		seedQueries = payload.QueryPlan.Queries
	}
	if len(seedQueries) != 0 {
		out = append(out, label)
		for i, query := range seedQueries {
			out = append(out, fmt.Sprintf("  %d. %s", i+1, query))
		}
	}
	if len(payload.ExpansionQueries) != 0 {
		out = append(out, "agent broad expansion queries:")
		for i, query := range payload.ExpansionQueries {
			out = append(out, fmt.Sprintf("  %d. %s", i+1, query))
		}
	}
	if len(payload.SuggestedQueries) != 0 {
		out = append(out, "agent suggested follow-up queries:")
		for i, query := range payload.SuggestedQueries {
			out = append(out, fmt.Sprintf("  %d. %s", i+1, query))
		}
	}
	return out
}

func formatAgentSearchPromotions(promotions []agentSearchDocumentPromotion, out []string) []string {
	for _, promotion := range promotions {
		source := rag.SourceChunk{
			SourceNumber: promotion.SourceNumber,
			Path:         promotion.Path,
			StartLine:    promotion.StartLine,
			EndLine:      promotion.EndLine,
		}
		switch promotion.Status {
		case "ok":
			out = append(out, fmt.Sprintf("agent document promotion: reading whole %s", SourceReference(source)))
			if promotion.Lines > 0 {
				out[len(out)-1] += fmt.Sprintf(", lines=%d, estimated_tokens=%d", promotion.Lines, promotion.EstimatedTokens)
			}
			if promotion.Reason != "" {
				out = append(out, "reason: "+promotion.Reason)
			}
		case "too_large":
			out = append(out, fmt.Sprintf("agent document promotion: skipped too_large %s, lines=%d, estimated_tokens=%d, max_tokens=%d", SourceReference(source), promotion.Lines, promotion.EstimatedTokens, promotion.MaxTokens))
			if promotion.Reason != "" {
				out = append(out, "reason: "+promotion.Reason)
			}
		case "budget_exhausted":
			out = append(out, fmt.Sprintf("agent document promotion: skipped budget %s", SourceReference(source)))
			if promotion.Reason != "" {
				out = append(out, "reason: "+promotion.Reason)
			}
		}
	}
	return out
}

func formatAgentSearchWarningsAndErrors(result falken.ToolResult, payload *agentSearchPayload, out []string) []string {
	for _, warning := range payload.Warnings {
		out = append(out, fmt.Sprintf("agent tool warning: %s: %s", result.Name, warning))
	}
	if payload.Error != "" {
		out = append(out, fmt.Sprintf("agent tool error: %s: %s", result.Name, payload.Error))
	}
	return out
}

func formatAgentSearchToolResult(result falken.ToolResult) ([]string, bool) {
	var payload agentSearchPayload
	if len(result.Payload) == 0 || json.Unmarshal(result.Payload, &payload) != nil {
		return nil, false
	}
	out := []string{formatAgentSearchSummary(result, &payload)}
	out = formatAgentSearchQueries(&payload, out)
	out = formatAgentSearchPromotions(payload.DocumentPromotions, out)
	out = formatAgentSearchWarningsAndErrors(result, &payload, out)
	return out, true
}

func formatAgentReadDocumentToolResult(result falken.ToolResult) ([]string, bool) {
	var payload struct {
		Success         bool     `json:"success"`
		Status          string   `json:"status"`
		SourceNumber    int      `json:"source_number"`
		Path            string   `json:"path"`
		Mode            string   `json:"mode"`
		ParentKind      string   `json:"parent_kind"`
		StartLine       int      `json:"start_line"`
		EndLine         int      `json:"end_line"`
		Lines           int      `json:"lines"`
		EstimatedTokens int      `json:"estimated_tokens"`
		MaxTokens       int      `json:"max_tokens"`
		Warnings        []string `json:"warnings"`
		Error           string   `json:"error"`
	}
	if len(result.Payload) == 0 || json.Unmarshal(result.Payload, &payload) != nil {
		return nil, false
	}
	status := payload.Status
	if status == "" {
		status, _ = agentToolResultStatus(result)
	}
	source := rag.SourceChunk{
		SourceNumber: payload.SourceNumber,
		Path:         payload.Path,
		StartLine:    payload.StartLine,
		EndLine:      payload.EndLine,
	}
	mode := payload.Mode
	if mode == "" {
		mode = "whole"
	}
	summary := fmt.Sprintf("agent tool result: %s %s, %s, mode=%s, lines=%d, estimated_tokens=%d", result.Name, status, SourceReference(source), mode, payload.Lines, payload.EstimatedTokens)
	if status == "too_large" {
		summary = fmt.Sprintf("agent tool result: %s too_large, %s, lines=%d, estimated_tokens=%d, max_tokens=%d", result.Name, SourceReference(source), payload.Lines, payload.EstimatedTokens, payload.MaxTokens)
	}
	if payload.ParentKind != "" && mode == "parent" {
		summary += ", parent_kind=" + payload.ParentKind
	}
	out := []string{summary}
	for _, warning := range payload.Warnings {
		out = append(out, fmt.Sprintf("agent tool warning: %s: %s", result.Name, warning))
	}
	if payload.Error != "" {
		out = append(out, fmt.Sprintf("agent tool error: %s: %s", result.Name, payload.Error))
	}
	return out, true
}

func formatAgentReadSourceToolResult(result falken.ToolResult) ([]string, bool) {
	var payload struct {
		Success               bool     `json:"success"`
		Status                string   `json:"status"`
		SourceNumber          int      `json:"source_number"`
		CoveredBySourceNumber int      `json:"covered_by_source_number"`
		Path                  string   `json:"path"`
		StartLine             int      `json:"start_line"`
		EndLine               int      `json:"end_line"`
		CoveredByPath         string   `json:"covered_by_path"`
		CoveredByStartLine    int      `json:"covered_by_start_line"`
		CoveredByEndLine      int      `json:"covered_by_end_line"`
		MaxMergedLines        int      `json:"max_merged_read_source_lines"`
		Warnings              []string `json:"warnings"`
		Error                 string   `json:"error"`
	}
	if len(result.Payload) == 0 || json.Unmarshal(result.Payload, &payload) != nil {
		return nil, false
	}
	status := payload.Status
	if status == "" {
		status, _ = agentToolResultStatus(result)
	}
	source := rag.SourceChunk{
		SourceNumber: payload.SourceNumber,
		Path:         payload.Path,
		StartLine:    payload.StartLine,
		EndLine:      payload.EndLine,
	}
	summary := fmt.Sprintf("agent tool result: %s %s, %s", result.Name, status, SourceReference(source))
	if status == "already_covered" && payload.CoveredBySourceNumber > 0 {
		covered := rag.SourceChunk{
			SourceNumber: payload.CoveredBySourceNumber,
			Path:         payload.CoveredByPath,
			StartLine:    payload.CoveredByStartLine,
			EndLine:      payload.CoveredByEndLine,
		}
		summary = fmt.Sprintf("agent tool result: %s ok, [source %d] already covered by %s", result.Name, payload.SourceNumber, SourceReference(covered))
	}
	if status == "merge_existing" && payload.CoveredBySourceNumber > 0 {
		covered := rag.SourceChunk{
			SourceNumber: payload.CoveredBySourceNumber,
			Path:         payload.CoveredByPath,
			StartLine:    payload.CoveredByStartLine,
			EndLine:      payload.CoveredByEndLine,
		}
		summary = fmt.Sprintf("agent tool result: %s ok, merged [source %d] into expanded %s", result.Name, payload.SourceNumber, SourceReference(covered))
	}
	if status == "merge_too_large" && payload.CoveredBySourceNumber > 0 {
		summary = fmt.Sprintf("agent tool result: %s ok, merge skipped for [source %d]; merging into [source %d] would exceed max merged read range %d lines",
			result.Name,
			payload.SourceNumber,
			payload.CoveredBySourceNumber,
			payload.MaxMergedLines,
		)
	}
	out := []string{summary}
	for _, warning := range payload.Warnings {
		out = append(out, fmt.Sprintf("agent tool warning: %s: %s", result.Name, warning))
	}
	if payload.Error != "" {
		out = append(out, fmt.Sprintf("agent tool error: %s: %s", result.Name, payload.Error))
	}
	return out, true
}

func agentToolResultStatus(result falken.ToolResult) (string, bool) {
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
	if status == "" {
		if success {
			status = "ok"
		} else {
			status = "error"
		}
	}
	return status, success
}
