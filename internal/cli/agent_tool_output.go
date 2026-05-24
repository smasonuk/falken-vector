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
	}
	status, success := agentToolResultStatus(result)
	out := []string{fmt.Sprintf("agent tool result: %s %s", result.Name, status)}
	if !success && result.Error != "" {
		out = append(out, fmt.Sprintf("agent tool error: %s: %s", result.Name, result.Error))
	}
	return out
}

func formatAgentSearchToolResult(result falken.ToolResult) ([]string, bool) {
	var payload struct {
		Success   bool   `json:"success"`
		Status    string `json:"status"`
		Query     string `json:"query"`
		TopK      int    `json:"top_k"`
		QueryPlan struct {
			Mode    string   `json:"mode"`
			Queries []string `json:"queries"`
		} `json:"query_plan"`
		SeedQueryPlan struct {
			Mode    string   `json:"mode"`
			Queries []string `json:"queries"`
		} `json:"seed_query_plan"`
		Sources          []json.RawMessage `json:"sources"`
		ExpansionQueries []string          `json:"expansion_queries"`
		SuggestedQueries []string          `json:"suggested_queries"`
		NewSources       int               `json:"new_sources"`
		DuplicateSources int               `json:"duplicate_sources"`
		UniqueDocuments  int               `json:"unique_documents"`
		RetrievalCalls   int               `json:"retrieval_calls"`
		Warnings         []string          `json:"warnings"`
		Error            string            `json:"error"`
	}
	if len(result.Payload) == 0 || json.Unmarshal(result.Payload, &payload) != nil {
		return nil, false
	}
	status := payload.Status
	if status == "" {
		status, _ = agentToolResultStatus(result)
	}
	summary := fmt.Sprintf("agent tool result: %s %s, query=%s, top_k=%d, sources=%d", result.Name, status, strconv.Quote(payload.Query), payload.TopK, len(payload.Sources))
	if payload.RetrievalCalls > 0 {
		summary += fmt.Sprintf(", new=%d, duplicates=%d, docs=%d, retrievals=%d", payload.NewSources, payload.DuplicateSources, payload.UniqueDocuments, payload.RetrievalCalls)
	}
	out := []string{summary}
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
		Success      bool     `json:"success"`
		Status       string   `json:"status"`
		SourceNumber int      `json:"source_number"`
		Path         string   `json:"path"`
		StartLine    int      `json:"start_line"`
		EndLine      int      `json:"end_line"`
		Warnings     []string `json:"warnings"`
		Error        string   `json:"error"`
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
	out := []string{fmt.Sprintf("agent tool result: %s %s, %s", result.Name, status, SourceReference(source))}
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
