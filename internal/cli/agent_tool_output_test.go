package cli

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/smasonuk/falken-core/pkg/falken"
)

func TestFormatAgentToolArgumentsCompactsJSON(t *testing.T) {
	got := formatAgentToolArguments(json.RawMessage(`{
		"query": "hello",
		"top_k": 8
	}`))
	if got != `{"query":"hello","top_k":8}` {
		t.Fatalf("formatAgentToolArguments = %q", got)
	}
}

func TestFormatAgentToolArgumentsEmpty(t *testing.T) {
	got := formatAgentToolArguments(nil)
	if got != `{}` {
		t.Fatalf("formatAgentToolArguments = %q, want {}", got)
	}
}

func TestFormatAgentToolArgumentsInvalidJSON(t *testing.T) {
	got := formatAgentToolArguments(json.RawMessage(`{"query":`))
	if got != `<invalid arguments>` {
		t.Fatalf("formatAgentToolArguments = %q, want invalid marker", got)
	}
}

func TestFormatAgentToolArgumentsRedactsSensitiveKeysRecursively(t *testing.T) {
	got := formatAgentToolArguments(json.RawMessage(`{
		"query": "hello",
		"Authorization": "Bearer secret",
		"nested": {
			"api_key": "abc",
			"items": [
				{"password": "pw"},
				{"safe": "value"}
			]
		}
	}`))
	for _, leaked := range []string{"Bearer secret", "abc", "pw"} {
		if strings.Contains(got, leaked) {
			t.Fatalf("formatAgentToolArguments = %q, leaked %q", got, leaked)
		}
	}
	for _, want := range []string{`"Authorization":"[redacted]"`, `"api_key":"[redacted]"`, `"password":"[redacted]"`, `"safe":"value"`} {
		if !strings.Contains(got, want) {
			t.Fatalf("formatAgentToolArguments = %q, want %s", got, want)
		}
	}
}

func TestFormatAgentToolArgumentsTruncatesOversizedArguments(t *testing.T) {
	got := formatAgentToolArguments(json.RawMessage(`{"query":"` + strings.Repeat("x", maxAgentToolArgumentBytes+100) + `"}`))
	if len(got) != maxAgentToolArgumentBytes {
		t.Fatalf("length = %d, want %d", len(got), maxAgentToolArgumentBytes)
	}
	if !strings.HasSuffix(got, agentToolArgumentSuffix) {
		t.Fatalf("formatAgentToolArguments = %q, want truncation suffix", got)
	}
}

func TestFormatAgentToolResultSearchIndexSummary(t *testing.T) {
	lines := formatAgentToolResult(falken.ToolResult{
		Name: "search_index",
		Payload: json.RawMessage(`{
			"success": true,
			"status": "ok",
			"query": "AlphaFold folding",
			"top_k": 12,
			"query_plan": {"queries": ["AlphaFold folding", "A3M PDB"]},
			"sources": [
				{"text": "source text that must not leak"},
				{"text": "more source text"}
			],
			"warnings": ["top_k raised to configured floor 12"]
		}`),
	})
	got := strings.Join(lines, "\n")
	for _, want := range []string{
		`agent tool result: search_index ok, query="AlphaFold folding", top_k=12, sources=2`,
		"agent search query plan:",
		"  1. AlphaFold folding",
		"  2. A3M PDB",
		"agent tool warning: search_index: top_k raised to configured floor 12",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("lines = %q, want %q", got, want)
		}
	}
	if strings.Contains(got, "source text that must not leak") {
		t.Fatalf("lines = %q, leaked source text", got)
	}
}

func TestFormatAgentToolResultReadSourceSummaryOmitsText(t *testing.T) {
	lines := formatAgentToolResult(falken.ToolResult{
		Name: "read_index_source",
		Payload: json.RawMessage(`{
			"success": true,
			"status": "ok",
			"source_number": 1,
			"path": "README.md",
			"start_line": 1,
			"end_line": 30,
			"text": "expanded text that must not leak"
		}`),
	})
	got := strings.Join(lines, "\n")
	if !strings.Contains(got, "agent tool result: read_index_source ok, [source 1] README.md:1-30") {
		t.Fatalf("lines = %q, want read source summary", got)
	}
	if strings.Contains(got, "expanded text") {
		t.Fatalf("lines = %q, leaked expanded text", got)
	}
}
