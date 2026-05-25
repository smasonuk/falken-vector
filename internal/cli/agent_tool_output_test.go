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
			"seed_query_plan": {"queries": ["AlphaFold folding", "A3M PDB"]},
			"expansion_queries": ["AlphaFold folding A3M PDB error JSON"],
			"suggested_queries": ["AlphaFold folding UniProt metadata"],
			"new_sources": 2,
			"duplicate_sources": 1,
			"unique_documents": 2,
			"retrieval_calls": 4,
			"sources": [
				{"text": "source text that must not leak"},
				{"text": "more source text"}
			],
			"warnings": ["top_k raised to configured floor 12"]
		}`),
	})
	got := strings.Join(lines, "\n")
	for _, want := range []string{
		`agent tool result: search_index ok, query="AlphaFold folding", top_k=12, sources=2, new=2, duplicates=1, docs=2, retrievals=4`,
		"agent seed query plan:",
		"  1. AlphaFold folding",
		"  2. A3M PDB",
		"agent broad expansion queries:",
		"  1. AlphaFold folding A3M PDB error JSON",
		"agent suggested follow-up queries:",
		"  1. AlphaFold folding UniProt metadata",
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

func TestFormatAgentToolResultSearchIndexShowsNormalizedQuery(t *testing.T) {
	lines := formatAgentToolResult(falken.ToolResult{
		Name: "search_index",
		Payload: json.RawMessage(`{
			"success": true,
			"status": "ok",
			"query": "protein folding",
			"original_query": "protein folding folding",
			"query_normalized": true,
			"top_k": 12,
			"sources": []
		}`),
	})
	got := strings.Join(lines, "\n")
	if !strings.Contains(got, `query="protein folding" (normalized from "protein folding folding")`) {
		t.Fatalf("lines = %q, want normalized query note", got)
	}
}

func TestFormatAgentToolResultSearchIndexOmitsNormalizedQueryWhenUnchanged(t *testing.T) {
	lines := formatAgentToolResult(falken.ToolResult{
		Name: "search_index",
		Payload: json.RawMessage(`{
			"success": true,
			"status": "ok",
			"query": "protein folding",
			"top_k": 12,
			"sources": []
		}`),
	})
	got := strings.Join(lines, "\n")
	if strings.Contains(got, "normalized from") {
		t.Fatalf("lines = %q, did not want normalization note", got)
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

func TestFormatAgentToolResultReadSourceAlreadyCovered(t *testing.T) {
	lines := formatAgentToolResult(falken.ToolResult{
		Name: "read_index_source",
		Payload: json.RawMessage(`{
			"success": true,
			"status": "already_covered",
			"source_number": 2,
			"path": "alphafold/meetings/sdb.md",
			"start_line": 3,
			"end_line": 95,
			"covered_by_source_number": 11,
			"covered_by_path": "alphafold/meetings/sdb.md",
			"covered_by_start_line": 1,
			"covered_by_end_line": 86
		}`),
	})
	got := strings.Join(lines, "\n")
	if !strings.Contains(got, "agent tool result: read_index_source ok, [source 2] already covered by [source 11] alphafold/meetings/sdb.md:1-86") {
		t.Fatalf("lines = %q, want already-covered summary", got)
	}
}

func TestFormatAgentToolResultReadSourceMergeExisting(t *testing.T) {
	lines := formatAgentToolResult(falken.ToolResult{
		Name: "read_index_source",
		Payload: json.RawMessage(`{
			"success": true,
			"status": "merge_existing",
			"source_number": 2,
			"path": "alphafold/meetings/sdb.md",
			"start_line": 3,
			"end_line": 124,
			"covered_by_source_number": 11,
			"covered_by_path": "alphafold/meetings/sdb.md",
			"covered_by_start_line": 1,
			"covered_by_end_line": 124
		}`),
	})
	got := strings.Join(lines, "\n")
	if !strings.Contains(got, "agent tool result: read_index_source ok, merged [source 2] into expanded [source 11] alphafold/meetings/sdb.md:1-124") {
		t.Fatalf("lines = %q, want merge summary", got)
	}
}
