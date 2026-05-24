package cli

import (
	"encoding/json"
	"strings"
	"testing"
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
