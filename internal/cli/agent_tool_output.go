package cli

import (
	"bytes"
	"encoding/json"
	"strings"
	"unicode/utf8"
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
