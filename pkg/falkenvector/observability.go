package falkenvector

import "fmt"

const defaultMaxMessageBytes = 8 * 1024
const redactedPreviewBytes = 512

func normalizeObservabilityConfig(config ObservabilityConfig) ObservabilityConfig {
	if config.MaxMessageBytes <= 0 {
		config.MaxMessageBytes = defaultMaxMessageBytes
	}
	return config
}

func observedText(content string, config ObservabilityConfig, include bool) string {
	if !include || content == "" {
		return ""
	}
	config = normalizeObservabilityConfig(config)
	if config.RawPayloads {
		return truncateBytes(content, config.MaxMessageBytes)
	}
	limit := redactedPreviewBytes
	if config.MaxMessageBytes < limit {
		limit = config.MaxMessageBytes
	}
	if len(content) <= limit {
		return fmt.Sprintf("[redacted %d bytes]", len(content))
	}
	return truncateBytes(content, limit) + "\n[truncated/redacted]"
}

func truncateBytes(content string, maxBytes int) string {
	if maxBytes <= 0 || content == "" {
		return ""
	}
	if len(content) <= maxBytes {
		return content
	}
	out := make([]rune, 0, maxBytes)
	bytes := 0
	for _, r := range content {
		runeBytes := len(string(r))
		if bytes+runeBytes > maxBytes {
			break
		}
		out = append(out, r)
		bytes += runeBytes
	}
	return string(out)
}
