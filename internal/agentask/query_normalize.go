package agentask

import "strings"

func normalizeSearchQuery(query string) string {
	words := strings.Fields(query)
	out := make([]string, 0, len(words))
	seen := map[string]struct{}{}
	for _, word := range words {
		word = strings.Trim(word, " \t\r\n.,:;()[]{}")
		key := strings.ToLower(word)
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, word)
	}
	return strings.Join(out, " ")
}
