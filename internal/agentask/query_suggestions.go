package agentask

import (
	"regexp"
	"strings"

	"github.com/smasonuk/falken-vector/internal/rag"
)

var (
	quotedPhrasePattern = regexp.MustCompile(`"([^"]{3,80})"|'([^']{3,80})'`)
	allCapsTokenPattern = regexp.MustCompile(`\b[A-Z0-9][A-Z0-9_-]{1,}\b`)
	camelTokenPattern   = regexp.MustCompile(`\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b`)
	wordTokenPattern    = regexp.MustCompile(`\b[A-Za-z][A-Za-z0-9_-]{4,}\b`)
)

func SuggestFollowupQueries(question string, sources []rag.SourceChunk, limit int) []string {
	if limit <= 0 {
		return nil
	}
	terms := suggestedTerms(question, sources, limit*4)
	if len(terms) == 0 {
		return nil
	}
	queries := make([]string, 0, limit)
	for len(terms) > 0 && len(queries) < limit {
		n := 4
		if len(terms) < n {
			n = len(terms)
		}
		queries = append(queries, strings.Join(terms[:n], " "))
		terms = terms[n:]
	}
	return queries
}

func suggestedTerms(question string, sources []rag.SourceChunk, limit int) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, limit)
	add := func(value string) {
		if len(out) >= limit {
			return
		}
		value = strings.Trim(value, " \t\r\n.,:;()[]{}")
		if !usefulSuggestionTerm(value, question) {
			return
		}
		key := strings.ToLower(value)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		out = append(out, value)
	}

	for _, source := range sources {
		for _, part := range splitPathTerms(source.Path) {
			add(part)
		}
		for _, match := range quotedPhrasePattern.FindAllStringSubmatch(source.Text, -1) {
			if match[1] != "" {
				add(match[1])
			} else {
				add(match[2])
			}
		}
		for _, token := range allCapsTokenPattern.FindAllString(source.Text, -1) {
			add(token)
		}
		for _, token := range camelTokenPattern.FindAllString(source.Text, -1) {
			add(token)
		}
		for _, token := range wordTokenPattern.FindAllString(source.Text, -1) {
			add(token)
		}
	}
	return out
}

func splitPathTerms(path string) []string {
	path = strings.NewReplacer("/", " ", "\\", " ", ".", " ", "-", " ", "_", " ").Replace(path)
	return strings.Fields(path)
}

func usefulSuggestionTerm(term, question string) bool {
	if len(term) < 3 {
		return false
	}
	normalized := strings.ToLower(term)
	if _, ok := suggestionStopWords[normalized]; ok {
		return false
	}
	if strings.Contains(strings.ToLower(question), normalized) && len(term) < 6 {
		return false
	}
	return true
}

var suggestionStopWords = map[string]struct{}{
	"about":       {},
	"after":       {},
	"before":      {},
	"chunk":       {},
	"chunks":      {},
	"document":    {},
	"files":       {},
	"index":       {},
	"indexed":     {},
	"information": {},
	"internal":    {},
	"line":        {},
	"lines":       {},
	"local":       {},
	"query":       {},
	"result":      {},
	"results":     {},
	"search":      {},
	"source":      {},
	"sources":     {},
	"summary":     {},
	"that":        {},
	"their":       {},
	"there":       {},
	"these":       {},
	"thing":       {},
	"those":       {},
	"using":       {},
	"with":        {},
}
