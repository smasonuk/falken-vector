package agentask

import (
	"path/filepath"
	"regexp"
	"strings"

	"github.com/smasonuk/falken-vector/internal/rag"
)

var (
	quotedPhrasePattern = regexp.MustCompile(`"([^"]{3,80})"|'([^']{3,80})'`)
	allCapsTokenPattern = regexp.MustCompile(`\b[A-Z0-9][A-Z0-9_-]{1,}\b`)
	camelTokenPattern   = regexp.MustCompile(`\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b`)
	wordTokenPattern    = regexp.MustCompile(`\b[A-Za-z][A-Za-z0-9_-]{4,}\b`)
	usefulPhrasePattern = regexp.MustCompile(`(?i)\b(error\s+JSON|structure\s+DB|shared\s+completed|shared\s+in[- ]progress|monomer|dimer|ligand|metadata|protein|proteins|workflow|pipeline)\b`)
)

func SuggestFollowupQueries(question string, sources []rag.SourceChunk, limit int) []string {
	if limit <= 0 {
		return nil
	}
	buckets := rankedSuggestionBuckets(question, sources, limit*6)
	groups := groupTermsForQueries(buckets)
	if len(groups) == 0 {
		return nil
	}
	topic := compactQueryTopic(question)
	queries := make([]string, 0, limit)
	for _, group := range groups {
		query := normalizeSearchQuery(joinUnique(append([]string{topic}, group...)))
		if query == "" || !materiallyDifferentQuery(query, question) {
			continue
		}
		queries = append(queries, query)
		if len(queries) >= limit {
			break
		}
	}
	return queries
}

func suggestedTerms(question string, sources []rag.SourceChunk, limit int) []string {
	return rankedSuggestionBuckets(question, sources, limit).flatten(limit)
}

type suggestionBuckets struct {
	phrases       []string
	allCaps       []string
	camel         []string
	usefulPhrases []string
	ordinary      []string
	path          []string
}

func (b suggestionBuckets) flatten(limit int) []string {
	out := make([]string, 0, limit)
	for _, values := range [][]string{b.phrases, b.allCaps, b.camel, b.usefulPhrases, b.ordinary, b.path} {
		for _, value := range values {
			if limit > 0 && len(out) >= limit {
				return out
			}
			out = append(out, value)
		}
	}
	return out
}

func rankedSuggestionBuckets(question string, sources []rag.SourceChunk, limit int) suggestionBuckets {
	seen := map[string]struct{}{}
	var buckets suggestionBuckets
	add := func(values *[]string, value string) {
		if limit > 0 && len(seen) >= limit {
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
		*values = append(*values, value)
	}

	for _, source := range sources {
		for _, match := range quotedPhrasePattern.FindAllStringSubmatch(source.Text, -1) {
			if match[1] != "" {
				add(&buckets.phrases, match[1])
			} else {
				add(&buckets.phrases, match[2])
			}
		}
		for _, token := range allCapsTokenPattern.FindAllString(source.Text, -1) {
			add(&buckets.allCaps, token)
		}
		for _, token := range camelTokenPattern.FindAllString(source.Text, -1) {
			add(&buckets.camel, token)
		}
		for _, phrase := range usefulPhrasePattern.FindAllString(source.Text, -1) {
			add(&buckets.usefulPhrases, phrase)
		}
		for _, token := range wordTokenPattern.FindAllString(source.Text, -1) {
			add(&buckets.ordinary, token)
		}
		for _, part := range splitPathTerms(source.Path) {
			add(&buckets.path, part)
		}
	}
	return buckets
}

func groupTermsForQueries(b suggestionBuckets) [][]string {
	groups := make([][]string, 0, 6)
	if group := preferredTerms([]string{"A3M", "PDB", "error JSON", "JSON", "FASTA"}, b.allCaps, b.usefulPhrases); len(group) != 0 {
		groups = append(groups, group)
	}
	if group := preferredTerms([]string{"UniProt", "metadata", "protein", "proteins"}, b.camel, b.usefulPhrases, b.ordinary); len(group) != 0 {
		groups = append(groups, group)
	}
	if group := preferredTerms([]string{"monomer", "dimer", "ligand", "workflow", "pipeline"}, b.usefulPhrases, b.ordinary); len(group) != 0 {
		groups = append(groups, group)
	}
	for _, values := range [][]string{b.phrases, b.allCaps, b.camel, b.usefulPhrases, b.ordinary, b.path} {
		for len(values) != 0 {
			n := 4
			if len(values) < n {
				n = len(values)
			}
			groups = append(groups, values[:n])
			values = values[n:]
		}
	}
	return dedupeTermGroups(groups)
}

func preferredTerms(preferred []string, sources ...[]string) []string {
	all := make([]string, 0)
	for _, values := range sources {
		all = append(all, values...)
	}
	out := make([]string, 0, 4)
	for _, want := range preferred {
		for _, value := range all {
			if strings.EqualFold(value, want) || strings.Contains(strings.ToLower(value), strings.ToLower(want)) {
				out = append(out, value)
				break
			}
		}
	}
	return uniqueTerms(out)
}

func dedupeTermGroups(groups [][]string) [][]string {
	out := make([][]string, 0, len(groups))
	seen := map[string]struct{}{}
	for _, group := range groups {
		group = uniqueTerms(group)
		if len(group) == 0 {
			continue
		}
		key := strings.ToLower(strings.Join(group, " "))
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, group)
	}
	return out
}

func uniqueTerms(values []string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]struct{}{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		key := strings.ToLower(value)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, value)
	}
	return out
}

func splitPathTerms(path string) []string {
	if filepath.IsAbs(path) {
		return nil
	}
	path = strings.NewReplacer("/", " ", "\\", " ", ".", " ", "-", " ", "_", " ").Replace(path)
	return strings.Fields(path)
}

func compactQueryTopic(question string) string {
	question = normalizeSearchQuery(question)
	tokens := wordTokenPattern.FindAllString(question, -1)
	out := make([]string, 0, 3)
	for _, token := range tokens {
		if !usefulTopicTerm(token) {
			continue
		}
		out = append(out, token)
		if len(out) >= 3 {
			break
		}
	}
	return strings.Join(uniqueTerms(out), " ")
}

func joinUnique(values []string) string {
	return strings.Join(uniqueTerms(values), " ")
}

func materiallyDifferentQuery(query, question string) bool {
	return !strings.EqualFold(normalizeSearchQuery(query), normalizeSearchQuery(question))
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

func usefulTopicTerm(term string) bool {
	normalized := strings.ToLower(strings.Trim(term, " \t\r\n.,:;()[]{}"))
	if normalized == "" {
		return false
	}
	if _, ok := topicStopWords[normalized]; ok {
		return false
	}
	return usefulSuggestionTerm(term, "")
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

var topicStopWords = map[string]struct{}{
	"anything":    {},
	"about":       {},
	"information": {},
	"overview":    {},
	"related":     {},
	"summarize":   {},
	"summarise":   {},
	"summary":     {},
}
