package agentask

import (
	"path/filepath"
	"regexp"
	"strings"
	"unicode"

	"github.com/smasonuk/falken-vector/internal/rag"
)

var (
	quotedPhrasePattern = regexp.MustCompile(`"([^"]{3,80})"|'([^']{3,80})'`)
	allCapsTokenPattern = regexp.MustCompile(`\b[A-Z0-9][A-Z0-9_-]{1,}\b`)
	camelTokenPattern   = regexp.MustCompile(`\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b`)
	wordTokenPattern    = regexp.MustCompile(`\b[A-Za-z][A-Za-z0-9_-]{4,}\b`)
	usefulPhrasePattern = regexp.MustCompile(`(?i)\b(error\s+JSON|structure\s+DB|shared\s+completed|shared\s+in[- ]progress|monomer|dimer|ligand|metadata|protein|proteins|workflow|pipeline)\b`)
)

type SuggestionPurpose int

const (
	SuggestionPurposeBroadExpansion SuggestionPurpose = iota
	SuggestionPurposeAgentHint
)

func SuggestFollowupQueries(question string, sources []rag.SourceChunk, limit int) []string {
	return SuggestBroadExpansionQueries(question, sources, limit)
}

func SuggestBroadExpansionQueries(question string, sources []rag.SourceChunk, limit int) []string {
	return suggestFollowupQueries(question, sources, limit, SuggestionPurposeBroadExpansion)
}

func SuggestAgentFollowupHints(question string, sources []rag.SourceChunk, limit int) []string {
	return suggestFollowupQueries(question, sources, limit, SuggestionPurposeAgentHint)
}

func suggestFollowupQueries(question string, sources []rag.SourceChunk, limit int, purpose SuggestionPurpose) []string {
	if limit <= 0 {
		return nil
	}
	buckets := rankedSuggestionBuckets(question, sources, limit*6)
	groups := groupTermsForQueries(buckets)
	if len(groups) == 0 {
		return nil
	}
	topic := compactQueryTopic(question)
	if purpose == SuggestionPurposeBroadExpansion {
		cleanedGroups := make([][]string, 0, len(groups))
		for _, group := range groups {
			if cleaned := cleanBroadExpansionGroup(group); len(cleaned) != 0 {
				cleanedGroups = append(cleanedGroups, cleaned)
			}
		}
		availableTerms := flattenTermGroups(cleanedGroups)
		candidates := make([]expansionQueryCandidate, 0, len(groups))
		for _, group := range cleanedGroups {
			group = enrichExpansionGroup(group, availableTerms)
			query := normalizeSearchQuery(joinUnique(append([]string{topic}, group...)))
			if query == "" || !materiallyDifferentQuery(query, question) {
				continue
			}
			if !usefulExpansionGroup(group) || !usefulExpansionQuery(query, question) {
				continue
			}
			candidates = append(candidates, expansionQueryCandidate{
				Query:    query,
				Category: expansionCategoryForGroup(group),
			})
		}
		return selectExpansionQueries(candidates, limit)
	}

	queries := make([]string, 0, limit)
	for _, group := range groups {
		query := normalizeSearchQuery(joinUnique(append([]string{topic}, group...)))
		if query == "" || !materiallyDifferentQuery(query, question) {
			continue
		}
		switch purpose {
		case SuggestionPurposeAgentHint:
			if !usefulHintGroup(group) || !usefulHintQuery(query, question) {
				continue
			}
		}
		queries = append(queries, query)
		if len(queries) >= limit {
			break
		}
	}
	return queries
}

func flattenTermGroups(groups [][]string) []string {
	out := make([]string, 0)
	for _, group := range groups {
		out = append(out, group...)
	}
	return uniqueTerms(out)
}

type expansionQueryCandidate struct {
	Query    string
	Category ExpansionCategory
}

func selectExpansionQueries(candidates []expansionQueryCandidate, limit int) []string {
	if limit <= 0 {
		return nil
	}
	selected := make([]string, 0, limit)
	usedCategories := map[ExpansionCategory]struct{}{}
	deferred := make([]expansionQueryCandidate, 0)
	for _, candidate := range candidates {
		if _, ok := usedCategories[candidate.Category]; ok && candidate.Category != CategoryUnknown {
			deferred = append(deferred, candidate)
			continue
		}
		if tooSimilarExpansionQuery(candidate.Query, selected, 0.75) {
			continue
		}
		selected = append(selected, candidate.Query)
		usedCategories[candidate.Category] = struct{}{}
		if len(selected) >= limit {
			return selected
		}
	}
	for _, candidate := range deferred {
		if tooSimilarExpansionQuery(candidate.Query, selected, 0.75) {
			continue
		}
		selected = append(selected, candidate.Query)
		if len(selected) >= limit {
			break
		}
	}
	return selected
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
	if group := preferredTerms([]string{"A3M", "PDB", "error JSON", "JSON", "FASTA", "NCBI"}, b.allCaps, b.usefulPhrases); len(group) != 0 {
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

func usefulExpansionGroup(group []string) bool {
	joined := strings.Join(group, " ")
	return len(group) != 0 &&
		!containsTranscriptFragment(joined) &&
		highSignalTermCount(joined) > 0
}

func cleanExpansionGroup(group []string) []string {
	return cleanBroadExpansionGroup(group)
}

func cleanBroadExpansionGroup(group []string) []string {
	out := make([]string, 0, len(group))
	for _, term := range group {
		term = strings.Trim(term, " \t\r\n.,:;()[]{}")
		if term == "" {
			continue
		}
		if usefulExpansionTerm(term) {
			out = append(out, term)
			continue
		}
		words := strings.Fields(term)
		if len(words) > 1 {
			cleanedWords := make([]string, 0, len(words))
			for _, word := range words {
				if usefulExpansionTerm(word) {
					cleanedWords = append(cleanedWords, word)
				}
			}
			if len(cleanedWords) != 0 {
				out = append(out, strings.Join(cleanedWords, " "))
			}
			continue
		}
	}
	return uniqueTerms(out)
}

func enrichExpansionGroup(group []string, availableTerms []string) []string {
	category := expansionCategoryForGroup(group)
	if category == CategoryUnknown {
		return group
	}
	out := append([]string(nil), group...)
	for _, fallback := range categoryFallbackTerms[category] {
		if containsTermFold(out, fallback) || !containsTermFold(availableTerms, fallback) {
			continue
		}
		out = append(out, fallback)
		signal := expansionQuerySignalForTerms(out, nil)
		if signal.SemanticTerms >= 2 || signal.TechnicalTerms >= 1 || signal.UsefulPhrases >= 1 {
			break
		}
	}
	return uniqueTerms(out)
}

func containsTermFold(values []string, term string) bool {
	for _, value := range values {
		if strings.EqualFold(value, term) {
			return true
		}
	}
	return false
}

func usefulExpansionTerm(term string) bool {
	term = strings.Trim(term, " \t\r\n.,:;()[]{}")
	if term == "" {
		return false
	}
	if isKnownTechnicalToken(term) || isKnownUsefulExpansionWord(term) {
		return true
	}
	if isNumericNoiseToken(term) || isUnknownAllCapsNoise(term) {
		return false
	}
	return false
}

func isKnownTechnicalToken(token string) bool {
	normalized := strings.ToLower(strings.Trim(token, " \t\r\n.,:;()[]{}"))
	if normalized == "" {
		return false
	}
	if _, ok := highSignalExpansionTokens[normalized]; ok {
		return true
	}
	for _, phrase := range highSignalExpansionPhrases {
		if normalized == phrase {
			return true
		}
	}
	return false
}

func isKnownUsefulExpansionWord(token string) bool {
	normalized := strings.ToLower(strings.Trim(token, " \t\r\n.,:;()[]{}"))
	if _, ok := usefulExpansionWords[normalized]; ok {
		return true
	}
	return false
}

func isNumericNoiseToken(token string) bool {
	token = strings.Trim(token, " \t\r\n.,:;()[]{}")
	if token == "" {
		return false
	}
	digits := 0
	for _, r := range token {
		if unicode.IsDigit(r) {
			digits++
			continue
		}
		if r == '-' || r == '_' || r == '.' {
			continue
		}
		return false
	}
	return digits > 0
}

func isUnknownAllCapsNoise(token string) bool {
	token = strings.Trim(token, " \t\r\n.,:;()[]{}")
	if token == "" || isKnownTechnicalToken(token) {
		return false
	}
	hasLetter := false
	for _, r := range token {
		if unicode.IsLetter(r) {
			hasLetter = true
			if !unicode.IsUpper(r) {
				return false
			}
			continue
		}
		if unicode.IsDigit(r) || r == '-' || r == '_' {
			continue
		}
		return false
	}
	return hasLetter && len(token) >= 2
}

func tooSimilarExpansionQuery(candidate string, existing []string, threshold float64) bool {
	if threshold <= 0 {
		threshold = 0.75
	}
	candidateTokens := expansionQueryTokenSet(candidate)
	if len(candidateTokens) == 0 {
		return false
	}
	for _, query := range existing {
		if tokenJaccardSimilarity(candidateTokens, expansionQueryTokenSet(query)) >= threshold {
			return true
		}
	}
	return false
}

func expansionQueryTokenSet(query string) map[string]struct{} {
	tokens := normalizedQueryTokens(query)
	out := make(map[string]struct{}, len(tokens))
	for _, token := range tokens {
		if token == "" {
			continue
		}
		out[token] = struct{}{}
	}
	return out
}

func tokenJaccardSimilarity(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	intersection := 0
	for token := range a {
		if _, ok := b[token]; ok {
			intersection++
		}
	}
	union := len(a) + len(b) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func usefulExpansionQuery(query string, originalQuestion string) bool {
	query = normalizeSearchQuery(query)
	if query == "" || !materiallyDifferentQuery(query, originalQuestion) {
		return false
	}
	if len(query) > 100 {
		return false
	}
	tokens := normalizedQueryTokens(query)
	if len(tokens) == 0 || len(tokens) > 10 {
		return false
	}
	if containsTranscriptFragment(query) {
		return false
	}
	if mostlySpeechTokens(tokens) {
		return false
	}
	signal := expansionQuerySignalScore(query, compactQueryTopic(originalQuestion))
	return signal.TechnicalTerms >= 1 || signal.UsefulPhrases >= 1 || signal.SemanticTerms >= 2
}

func usefulHintGroup(group []string) bool {
	return len(group) != 0 && !containsTranscriptFragment(strings.Join(group, " "))
}

func usefulHintQuery(query string, originalQuestion string) bool {
	query = normalizeSearchQuery(query)
	if query == "" || !materiallyDifferentQuery(query, originalQuestion) {
		return false
	}
	if len(query) > 120 {
		return false
	}
	tokens := normalizedQueryTokens(query)
	if len(tokens) == 0 || len(tokens) > 12 {
		return false
	}
	if containsTranscriptFragment(query) {
		return false
	}
	if highSignalTermCount(query) == 0 && mostlySpeechTokens(tokens) {
		return false
	}
	return true
}

type ExpansionCategory string

const (
	CategoryUnknown  ExpansionCategory = ""
	CategoryArtifact ExpansionCategory = "artifact"
	CategoryMetadata ExpansionCategory = "metadata"
	CategorySpecies  ExpansionCategory = "species"
	CategoryModality ExpansionCategory = "modality"
	CategoryPipeline ExpansionCategory = "pipeline"
)

type expansionSignal struct {
	TechnicalTerms int
	UsefulPhrases  int
	SemanticTerms  int
}

func expansionQuerySignalScore(query string, topic string) expansionSignal {
	topicTokens := normalizedQueryTokenSet(topic)
	return expansionQuerySignalForTerms(strings.Fields(query), topicTokens)
}

func expansionQuerySignalForTerms(terms []string, topicTokens map[string]struct{}) expansionSignal {
	var signal expansionSignal
	joined := strings.ToLower(strings.Join(terms, " "))
	for _, phrase := range highSignalExpansionPhrases {
		if strings.Contains(joined, phrase) {
			signal.UsefulPhrases++
		}
	}
	seenSemantic := map[string]struct{}{}
	for _, term := range terms {
		for _, token := range normalizedQueryTokens(term) {
			if _, ok := topicTokens[token]; ok {
				continue
			}
			if _, ok := technicalExpansionTokens[token]; ok {
				signal.TechnicalTerms++
				continue
			}
			if _, ok := semanticExpansionTokens[token]; ok {
				if _, seen := seenSemantic[token]; seen {
					continue
				}
				seenSemantic[token] = struct{}{}
				signal.SemanticTerms++
			}
		}
	}
	return signal
}

func normalizedQueryTokenSet(query string) map[string]struct{} {
	tokens := normalizedQueryTokens(query)
	out := make(map[string]struct{}, len(tokens))
	for _, token := range tokens {
		out[token] = struct{}{}
	}
	return out
}

func expansionCategoryForGroup(group []string) ExpansionCategory {
	counts := map[ExpansionCategory]int{}
	for _, term := range group {
		for _, token := range normalizedQueryTokens(term) {
			category := expansionCategoryForToken(token)
			if category == CategoryUnknown {
				continue
			}
			counts[category]++
		}
	}
	best := CategoryUnknown
	bestCount := 0
	for _, category := range []ExpansionCategory{CategoryArtifact, CategoryMetadata, CategorySpecies, CategoryModality, CategoryPipeline} {
		if counts[category] > bestCount {
			best = category
			bestCount = counts[category]
		}
	}
	return best
}

func expansionCategoryForToken(token string) ExpansionCategory {
	switch strings.ToLower(token) {
	case "a3m", "fasta", "pdb", "json":
		return CategoryArtifact
	case "metadata", "method", "methods", "version", "versions", "structure", "structures":
		return CategoryMetadata
	case "alignment", "alignments", "mmseqs", "ncbi", "sequence", "sequences", "species", "uniprot":
		return CategorySpecies
	case "dimer", "ligand", "monomer":
		return CategoryModality
	case "pipeline", "workflow":
		return CategoryPipeline
	default:
		return CategoryUnknown
	}
}

func normalizedQueryTokens(query string) []string {
	words := strings.Fields(normalizeSearchQuery(query))
	out := make([]string, 0, len(words))
	for _, word := range words {
		word = strings.ToLower(strings.Trim(word, " \t\r\n.,:;()[]{}"))
		if word == "" {
			continue
		}
		out = append(out, word)
	}
	return out
}

func containsTranscriptFragment(value string) bool {
	normalized := " " + strings.ToLower(strings.Join(strings.Fields(value), " ")) + " "
	for _, fragment := range transcriptNoiseFragments {
		if strings.Contains(normalized, " "+fragment+" ") {
			return true
		}
	}
	return false
}

func highSignalTermCount(value string) int {
	normalized := strings.ToLower(strings.Join(strings.Fields(value), " "))
	count := 0
	for _, phrase := range highSignalExpansionPhrases {
		if strings.Contains(normalized, phrase) {
			count++
		}
	}
	for _, token := range normalizedQueryTokens(value) {
		if _, ok := highSignalExpansionTokens[token]; ok {
			count++
		}
	}
	for _, token := range allCapsTokenPattern.FindAllString(value, -1) {
		if isKnownTechnicalToken(token) {
			count++
		}
	}
	return count
}

func mostlySpeechTokens(tokens []string) bool {
	if len(tokens) == 0 {
		return true
	}
	speech := 0
	for _, token := range tokens {
		if len(token) <= 2 {
			speech++
			continue
		}
		if _, ok := expansionSpeechStopWords[token]; ok {
			speech++
		}
	}
	return speech*2 >= len(tokens)
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

var highSignalExpansionPhrases = []string{
	"error json",
	"structure db",
	"shared completed",
	"shared in progress",
}

var highSignalExpansionTokens = map[string]struct{}{
	"a3m":       {},
	"alphafold": {},
	"dimer":     {},
	"fasta":     {},
	"json":      {},
	"ligand":    {},
	"metadata":  {},
	"monomer":   {},
	"mmseqs":    {},
	"ncbi":      {},
	"pdb":       {},
	"pipeline":  {},
	"protein":   {},
	"proteins":  {},
	"uniprot":   {},
	"workflow":  {},
}

var technicalExpansionTokens = map[string]struct{}{
	"a3m":     {},
	"fasta":   {},
	"json":    {},
	"mmseqs":  {},
	"ncbi":    {},
	"pdb":     {},
	"uniprot": {},
}

var semanticExpansionTokens = map[string]struct{}{
	"alignment":  {},
	"alignments": {},
	"dimer":      {},
	"ligand":     {},
	"metadata":   {},
	"method":     {},
	"methods":    {},
	"monomer":    {},
	"pipeline":   {},
	"protein":    {},
	"proteins":   {},
	"sequence":   {},
	"sequences":  {},
	"species":    {},
	"structure":  {},
	"structures": {},
	"version":    {},
	"versions":   {},
	"workflow":   {},
}

var usefulExpansionWords = map[string]struct{}{
	"a3m":        {},
	"alignment":  {},
	"alignments": {},
	"alphafold":  {},
	"dimer":      {},
	"fasta":      {},
	"json":       {},
	"ligand":     {},
	"metadata":   {},
	"method":     {},
	"methods":    {},
	"mmseqs":     {},
	"monomer":    {},
	"ncbi":       {},
	"pdb":        {},
	"pipeline":   {},
	"protein":    {},
	"proteins":   {},
	"sequence":   {},
	"sequences":  {},
	"species":    {},
	"structure":  {},
	"structures": {},
	"uniprot":    {},
	"version":    {},
	"versions":   {},
	"workflow":   {},
}

var categoryFallbackTerms = map[ExpansionCategory][]string{
	CategoryArtifact: {"PDB", "JSON", "FASTA", "A3M"},
	CategoryMetadata: {"method", "version", "metadata"},
	CategorySpecies:  {"species", "NCBI", "UniProt", "sequence", "alignment"},
	CategoryModality: {"monomer", "dimer", "ligand"},
	CategoryPipeline: {"workflow", "pipeline", "alignment", "A3M"},
}

var transcriptNoiseFragments = []string{
	"for example",
	"going to",
	"got any",
	"these two",
	"probably about",
	"what was",
	"which are",
	"which is",
	"kind of",
	"sort of",
	"you know",
	"i think",
}

var expansionSpeechStopWords = map[string]struct{}{
	"about":    {},
	"again":    {},
	"also":     {},
	"and":      {},
	"any":      {},
	"are":      {},
	"because":  {},
	"been":     {},
	"being":    {},
	"don":      {},
	"example":  {},
	"for":      {},
	"from":     {},
	"got":      {},
	"going":    {},
	"here":     {},
	"into":     {},
	"probably": {},
	"really":   {},
	"run":      {},
	"say":      {},
	"several":  {},
	"that":     {},
	"these":    {},
	"this":     {},
	"those":    {},
	"two":      {},
	"used":     {},
	"was":      {},
	"what":     {},
	"which":    {},
	"with":     {},
	"would":    {},
	"yeah":     {},
}
