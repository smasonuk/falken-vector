package rag

import (
	"context"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

type Reranker interface {
	Rerank(ctx context.Context, question string, candidates []RerankCandidate, limit int) ([]RerankResult, error)
}

type RerankCandidate struct {
	ChunkID   string
	Path      string
	Text      string
	StartLine int
	EndLine   int
	Sources   []string
	Score     float32
	RankScore float64
}

type RerankResult struct {
	ChunkID     string
	Score       float64
	Explanation string
}

type HeuristicReranker struct{}

type scoredRerankCandidate struct {
	result        RerankResult
	originalRank  float64
	originalIndex int
}

var rerankTermPattern = regexp.MustCompile(`[A-Za-z0-9_./-]+`)
var quotedPhrasePattern = regexp.MustCompile(`"([^"]+)"`)

func (HeuristicReranker) Rerank(ctx context.Context, question string, candidates []RerankCandidate, limit int) ([]RerankResult, error) {
	if len(candidates) == 0 {
		return nil, nil
	}
	questionTerms := tokenizeRerankTerms(question)
	phrases := rerankPhrases(question)
	normalizedOriginal := normalizeOriginalScores(candidates)
	scored := make([]scoredRerankCandidate, 0, len(candidates))
	for i, candidate := range candidates {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		originalRank := originalRankScore(candidate)
		tokenOverlap := termOverlap(questionTerms, tokenizeRerankTerms(candidate.Text))
		phraseScore := phraseMatchScore(phrases, candidate)
		pathScore := termOverlap(questionTerms, tokenizeRerankTerms(candidate.Path+" "+filepath.Base(candidate.Path)))
		sourceScore := 0.0
		if len(uniqueStrings(candidate.Sources)) >= 2 {
			sourceScore = 1
		}
		score := 0.40*normalizedOriginal[i] +
			0.25*tokenOverlap +
			0.15*phraseScore +
			0.10*pathScore +
			0.10*sourceScore
		scored = append(scored, scoredRerankCandidate{
			result: RerankResult{
				ChunkID:     candidate.ChunkID,
				Score:       score,
				Explanation: rerankExplanation(normalizedOriginal[i], tokenOverlap, phraseScore, pathScore, sourceScore),
			},
			originalRank:  originalRank,
			originalIndex: i,
		})
	}
	sort.SliceStable(scored, func(i, j int) bool {
		if scored[i].result.Score != scored[j].result.Score {
			return scored[i].result.Score > scored[j].result.Score
		}
		if scored[i].originalRank != scored[j].originalRank {
			return scored[i].originalRank > scored[j].originalRank
		}
		if scored[i].originalIndex != scored[j].originalIndex {
			return scored[i].originalIndex < scored[j].originalIndex
		}
		return scored[i].result.ChunkID < scored[j].result.ChunkID
	})
	if limit > 0 && len(scored) > limit {
		scored = scored[:limit]
	}
	out := make([]RerankResult, 0, len(scored))
	for _, candidate := range scored {
		out = append(out, candidate.result)
	}
	return out, nil
}

func normalizeOriginalScores(candidates []RerankCandidate) []float64 {
	values := make([]float64, len(candidates))
	minScore := 0.0
	maxScore := 0.0
	for i, candidate := range candidates {
		value := originalRankScore(candidate)
		values[i] = value
		if i == 0 || value < minScore {
			minScore = value
		}
		if i == 0 || value > maxScore {
			maxScore = value
		}
	}
	if maxScore == minScore {
		for i := range values {
			values[i] = 0.5
		}
		return values
	}
	for i, value := range values {
		values[i] = (value - minScore) / (maxScore - minScore)
	}
	return values
}

func originalRankScore(candidate RerankCandidate) float64 {
	if candidate.RankScore != 0 {
		return candidate.RankScore
	}
	return float64(candidate.Score)
}

func tokenizeRerankTerms(input string) map[string]struct{} {
	matches := rerankTermPattern.FindAllString(input, -1)
	terms := make(map[string]struct{})
	add := func(term string) {
		term = strings.ToLower(strings.Trim(term, "_-./"))
		if !usableRerankTerm(term) {
			return
		}
		terms[term] = struct{}{}
	}
	for _, match := range matches {
		add(match)
		for _, part := range strings.FieldsFunc(match, func(r rune) bool {
			return r == '_' || r == '-' || r == '.' || r == '/'
		}) {
			add(part)
		}
	}
	return terms
}

func usableRerankTerm(term string) bool {
	if len(term) < 2 {
		return false
	}
	switch term {
	case "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what", "where", "why", "with":
		return false
	default:
		return true
	}
}

func termOverlap(questionTerms map[string]struct{}, candidateTerms map[string]struct{}) float64 {
	if len(questionTerms) == 0 {
		return 0
	}
	matches := 0
	for term := range questionTerms {
		if _, ok := candidateTerms[term]; ok {
			matches++
		}
	}
	return float64(matches) / float64(len(questionTerms))
}

func rerankPhrases(question string) []string {
	seen := make(map[string]struct{})
	phrases := make([]string, 0)
	add := func(phrase string) {
		phrase = normalizePhrase(phrase)
		if len(phrase) < 2 {
			return
		}
		if _, ok := seen[phrase]; ok {
			return
		}
		seen[phrase] = struct{}{}
		phrases = append(phrases, phrase)
	}
	for _, match := range quotedPhrasePattern.FindAllStringSubmatch(question, -1) {
		if len(match) > 1 {
			add(match[1])
		}
	}
	add(question)
	return phrases
}

func phraseMatchScore(phrases []string, candidate RerankCandidate) float64 {
	haystack := normalizePhrase(candidate.Path + " " + candidate.Text)
	for _, phrase := range phrases {
		if phrase != "" && strings.Contains(haystack, phrase) {
			return 1
		}
	}
	return 0
}

func normalizePhrase(input string) string {
	return strings.ToLower(strings.Join(strings.Fields(input), " "))
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func rerankExplanation(original float64, overlap float64, phrase float64, path float64, source float64) string {
	parts := make([]string, 0, 5)
	if original > 0 {
		parts = append(parts, "original")
	}
	if overlap > 0 {
		parts = append(parts, "token_overlap")
	}
	if phrase > 0 {
		parts = append(parts, "phrase")
	}
	if path > 0 {
		parts = append(parts, "path")
	}
	if source > 0 {
		parts = append(parts, "source_agreement")
	}
	return strings.Join(parts, ",")
}
