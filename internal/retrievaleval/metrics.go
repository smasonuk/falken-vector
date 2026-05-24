package retrievaleval

import (
	"path/filepath"
	"strings"

	"github.com/smasonuk/falken-vector/internal/rag"
)

type RetrievedMatch struct {
	Rank        int     `json:"rank"`
	ChunkID     string  `json:"chunk_id"`
	Path        string  `json:"path"`
	StartLine   int     `json:"start_line"`
	EndLine     int     `json:"end_line"`
	Score       float32 `json:"score"`
	Relevant    bool    `json:"relevant"`
	MatchReason string  `json:"match_reason,omitempty"`
}

type CaseResult struct {
	CaseID            string           `json:"case_id"`
	Question          string           `json:"question"`
	ExpectedRelevant  int              `json:"expected_relevant"`
	Retrieved         []RetrievedMatch `json:"retrieved"`
	Hit               bool             `json:"hit"`
	RelevantRetrieved int              `json:"relevant_retrieved"`
	Precision         float64          `json:"precision"`
	Recall            float64          `json:"recall"`
	ReciprocalRank    float64          `json:"reciprocal_rank"`
	FirstRelevantRank int              `json:"first_relevant_rank,omitempty"`
	QueryPlan         []string         `json:"query_plan,omitempty"`
	Error             string           `json:"error,omitempty"`
}

type Summary struct {
	Cases          int          `json:"cases"`
	EvaluatedCases int          `json:"evaluated_cases"`
	FailedCases    int          `json:"failed_cases"`
	TopK           int          `json:"top_k"`
	HitAtK         float64      `json:"hit_at_k"`
	RecallAtK      float64      `json:"recall_at_k"`
	PrecisionAtK   float64      `json:"precision_at_k"`
	MRRAtK         float64      `json:"mrr_at_k"`
	CaseResults    []CaseResult `json:"case_results,omitempty"`
}

func EvaluateCase(c Case, results []rag.RetrievedChunk) CaseResult {
	return EvaluateCaseWithPlan(c, results, nil)
}

func EvaluateCaseWithPlan(c Case, results []rag.RetrievedChunk, plan []string) CaseResult {
	expectations := buildExpectations(c)
	result := CaseResult{
		CaseID:           c.ID,
		Question:         c.Question,
		ExpectedRelevant: len(expectations),
		Retrieved:        make([]RetrievedMatch, 0, len(results)),
		QueryPlan:        append([]string(nil), plan...),
	}
	matchedExpectations := make(map[string]struct{})
	for i, retrieved := range results {
		relevant, reason, matchedKeys := matchRetrieved(retrieved, expectations)
		match := RetrievedMatch{
			Rank:        i + 1,
			ChunkID:     retrieved.Chunk.ID,
			Path:        retrieved.Path,
			StartLine:   retrieved.Chunk.StartLine,
			EndLine:     retrieved.Chunk.EndLine,
			Score:       retrieved.Score,
			Relevant:    relevant,
			MatchReason: reason,
		}
		result.Retrieved = append(result.Retrieved, match)
		if !relevant {
			continue
		}
		result.Hit = true
		result.RelevantRetrieved++
		if result.FirstRelevantRank == 0 {
			result.FirstRelevantRank = match.Rank
			result.ReciprocalRank = 1 / float64(match.Rank)
		}
		for _, key := range matchedKeys {
			matchedExpectations[key] = struct{}{}
		}
	}
	if len(results) != 0 {
		result.Precision = float64(result.RelevantRetrieved) / float64(len(results))
	}
	if result.ExpectedRelevant != 0 {
		result.Recall = float64(len(matchedExpectations)) / float64(result.ExpectedRelevant)
	}
	return result
}

func Summarize(results []CaseResult, topK int, includeDetails bool) Summary {
	summary := Summary{
		Cases:          len(results),
		EvaluatedCases: len(results),
		TopK:           topK,
	}
	for _, result := range results {
		if result.Error != "" {
			summary.FailedCases++
		}
		if result.Hit {
			summary.HitAtK++
		}
		summary.RecallAtK += result.Recall
		summary.PrecisionAtK += result.Precision
		summary.MRRAtK += result.ReciprocalRank
	}
	if len(results) != 0 {
		denominator := float64(len(results))
		summary.HitAtK /= denominator
		summary.RecallAtK /= denominator
		summary.PrecisionAtK /= denominator
		summary.MRRAtK /= denominator
	}
	if includeDetails {
		summary.CaseResults = results
	}
	return summary
}

type expectation struct {
	Key    string
	Kind   string
	Value  string
	Slash  string
	Reason string
}

func buildExpectations(c Case) []expectation {
	var expectations []expectation
	seen := make(map[string]struct{})
	add := func(kind string, value string) {
		if strings.TrimSpace(value) == "" {
			return
		}
		key := kind + ":" + value
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		exp := expectation{Key: key, Kind: kind, Value: value, Slash: filepath.ToSlash(value)}
		switch kind {
		case "chunk_id":
			exp.Reason = "chunk_id"
		case "path":
			exp.Reason = "path"
		case "path_suffix":
			exp.Reason = "path_suffix"
		}
		expectations = append(expectations, exp)
	}
	for _, id := range c.ExpectedChunkIDs {
		add("chunk_id", id)
	}
	for _, path := range c.ExpectedPaths {
		add("path", path)
	}
	for _, suffix := range c.ExpectedPathSuffixes {
		add("path_suffix", suffix)
	}
	return expectations
}

func matchRetrieved(retrieved rag.RetrievedChunk, expectations []expectation) (bool, string, []string) {
	pathSlash := filepath.ToSlash(retrieved.Path)
	var reasons []string
	var keys []string
	for _, exp := range expectations {
		matched := false
		switch exp.Kind {
		case "chunk_id":
			matched = retrieved.Chunk.ID == exp.Value
		case "path":
			matched = retrieved.Path == exp.Value
		case "path_suffix":
			matched = strings.HasSuffix(pathSlash, exp.Slash)
		}
		if matched {
			reasons = append(reasons, exp.Reason)
			keys = append(keys, exp.Key)
		}
	}
	return len(keys) != 0, strings.Join(reasons, ","), keys
}
