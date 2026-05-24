package rag

import (
	"context"
	"reflect"
	"testing"
)

func TestParseRerankerMode(t *testing.T) {
	tests := []struct {
		input string
		want  RerankerMode
		ok    bool
	}{
		{"", RerankerModeNone, true},
		{"none", RerankerModeNone, true},
		{"heuristic", RerankerModeHeuristic, true},
		{"cross-encoder", "", false},
	}
	for _, tt := range tests {
		got, err := ParseRerankerMode(tt.input)
		if tt.ok && err != nil {
			t.Fatalf("ParseRerankerMode(%q): %v", tt.input, err)
		}
		if !tt.ok && err == nil {
			t.Fatalf("ParseRerankerMode(%q) succeeded, want error", tt.input)
		}
		if got != tt.want {
			t.Fatalf("ParseRerankerMode(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestHeuristicRerankerBoostsTokenOverlap(t *testing.T) {
	results := rerankForTest(t, "where is write lock acquired", []RerankCandidate{
		{ChunkID: "low", Text: "unrelated content", RankScore: 0.5},
		{ChunkID: "high", Text: "the write lock is acquired before reset", RankScore: 0.5},
	})
	if results[0].ChunkID != "high" {
		t.Fatalf("results = %+v, want token-overlap candidate first", results)
	}
}

func TestHeuristicRerankerBoostsPathMatch(t *testing.T) {
	results := rerankForTest(t, "internal/rag/retrieve.go", []RerankCandidate{
		{ChunkID: "text", Path: "internal/config/paths.go", Text: "unrelated content", RankScore: 0.5},
		{ChunkID: "path", Path: "internal/rag/retrieve.go", Text: "unrelated content", RankScore: 0.5},
	})
	if results[0].ChunkID != "path" {
		t.Fatalf("results = %+v, want path-match candidate first", results)
	}
}

func TestHeuristicRerankerBoostsSourceAgreement(t *testing.T) {
	results := rerankForTest(t, "write lock", []RerankCandidate{
		{ChunkID: "single", Text: "write lock", RankScore: 0.2, Sources: []string{"vector"}},
		{ChunkID: "both", Text: "write lock", RankScore: 0.2, Sources: []string{"vector", "lexical"}},
	})
	if results[0].ChunkID != "both" {
		t.Fatalf("results = %+v, want source-agreement candidate first", results)
	}
}

func TestHeuristicRerankerIsDeterministic(t *testing.T) {
	candidates := []RerankCandidate{
		{ChunkID: "a", Path: "a.md", Text: "alpha beta", RankScore: 0.2},
		{ChunkID: "b", Path: "b.md", Text: "alpha gamma", RankScore: 0.3},
		{ChunkID: "c", Path: "c.md", Text: "delta", RankScore: 0.1},
	}
	first := rerankForTest(t, "alpha", candidates)
	second := rerankForTest(t, "alpha", candidates)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("rerank not deterministic\nfirst=%+v\nsecond=%+v", first, second)
	}
}

func TestHeuristicRerankerPreservesAllCandidates(t *testing.T) {
	candidates := []RerankCandidate{
		{ChunkID: "a", Text: "alpha"},
		{ChunkID: "b", Text: "beta"},
		{ChunkID: "c", Text: "gamma"},
	}
	results := rerankForTest(t, "alpha", candidates)
	if len(results) != len(candidates) {
		t.Fatalf("result count = %d, want %d", len(results), len(candidates))
	}
}

func rerankForTest(t *testing.T, question string, candidates []RerankCandidate) []RerankResult {
	t.Helper()
	results, err := (HeuristicReranker{}).Rerank(context.Background(), question, candidates, 0)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	return results
}
