package retrievaleval

import (
	"testing"

	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
)

func TestEvaluateCasePerfectFirstRankHit(t *testing.T) {
	result := EvaluateCase(Case{
		ID:               "case",
		Question:         "question",
		ExpectedChunkIDs: []string{"chunk-1"},
	}, []rag.RetrievedChunk{
		retrieved("chunk-1", "doc", "README.md", 0.9),
	})
	if !result.Hit || result.Recall != 1 || result.Precision != 1 || result.ReciprocalRank != 1 {
		t.Fatalf("result = %+v, want perfect hit", result)
	}
}

func TestEvaluateCaseRelevantAtRankThree(t *testing.T) {
	result := EvaluateCase(Case{
		Question:         "question",
		ExpectedChunkIDs: []string{"chunk-3"},
	}, []rag.RetrievedChunk{
		retrieved("chunk-1", "doc", "a.md", 0.9),
		retrieved("chunk-2", "doc", "b.md", 0.8),
		retrieved("chunk-3", "doc", "c.md", 0.7),
	})
	if !result.Hit || result.FirstRelevantRank != 3 || result.ReciprocalRank != float64(1)/3 {
		t.Fatalf("result = %+v, want rank-three reciprocal rank", result)
	}
	if result.Precision != float64(1)/3 {
		t.Fatalf("precision = %v, want 1/3", result.Precision)
	}
}

func TestEvaluateCaseNoRelevantResults(t *testing.T) {
	result := EvaluateCase(Case{
		Question:         "question",
		ExpectedChunkIDs: []string{"missing"},
	}, []rag.RetrievedChunk{
		retrieved("chunk-1", "doc", "README.md", 0.9),
	})
	if result.Hit || result.Recall != 0 || result.ReciprocalRank != 0 {
		t.Fatalf("result = %+v, want no hit", result)
	}
}

func TestEvaluateCaseMatchesChunkIDPathAndSuffix(t *testing.T) {
	cases := []Case{
		{Question: "id", ExpectedChunkIDs: []string{"chunk-1"}},
		{Question: "path", ExpectedPaths: []string{"/repo/internal/rag/retrieve.go"}},
		{Question: "suffix", ExpectedPathSuffixes: []string{"internal/rag/retrieve.go"}},
	}
	for _, c := range cases {
		result := EvaluateCase(c, []rag.RetrievedChunk{
			retrieved("chunk-1", "doc", "/repo/internal/rag/retrieve.go", 0.9),
		})
		if !result.Hit || result.Recall != 1 {
			t.Fatalf("case %+v result = %+v, want match", c, result)
		}
	}
}

func TestEvaluateCaseDeduplicatesExpectations(t *testing.T) {
	result := EvaluateCase(Case{
		Question:             "question",
		ExpectedChunkIDs:     []string{"chunk-1", "chunk-1"},
		ExpectedPathSuffixes: []string{"internal/rag/retrieve.go", "internal/rag/retrieve.go"},
	}, []rag.RetrievedChunk{
		retrieved("chunk-1", "doc", "/repo/internal/rag/retrieve.go", 0.9),
	})
	if result.ExpectedRelevant != 2 {
		t.Fatalf("ExpectedRelevant = %d, want unique id + suffix expectations", result.ExpectedRelevant)
	}
	if result.Recall != 1 {
		t.Fatalf("Recall = %v, want 1 without double-counting duplicate expectations", result.Recall)
	}
}

func TestEvaluateCaseWithPlanCopiesQueryPlan(t *testing.T) {
	result := EvaluateCaseWithPlan(Case{
		Question:         "question",
		ExpectedChunkIDs: []string{"chunk-1"},
	}, []rag.RetrievedChunk{
		retrieved("chunk-1", "doc", "README.md", 0.9),
	}, []string{"question", "expanded"})
	if len(result.QueryPlan) != 2 || result.QueryPlan[1] != "expanded" {
		t.Fatalf("QueryPlan = %+v, want copied plan", result.QueryPlan)
	}
}

func TestSummarizeAveragesCaseMetrics(t *testing.T) {
	summary := Summarize([]CaseResult{
		{Hit: true, Recall: 1, Precision: 0.5, ReciprocalRank: 1},
		{Hit: false, Recall: 0, Precision: 0, ReciprocalRank: 0, Error: "boom"},
	}, 8, false)
	if summary.Cases != 2 || summary.FailedCases != 1 || summary.HitAtK != 0.5 || summary.RecallAtK != 0.5 || summary.PrecisionAtK != 0.25 || summary.MRRAtK != 0.5 {
		t.Fatalf("summary = %+v, want averaged metrics", summary)
	}
	if summary.CaseResults != nil {
		t.Fatalf("CaseResults = %+v, want omitted without details", summary.CaseResults)
	}
}

func retrieved(chunkID string, docID string, path string, score float32) rag.RetrievedChunk {
	return rag.RetrievedChunk{
		Chunk: manifest.Chunk{
			ID:         chunkID,
			DocumentID: docID,
			StartLine:  10,
			EndLine:    20,
		},
		Path:  path,
		Score: score,
	}
}
