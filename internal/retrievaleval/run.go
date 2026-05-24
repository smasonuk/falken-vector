package retrievaleval

import (
	"context"
	"errors"
	"fmt"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/smasonuk/falken-vector/internal/vectorstore"
)

type Options struct {
	Paths             config.Paths
	TopK              int
	CandidateK        int
	VectorCandidateK  int
	LexicalCandidateK int
	Mode              rag.RetrievalMode
	RerankerMode      rag.RerankerMode
	Reranker          rag.Reranker
	QueryPlannerMode  rag.QueryPlannerMode
	QueryPlanner      rag.QueryPlanner
	MaxSubqueries     int
	Embedder          llm.Embedder
	Vector            vectorstore.Store
	OpenVector        func(context.Context, string, int) (vectorstore.Store, error)
	Details           bool
}

func Run(ctx context.Context, store manifest.Store, cases []Case, opts Options) (Summary, error) {
	if opts.TopK <= 0 {
		opts.TopK = 8
	}
	if opts.Mode == "" {
		opts.Mode = rag.RetrievalModeVector
	}
	if _, err := rag.ParseRetrievalMode(string(opts.Mode)); err != nil {
		return Summary{}, err
	}
	if _, err := rag.ParseRerankerMode(string(opts.RerankerMode)); err != nil {
		return Summary{}, err
	}
	if _, err := rag.ParseQueryPlannerMode(string(opts.QueryPlannerMode)); err != nil {
		return Summary{}, err
	}
	if rag.RetrievalModeUsesVector(opts.Mode) && opts.Embedder == nil {
		return Summary{}, errors.New("embedder is required")
	}
	if err := rag.CheckIndexForMode(opts.Paths, opts.Mode); err != nil {
		return Summary{}, err
	}
	results := make([]CaseResult, 0, len(cases))
	for _, c := range cases {
		retrieved, err := rag.RetrieveWithPlan(ctx, store, rag.RetrieveOptions{
			Question:          c.Question,
			TopK:              opts.TopK,
			CandidateK:        opts.CandidateK,
			VectorCandidateK:  opts.VectorCandidateK,
			LexicalCandidateK: opts.LexicalCandidateK,
			Mode:              opts.Mode,
			RerankerMode:      opts.RerankerMode,
			Reranker:          opts.Reranker,
			QueryPlannerMode:  opts.QueryPlannerMode,
			QueryPlanner:      opts.QueryPlanner,
			MaxSubqueries:     opts.MaxSubqueries,
			Paths:             opts.Paths,
			Embedder:          opts.Embedder,
			Vector:            opts.Vector,
			OpenVector:        opts.OpenVector,
		})
		if err != nil {
			results = append(results, CaseResult{
				CaseID:           c.ID,
				Question:         c.Question,
				ExpectedRelevant: len(buildExpectations(c)),
				Error:            fmt.Sprintf("%v", err),
			})
			continue
		}
		results = append(results, EvaluateCaseWithPlan(c, retrieved.Chunks, retrieved.Plan.Queries))
	}
	return Summarize(results, opts.TopK, opts.Details), nil
}
