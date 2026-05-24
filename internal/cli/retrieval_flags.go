package cli

import (
	"fmt"
	"os"

	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/spf13/cobra"
)

type retrievalFlagValues struct {
	mode              string
	reranker          string
	queryPlanner      string
	maxSubqueries     int
	showQueryPlan     bool
	candidateK        int
	vectorCandidateK  int
	lexicalCandidateK int
}

type sourceFilterFlagValues struct {
	include []string
	exclude []string
	roots   []string
}

func addRetrievalFlags(cmd *cobra.Command, values *retrievalFlagValues) {
	values.mode = string(rag.RetrievalModeVector)
	values.reranker = string(rag.RerankerModeNone)
	values.queryPlanner = string(rag.QueryPlannerModeNone)
	values.maxSubqueries = 4
	cmd.Flags().StringVar(&values.mode, "retrieval", values.mode, "retrieval mode: vector, lexical, or hybrid")
	cmd.Flags().StringVar(&values.reranker, "reranker", values.reranker, "reranker mode: none or heuristic")
	cmd.Flags().StringVar(&values.queryPlanner, "query-planner", values.queryPlanner, "query planner: none, heuristic, or llm")
	cmd.Flags().IntVar(&values.maxSubqueries, "max-subqueries", values.maxSubqueries, "maximum number of planned retrieval queries")
	cmd.Flags().BoolVar(&values.showQueryPlan, "show-query-plan", false, "print planned retrieval queries; eval includes per-case plans")
	cmd.Flags().IntVar(&values.candidateK, "candidate-k", 0, "candidate pool size for vector and lexical retrieval")
	cmd.Flags().IntVar(&values.vectorCandidateK, "vector-candidate-k", 0, "vector candidate pool size")
	cmd.Flags().IntVar(&values.lexicalCandidateK, "lexical-candidate-k", 0, "lexical candidate pool size")
}

func retrievalOptionsFromFlags(cmd *cobra.Command, values retrievalFlagValues) (rag.RetrieveOptions, error) {
	mode, err := rag.ParseRetrievalMode(values.mode)
	if err != nil {
		return rag.RetrieveOptions{}, err
	}
	rerankerMode, err := rag.ParseRerankerMode(values.reranker)
	if err != nil {
		return rag.RetrieveOptions{}, err
	}
	queryPlannerMode, err := rag.ParseQueryPlannerMode(values.queryPlanner)
	if err != nil {
		return rag.RetrieveOptions{}, err
	}
	if values.maxSubqueries < 1 || values.maxSubqueries > 8 {
		return rag.RetrieveOptions{}, fmt.Errorf("--max-subqueries must be between 1 and 8")
	}
	if cmd.Flags().Changed("candidate-k") && values.candidateK <= 0 {
		return rag.RetrieveOptions{}, fmt.Errorf("--candidate-k must be > 0")
	}
	if cmd.Flags().Changed("vector-candidate-k") && values.vectorCandidateK <= 0 {
		return rag.RetrieveOptions{}, fmt.Errorf("--vector-candidate-k must be > 0")
	}
	if cmd.Flags().Changed("lexical-candidate-k") && values.lexicalCandidateK <= 0 {
		return rag.RetrieveOptions{}, fmt.Errorf("--lexical-candidate-k must be > 0")
	}
	return rag.RetrieveOptions{
		Mode:              mode,
		RerankerMode:      rerankerMode,
		QueryPlannerMode:  queryPlannerMode,
		MaxSubqueries:     values.maxSubqueries,
		CandidateK:        values.candidateK,
		VectorCandidateK:  values.vectorCandidateK,
		LexicalCandidateK: values.lexicalCandidateK,
	}, nil
}

func configureQueryPlanner(opts *rag.RetrieveOptions) error {
	if opts.QueryPlannerMode != rag.QueryPlannerModeLLM {
		return nil
	}
	client, err := newCLIChatClient()
	if err != nil {
		return fmt.Errorf("configure query planner LLM: %w", err)
	}
	opts.QueryPlanner = rag.LLMQueryPlanner{LLM: client}
	return nil
}

func addSourceFilterFlags(cmd *cobra.Command, values *sourceFilterFlagValues) {
	cmd.Flags().StringArrayVar(&values.include, "include", nil, "include only matching source paths; can be repeated")
	cmd.Flags().StringArrayVar(&values.exclude, "exclude", nil, "exclude matching source paths; can be repeated")
	cmd.Flags().StringArrayVar(&values.roots, "source-root", nil, "restrict retrieval to files under this source root; can be repeated")
}

func sourceFilterFromFlags(values sourceFilterFlagValues) (rag.SourceFilter, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return rag.SourceFilter{}, fmt.Errorf("get current directory: %w", err)
	}
	return rag.ParseSourceFilter(values.include, values.exclude, values.roots, cwd)
}
