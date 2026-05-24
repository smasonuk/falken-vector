package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/smasonuk/falken-vector/internal/llm"
)

type LLMQueryPlanner struct {
	LLM   llm.Client
	Model string
}

type llmQueryPlanResponse struct {
	Subqueries []string `json:"subqueries"`
}

func (p LLMQueryPlanner) Plan(ctx context.Context, question string, opts QueryPlanOptions) (QueryPlan, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return QueryPlan{}, fmt.Errorf("question is required")
	}
	if p.LLM == nil {
		return QueryPlan{}, fmt.Errorf("query planner LLM client is required")
	}
	response, err := p.LLM.Complete(ctx, llm.CompletionRequest{
		Model:       p.Model,
		System:      "You generate search queries for retrieval over a local source/document index.\nReturn strict JSON only.\nGenerate concise retrieval subqueries that preserve code identifiers, file paths, symbols, config keys, and error names.\nDo not answer the question.",
		User:        "Question:\n" + question + "\n\nReturn JSON:\n{\"subqueries\":[\"...\"]}",
		Temperature: 0,
	})
	if err != nil {
		return fallbackLLMPlan(question, opts.MaxSubqueries, fmt.Sprintf("query planner LLM failed: %v", err)), nil
	}
	var parsed llmQueryPlanResponse
	if err := json.Unmarshal([]byte(strings.TrimSpace(response.Text)), &parsed); err != nil {
		return fallbackLLMPlan(question, opts.MaxSubqueries, fmt.Sprintf("query planner returned invalid JSON: %v", err)), nil
	}
	queries := normalizeQueries(question, parsed.Subqueries, normalizeMaxSubqueries(opts.MaxSubqueries))
	if len(queries) == 0 {
		return fallbackLLMPlan(question, opts.MaxSubqueries, "query planner returned no subqueries"), nil
	}
	return QueryPlan{
		OriginalQuestion: question,
		Queries:          queries,
		Mode:             string(QueryPlannerModeLLM),
	}, nil
}

func fallbackLLMPlan(question string, max int, warning string) QueryPlan {
	return QueryPlan{
		OriginalQuestion: question,
		Queries:          normalizeQueries(question, nil, normalizeMaxSubqueries(max)),
		Mode:             string(QueryPlannerModeLLM),
		Warning:          warning,
	}
}
