package rag

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/smasonuk/falken-vector/internal/llm"
)

func TestParseQueryPlannerMode(t *testing.T) {
	for _, value := range []string{"", "none", "heuristic", "llm"} {
		if _, err := ParseQueryPlannerMode(value); err != nil {
			t.Fatalf("ParseQueryPlannerMode(%q): %v", value, err)
		}
	}
	if _, err := ParseQueryPlannerMode("bogus"); err == nil || !strings.Contains(err.Error(), "--query-planner must be none, heuristic, or llm") {
		t.Fatalf("ParseQueryPlannerMode invalid error = %v", err)
	}
}

func TestNormalizeRetrieveOptionsDefaultsPlannerNone(t *testing.T) {
	opts, err := normalizeRetrieveOptions(RetrieveOptions{})
	if err != nil {
		t.Fatalf("normalizeRetrieveOptions: %v", err)
	}
	if opts.QueryPlannerMode != QueryPlannerModeNone || opts.MaxSubqueries != 4 {
		t.Fatalf("opts = %+v, want planner none and max 4", opts)
	}
}

func TestNormalizeRetrieveOptionsRejectsInvalidPlanner(t *testing.T) {
	_, err := normalizeRetrieveOptions(RetrieveOptions{QueryPlannerMode: QueryPlannerMode("bad")})
	if err == nil || !strings.Contains(err.Error(), "--query-planner must be none, heuristic, or llm") {
		t.Fatalf("normalizeRetrieveOptions error = %v, want invalid planner", err)
	}
}

func TestHeuristicPlannerAlwaysIncludesOriginalQuestion(t *testing.T) {
	plan, err := HeuristicQueryPlanner{}.Plan(context.Background(), "How does compact avoid stale vectors?", QueryPlanOptions{MaxSubqueries: 4})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if len(plan.Queries) == 0 || plan.Queries[0] != "How does compact avoid stale vectors?" {
		t.Fatalf("queries = %+v, want original first", plan.Queries)
	}
}

func TestHeuristicPlannerExtractsCodeSymbolsFlagsEnvVarsAndPaths(t *testing.T) {
	terms := ExtractQueryTerms(`Where is ErrPendingRunDetected checked with --state-dir and FALKENGO_EMBEDDING_MODEL in internal/rag/retrieve.go?`)
	for _, want := range []string{"ErrPendingRunDetected", "--state-dir", "FALKENGO_EMBEDDING_MODEL", "internal/rag/retrieve.go"} {
		if !containsString(terms.CodeTokens, want) {
			t.Fatalf("code tokens = %+v, missing %q", terms.CodeTokens, want)
		}
	}
}

func TestHeuristicPlannerDeduplicatesAndRespectsMaxSubqueries(t *testing.T) {
	queries := BuildHeuristicSubqueries(`ErrPendingRunDetected ErrPendingRunDetected "ErrPendingRunDetected"`, 2)
	if len(queries) != 2 {
		t.Fatalf("queries = %+v, want max 2", queries)
	}
	if strings.EqualFold(queries[0], queries[1]) {
		t.Fatalf("queries = %+v, want deduped", queries)
	}
}

func TestHeuristicPlannerNormalizesCaseDuplicateTerms(t *testing.T) {
	queries := BuildHeuristicSubqueries("AlphaFold alphafold", 4)
	if len(queries) == 0 || queries[0] != "AlphaFold" {
		t.Fatalf("queries = %+v, want original query normalized to AlphaFold", queries)
	}
	for _, query := range queries {
		if strings.Contains(query, "AlphaFold alphafold") {
			t.Fatalf("queries = %+v, leaked case duplicate", queries)
		}
	}
}

func TestLLMPlannerParsesStrictJSONAndPrependsOriginalQuestion(t *testing.T) {
	planner := LLMQueryPlanner{LLM: fakePlannerLLM{text: `{"subqueries":["compact vector database","manifest activation"]}`}}
	plan, err := planner.Plan(context.Background(), "How does compact work?", QueryPlanOptions{MaxSubqueries: 3})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if !sameStringSlice(plan.Queries, []string{"How does compact work?", "compact vector database", "manifest activation"}) {
		t.Fatalf("queries = %+v", plan.Queries)
	}
}

func TestLLMPlannerFallsBackOnInvalidJSON(t *testing.T) {
	planner := LLMQueryPlanner{LLM: fakePlannerLLM{text: `not-json`}}
	plan, err := planner.Plan(context.Background(), "question", QueryPlanOptions{MaxSubqueries: 4})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if !sameStringSlice(plan.Queries, []string{"question"}) || plan.Warning == "" {
		t.Fatalf("plan = %+v, want original-only fallback with warning", plan)
	}
}

func TestLLMPlannerDeduplicatesAndLimits(t *testing.T) {
	planner := LLMQueryPlanner{LLM: fakePlannerLLM{text: `{"subqueries":["question","one","one","two","three"]}`}}
	plan, err := planner.Plan(context.Background(), "question", QueryPlanOptions{MaxSubqueries: 3})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if !sameStringSlice(plan.Queries, []string{"question", "one", "two"}) {
		t.Fatalf("queries = %+v", plan.Queries)
	}
}

func TestLLMPlannerFallsBackOnLLMError(t *testing.T) {
	planner := LLMQueryPlanner{LLM: fakePlannerLLM{err: errors.New("boom")}}
	plan, err := planner.Plan(context.Background(), "question", QueryPlanOptions{MaxSubqueries: 4})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if plan.Warning == "" || !sameStringSlice(plan.Queries, []string{"question"}) {
		t.Fatalf("plan = %+v, want fallback warning", plan)
	}
}

type fakePlannerLLM struct {
	text string
	err  error
}

func (f fakePlannerLLM) Complete(context.Context, llm.CompletionRequest) (llm.CompletionResponse, error) {
	if f.err != nil {
		return llm.CompletionResponse{}, f.err
	}
	return llm.CompletionResponse{Text: f.text}, nil
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func sameStringSlice(got []string, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
