package rag

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
)

type QueryPlannerMode string

const (
	QueryPlannerModeNone      QueryPlannerMode = "none"
	QueryPlannerModeHeuristic QueryPlannerMode = "heuristic"
	QueryPlannerModeLLM       QueryPlannerMode = "llm"
)

type QueryPlanner interface {
	Plan(ctx context.Context, question string, opts QueryPlanOptions) (QueryPlan, error)
}

type QueryPlanOptions struct {
	MaxSubqueries int
}

type QueryPlan struct {
	OriginalQuestion string   `json:"original_question"`
	Queries          []string `json:"queries"`
	Mode             string   `json:"mode"`
	Warning          string   `json:"warning,omitempty"`
}

type QueryTerms struct {
	CodeTokens []string
	Phrases    []string
	Terms      []string
}

type HeuristicQueryPlanner struct{}

var (
	queryCodeTokenPattern = regexp.MustCompile(`--[A-Za-z0-9][A-Za-z0-9-]*|[A-Z][A-Z0-9_]{2,}|[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+|[A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]+|[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+`)
	queryTermPattern      = regexp.MustCompile(`[A-Za-z0-9_./-]+`)
)

func ParseQueryPlannerMode(value string) (QueryPlannerMode, error) {
	switch QueryPlannerMode(strings.TrimSpace(value)) {
	case "", QueryPlannerModeNone:
		return QueryPlannerModeNone, nil
	case QueryPlannerModeHeuristic:
		return QueryPlannerModeHeuristic, nil
	case QueryPlannerModeLLM:
		return QueryPlannerModeLLM, nil
	default:
		return "", fmt.Errorf("--query-planner must be none, heuristic, or llm")
	}
}

func (HeuristicQueryPlanner) Plan(ctx context.Context, question string, opts QueryPlanOptions) (QueryPlan, error) {
	select {
	case <-ctx.Done():
		return QueryPlan{}, ctx.Err()
	default:
	}
	return QueryPlan{
		OriginalQuestion: strings.TrimSpace(question),
		Queries:          BuildHeuristicSubqueries(question, normalizeMaxSubqueries(opts.MaxSubqueries)),
		Mode:             string(QueryPlannerModeHeuristic),
	}, nil
}

func ExtractQueryTerms(question string) QueryTerms {
	seenCode := make(map[string]struct{})
	codeTokens := make([]string, 0)
	addCode := func(value string) {
		value = strings.TrimSpace(value)
		if value == "" {
			return
		}
		if _, ok := seenCode[value]; ok {
			return
		}
		seenCode[value] = struct{}{}
		codeTokens = append(codeTokens, value)
	}
	for _, token := range queryCodeTokenPattern.FindAllString(question, -1) {
		addCode(strings.Trim(token, ".,:;()[]{}"))
	}

	phrases := make([]string, 0)
	seenPhrases := make(map[string]struct{})
	for _, match := range quotedPhrasePattern.FindAllStringSubmatch(question, -1) {
		if len(match) < 2 {
			continue
		}
		phrase := strings.TrimSpace(match[1])
		key := strings.ToLower(phrase)
		if phrase == "" {
			continue
		}
		if _, ok := seenPhrases[key]; ok {
			continue
		}
		seenPhrases[key] = struct{}{}
		phrases = append(phrases, phrase)
	}

	terms := make([]string, 0)
	seenTerms := make(map[string]struct{})
	addTerm := func(term string) {
		term = strings.Trim(strings.ToLower(term), "_-./")
		if !usableRerankTerm(term) {
			return
		}
		if _, ok := seenTerms[term]; ok {
			return
		}
		seenTerms[term] = struct{}{}
		terms = append(terms, term)
	}
	for _, token := range queryTermPattern.FindAllString(question, -1) {
		addTerm(token)
		for _, part := range strings.FieldsFunc(token, func(r rune) bool {
			return r == '_' || r == '-' || r == '.' || r == '/'
		}) {
			addTerm(part)
		}
	}
	return QueryTerms{CodeTokens: codeTokens, Phrases: phrases, Terms: terms}
}

func BuildHeuristicSubqueries(question string, max int) []string {
	max = normalizeMaxSubqueries(max)
	question = normalizeQueryText(question)
	queries := make([]string, 0, max)
	add := func(query string) {
		query = normalizeQueryText(query)
		if query == "" {
			return
		}
		for _, existing := range queries {
			if strings.EqualFold(existing, query) {
				return
			}
		}
		if len(queries) < max {
			queries = append(queries, query)
		}
	}
	add(question)
	terms := ExtractQueryTerms(question)
	if len(terms.CodeTokens) != 0 {
		add(joinLimited(append(append([]string{}, terms.CodeTokens...), terms.Terms...), 8))
		for _, token := range terms.CodeTokens {
			add(joinLimited(append([]string{token}, terms.Terms...), 5))
		}
	}
	for _, phrase := range terms.Phrases {
		add(phrase)
	}
	if len(terms.Terms) != 0 {
		add(joinLimited(terms.Terms, 8))
	}
	return queries
}

func buildQueryPlan(ctx context.Context, opts RetrieveOptions) (QueryPlan, error) {
	question := normalizeQueryText(opts.Question)
	if question == "" {
		return QueryPlan{}, errors.New("question is required")
	}
	switch opts.QueryPlannerMode {
	case "", QueryPlannerModeNone:
		return QueryPlan{OriginalQuestion: question, Queries: []string{question}, Mode: string(QueryPlannerModeNone)}, nil
	case QueryPlannerModeHeuristic:
		planner := opts.QueryPlanner
		if planner == nil {
			planner = HeuristicQueryPlanner{}
		}
		return normalizePlan(ctx, planner, question, QueryPlanOptions{MaxSubqueries: opts.MaxSubqueries}, string(QueryPlannerModeHeuristic))
	case QueryPlannerModeLLM:
		if opts.QueryPlanner == nil {
			return QueryPlan{}, fmt.Errorf("query planner is required for llm mode")
		}
		return normalizePlan(ctx, opts.QueryPlanner, question, QueryPlanOptions{MaxSubqueries: opts.MaxSubqueries}, string(QueryPlannerModeLLM))
	default:
		return QueryPlan{}, fmt.Errorf("invalid query planner mode %q", opts.QueryPlannerMode)
	}
}

func normalizePlan(ctx context.Context, planner QueryPlanner, question string, opts QueryPlanOptions, mode string) (QueryPlan, error) {
	plan, err := planner.Plan(ctx, question, opts)
	if err != nil {
		return QueryPlan{}, err
	}
	plan.OriginalQuestion = question
	plan.Mode = mode
	plan.Queries = normalizeQueries(question, plan.Queries, normalizeMaxSubqueries(opts.MaxSubqueries))
	return plan, nil
}

func normalizeQueries(question string, queries []string, max int) []string {
	max = normalizeMaxSubqueries(max)
	out := make([]string, 0, max)
	add := func(query string) {
		query = normalizeQueryText(query)
		if query == "" {
			return
		}
		for _, existing := range out {
			if strings.EqualFold(existing, query) {
				return
			}
		}
		if len(out) < max {
			out = append(out, query)
		}
	}
	add(question)
	for _, query := range queries {
		add(query)
	}
	return out
}

func normalizeQueryText(query string) string {
	words := strings.Fields(query)
	out := make([]string, 0, len(words))
	seen := map[string]struct{}{}
	for _, word := range words {
		word = strings.Trim(word, " \t\r\n.,:;()[]{}")
		key := strings.ToLower(word)
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, word)
	}
	return strings.Join(out, " ")
}

func joinLimited(values []string, limit int) string {
	if limit <= 0 || len(values) <= limit {
		return strings.Join(values, " ")
	}
	return strings.Join(values[:limit], " ")
}

func normalizeMaxSubqueries(max int) int {
	if max <= 0 {
		return 4
	}
	if max > 8 {
		return 8
	}
	return max
}
