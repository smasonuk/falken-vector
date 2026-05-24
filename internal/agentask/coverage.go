package agentask

import (
	"fmt"
	"strings"
)

type coveragePolicy struct {
	Enabled             bool
	MinBroadSearchCalls int
	MaxRetries          int
}

type coverageSearchStats struct {
	SearchCalls           int
	SuccessfulSearchCalls int
}

type coverageDecision struct {
	Nudge           bool
	Reason          string
	RemainingNeeded int
}

func normalizeCoveragePolicy(opts Options) coveragePolicy {
	enabled := true
	if opts.CoverageNudge != nil {
		enabled = *opts.CoverageNudge
	}
	minSearchCalls := opts.MinBroadSearchCalls
	if minSearchCalls <= 0 {
		minSearchCalls = 2
	}
	maxRetries := opts.MaxCoverageRetries
	if maxRetries <= 0 {
		maxRetries = 1
	}
	return coveragePolicy{
		Enabled:             enabled,
		MinBroadSearchCalls: minSearchCalls,
		MaxRetries:          maxRetries,
	}
}

func isBroadCoverageQuestion(question string) bool {
	normalized := strings.ToLower(strings.Join(strings.Fields(question), " "))
	if normalized == "" {
		return false
	}
	phrases := []string{
		"summarize",
		"summarise",
		"summary",
		"overview",
		"anything related",
		"related to",
		"tell me about",
		"what do we know",
		"all mentions",
		"all information",
		"everything about",
	}
	for _, phrase := range phrases {
		if strings.Contains(normalized, phrase) {
			return true
		}
	}
	return false
}

func searchStats(trace AgentTrace) coverageSearchStats {
	stats := coverageSearchStats{}
	for _, call := range trace.ToolCalls {
		if call.Name == SearchIndexToolName {
			stats.SearchCalls++
		}
	}
	for _, result := range trace.ToolResults {
		if result.Name == SearchIndexToolName && result.Success {
			stats.SuccessfulSearchCalls++
		}
	}
	return stats
}

func shouldCoverageNudge(
	question string,
	policy coveragePolicy,
	searchCalls int,
	successfulSearchCalls int,
	sourceCount int,
	maxSearchCalls int,
	coverageRetries int,
) bool {
	return coverageNudgeDecision(question, policy, searchCalls, successfulSearchCalls, sourceCount, maxSearchCalls, coverageRetries).Nudge
}

func coverageNudgeDecision(
	question string,
	policy coveragePolicy,
	searchCalls int,
	successfulSearchCalls int,
	sourceCount int,
	maxSearchCalls int,
	coverageRetries int,
) coverageDecision {
	if !policy.Enabled {
		return coverageDecision{Reason: "disabled"}
	}
	if !isBroadCoverageQuestion(question) {
		return coverageDecision{Reason: "not_broad"}
	}
	if searchCalls == 0 {
		return coverageDecision{Reason: "no_search"}
	}
	if sourceCount == 0 {
		return coverageDecision{Reason: "no_sources"}
	}
	if successfulSearchCalls >= policy.MinBroadSearchCalls {
		return coverageDecision{Reason: "enough_searches"}
	}
	remainingNeeded := coverageRemainingSearches(policy, successfulSearchCalls, searchCalls, maxSearchCalls)
	if remainingNeeded <= 0 {
		return coverageDecision{Reason: "search_budget_exhausted"}
	}
	if coverageRetries >= policy.MaxRetries {
		return coverageDecision{Reason: "retry_limit_reached", RemainingNeeded: remainingNeeded}
	}
	return coverageDecision{Nudge: true, RemainingNeeded: remainingNeeded}
}

func coverageRemainingSearches(policy coveragePolicy, successfulSearchCalls int, searchCalls int, maxSearchCalls int) int {
	needed := policy.MinBroadSearchCalls - successfulSearchCalls
	if needed <= 0 {
		return 0
	}
	remainingBudget := maxSearchCalls - searchCalls
	if remainingBudget <= 0 {
		return 0
	}
	if needed > remainingBudget {
		return remainingBudget
	}
	return needed
}

func coverageNudgePrompt(question string, previousAnswer string, searchCalls int, remainingNeeded int) string {
	return fmt.Sprintf(`Your previous answer was produced after only %d search_index call(s).

The user's question is broad or exploratory:
%s

Before finalizing, improve coverage:
1. Make at least %d additional search_index call(s) using materially different queries.
2. Base the new queries on terms, acronyms, paths, filenames, headings, entities, and concepts visible in the sources already returned.
3. Look for adjacent or related material, not just duplicates of the first search.
4. Then rewrite the answer using all relevant evidence found.
5. If the additional searches add no useful evidence, say that briefly.

Do not invent facts. Cite only [source N] IDs returned by search_index.

Previous answer:
%s`, searchCalls, question, remainingNeeded, previousAnswer)
}

func coverageNudgeWarning(successfulSearchCalls, remainingNeeded int) string {
	return fmt.Sprintf("coverage nudge: broad question answered after %d successful search_index call(s); requesting at least %d additional search_index call(s)", successfulSearchCalls, remainingNeeded)
}

func coverageSkippedWarning(reason string) string {
	switch reason {
	case "search_budget_exhausted":
		return "coverage nudge skipped: search call limit reached"
	case "retry_limit_reached":
		return "coverage nudge skipped: coverage retry limit reached"
	default:
		return ""
	}
}
