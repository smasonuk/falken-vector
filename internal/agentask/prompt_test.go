package agentask

import (
	"strings"
	"testing"
)

func TestAgentSystemPromptDirectsBroadQuestionsToSearchMore(t *testing.T) {
	prompt := agentSystemPrompt(false)
	for _, want := range []string{
		"broad, exploratory, summary",
		"at least two materially different queries",
		"suggested follow-up queries",
		"mostly duplicates or no new evidence",
		"Do not invent source IDs",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt = %q, want substring %q", prompt, want)
		}
	}
}

func TestReadSourcePromptIncludesBroadSummaryThinSpanGuidance(t *testing.T) {
	prompt := agentSystemPrompt(true)
	for _, want := range []string{
		"read_index_source only accepts source numbers",
		"For broad summaries",
		"one or two lines",
		"several relevant sources come from the same document",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt = %q, want substring %q", prompt, want)
		}
	}
}
