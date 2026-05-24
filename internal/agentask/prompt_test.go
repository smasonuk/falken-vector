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
		"Do not invent source IDs",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt = %q, want substring %q", prompt, want)
		}
	}
}
