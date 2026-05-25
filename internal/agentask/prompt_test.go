package agentask

import (
	"strings"
	"testing"
)

func TestAgentSystemPromptDirectsBroadQuestionsToSearchMore(t *testing.T) {
	prompt := agentSystemPrompt(false, false)
	for _, want := range []string{
		"broad, exploratory, summary",
		"at least two materially different queries",
		"suggested follow-up queries",
		"mostly duplicates or no new evidence",
		"fewer than 3 new sources",
		"already searched twice",
		"Do not invent source IDs",
		"[source 2] [source 3]",
		"Never write [sources 2, 3]",
		"one bracket per cited source",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt = %q, want substring %q", prompt, want)
		}
	}
}

func TestCorrectiveCitationPromptForbidsGroupedCitations(t *testing.T) {
	prompt := correctiveCitationPrompt([]string{"bad citation"}, "Previous [sources 2, 3].")
	for _, want := range []string{
		"[source 2] [source 3]",
		"Never write [sources 2, 3]",
		"one bracket per cited source",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt = %q, want substring %q", prompt, want)
		}
	}
}

func TestReadSourcePromptIncludesBroadSummaryThinSpanGuidance(t *testing.T) {
	prompt := agentSystemPrompt(true, false)
	for _, want := range []string{
		"read_index_source only accepts source numbers",
		"For broad summaries",
		"one or two lines",
		"several relevant sources come from the same document",
		"read one representative source",
		"expanded line ranges would largely overlap",
		"already covered by another source",
		"merged into another expanded source",
		"cite the covering source number",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt = %q, want substring %q", prompt, want)
		}
	}
}

func TestReadDocumentPromptIncludesDocumentGuidance(t *testing.T) {
	prompt := agentSystemPrompt(true, true)
	for _, want := range []string{
		"Use read_index_document when",
		"summary, overview, audit, timeline, comparison",
		"many returned sources are from the same file",
		"Do not use read_index_document when",
		"precise lookup",
		"Use read_index_source for local expansion",
		"Use read_index_document for whole-file",
		"Never invent paths",
		"only accepts source_number values",
		"[source 2] [source 3]",
		"Never write [sources 2, 3]",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt = %q, want substring %q", prompt, want)
		}
	}
	if strings.Contains(prompt, "path argument") {
		t.Fatalf("prompt = %q, should not suggest raw path arguments", prompt)
	}
}
