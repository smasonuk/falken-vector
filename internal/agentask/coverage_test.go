package agentask

import (
	"strings"
	"testing"
)

func TestIsBroadCoverageQuestion(t *testing.T) {
	tests := []struct {
		name     string
		question string
		want     bool
	}{
		{name: "summarize", question: "Summarize the AlphaFold notes.", want: true},
		{name: "overview", question: "Give me an overview of citation validation.", want: true},
		{name: "anything related", question: "Find anything related to folding.", want: true},
		{name: "narrow implementation", question: "Where is read_index_source implemented?", want: false},
		{name: "narrow definition", question: "What does MaxSearchCalls default to?", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isBroadCoverageQuestion(tt.question); got != tt.want {
				t.Fatalf("isBroadCoverageQuestion(%q) = %v, want %v", tt.question, got, tt.want)
			}
		})
	}
}

func TestCoverageNudgePromptContainsInstructions(t *testing.T) {
	prompt := coverageNudgePrompt(
		"summarize information on alphafold or anything related to folding",
		"AlphaFold outputs include PDB files [source 1].",
		1,
		2,
	)
	for _, want := range []string{
		"summarize information on alphafold or anything related to folding",
		"AlphaFold outputs include PDB files [source 1].",
		"additional search_index",
		"materially different queries",
		"terms, acronyms, paths, filenames, headings, entities, and concepts visible in the sources already returned",
		"[source 2] [source 3]",
		"Never write [sources 2, 3]",
		"one bracket per cited source",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt = %q, want substring %q", prompt, want)
		}
	}
}

func TestNormalizeCoveragePolicyDefaults(t *testing.T) {
	policy := normalizeCoveragePolicy(Options{})
	if !policy.Enabled || policy.MinBroadSearchCalls != 2 || policy.MaxRetries != 1 {
		t.Fatalf("policy = %+v, want default enabled with min 2 and max retries 1", policy)
	}
	disabled := normalizeCoveragePolicy(Options{
		CoverageNudge:       boolPtr(false),
		MinBroadSearchCalls: -1,
		MaxCoverageRetries:  -1,
	})
	if disabled.Enabled || disabled.MinBroadSearchCalls != 2 || disabled.MaxRetries != 1 {
		t.Fatalf("disabled policy = %+v, want disabled with normalized counts", disabled)
	}
}

func TestShouldCoverageNudge(t *testing.T) {
	policy := coveragePolicy{Enabled: true, MinBroadSearchCalls: 2, MaxRetries: 1}
	if !shouldCoverageNudge("summarize alphafold", policy, 1, 1, 1, 6, 0) {
		t.Fatal("shouldCoverageNudge = false, want true for broad one-search answer")
	}
	if shouldCoverageNudge("Where is citation validation implemented?", policy, 1, 1, 1, 6, 0) {
		t.Fatal("shouldCoverageNudge = true, want false for narrow question")
	}
	if shouldCoverageNudge("summarize alphafold", policy, 2, 2, 2, 6, 0) {
		t.Fatal("shouldCoverageNudge = true, want false after enough searches")
	}
	if shouldCoverageNudge("summarize alphafold", policy, 1, 1, 1, 1, 0) {
		t.Fatal("shouldCoverageNudge = true, want false when search budget is spent")
	}
	if shouldCoverageNudge("summarize alphafold", policy, 1, 1, 0, 6, 0) {
		t.Fatal("shouldCoverageNudge = true, want false when first search returned no sources")
	}
	if shouldCoverageNudge("summarize alphafold", policy, 1, 1, 1, 6, 1) {
		t.Fatal("shouldCoverageNudge = true, want false after retry limit")
	}
}
