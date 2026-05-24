package rag

import (
	"strings"
	"testing"
)

func TestBuildPrompt(t *testing.T) {
	prompt, err := BuildPrompt("What is indexed?", []SourceChunk{{
		Path:      "README.md",
		StartLine: 10,
		EndLine:   12,
		Text:      "Falkengo indexes text files.",
	}})
	if err != nil {
		t.Fatalf("BuildPrompt: %v", err)
	}
	if prompt.System != SystemPrompt {
		t.Fatalf("system prompt changed")
	}
	wantParts := []string{
		"[source 1]",
		"Path: README.md",
		"Lines: 10-12",
		"Text:\nFalkengo indexes text files.",
		"Question:\nWhat is indexed?",
	}
	for _, part := range wantParts {
		if !strings.Contains(prompt.User, part) {
			t.Fatalf("prompt missing %q:\n%s", part, prompt.User)
		}
	}
	if prompt.Sources[0].SourceNumber != 1 {
		t.Fatalf("source number = %d, want 1", prompt.Sources[0].SourceNumber)
	}
}

func TestBuildPromptRequiresQuestionAndChunks(t *testing.T) {
	if _, err := BuildPrompt("", []SourceChunk{{Text: "x"}}); err == nil {
		t.Fatal("empty question succeeded")
	}
	if _, err := BuildPrompt("q", nil); err == nil {
		t.Fatal("empty chunks succeeded")
	}
}
