package cli

import (
	"errors"
	"strings"
	"testing"

	"github.com/smasonuk/falken-vector/internal/llm"
)

func TestNewCLIAgentLLMMissingAPIKey(t *testing.T) {
	_, err := newCLIAgentLLMFromEnv(func(string) string { return "" })
	if !errors.Is(err, llm.ErrMissingAPIKey) {
		t.Fatalf("error = %v, want missing API key", err)
	}
	if err == nil || !strings.Contains(err.Error(), llm.EnvLLMAPIKey) {
		t.Fatalf("error = %v, want clear env var guidance", err)
	}
}

func TestNewCLIAgentLLMReadsAPIKey(t *testing.T) {
	got, err := newCLIAgentLLMFromEnv(func(key string) string {
		if key == llm.EnvLLMAPIKey {
			return "test-key"
		}
		return ""
	})
	if err != nil {
		t.Fatalf("newCLIAgentLLMFromEnv: %v", err)
	}
	if got == nil {
		t.Fatal("newCLIAgentLLMFromEnv returned nil LLM")
	}
}

func TestNewCLIAgentLLMAcceptsBaseURLAndModel(t *testing.T) {
	got, err := newCLIAgentLLMFromEnv(func(key string) string {
		switch key {
		case llm.EnvLLMAPIKey:
			return "test-key"
		case llm.EnvLLMBaseURL:
			return "http://127.0.0.1:9999/v1"
		case llm.EnvLLMModel:
			return "local-model"
		default:
			return ""
		}
	})
	if err != nil {
		t.Fatalf("newCLIAgentLLMFromEnv: %v", err)
	}
	if got == nil {
		t.Fatal("newCLIAgentLLMFromEnv returned nil LLM")
	}
}
