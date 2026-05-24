package cli

import (
	"errors"
	"strings"
	"testing"

	"github.com/smasonuk/falken-vector/internal/llm"
)

func TestNewCLIAgentLLMMissingBaseURL(t *testing.T) {
	_, err := newCLIAgentLLMFromEnv(func(string) string { return "" })
	if !errors.Is(err, llm.ErrBaseURLRequired) {
		t.Fatalf("error = %v, want missing base URL", err)
	}
	if err == nil || !strings.Contains(err.Error(), llm.EnvLLMBaseURL) {
		t.Fatalf("error = %v, want clear env var guidance", err)
	}
}

func TestNewCLIAgentLLMMissingModel(t *testing.T) {
	_, err := newCLIAgentLLMFromEnv(func(key string) string {
		if key == llm.EnvLLMBaseURL {
			return "http://127.0.0.1:9999/v1"
		}
		return ""
	})
	if !errors.Is(err, llm.ErrModelRequired) {
		t.Fatalf("error = %v, want missing model", err)
	}
	if err == nil || !strings.Contains(err.Error(), llm.EnvLLMModel) {
		t.Fatalf("error = %v, want clear env var guidance", err)
	}
}

func TestNewCLIAgentLLMReadsAPIKey(t *testing.T) {
	got, err := newCLIAgentLLMFromEnv(func(key string) string {
		switch key {
		case llm.EnvLLMAPIKey:
			return "test-key"
		case llm.EnvLLMBaseURL:
			return "http://127.0.0.1:9999/v1"
		case llm.EnvLLMModel:
			return "chat-model"
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

func TestNewCLIAgentLLMAcceptsLocalEndpointWithoutAPIKey(t *testing.T) {
	got, err := newCLIAgentLLMFromEnv(func(key string) string {
		switch key {
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

func TestNewCLIAgentLLMParsesHeaders(t *testing.T) {
	got, err := newCLIAgentLLMFromEnv(func(key string) string {
		switch key {
		case llm.EnvLLMBaseURL:
			return "http://127.0.0.1:9999/v1"
		case llm.EnvLLMModel:
			return "local-model"
		case llm.EnvLLMHeaders:
			return `{"X-Test-Provider":"test-provider"}`
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
