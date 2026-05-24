package cli

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/tmc/langchaingo/httputil"
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

func TestNewCLIAgentLLMAcceptsNonstandardSuccessStatus(t *testing.T) {
	oldDefaultClient := httputil.DefaultClient
	t.Cleanup(func() { httputil.DefaultClient = oldDefaultClient })
	httputil.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("path = %q, want /chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("Authorization = %q, want bearer token", got)
		}
		return &http.Response{
			StatusCode: 246,
			Status:     "246 Custom Success",
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"chatcmpl-test","object":"chat.completion","created":0,"model":"chat-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)),
			Request:    r,
		}, nil
	})}

	got, err := newCLIAgentLLMFromEnv(func(key string) string {
		switch key {
		case llm.EnvLLMAPIKey:
			return "test-key"
		case llm.EnvLLMBaseURL:
			return "http://llm.test"
		case llm.EnvLLMModel:
			return "chat-model"
		default:
			return ""
		}
	})
	if err != nil {
		t.Fatalf("newCLIAgentLLMFromEnv: %v", err)
	}
	response, err := got.Complete(context.Background(), falken.CompletionRequest{
		Messages: []falken.Message{{Role: falken.RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.AssistantText != "ok" {
		t.Fatalf("AssistantText = %q, want ok", response.AssistantText)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
