package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/smasonuk/falken-vector/pkg/embeddings"
)

func TestEmbedderConvertsEmbedding(t *testing.T) {
	embedder := NewEmbedder(fakeEmbeddingClient{response: embeddings.EmbeddingResponse{
		Model:     "text-embedding-3-small",
		Embedding: []float64{1.5, -2.25},
	}})
	got, err := embedder.EmbedText(context.Background(), "hello")
	if err != nil {
		t.Fatalf("EmbedText: %v", err)
	}
	if got.Model != "text-embedding-3-small" || len(got.Vector) != 2 || got.Vector[0] != 1.5 || got.Vector[1] != -2.25 {
		t.Fatalf("embedding = %+v", got)
	}
}

func TestEmbedderRejectsEmptyVector(t *testing.T) {
	embedder := NewEmbedder(fakeEmbeddingClient{response: embeddings.EmbeddingResponse{Model: "m"}})
	_, err := embedder.EmbedText(context.Background(), "hello")
	if err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("EmbedText error = %v, want empty vector", err)
	}
}

func TestChatClientCompleteSuccess(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer key" {
			t.Errorf("authorization = %q", got)
		}
		if got := r.Header.Get("X-Portkey-Provider"); got != embeddings.DefaultPortkeyProvider {
			t.Errorf("provider = %q", got)
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"model":"gpt-test","choices":[{"message":{"role":"assistant","content":"answer"}}]}`)
	}))
	defer server.Close()

	client, err := NewChatClient(ChatConfig{
		BaseURL: server.URL,
		APIKey:  "key",
		Model:   "gpt-test",
		Headers: map[string]string{"X-Portkey-Provider": embeddings.DefaultPortkeyProvider},
	})
	if err != nil {
		t.Fatalf("NewChatClient: %v", err)
	}
	response, err := client.Complete(context.Background(), CompletionRequest{System: "sys", User: "user"})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Text != "answer" || response.Model != "gpt-test" {
		t.Fatalf("response = %+v", response)
	}
	if body["model"] != "gpt-test" {
		t.Fatalf("model body = %v", body["model"])
	}
}

func TestChatClientReportsNon2xx(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"bad request","type":"invalid_request_error"}}`)
	}))
	defer server.Close()
	client, err := NewChatClient(ChatConfig{BaseURL: server.URL, APIKey: "key"})
	if err != nil {
		t.Fatalf("NewChatClient: %v", err)
	}
	_, err = client.Complete(context.Background(), CompletionRequest{System: "s", User: "u"})
	if err == nil || !strings.Contains(err.Error(), "bad request") {
		t.Fatalf("Complete error = %v", err)
	}
}

func TestChatClientRequiresAPIKey(t *testing.T) {
	_, err := NewChatClient(ChatConfig{})
	if !errors.Is(err, ErrMissingAPIKey) {
		t.Fatalf("NewChatClient error = %v, want missing key", err)
	}
}

func TestEnvEmbedderRequiresEmbeddingAPIKey(t *testing.T) {
	_, err := NewEnvEmbedder(func(string) string { return "" })
	if !errors.Is(err, ErrMissingAPIKey) {
		t.Fatalf("NewEnvEmbedder error = %v, want missing API key", err)
	}
	if err == nil || !strings.Contains(err.Error(), EnvEmbeddingModelAPIKey) {
		t.Fatalf("NewEnvEmbedder error = %v, want %s guidance", err, EnvEmbeddingModelAPIKey)
	}
}

func TestEnvChatClientOnlyAddsPortkeyHeaderForDefaultBaseURL(t *testing.T) {
	defaultClient, err := NewEnvChatClient(func(key string) string {
		if key == EnvEmbeddingModelAPIKey {
			return "key"
		}
		return ""
	})
	if err != nil {
		t.Fatalf("NewEnvChatClient default: %v", err)
	}
	if defaultClient.headers["X-Portkey-Provider"] != embeddings.DefaultPortkeyProvider {
		t.Fatalf("default headers = %+v, want Portkey provider", defaultClient.headers)
	}

	customClient, err := NewEnvChatClient(func(key string) string {
		switch key {
		case EnvEmbeddingModelAPIKey:
			return "key"
		case EnvLLMBaseURL:
			return "https://example.test/v1"
		default:
			return ""
		}
	})
	if err != nil {
		t.Fatalf("NewEnvChatClient custom: %v", err)
	}
	if _, ok := customClient.headers["X-Portkey-Provider"]; ok {
		t.Fatalf("custom headers = %+v, want no Portkey provider", customClient.headers)
	}
}

type fakeEmbeddingClient struct {
	response embeddings.EmbeddingResponse
	err      error
}

func (f fakeEmbeddingClient) Embed(context.Context, embeddings.EmbeddingRequest) (embeddings.EmbeddingResponse, error) {
	return f.response, f.err
}
