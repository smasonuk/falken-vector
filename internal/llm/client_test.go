package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
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

func TestEnvEmbedderRequiresEmbeddingURL(t *testing.T) {
	_, err := NewEnvEmbedder(func(string) string { return "" })
	if !errors.Is(err, embeddings.ErrBaseURLRequired) {
		t.Fatalf("NewEnvEmbedder error = %v, want base URL required", err)
	}
	if err == nil || !strings.Contains(err.Error(), EnvEmbeddingModelURL) {
		t.Fatalf("NewEnvEmbedder error = %v, want %s guidance", err, EnvEmbeddingModelURL)
	}
}

func TestEnvEmbedderRequiresEmbeddingModel(t *testing.T) {
	_, err := NewEnvEmbedder(func(key string) string {
		if key == EnvEmbeddingModelURL {
			return "https://embed.test/v1"
		}
		return ""
	})
	if !errors.Is(err, embeddings.ErrModelRequired) {
		t.Fatalf("NewEnvEmbedder error = %v, want model required", err)
	}
	if err == nil || !strings.Contains(err.Error(), EnvEmbeddingModel) {
		t.Fatalf("NewEnvEmbedder error = %v, want %s guidance", err, EnvEmbeddingModel)
	}
}

func TestEnvEmbedderAllowsMissingAPIKeyAndPassesHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("authorization = %q, want no bearer token", got)
			http.Error(w, "bad authorization", http.StatusBadRequest)
			return
		}
		if got := r.Header.Get("X-Test-Provider"); got != "test-provider" {
			t.Errorf("X-Test-Provider = %q, want configured header", got)
			http.Error(w, "bad header", http.StatusBadRequest)
			return
		}
		fmt.Fprint(w, `{"model":"embed-model","data":[{"embedding":[1,2]}],"usage":{}}`)
	}))
	defer server.Close()

	embedder, err := NewEnvEmbedder(func(key string) string {
		switch key {
		case EnvEmbeddingModelURL:
			return server.URL
		case EnvEmbeddingModel:
			return "embed-model"
		case EnvEmbeddingModelHeaders:
			return `{"X-Test-Provider":"test-provider"}`
		default:
			return ""
		}
	})
	if err != nil {
		t.Fatalf("NewEnvEmbedder: %v", err)
	}
	embedding, err := embedder.EmbedText(context.Background(), "hello")
	if err != nil {
		t.Fatalf("EmbedText: %v", err)
	}
	if len(embedding.Vector) != 2 {
		t.Fatalf("embedding = %+v, want two values", embedding)
	}
}

func TestHeadersFromJSONEnv(t *testing.T) {
	headers, err := HeadersFromJSONEnv(func(key string) string {
		if key == "HEADERS" {
			return `{" X-Test-Provider ":" test-provider ","Empty":""," ":"ignored"}`
		}
		return ""
	}, "HEADERS")
	if err != nil {
		t.Fatalf("HeadersFromJSONEnv: %v", err)
	}
	want := map[string]string{"X-Test-Provider": "test-provider"}
	if !reflect.DeepEqual(headers, want) {
		t.Fatalf("headers = %+v, want %+v", headers, want)
	}
}

func TestHeadersFromJSONEnvEmpty(t *testing.T) {
	headers, err := HeadersFromJSONEnv(func(string) string { return "" }, "HEADERS")
	if err != nil {
		t.Fatalf("HeadersFromJSONEnv: %v", err)
	}
	if headers != nil {
		t.Fatalf("headers = %+v, want nil", headers)
	}
}

func TestHeadersFromJSONEnvRejectsInvalidJSON(t *testing.T) {
	_, err := HeadersFromJSONEnv(func(string) string { return `{bad json` }, EnvLLMHeaders)
	if err == nil {
		t.Fatal("HeadersFromJSONEnv succeeded, want error")
	}
	if !strings.Contains(err.Error(), EnvLLMHeaders) {
		t.Fatalf("error = %v, want env var name", err)
	}
}

func TestHeadersFromJSONEnvRejectsNonObject(t *testing.T) {
	_, err := HeadersFromJSONEnv(func(string) string { return `["x"]` }, EnvLLMHeaders)
	if err == nil {
		t.Fatal("HeadersFromJSONEnv succeeded, want error")
	}
	if !strings.Contains(err.Error(), EnvLLMHeaders) {
		t.Fatalf("error = %v, want env var name", err)
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
		if got := r.Header.Get("X-Test-Provider"); got != "test-provider" {
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
		Headers: map[string]string{"X-Test-Provider": "test-provider"},
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
	client, err := NewChatClient(ChatConfig{BaseURL: server.URL, Model: "chat-model"})
	if err != nil {
		t.Fatalf("NewChatClient: %v", err)
	}
	_, err = client.Complete(context.Background(), CompletionRequest{System: "s", User: "u"})
	if err == nil || !strings.Contains(err.Error(), "bad request") {
		t.Fatalf("Complete error = %v", err)
	}
}

func TestChatClientRequiresBaseURL(t *testing.T) {
	_, err := NewChatClient(ChatConfig{})
	if !errors.Is(err, embeddings.ErrBaseURLRequired) {
		t.Fatalf("NewChatClient error = %v, want base URL required", err)
	}
}

func TestChatClientCompleteRequiresModel(t *testing.T) {
	client, err := NewChatClient(ChatConfig{BaseURL: "https://chat.test/v1"})
	if err != nil {
		t.Fatalf("NewChatClient: %v", err)
	}
	_, err = client.Complete(context.Background(), CompletionRequest{System: "s", User: "u"})
	if !errors.Is(err, embeddings.ErrModelRequired) {
		t.Fatalf("Complete error = %v, want model required", err)
	}
}

func TestChatClientRequestModelOverridesClientModel(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		fmt.Fprint(w, `{"model":"request-model","choices":[{"message":{"role":"assistant","content":"answer"}}]}`)
	}))
	defer server.Close()

	client, err := NewChatClient(ChatConfig{BaseURL: server.URL, Model: "client-model"})
	if err != nil {
		t.Fatalf("NewChatClient: %v", err)
	}
	if _, err := client.Complete(context.Background(), CompletionRequest{System: "s", User: "u", Model: "request-model"}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := body["model"]; got != "request-model" {
		t.Fatalf("model = %v, want request override", got)
	}
}

func TestChatClientCanRunWithoutAPIKeyWithCustomAuthorizationHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Provider custom-token" {
			t.Errorf("authorization = %q, want custom header", got)
			http.Error(w, "bad authorization", http.StatusBadRequest)
			return
		}
		fmt.Fprint(w, `{"model":"chat-model","choices":[{"message":{"role":"assistant","content":"answer"}}]}`)
	}))
	defer server.Close()

	client, err := NewChatClient(ChatConfig{
		BaseURL: server.URL,
		Model:   "chat-model",
		Headers: map[string]string{"Authorization": "Provider custom-token"},
	})
	if err != nil {
		t.Fatalf("NewChatClient: %v", err)
	}
	if _, err := client.Complete(context.Background(), CompletionRequest{System: "s", User: "u"}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
}

func TestEnvChatClientRequiresBaseURL(t *testing.T) {
	_, err := NewEnvChatClient(func(string) string { return "" })
	if !errors.Is(err, embeddings.ErrBaseURLRequired) {
		t.Fatalf("NewEnvChatClient error = %v, want base URL required", err)
	}
	if err == nil || !strings.Contains(err.Error(), EnvLLMBaseURL) {
		t.Fatalf("NewEnvChatClient error = %v, want %s guidance", err, EnvLLMBaseURL)
	}
}

func TestEnvChatClientRequiresModel(t *testing.T) {
	_, err := NewEnvChatClient(func(key string) string {
		if key == EnvLLMBaseURL {
			return "https://chat.test/v1"
		}
		return ""
	})
	if !errors.Is(err, embeddings.ErrModelRequired) {
		t.Fatalf("NewEnvChatClient error = %v, want model required", err)
	}
	if err == nil || !strings.Contains(err.Error(), EnvLLMModel) {
		t.Fatalf("NewEnvChatClient error = %v, want %s guidance", err, EnvLLMModel)
	}
}

func TestEnvChatClientReadsHeadersAndAllowsMissingAPIKey(t *testing.T) {
	client, err := NewEnvChatClient(func(key string) string {
		switch key {
		case EnvLLMBaseURL:
			return "https://chat.test/v1"
		case EnvLLMModel:
			return "chat-model"
		case EnvLLMHeaders:
			return `{"X-Test-Provider":"test-provider"}`
		default:
			return ""
		}
	})
	if err != nil {
		t.Fatalf("NewEnvChatClient: %v", err)
	}
	if client.apiKey != "" {
		t.Fatalf("apiKey = %q, want empty", client.apiKey)
	}
	if got := client.headers["X-Test-Provider"]; got != "test-provider" {
		t.Fatalf("headers = %+v, want configured header", client.headers)
	}
}

type fakeEmbeddingClient struct {
	response embeddings.EmbeddingResponse
	err      error
}

func (f fakeEmbeddingClient) Embed(context.Context, embeddings.EmbeddingRequest) (embeddings.EmbeddingResponse, error) {
	return f.response, f.err
}
