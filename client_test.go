package falkenvector_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	falkenvector "github.com/smasonuk/falken-vector"
)

func TestClientEmbedSendsOpenAICompatiblePortkeyRequest(t *testing.T) {
	var requestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %q, want POST", r.Method)
			http.Error(w, "bad method", http.StatusBadRequest)
			return
		}
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %q, want /embeddings", r.URL.Path)
			http.Error(w, "bad path", http.StatusBadRequest)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization = %q, want bearer token", got)
			http.Error(w, "bad authorization", http.StatusBadRequest)
			return
		}
		if got := r.Header.Get("X-Portkey-Provider"); got != falkenvector.DefaultPortkeyProvider {
			t.Errorf("X-Portkey-Provider = %q, want default provider", got)
			http.Error(w, "bad provider header", http.StatusBadRequest)
			return
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", got)
			http.Error(w, "bad content type", http.StatusBadRequest)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"object":"list",
			"model":"text-embedding-3-small",
			"data":[{"object":"embedding","index":0,"embedding":[0.25,-0.5,0.75]}],
			"usage":{"prompt_tokens":4,"total_tokens":4}
		}`)
	}))
	defer server.Close()

	client, err := falkenvector.New(falkenvector.Config{
		BaseURL: server.URL + "/",
		APIKey:  " test-key ",
		Headers: map[string]string{
			"X-Portkey-Provider": falkenvector.DefaultPortkeyProvider,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	response, err := client.Embed(context.Background(), falkenvector.EmbeddingRequest{
		Input: " hello Falken vectors ",
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if got := requestBody["model"]; got != falkenvector.DefaultEmbeddingModel {
		t.Fatalf("model = %v, want default embedding model", got)
	}
	if got := requestBody["input"]; got != " hello Falken vectors " {
		t.Fatalf("input = %v, want exact input", got)
	}
	if got := requestBody["encoding_format"]; got != "float" {
		t.Fatalf("encoding_format = %v, want float", got)
	}
	if response.Model != falkenvector.DefaultEmbeddingModel {
		t.Fatalf("response model = %q, want default embedding model", response.Model)
	}
	if len(response.Embedding) != 3 || response.Embedding[0] != 0.25 || response.Embedding[1] != -0.5 || response.Embedding[2] != 0.75 {
		t.Fatalf("embedding = %+v, want parsed vector", response.Embedding)
	}
	if response.Usage.PromptTokens != 4 || response.Usage.TotalTokens != 4 {
		t.Fatalf("usage = %+v, want token counts", response.Usage)
	}
}

func TestPortkeyConfigSetsDefaults(t *testing.T) {
	config := falkenvector.PortkeyConfig("test-key")
	if config.BaseURL != falkenvector.DefaultPortkeyBaseURL {
		t.Fatalf("base URL = %q, want default Portkey URL", config.BaseURL)
	}
	if config.APIKey != "test-key" {
		t.Fatalf("api key = %q, want provided key", config.APIKey)
	}
	if config.Model != falkenvector.DefaultEmbeddingModel {
		t.Fatalf("model = %q, want default embedding model", config.Model)
	}
	if config.Headers["X-Portkey-Provider"] != falkenvector.DefaultPortkeyProvider {
		t.Fatalf("provider header = %q, want default provider", config.Headers["X-Portkey-Provider"])
	}
}

func TestNewRequiresAPIKey(t *testing.T) {
	_, err := falkenvector.New(falkenvector.Config{})
	if !errors.Is(err, falkenvector.ErrAPIKeyRequired) {
		t.Fatalf("New error = %v, want ErrAPIKeyRequired", err)
	}
}

func TestClientEmbedReportsAPIErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":{"message":"bad key","type":"invalid_request_error"}}`)
	}))
	defer server.Close()

	client, err := falkenvector.New(falkenvector.Config{
		BaseURL: server.URL,
		APIKey:  "test-key",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = client.Embed(context.Background(), falkenvector.EmbeddingRequest{
		Input: "hello",
	})
	if err == nil {
		t.Fatal("Embed succeeded, want API error")
	}
	if !strings.Contains(err.Error(), "status 401") || !strings.Contains(err.Error(), "bad key") {
		t.Fatalf("Embed error = %v, want status and API message", err)
	}
}

func TestClientEmbedRequiresInput(t *testing.T) {
	client, err := falkenvector.New(falkenvector.Config{
		APIKey: "test-key",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = client.Embed(context.Background(), falkenvector.EmbeddingRequest{})
	if err == nil || !strings.Contains(err.Error(), "input") {
		t.Fatalf("Embed error = %v, want missing input error", err)
	}
}
