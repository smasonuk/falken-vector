package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/smasonuk/falken-vector/pkg/embeddings"
)

const (
	EnvEmbeddingModelAPIKey  = "FALKENGO_EMBEDDING_MODEL_API_KEY"
	EnvEmbeddingModel        = "FALKENGO_EMBEDDING_MODEL"
	EnvEmbeddingModelURL     = "FALKENGO_EMBEDDING_MODEL_URL"
	EnvEmbeddingModelHeaders = "FALKENGO_EMBEDDING_MODEL_HEADERS"
	EnvLLMAPIKey             = "FALKENGO_LLM_API_KEY"
	EnvLLMBaseURL            = "FALKENGO_LLM_BASE_URL"
	EnvLLMModel              = "FALKENGO_LLM_MODEL"
	EnvLLMHeaders            = "FALKENGO_LLM_HEADERS"
	defaultTemperature       = 0.1
)

var (
	ErrBaseURLRequired = embeddings.ErrBaseURLRequired
	ErrModelRequired   = embeddings.ErrModelRequired
)

func HeadersFromJSONEnv(getenv func(string) string, key string) (map[string]string, error) {
	if getenv == nil {
		getenv = func(string) string { return "" }
	}
	raw := strings.TrimSpace(getenv(key))
	if raw == "" {
		return nil, nil
	}
	if !strings.HasPrefix(raw, "{") {
		return nil, fmt.Errorf("%s must be a JSON object with string header values", key)
	}
	var decoded map[string]string
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		return nil, fmt.Errorf("parse %s as JSON object with string header values: %w", key, err)
	}
	if decoded == nil {
		return nil, fmt.Errorf("%s must be a JSON object with string header values", key)
	}
	headers := make(map[string]string, len(decoded))
	for name, value := range decoded {
		name = strings.TrimSpace(name)
		value = strings.TrimSpace(value)
		if name == "" || value == "" {
			continue
		}
		headers[name] = value
	}
	if len(headers) == 0 {
		return nil, nil
	}
	return headers, nil
}

type OpenAIEmbedder struct {
	client interface {
		Embed(context.Context, embeddings.EmbeddingRequest) (embeddings.EmbeddingResponse, error)
	}
}

func NewEnvEmbedder(getenv func(string) string) (*OpenAIEmbedder, error) {
	if getenv == nil {
		getenv = func(key string) string { return "" }
	}
	baseURL := strings.TrimSpace(getenv(EnvEmbeddingModelURL))
	if baseURL == "" {
		return nil, fmt.Errorf("%w: set %s", embeddings.ErrBaseURLRequired, EnvEmbeddingModelURL)
	}
	model := strings.TrimSpace(getenv(EnvEmbeddingModel))
	if model == "" {
		return nil, fmt.Errorf("%w: set %s", embeddings.ErrModelRequired, EnvEmbeddingModel)
	}
	headers, err := HeadersFromJSONEnv(getenv, EnvEmbeddingModelHeaders)
	if err != nil {
		return nil, err
	}
	client, err := embeddings.New(embeddings.Config{
		BaseURL: baseURL,
		APIKey:  strings.TrimSpace(getenv(EnvEmbeddingModelAPIKey)),
		Model:   model,
		Headers: headers,
	})
	if err != nil {
		return nil, err
	}
	return &OpenAIEmbedder{client: client}, nil
}

func NewEmbedder(client interface {
	Embed(context.Context, embeddings.EmbeddingRequest) (embeddings.EmbeddingResponse, error)
}) *OpenAIEmbedder {
	return &OpenAIEmbedder{client: client}
}

func (e *OpenAIEmbedder) EmbedText(ctx context.Context, input string) (Embedding, error) {
	if e == nil || e.client == nil {
		return Embedding{}, errors.New("embedder client is required")
	}
	if strings.TrimSpace(input) == "" {
		return Embedding{}, errors.New("embedding input is required")
	}
	response, err := e.client.Embed(ctx, embeddings.EmbeddingRequest{Input: input})
	if err != nil {
		return Embedding{}, err
	}
	vector := float64ToFloat32(response.Embedding)
	if len(vector) == 0 {
		return Embedding{}, errors.New("embedding response vector is empty")
	}
	return Embedding{Model: response.Model, Vector: vector}, nil
}

type ChatClient struct {
	baseURL    string
	apiKey     string
	model      string
	headers    map[string]string
	httpClient interface {
		Do(*http.Request) (*http.Response, error)
	}
}

func NewEnvChatClient(getenv func(string) string) (*ChatClient, error) {
	if getenv == nil {
		getenv = func(key string) string { return "" }
	}
	apiKey := strings.TrimSpace(getenv(EnvLLMAPIKey))
	if apiKey == "" {
		apiKey = strings.TrimSpace(getenv(EnvEmbeddingModelAPIKey))
	}
	baseURL := strings.TrimSpace(getenv(EnvLLMBaseURL))
	if baseURL == "" {
		return nil, fmt.Errorf("%w: set %s", embeddings.ErrBaseURLRequired, EnvLLMBaseURL)
	}
	model := strings.TrimSpace(getenv(EnvLLMModel))
	if model == "" {
		return nil, fmt.Errorf("%w: set %s", embeddings.ErrModelRequired, EnvLLMModel)
	}
	headers, err := HeadersFromJSONEnv(getenv, EnvLLMHeaders)
	if err != nil {
		return nil, err
	}
	return NewChatClient(ChatConfig{
		BaseURL: baseURL,
		APIKey:  apiKey,
		Model:   model,
		Headers: headers,
	})
}

type ChatConfig struct {
	BaseURL    string
	APIKey     string
	Model      string
	Headers    map[string]string
	HTTPClient interface {
		Do(*http.Request) (*http.Response, error)
	}
}

func NewChatClient(config ChatConfig) (*ChatClient, error) {
	apiKey := strings.TrimSpace(config.APIKey)
	baseURL := strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	if baseURL == "" {
		return nil, embeddings.ErrBaseURLRequired
	}
	model := strings.TrimSpace(config.Model)
	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	headers := copyHeaders(config.Headers)
	return &ChatClient{baseURL: baseURL, apiKey: apiKey, model: model, headers: headers, httpClient: httpClient}, nil
}

func (c *ChatClient) Complete(ctx context.Context, req CompletionRequest) (CompletionResponse, error) {
	if c == nil {
		return CompletionResponse{}, errors.New("chat client is nil")
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = c.model
	}
	if model == "" {
		return CompletionResponse{}, embeddings.ErrModelRequired
	}
	temp := req.Temperature
	if temp == 0 {
		temp = defaultTemperature
	}
	body, err := json.Marshal(chatRequest{
		Model: model,
		Messages: []chatMessage{
			{Role: "system", Content: req.System},
			{Role: "user", Content: req.User},
		},
		Temperature: temp,
	})
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("encode chat completion request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("create chat completion request: %w", err)
	}
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	for name, value := range c.headers {
		httpReq.Header.Set(name, value)
	}
	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("post chat completion request: %w", err)
	}
	defer httpResp.Body.Close()
	responseBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("read chat completion response: %w", err)
	}
	if httpResp.StatusCode < http.StatusOK || httpResp.StatusCode >= http.StatusMultipleChoices {
		return CompletionResponse{}, apiError("chat completions API", httpResp.StatusCode, responseBody)
	}
	var response chatResponse
	if err := json.Unmarshal(responseBody, &response); err != nil {
		return CompletionResponse{}, fmt.Errorf("decode chat completion response: %w", err)
	}
	if len(response.Choices) == 0 {
		return CompletionResponse{}, errors.New("chat completion response did not include choices")
	}
	return CompletionResponse{Model: response.Model, Text: response.Choices[0].Message.Content}, nil
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float32       `json:"temperature"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func float64ToFloat32(values []float64) []float32 {
	out := make([]float32, len(values))
	for i, value := range values {
		out[i] = float32(value)
	}
	return out
}

func apiError(label string, statusCode int, body []byte) error {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err == nil && response.Error.Message != "" {
		if response.Error.Type != "" {
			return fmt.Errorf("%s failed with status %d: %s (%s)", label, statusCode, response.Error.Message, response.Error.Type)
		}
		return fmt.Errorf("%s failed with status %d: %s", label, statusCode, response.Error.Message)
	}
	bodyText := strings.TrimSpace(string(body))
	if bodyText == "" {
		return fmt.Errorf("%s failed with status %d", label, statusCode)
	}
	return fmt.Errorf("%s failed with status %d: %s", label, statusCode, bodyText)
}

func copyHeaders(headers map[string]string) map[string]string {
	out := make(map[string]string, len(headers))
	for name, value := range headers {
		name = strings.TrimSpace(name)
		value = strings.TrimSpace(value)
		if name == "" || value == "" {
			continue
		}
		out[name] = value
	}
	return out
}
