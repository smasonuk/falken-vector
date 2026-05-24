// Package embeddings provides provider-neutral helpers for retrieving vector
// embeddings from OpenAI-compatible APIs.
package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

var (
	ErrBaseURLRequired = errors.New("base URL is required")
	ErrModelRequired   = errors.New("embedding model is required")
)

// Provider is the minimal embeddings provider contract.
type Provider interface {
	Embed(context.Context, EmbeddingRequest) (EmbeddingResponse, error)
}

// HTTPClient is the minimal HTTP client contract used by Client.
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

// Config describes an OpenAI-compatible embeddings client.
type Config struct {
	BaseURL    string
	APIKey     string
	Model      string
	Headers    map[string]string
	HTTPClient HTTPClient
}

// EmbeddingRequest is a request for one text embedding.
type EmbeddingRequest struct {
	Input string
	Model string
}

// EmbeddingResponse is the provider-neutral result for one text embedding.
type EmbeddingResponse struct {
	Model     string
	Embedding []float64
	Usage     Usage
}

// Usage captures token counts returned by the embeddings API.
type Usage struct {
	PromptTokens int
	TotalTokens  int
}

// Client retrieves embeddings from an OpenAI-compatible API.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	headers    map[string]string
	httpClient HTTPClient
}

// New creates an embeddings client.
func New(config Config) (*Client, error) {
	baseURL := strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	if baseURL == "" {
		return nil, ErrBaseURLRequired
	}

	model := strings.TrimSpace(config.Model)

	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	headers := make(map[string]string, len(config.Headers))
	for name, value := range config.Headers {
		name = strings.TrimSpace(name)
		value = strings.TrimSpace(value)
		if name == "" || value == "" {
			continue
		}
		headers[name] = value
	}

	return &Client{
		baseURL:    baseURL,
		apiKey:     strings.TrimSpace(config.APIKey),
		model:      model,
		headers:    headers,
		httpClient: httpClient,
	}, nil
}

// Embed retrieves an embedding for request.Input.
func (c *Client) Embed(ctx context.Context, request EmbeddingRequest) (EmbeddingResponse, error) {
	if c == nil {
		return EmbeddingResponse{}, errors.New("embeddings client is nil")
	}
	if strings.TrimSpace(request.Input) == "" {
		return EmbeddingResponse{}, errors.New("embedding input is required")
	}
	model := strings.TrimSpace(request.Model)
	if model == "" {
		model = c.model
	}
	if model == "" {
		return EmbeddingResponse{}, ErrModelRequired
	}

	body, err := json.Marshal(embeddingAPIRequest{
		Model:          model,
		Input:          request.Input,
		EncodingFormat: "float",
	})
	if err != nil {
		return EmbeddingResponse{}, fmt.Errorf("encode embeddings request: %w", err)
	}

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return EmbeddingResponse{}, fmt.Errorf("create embeddings request: %w", err)
	}
	if c.apiKey != "" {
		httpRequest.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	for name, value := range c.headers {
		httpRequest.Header.Set(name, value)
	}

	httpResponse, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return EmbeddingResponse{}, fmt.Errorf("post embeddings request: %w", err)
	}
	defer httpResponse.Body.Close()

	responseBody, err := io.ReadAll(httpResponse.Body)
	if err != nil {
		return EmbeddingResponse{}, fmt.Errorf("read embeddings response: %w", err)
	}
	if httpResponse.StatusCode < http.StatusOK || httpResponse.StatusCode >= http.StatusMultipleChoices {
		return EmbeddingResponse{}, apiError(httpResponse.StatusCode, responseBody)
	}

	var apiResponse embeddingAPIResponse
	if err := json.Unmarshal(responseBody, &apiResponse); err != nil {
		return EmbeddingResponse{}, fmt.Errorf("decode embeddings response: %w", err)
	}
	if len(apiResponse.Data) == 0 {
		return EmbeddingResponse{}, errors.New("embeddings response did not include data")
	}

	return EmbeddingResponse{
		Model:     apiResponse.Model,
		Embedding: append([]float64(nil), apiResponse.Data[0].Embedding...),
		Usage: Usage{
			PromptTokens: apiResponse.Usage.PromptTokens,
			TotalTokens:  apiResponse.Usage.TotalTokens,
		},
	}, nil
}

type embeddingAPIRequest struct {
	Model          string `json:"model"`
	Input          string `json:"input"`
	EncodingFormat string `json:"encoding_format"`
}

type embeddingAPIResponse struct {
	Model string `json:"model"`
	Data  []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

type embeddingAPIError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func apiError(statusCode int, body []byte) error {
	var response embeddingAPIError
	if err := json.Unmarshal(body, &response); err == nil && response.Error.Message != "" {
		if response.Error.Type != "" {
			return fmt.Errorf("embeddings API failed with status %d: %s (%s)", statusCode, response.Error.Message, response.Error.Type)
		}
		return fmt.Errorf("embeddings API failed with status %d: %s", statusCode, response.Error.Message)
	}
	bodyText := strings.TrimSpace(string(body))
	if bodyText == "" {
		return fmt.Errorf("embeddings API failed with status %d", statusCode)
	}
	return fmt.Errorf("embeddings API failed with status %d: %s", statusCode, bodyText)
}
