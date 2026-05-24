package cli

import (
	"fmt"
	"os"
	"strings"

	"github.com/smasonuk/falken-core/pkg/falken"
	falkenlangchain "github.com/smasonuk/falken-extra/llm/langchaingo"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/openaihttp"
	"github.com/tmc/langchaingo/llms/openai"
)

var newCLIAgentLLM = func() (falken.LLM, error) {
	return newCLIAgentLLMFromEnv(os.Getenv)
}

var newCLIAgentLLMWithModel = newCLIAgentLLMWithModelOverride

func newCLIAgentLLMWithModelOverride(model string) (falken.LLM, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return newCLIAgentLLM()
	}
	return newCLIAgentLLMFromEnv(func(key string) string {
		if key == llm.EnvLLMModel {
			return model
		}
		return os.Getenv(key)
	})
}

func newCLIAgentLLMFromEnv(getenv func(string) string) (falken.LLM, error) {
	if getenv == nil {
		getenv = func(string) string { return "" }
	}
	apiKey := strings.TrimSpace(getenv(llm.EnvLLMAPIKey))
	if apiKey == "" {
		apiKey = strings.TrimSpace(getenv(llm.EnvEmbeddingModelAPIKey))
	}

	baseURL := strings.TrimSpace(getenv(llm.EnvLLMBaseURL))
	if baseURL == "" {
		return nil, fmt.Errorf("%w: set %s", llm.ErrBaseURLRequired, llm.EnvLLMBaseURL)
	}
	modelName := strings.TrimSpace(getenv(llm.EnvLLMModel))
	if modelName == "" {
		return nil, fmt.Errorf("%w: set %s", llm.ErrModelRequired, llm.EnvLLMModel)
	}
	headers, err := llm.HeadersFromJSONEnv(getenv, llm.EnvLLMHeaders)
	if err != nil {
		return nil, err
	}

	options := []openai.Option{openai.WithModel(modelName), openai.WithBaseURL(baseURL)}
	if apiKey == "" {
		model, err := newOpenAIModelWithoutAPIKey(modelName, baseURL, headers)
		if err != nil {
			return nil, fmt.Errorf("configure OpenAI-compatible agent LLM: %w", err)
		}
		return falkenlangchain.New(model), nil
	}
	options = append(options, openai.WithToken(apiKey))
	options = append(options, openai.WithHTTPClient(openaihttp.SuccessStatusClient{
		Next: falkenlangchain.NewHeaderHTTPClient(nil, headers),
	}))
	model, err := openai.New(options...)
	if err != nil {
		return nil, fmt.Errorf("configure OpenAI-compatible agent LLM: %w", err)
	}
	return falkenlangchain.New(model), nil
}

func newOpenAIModelWithoutAPIKey(modelName, baseURL string, headers map[string]string) (*openai.LLM, error) {
	const placeholderToken = "unused-local-openai-compatible-token"
	// LangChainGo requires a non-empty token during construction. Strip the
	// placeholder before transport so local endpoints see no bearer auth.
	httpClient := openaihttp.SuccessStatusClient{
		Next: openaihttp.StripPlaceholderAuthorizationClient{
			Next:             falkenlangchain.NewHeaderHTTPClient(nil, headers),
			PlaceholderToken: placeholderToken,
		},
	}
	return openai.New(
		openai.WithToken(placeholderToken),
		openai.WithModel(modelName),
		openai.WithBaseURL(baseURL),
		openai.WithHTTPClient(httpClient),
	)
}
