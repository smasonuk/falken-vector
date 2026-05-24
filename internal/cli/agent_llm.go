package cli

import (
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/smasonuk/falken-core/pkg/falken"
	falkenlangchain "github.com/smasonuk/falken-extra/llm/langchaingo"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/pkg/embeddings"
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
	if apiKey == "" {
		return nil, fmt.Errorf("%w: set %s or %s", llm.ErrMissingAPIKey, llm.EnvLLMAPIKey, llm.EnvEmbeddingModelAPIKey)
	}

	baseURL := strings.TrimSpace(getenv(llm.EnvLLMBaseURL))
	if baseURL == "" {
		baseURL = embeddings.DefaultPortkeyBaseURL
	}
	modelName := strings.TrimSpace(getenv(llm.EnvLLMModel))
	if modelName == "" {
		modelName = llm.DefaultChatModel
	}

	options := []openai.Option{openai.WithToken(apiKey), openai.WithModel(modelName), openai.WithBaseURL(baseURL)}
	if isDefaultPortkeyBaseURLForCLI(baseURL) {
		options = append(options, openai.WithHTTPClient(falkenlangchain.NewHeaderHTTPClient(http.DefaultClient, map[string]string{
			"X-Portkey-Provider": embeddings.DefaultPortkeyProvider,
		})))
	}
	model, err := openai.New(options...)
	if err != nil {
		return nil, fmt.Errorf("configure OpenAI-compatible agent LLM: %w", err)
	}
	return falkenlangchain.New(model), nil
}

func isDefaultPortkeyBaseURLForCLI(baseURL string) bool {
	return strings.TrimRight(strings.TrimSpace(baseURL), "/") == strings.TrimRight(embeddings.DefaultPortkeyBaseURL, "/")
}
