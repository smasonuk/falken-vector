package rag

import (
	"context"
	"errors"

	"github.com/smasonuk/falken-vector/internal/llm"
)

type AskOptions struct {
	Question       string
	Model          string
	Chunks         []RetrievedChunk
	LLM            llm.Client
	CitationPolicy CitationPolicy
}

type AskResult struct {
	Answer           string
	Sources          []SourceChunk
	CitationWarnings []string
	CitationValid    bool
	Retried          bool
}

func Ask(ctx context.Context, opts AskOptions) (AskResult, error) {
	if opts.LLM == nil {
		return AskResult{}, errors.New("llm client is required")
	}
	sources := SourceChunksFromRetrieved(opts.Chunks)
	prompt, err := BuildPrompt(opts.Question, sources)
	if err != nil {
		return AskResult{}, err
	}
	response, err := opts.LLM.Complete(ctx, llm.CompletionRequest{
		Model:       opts.Model,
		System:      prompt.System,
		User:        prompt.User,
		Temperature: 0.1,
	})
	if err != nil {
		return AskResult{Sources: prompt.Sources}, err
	}
	result := AskResult{Answer: response.Text, Sources: prompt.Sources}
	policy := normalizeCitationPolicy(opts.CitationPolicy)
	if policy == CitationPolicyOff {
		result.CitationValid = true
		return result, nil
	}
	validation := ValidateAnswerCitations(response.Text, len(prompt.Sources))
	if validation.Valid {
		result.CitationValid = true
		return result, nil
	}
	result.CitationWarnings = append(result.CitationWarnings, validation.Warnings...)
	if policy != CitationPolicyValidateAndRetry {
		return result, nil
	}
	retryResponse, err := opts.LLM.Complete(ctx, llm.CompletionRequest{
		Model:       opts.Model,
		System:      StrictCitationSystemPrompt,
		User:        prompt.User,
		Temperature: 0.1,
	})
	if err != nil {
		result.CitationWarnings = append(result.CitationWarnings, "citation retry failed: "+err.Error())
		return result, nil
	}
	result.Retried = true
	result.Answer = retryResponse.Text
	result.CitationWarnings = append(result.CitationWarnings, "retried with stricter citation prompt")
	retryValidation := ValidateAnswerCitations(retryResponse.Text, len(prompt.Sources))
	result.CitationValid = retryValidation.Valid
	if !retryValidation.Valid {
		result.CitationWarnings = append(result.CitationWarnings, "final answer may have citation issues: "+joinCitationWarnings(retryValidation.Warnings))
	}
	return result, nil
}
