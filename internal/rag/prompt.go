package rag

import (
	"errors"
	"fmt"
	"strings"
)

const SystemPrompt = `You are answering questions using only the supplied context.

Rules:
- Use only the context below.
- If the answer is not in the context, say you do not know.
- Cite sources using [source N].
- Be concise.`

const StrictCitationSystemPrompt = SystemPrompt + `

Your previous answer did not correctly cite the provided sources.
Rewrite the answer using only the supplied context.
Every factual claim must include a citation like [source N].
Only cite source numbers that exist in the context.
If the answer is not supported by the context, say you do not know and cite the closest relevant source if applicable.`

type SourceChunk struct {
	SourceNumber int
	Path         string
	SourceRoot   string
	StartLine    int
	EndLine      int
	Text         string
	Score        float32
	Provenance   *SourceProvenance
}

type SourceProvenance struct {
	ToolName      string
	Query         string
	Strategy      string
	Rank          int
	RetrievalCall int
}

type BuiltPrompt struct {
	System  string
	User    string
	Sources []SourceChunk
}

func BuildPrompt(question string, chunks []SourceChunk) (BuiltPrompt, error) {
	if strings.TrimSpace(question) == "" {
		return BuiltPrompt{}, errors.New("question is required")
	}
	if len(chunks) == 0 {
		return BuiltPrompt{}, errors.New("retrieved chunks are required")
	}

	sources := make([]SourceChunk, len(chunks))
	var b strings.Builder
	b.WriteString("Context:\n\n")
	for i, chunk := range chunks {
		chunk.SourceNumber = i + 1
		sources[i] = chunk
		fmt.Fprintf(&b, "[source %d]\n", chunk.SourceNumber)
		fmt.Fprintf(&b, "Path: %s\n", DisplayPath(chunk.Path, chunk.SourceRoot))
		fmt.Fprintf(&b, "Lines: %d-%d\n", chunk.StartLine, chunk.EndLine)
		b.WriteString("Text:\n")
		b.WriteString(chunk.Text)
		b.WriteString("\n\n")
	}
	b.WriteString("Question:\n")
	b.WriteString(question)

	return BuiltPrompt{System: SystemPrompt, User: b.String(), Sources: sources}, nil
}

func SourceChunksFromRetrieved(chunks []RetrievedChunk) []SourceChunk {
	out := make([]SourceChunk, 0, len(chunks))
	for i, chunk := range chunks {
		out = append(out, SourceChunk{
			SourceNumber: i + 1,
			Path:         chunk.Path,
			SourceRoot:   chunk.SourceRoot,
			StartLine:    chunk.Chunk.StartLine,
			EndLine:      chunk.Chunk.EndLine,
			Text:         chunk.Chunk.ChunkText,
			Score:        chunk.Score,
		})
	}
	return out
}
