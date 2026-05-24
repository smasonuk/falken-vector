package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/smasonuk/falken-vector/internal/rag"
)

func printQueryResults(w io.Writer, question string, results []rag.RetrievedChunk) {
	fmt.Fprintf(w, "Query: %s\n\n", question)
	if len(results) == 0 {
		fmt.Fprintln(w, "No relevant chunks found.")
		return
	}
	for i, result := range results {
		fmt.Fprintf(w, "Result %d\n", i+1)
		fmt.Fprintf(w, "Score: %.4f\n", result.Score)
		if result.Reranker != "" {
			fmt.Fprintf(w, "Reranker: %s\n", result.Reranker)
			fmt.Fprintf(w, "Rerank score: %.4f\n", result.RerankScore)
			fmt.Fprintf(w, "Rank score: %.4f\n", result.RankScore)
		}
		if len(result.Sources) != 0 {
			fmt.Fprintf(w, "Retrieval: %s\n", strings.Join(result.Sources, ","))
		}
		fmt.Fprintf(w, "Source: %s\n\n", SourceReference(rag.SourceChunk{
			SourceNumber: i + 1,
			Path:         result.Path,
			StartLine:    result.Chunk.StartLine,
			EndLine:      result.Chunk.EndLine,
		}))
		fmt.Fprintln(w, result.Chunk.ChunkText)
		if i != len(results)-1 {
			fmt.Fprintln(w)
		}
	}
}

func printQueryPlan(w io.Writer, plan rag.QueryPlan) {
	if len(plan.Queries) == 0 {
		return
	}
	fmt.Fprintln(w, "Query plan:")
	for i, query := range plan.Queries {
		fmt.Fprintf(w, "  %d. %s\n", i+1, query)
	}
	if plan.Warning != "" {
		fmt.Fprintf(w, "warning: %s\n", plan.Warning)
	}
	fmt.Fprintln(w)
}

func SourceReference(source rag.SourceChunk) string {
	if source.StartLine > 0 && source.EndLine > 0 {
		return fmt.Sprintf("[source %d] %s:%d-%d", source.SourceNumber, source.Path, source.StartLine, source.EndLine)
	}
	return fmt.Sprintf("[source %d] %s", source.SourceNumber, source.Path)
}

func printSources(w io.Writer, sources []rag.SourceChunk) {
	if len(sources) == 0 {
		return
	}
	fmt.Fprintln(w, "Sources:")
	for _, source := range sources {
		fmt.Fprintln(w, SourceReference(source))
	}
}

type queryJSONOutput struct {
	Question     string             `json:"question"`
	Retrieval    queryJSONRetrieval `json:"retrieval"`
	QueryPlan    *rag.QueryPlan     `json:"query_plan,omitempty"`
	SourceFilter *rag.SourceFilter  `json:"source_filter,omitempty"`
	Chunks       []queryJSONChunk   `json:"chunks"`
}

type queryJSONRetrieval struct {
	Mode              string `json:"mode"`
	Reranker          string `json:"reranker"`
	QueryPlanner      string `json:"query_planner"`
	TopK              int    `json:"top_k"`
	CandidateK        int    `json:"candidate_k"`
	VectorCandidateK  int    `json:"vector_candidate_k"`
	LexicalCandidateK int    `json:"lexical_candidate_k"`
}

type queryJSONChunk struct {
	SourceNumber int      `json:"source_number"`
	Path         string   `json:"path"`
	StartLine    int      `json:"start_line"`
	EndLine      int      `json:"end_line"`
	Score        float32  `json:"score"`
	RankScore    float64  `json:"rank_score"`
	Sources      []string `json:"sources,omitempty"`
	Reranker     string   `json:"reranker,omitempty"`
	RerankScore  float64  `json:"rerank_score,omitempty"`
	ChunkID      string   `json:"chunk_id"`
	DocumentID   string   `json:"document_id"`
	ChunkIndex   int      `json:"chunk_index"`
	Chunker      string   `json:"chunker,omitempty"`
	Language     string   `json:"language,omitempty"`
	SymbolName   string   `json:"symbol_name,omitempty"`
	SymbolKind   string   `json:"symbol_kind,omitempty"`
	HeadingPath  []string `json:"heading_path,omitempty"`
	Text         string   `json:"text"`
}

func writeQueryJSON(w io.Writer, question string, opts rag.RetrieveOptions, result rag.RetrieveResult) error {
	out := queryJSONOutput{
		Question: question,
		Retrieval: queryJSONRetrieval{
			Mode:              string(opts.Mode),
			Reranker:          string(opts.RerankerMode),
			QueryPlanner:      string(opts.QueryPlannerMode),
			TopK:              opts.TopK,
			CandidateK:        opts.CandidateK,
			VectorCandidateK:  opts.VectorCandidateK,
			LexicalCandidateK: opts.LexicalCandidateK,
		},
		Chunks: make([]queryJSONChunk, 0, len(result.Chunks)),
	}
	if len(result.Plan.Queries) != 0 {
		plan := result.Plan
		out.QueryPlan = &plan
	}
	if !opts.SourceFilter.IsEmpty() {
		filter := opts.SourceFilter
		out.SourceFilter = &filter
	}
	for i, chunk := range result.Chunks {
		out.Chunks = append(out.Chunks, queryJSONChunk{
			SourceNumber: i + 1,
			Path:         chunk.Path,
			StartLine:    chunk.Chunk.StartLine,
			EndLine:      chunk.Chunk.EndLine,
			Score:        chunk.Score,
			RankScore:    chunk.RankScore,
			Sources:      append([]string(nil), chunk.Sources...),
			Reranker:     chunk.Reranker,
			RerankScore:  chunk.RerankScore,
			ChunkID:      chunk.Chunk.ID,
			DocumentID:   chunk.Chunk.DocumentID,
			ChunkIndex:   chunk.Chunk.ChunkIndex,
			Chunker:      chunk.Chunk.Chunker,
			Language:     chunk.Chunk.Language,
			SymbolName:   chunk.Chunk.SymbolName,
			SymbolKind:   chunk.Chunk.SymbolKind,
			HeadingPath:  append([]string(nil), chunk.Chunk.HeadingPath...),
			Text:         chunk.Chunk.ChunkText,
		})
	}
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return encoder.Encode(out)
}

func printRetrievalDebug(w io.Writer, opts rag.RetrieveOptions, plan rag.QueryPlan) {
	fmt.Fprintln(w, "Retrieval debug:")
	fmt.Fprintf(w, "  mode: %s\n", opts.Mode)
	fmt.Fprintf(w, "  reranker: %s\n", opts.RerankerMode)
	fmt.Fprintf(w, "  query planner: %s\n", opts.QueryPlannerMode)
	fmt.Fprintf(w, "  top-k: %d\n", opts.TopK)
	fmt.Fprintf(w, "  candidate-k: %d\n", opts.CandidateK)
	fmt.Fprintf(w, "  vector-candidate-k: %d\n", opts.VectorCandidateK)
	fmt.Fprintf(w, "  lexical-candidate-k: %d\n", opts.LexicalCandidateK)
	fmt.Fprintf(w, "  max-per-document: %d\n", opts.MaxPerDocument)
	fmt.Fprintf(w, "  diversify: %t\n", !opts.DisableDiversify && opts.TopK >= 3 && opts.MaxPerDocument > 0)
	if !opts.SourceFilter.IsEmpty() {
		fmt.Fprintln(w, "source filter:")
		if len(opts.SourceFilter.IncludeGlobs) != 0 {
			fmt.Fprintf(w, "  include: %s\n", strings.Join(opts.SourceFilter.IncludeGlobs, ", "))
		}
		if len(opts.SourceFilter.ExcludeGlobs) != 0 {
			fmt.Fprintf(w, "  exclude: %s\n", strings.Join(opts.SourceFilter.ExcludeGlobs, ", "))
		}
		if len(opts.SourceFilter.SourceRoots) != 0 {
			fmt.Fprintf(w, "  source roots: %s\n", strings.Join(opts.SourceFilter.SourceRoots, ", "))
		}
	}
	fmt.Fprintln(w)
	if len(plan.Queries) != 0 {
		printQueryPlan(w, plan)
	}
}
