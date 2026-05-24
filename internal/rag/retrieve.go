package rag

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/vectorstore"
)

var ErrNoIndex = errors.New("No index found. Run `falkengo ingest <directory>` first.")

type RetrievalMode string

const (
	RetrievalModeVector  RetrievalMode = "vector"
	RetrievalModeLexical RetrievalMode = "lexical"
	RetrievalModeHybrid  RetrievalMode = "hybrid"
)

type RerankerMode string

const (
	RerankerModeNone      RerankerMode = "none"
	RerankerModeHeuristic RerankerMode = "heuristic"
)

const reciprocalRankFusionK = 60.0
const maxTotalCandidates = 500

type RetrieveOptions struct {
	Question          string
	TopK              int
	CandidateK        int
	VectorCandidateK  int
	LexicalCandidateK int
	Mode              RetrievalMode
	Paths             config.Paths
	Embedder          llm.Embedder
	Vector            vectorstore.Store
	OpenVector        func(context.Context, string, int) (vectorstore.Store, error)
	MaxPerDocument    int
	DisableDiversify  bool
	RerankerMode      RerankerMode
	Reranker          Reranker
	QueryPlannerMode  QueryPlannerMode
	QueryPlanner      QueryPlanner
	MaxSubqueries     int
	SourceFilter      SourceFilter
}

type RetrievedChunk struct {
	Chunk       manifest.Chunk
	Path        string
	SourceRoot  string
	Score       float32
	RankScore   float64
	Sources     []string
	RerankScore float64
	Reranker    string
}

type candidateHit struct {
	ChunkID    string
	Source     string
	Rank       int
	BestRank   int
	Score      float32
	FusedScore float64
	Sources    []string
	Query      string
	QueryIndex int
}

type candidateResult struct {
	candidate   candidateHit
	chunk       manifest.Chunk
	document    *manifest.Document
	rerankScore float64
	reranker    string
}

func Retrieve(ctx context.Context, store manifest.Store, opts RetrieveOptions) ([]RetrievedChunk, error) {
	result, err := RetrieveWithPlan(ctx, store, opts)
	if err != nil {
		return nil, err
	}
	return result.Chunks, nil
}

type RetrieveResult struct {
	Chunks []RetrievedChunk
	Plan   QueryPlan
}

func RetrieveWithPlan(ctx context.Context, store manifest.Store, opts RetrieveOptions) (RetrieveResult, error) {
	normalized, err := normalizeRetrieveOptions(opts)
	if err != nil {
		return RetrieveResult{}, err
	}
	if err := CheckIndexForMode(normalized.Paths, normalized.Mode); err != nil {
		return RetrieveResult{}, err
	}

	plan, err := buildQueryPlan(ctx, normalized)
	if err != nil {
		return RetrieveResult{}, err
	}
	allCandidates := make([]candidateHit, 0)
	for i, query := range plan.Queries {
		queryOpts := normalized
		queryOpts.Question = query
		candidates, err := retrieveCandidatesForQuery(ctx, store, queryOpts)
		if err != nil {
			return RetrieveResult{}, err
		}
		for j := range candidates {
			candidates[j].Query = query
			candidates[j].QueryIndex = i
		}
		allCandidates = append(allCandidates, candidates...)
	}
	if len(plan.Queries) > 1 {
		allCandidates = fuseCandidatesAcrossQueries(allCandidates)
	}
	if len(allCandidates) > maxTotalCandidates {
		allCandidates = allCandidates[:maxTotalCandidates]
	}
	chunks, err := finalizeCandidates(ctx, store, allCandidates, normalized)
	if err != nil {
		return RetrieveResult{}, err
	}
	return RetrieveResult{Chunks: chunks, Plan: plan}, nil
}

func retrieveCandidatesForQuery(ctx context.Context, store manifest.Store, opts RetrieveOptions) ([]candidateHit, error) {
	var candidates []candidateHit
	var err error
	switch opts.Mode {
	case RetrievalModeVector:
		candidates, err = vectorCandidates(ctx, opts)
	case RetrievalModeLexical:
		candidates, err = lexicalCandidates(ctx, store, opts)
	case RetrievalModeHybrid:
		var vectorHits []candidateHit
		vectorHits, err = vectorCandidates(ctx, opts)
		if err != nil {
			return nil, err
		}
		var lexicalHits []candidateHit
		lexicalHits, err = lexicalCandidates(ctx, store, opts)
		if err != nil {
			return nil, err
		}
		candidates = fuseCandidates(vectorHits, lexicalHits)
	default:
		err = fmt.Errorf("invalid retrieval mode %q", opts.Mode)
	}
	if err != nil {
		return nil, err
	}
	return candidates, nil
}

func vectorCandidates(ctx context.Context, opts RetrieveOptions) ([]candidateHit, error) {
	if opts.Embedder == nil {
		return nil, errors.New("embedder is required")
	}
	embedding, err := opts.Embedder.EmbedText(ctx, opts.Question)
	if err != nil {
		return nil, fmt.Errorf("embed question: %w", err)
	}
	vector := opts.Vector
	if vector == nil {
		openVector := opts.OpenVector
		if openVector == nil {
			openVector = vectorstore.Open
		}
		vector, err = openVector(ctx, opts.Paths.VecgoPath, len(embedding.Vector))
		if err != nil {
			return nil, fmt.Errorf("open vector database: %w", err)
		}
		defer vector.Close()
	}
	hits, err := vector.Search(ctx, embedding.Vector, opts.VectorCandidateK)
	if err != nil {
		return nil, fmt.Errorf("search vector database: %w", err)
	}
	candidates := make([]candidateHit, 0, len(hits))
	for i, hit := range hits {
		if hit.ChunkID == "" {
			continue
		}
		rank := i + 1
		candidates = append(candidates, candidateHit{
			ChunkID:    hit.ChunkID,
			Source:     "vector",
			Rank:       rank,
			BestRank:   rank,
			Score:      hit.Score,
			FusedScore: float64(hit.Score),
			Sources:    []string{"vector"},
			Query:      opts.Question,
		})
	}
	return candidates, nil
}

func lexicalCandidates(ctx context.Context, store manifest.Store, opts RetrieveOptions) ([]candidateHit, error) {
	lexicalStore, ok := store.(manifest.LexicalSearchStore)
	if !ok {
		return nil, errors.New("manifest store does not support lexical search")
	}
	hits, err := lexicalStore.SearchLexicalChunks(ctx, opts.Question, opts.LexicalCandidateK)
	if err != nil {
		return nil, fmt.Errorf("search lexical index: %w", err)
	}
	candidates := make([]candidateHit, 0, len(hits))
	for _, hit := range hits {
		if hit.ChunkID == "" {
			continue
		}
		rank := len(candidates) + 1
		score := reciprocalRankScore(rank)
		candidates = append(candidates, candidateHit{
			ChunkID:    hit.ChunkID,
			Source:     "lexical",
			Rank:       rank,
			BestRank:   rank,
			Score:      float32(score),
			FusedScore: score,
			Sources:    []string{"lexical"},
			Query:      opts.Question,
		})
	}
	return candidates, nil
}

func finalizeCandidates(ctx context.Context, store manifest.Store, candidates []candidateHit, opts RetrieveOptions) ([]RetrievedChunk, error) {
	if len(candidates) == 0 {
		return nil, nil
	}
	ids := make([]string, 0, len(candidates))
	uniqueCandidates := make([]candidateHit, 0, len(candidates))
	seenIDs := make(map[string]struct{}, len(candidates))
	for _, candidate := range candidates {
		if candidate.ChunkID == "" {
			continue
		}
		if _, ok := seenIDs[candidate.ChunkID]; ok {
			continue
		}
		seenIDs[candidate.ChunkID] = struct{}{}
		ids = append(ids, candidate.ChunkID)
		uniqueCandidates = append(uniqueCandidates, candidate)
	}
	chunks, err := store.GetActiveChunksByIDs(ctx, ids)
	if err != nil {
		return nil, fmt.Errorf("load active chunks: %w", err)
	}
	chunkByID := make(map[string]manifest.Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	docByID := make(map[string]*manifest.Document)
	candidateResults := make([]candidateResult, 0, len(chunks))
	for _, candidate := range uniqueCandidates {
		chunk, ok := chunkByID[candidate.ChunkID]
		if !ok || !chunk.Active {
			continue
		}
		doc, ok := docByID[chunk.DocumentID]
		if !ok {
			doc, err = store.GetDocumentByID(ctx, chunk.DocumentID)
			if errors.Is(err, manifest.ErrNotFound) {
				continue
			}
			if err != nil {
				return nil, fmt.Errorf("load document %s: %w", chunk.DocumentID, err)
			}
			docByID[chunk.DocumentID] = doc
		}
		if doc.Status != manifest.DocumentStatusIndexed {
			continue
		}
		if !opts.SourceFilter.Match(doc.Path, doc.SourceRoot) {
			continue
		}
		candidateResults = append(candidateResults, candidateResult{
			candidate: candidate,
			chunk:     chunk,
			document:  doc,
		})
	}
	candidateResults, err = maybeRerankCandidateResults(ctx, opts.Question, candidateResults, opts)
	if err != nil {
		return nil, err
	}
	if shouldDiversify(opts) {
		candidateResults = diversifyResults(candidateResults, opts.TopK, opts.MaxPerDocument)
	} else {
		candidateResults = trimResults(candidateResults, opts.TopK)
	}

	results := make([]RetrievedChunk, 0, len(candidateResults))
	for _, result := range candidateResults {
		results = append(results, toRetrievedChunk(result))
	}
	return results, nil
}

func toRetrievedChunk(result candidateResult) RetrievedChunk {
	score := result.candidate.Score
	if result.reranker != "" {
		score = float32(result.rerankScore)
	}
	return RetrievedChunk{
		Chunk:       result.chunk,
		Path:        result.document.Path,
		SourceRoot:  result.document.SourceRoot,
		Score:       score,
		RankScore:   result.candidate.FusedScore,
		Sources:     append([]string(nil), result.candidate.Sources...),
		RerankScore: result.rerankScore,
		Reranker:    result.reranker,
	}
}

func DisplayPath(path, sourceRoot string) string {
	path = strings.TrimSpace(path)
	sourceRoot = strings.TrimSpace(sourceRoot)
	if path == "" || sourceRoot == "" {
		return path
	}
	rel, err := filepath.Rel(sourceRoot, path)
	if err != nil || rel == "." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || rel == ".." || filepath.IsAbs(rel) {
		return path
	}
	return filepath.ToSlash(rel)
}

func maybeRerankCandidateResults(ctx context.Context, question string, results []candidateResult, opts RetrieveOptions) ([]candidateResult, error) {
	if len(results) == 0 {
		return results, nil
	}
	reranker := opts.Reranker
	rerankerName := string(opts.RerankerMode)
	if reranker != nil {
		rerankerName = "custom"
	}
	if reranker == nil {
		switch opts.RerankerMode {
		case "", RerankerModeNone:
			return results, nil
		case RerankerModeHeuristic:
			reranker = HeuristicReranker{}
		default:
			return nil, fmt.Errorf("invalid reranker mode %q", opts.RerankerMode)
		}
	}
	candidates := make([]RerankCandidate, 0, len(results))
	for _, result := range results {
		candidates = append(candidates, RerankCandidate{
			ChunkID:   result.chunk.ID,
			Path:      result.document.Path,
			Text:      result.chunk.ChunkText,
			StartLine: result.chunk.StartLine,
			EndLine:   result.chunk.EndLine,
			Sources:   append([]string(nil), result.candidate.Sources...),
			Score:     result.candidate.Score,
			RankScore: result.candidate.FusedScore,
		})
	}
	reranked, err := reranker.Rerank(ctx, question, candidates, 0)
	if err != nil {
		return nil, fmt.Errorf("rerank candidates: %w", err)
	}
	byID := make(map[string]candidateResult, len(results))
	for _, result := range results {
		byID[result.chunk.ID] = result
	}
	seen := make(map[string]struct{}, len(results))
	out := make([]candidateResult, 0, len(results))
	for _, rerankResult := range reranked {
		result, ok := byID[rerankResult.ChunkID]
		if !ok {
			continue
		}
		if _, ok := seen[rerankResult.ChunkID]; ok {
			continue
		}
		result.rerankScore = rerankResult.Score
		result.reranker = rerankerName
		out = append(out, result)
		seen[rerankResult.ChunkID] = struct{}{}
	}
	for _, result := range results {
		if _, ok := seen[result.chunk.ID]; ok {
			continue
		}
		out = append(out, result)
	}
	return out, nil
}

func diversifyResults(results []candidateResult, topK int, maxPerDocument int) []candidateResult {
	if topK <= 0 || len(results) <= topK {
		return results
	}
	selected := make([]candidateResult, 0, topK)
	selectedIDs := make(map[string]struct{}, topK)
	perDoc := make(map[string]int)
	for _, result := range results {
		docID := result.chunk.DocumentID
		if perDoc[docID] >= maxPerDocument {
			continue
		}
		selected = append(selected, result)
		selectedIDs[result.chunk.ID] = struct{}{}
		perDoc[docID]++
		if len(selected) == topK {
			return selected
		}
	}
	for _, result := range results {
		if _, ok := selectedIDs[result.chunk.ID]; ok {
			continue
		}
		selected = append(selected, result)
		selectedIDs[result.chunk.ID] = struct{}{}
		if len(selected) == topK {
			return selected
		}
	}
	return selected
}

func trimResults(results []candidateResult, topK int) []candidateResult {
	if topK <= 0 || len(results) <= topK {
		return results
	}
	return results[:topK]
}

func shouldDiversify(opts RetrieveOptions) bool {
	return !opts.DisableDiversify && opts.TopK >= 3 && opts.MaxPerDocument > 0
}

func CheckIndex(paths config.Paths) error {
	return CheckIndexForMode(paths, RetrievalModeVector)
}

func CheckIndexForMode(paths config.Paths, mode RetrievalMode) error {
	if mode == "" {
		mode = RetrievalModeVector
	}
	if _, err := ParseRetrievalMode(string(mode)); err != nil {
		return err
	}
	if _, err := os.Stat(paths.ManifestPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return ErrNoIndex
		}
		return err
	}
	if RetrievalModeUsesVector(mode) {
		if _, err := os.Stat(paths.VecgoPath); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return ErrNoIndex
			}
			return err
		}
	}
	return nil
}

func ParseRetrievalMode(value string) (RetrievalMode, error) {
	switch RetrievalMode(strings.TrimSpace(value)) {
	case "", RetrievalModeVector:
		return RetrievalModeVector, nil
	case RetrievalModeLexical:
		return RetrievalModeLexical, nil
	case RetrievalModeHybrid:
		return RetrievalModeHybrid, nil
	default:
		return "", fmt.Errorf("--retrieval must be vector, lexical, or hybrid")
	}
}

func ParseRerankerMode(value string) (RerankerMode, error) {
	switch RerankerMode(strings.TrimSpace(value)) {
	case "", RerankerModeNone:
		return RerankerModeNone, nil
	case RerankerModeHeuristic:
		return RerankerModeHeuristic, nil
	default:
		return "", fmt.Errorf("--reranker must be none or heuristic")
	}
}

func RetrievalModeUsesVector(mode RetrievalMode) bool {
	return mode == RetrievalModeVector || mode == RetrievalModeHybrid || mode == ""
}

func RetrievalModeUsesLexical(mode RetrievalMode) bool {
	return mode == RetrievalModeLexical || mode == RetrievalModeHybrid
}

func normalizeRetrieveOptions(opts RetrieveOptions) (RetrieveOptions, error) {
	if opts.TopK <= 0 {
		opts.TopK = 8
	}
	mode, err := ParseRetrievalMode(string(opts.Mode))
	if err != nil {
		return opts, err
	}
	opts.Mode = mode
	rerankerMode, err := ParseRerankerMode(string(opts.RerankerMode))
	if err != nil {
		return opts, err
	}
	opts.RerankerMode = rerankerMode
	queryPlannerMode, err := ParseQueryPlannerMode(string(opts.QueryPlannerMode))
	if err != nil {
		return opts, err
	}
	opts.QueryPlannerMode = queryPlannerMode
	if opts.MaxSubqueries <= 0 {
		opts.MaxSubqueries = 4
	}
	if opts.MaxSubqueries < 1 || opts.MaxSubqueries > 8 {
		return opts, fmt.Errorf("--max-subqueries must be between 1 and 8")
	}
	if opts.CandidateK <= 0 {
		if opts.SourceFilter.IsEmpty() {
			opts.CandidateK = maxInt(opts.TopK*5, 50)
		} else {
			opts.CandidateK = maxInt(opts.TopK*10, 100)
		}
	}
	if opts.VectorCandidateK <= 0 {
		opts.VectorCandidateK = opts.CandidateK
	}
	if opts.LexicalCandidateK <= 0 {
		opts.LexicalCandidateK = opts.CandidateK
	}
	if opts.MaxPerDocument <= 0 {
		opts.MaxPerDocument = 2
	}
	return opts, nil
}

func NormalizeRetrieveOptions(opts RetrieveOptions) (RetrieveOptions, error) {
	return normalizeRetrieveOptions(opts)
}

func fuseCandidates(vectorHits []candidateHit, lexicalHits []candidateHit) []candidateHit {
	byID := make(map[string]*candidateHit)
	add := func(hit candidateHit, weight float64) {
		if hit.ChunkID == "" {
			return
		}
		existing, ok := byID[hit.ChunkID]
		if !ok {
			copy := hit
			copy.BestRank = hit.Rank
			copy.FusedScore = 0
			copy.Sources = nil
			byID[hit.ChunkID] = &copy
			existing = &copy
		}
		existing.FusedScore += weight * reciprocalRankScore(hit.Rank)
		if existing.BestRank == 0 || hit.Rank < existing.BestRank {
			existing.BestRank = hit.Rank
		}
		if hit.Source == "vector" {
			existing.Score = float32(existing.FusedScore)
		}
		existing.Sources = appendSource(existing.Sources, hit.Source)
	}
	for _, hit := range vectorHits {
		add(hit, 1)
	}
	for _, hit := range lexicalHits {
		add(hit, 1)
	}
	out := make([]candidateHit, 0, len(byID))
	for _, hit := range byID {
		hit.Score = float32(hit.FusedScore)
		out = append(out, *hit)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].FusedScore != out[j].FusedScore {
			return out[i].FusedScore > out[j].FusedScore
		}
		if out[i].BestRank != out[j].BestRank {
			return out[i].BestRank < out[j].BestRank
		}
		return out[i].ChunkID < out[j].ChunkID
	})
	return out
}

func fuseCandidatesAcrossQueries(candidates []candidateHit) []candidateHit {
	byID := make(map[string]*candidateHit)
	add := func(hit candidateHit) {
		if hit.ChunkID == "" {
			return
		}
		existing, ok := byID[hit.ChunkID]
		if !ok {
			copy := hit
			copy.FusedScore = 0
			copy.Sources = nil
			copy.BestRank = hit.BestRank
			if copy.BestRank == 0 {
				copy.BestRank = hit.Rank
			}
			byID[hit.ChunkID] = &copy
			existing = &copy
		}
		weight := 1.0
		if hit.QueryIndex == 0 {
			weight = 1.25
			existing.Query = hit.Query
			existing.QueryIndex = 0
		} else if existing.Query == "" {
			existing.Query = hit.Query
			existing.QueryIndex = hit.QueryIndex
		}
		rank := hit.BestRank
		if rank <= 0 {
			rank = hit.Rank
		}
		if rank <= 0 {
			rank = 1
		}
		existing.FusedScore += weight * reciprocalRankScore(rank)
		if existing.BestRank == 0 || rank < existing.BestRank {
			existing.BestRank = rank
		}
		for _, source := range hit.Sources {
			existing.Sources = appendSource(existing.Sources, source)
		}
		if len(hit.Sources) == 0 {
			existing.Sources = appendSource(existing.Sources, hit.Source)
		}
	}
	for _, hit := range candidates {
		add(hit)
	}
	out := make([]candidateHit, 0, len(byID))
	for _, hit := range byID {
		hit.Score = float32(hit.FusedScore)
		out = append(out, *hit)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].FusedScore != out[j].FusedScore {
			return out[i].FusedScore > out[j].FusedScore
		}
		iOriginal := out[i].QueryIndex == 0
		jOriginal := out[j].QueryIndex == 0
		if iOriginal != jOriginal {
			return iOriginal
		}
		if out[i].BestRank != out[j].BestRank {
			return out[i].BestRank < out[j].BestRank
		}
		return out[i].ChunkID < out[j].ChunkID
	})
	return out
}

func reciprocalRankScore(rank int) float64 {
	return 1 / (reciprocalRankFusionK + float64(rank))
}

func appendSource(sources []string, source string) []string {
	for _, existing := range sources {
		if existing == source {
			return sources
		}
	}
	return append(sources, source)
}

func BuildFTSQuery(question string) string {
	return manifest.BuildFTSQuery(question)
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}
