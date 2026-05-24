package rag

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/vectorstore"
)

func TestRetrieveReturnsActiveChunks(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	paths, err := config.ResolvePaths(filepath.Join(tmp, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatalf("EnsureStateDirs: %v", err)
	}
	if err := os.WriteFile(paths.ManifestPath, []byte("test"), 0o644); err != nil {
		t.Fatalf("write manifest sentinel: %v", err)
	}
	touchDir(t, paths.VecgoPath)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"chunk": {ID: "chunk", DocumentID: "doc", ChunkText: "hello", Active: true, StartLine: 1, EndLine: 2},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question: "hello",
		TopK:     1,
		Paths:    paths,
		Embedder: fakeRAGEmbedder{},
		Vector:   fakeVector{hits: []vectorstore.Hit{{ChunkID: "chunk", Score: 0.9}}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(results) != 1 || results[0].Path != "README.md" || results[0].Score != 0.9 {
		t.Fatalf("results = %+v", results)
	}
}

func TestRetrieveSkipsChunksForNonIndexedDocuments(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	paths, err := config.ResolvePaths(filepath.Join(tmp, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatalf("EnsureStateDirs: %v", err)
	}
	if err := os.WriteFile(paths.ManifestPath, []byte("test"), 0o644); err != nil {
		t.Fatalf("write manifest sentinel: %v", err)
	}
	touchDir(t, paths.VecgoPath)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusError},
		},
		chunks: map[string]manifest.Chunk{
			"chunk": {ID: "chunk", DocumentID: "doc", ChunkText: "stale", Active: true, StartLine: 1, EndLine: 2},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question: "hello",
		TopK:     1,
		Paths:    paths,
		Embedder: fakeRAGEmbedder{},
		Vector:   fakeVector{hits: []vectorstore.Hit{{ChunkID: "chunk", Score: 0.9}}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("results = %+v, want none for non-indexed document", results)
	}
}

func TestRetrieveLexicalModeReturnsLexicalHits(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, false)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "internal/rag/retrieve.go", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"lex": {ID: "lex", DocumentID: "doc", ChunkText: "ErrPendingRunDetected", Active: true, StartLine: 10, EndLine: 12},
		},
		lexicalHits: []manifest.LexicalHit{{ChunkID: "lex", Score: -1.2}},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question: "ErrPendingRunDetected",
		TopK:     1,
		Paths:    paths,
		Mode:     RetrievalModeLexical,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(results) != 1 || results[0].Chunk.ID != "lex" || !sameSources(results[0].Sources, []string{"lexical"}) {
		t.Fatalf("results = %+v, want lexical result", results)
	}
	wantScore := float32(1.0 / 61.0)
	if math.Abs(float64(results[0].Score-wantScore)) > 0.000001 {
		t.Fatalf("lexical score = %.6f, want positive rank score %.6f", results[0].Score, wantScore)
	}
	if math.Abs(results[0].RankScore-float64(results[0].Score)) > 0.000001 {
		t.Fatalf("RankScore = %.6f, Score = %.6f, want same display score for lexical-only", results[0].RankScore, results[0].Score)
	}
}

func TestRetrieveHybridFusesAndDedupesSources(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"a": {ID: "a", DocumentID: "doc", Active: true, StartLine: 1, EndLine: 2},
			"b": {ID: "b", DocumentID: "doc", Active: true, StartLine: 3, EndLine: 4},
		},
		lexicalHits: []manifest.LexicalHit{{ChunkID: "b", Score: -2}, {ChunkID: "a", Score: -1}},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question: "hello",
		TopK:     2,
		Paths:    paths,
		Mode:     RetrievalModeHybrid,
		Embedder: fakeRAGEmbedder{},
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "a", Score: 0.9},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(results) != 2 || results[0].Chunk.ID != "a" {
		t.Fatalf("results = %+v, want chunk A first because it appears in both sources", results)
	}
	if !sameSources(results[0].Sources, []string{"vector", "lexical"}) {
		t.Fatalf("sources = %+v, want vector+lexical", results[0].Sources)
	}
}

func TestRetrieveHybridFallsBackWhenOneSourceHasNoHits(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"lex": {ID: "lex", DocumentID: "doc", Active: true},
		},
		lexicalHits: []manifest.LexicalHit{{ChunkID: "lex", Score: -1}},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question: "hello",
		TopK:     1,
		Paths:    paths,
		Mode:     RetrievalModeHybrid,
		Embedder: fakeRAGEmbedder{},
		Vector:   fakeVector{},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(results) != 1 || results[0].Chunk.ID != "lex" {
		t.Fatalf("results = %+v, want lexical fallback", results)
	}
}

func TestRetrieveWithPlanNoneMatchesRetrieve(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, false)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"chunk": {ID: "chunk", DocumentID: "doc", Active: true},
		},
		lexicalHits: []manifest.LexicalHit{{ChunkID: "chunk", Score: -1}},
	}
	opts := RetrieveOptions{Question: "hello", TopK: 1, Paths: paths, Mode: RetrievalModeLexical}
	plain, err := Retrieve(ctx, store, opts)
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	withPlan, err := RetrieveWithPlan(ctx, store, opts)
	if err != nil {
		t.Fatalf("RetrieveWithPlan: %v", err)
	}
	assertChunkIDs(t, plain, []string{"chunk"})
	assertChunkIDs(t, withPlan.Chunks, []string{"chunk"})
	if !sameStringSlice(withPlan.Plan.Queries, []string{"hello"}) || withPlan.Plan.Mode != string(QueryPlannerModeNone) {
		t.Fatalf("plan = %+v, want none plan with original query", withPlan.Plan)
	}
}

func TestRetrieveWithPlanRunsMultipleQueries(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, false)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"a": {ID: "a", DocumentID: "doc", Active: true},
			"b": {ID: "b", DocumentID: "doc", Active: true},
		},
		lexicalByQuery: map[string][]manifest.LexicalHit{
			"original": {{ChunkID: "a", Score: -1}},
			"sub":      {{ChunkID: "b", Score: -1}},
		},
	}
	result, err := RetrieveWithPlan(ctx, store, RetrieveOptions{
		Question:          "original",
		TopK:              2,
		DisableDiversify:  true,
		Paths:             paths,
		Mode:              RetrievalModeLexical,
		QueryPlannerMode:  QueryPlannerModeHeuristic,
		QueryPlanner:      staticQueryPlanner{queries: []string{"sub"}},
		MaxSubqueries:     2,
		LexicalCandidateK: 2,
	})
	if err != nil {
		t.Fatalf("RetrieveWithPlan: %v", err)
	}
	if !sameStringSlice(result.Plan.Queries, []string{"original", "sub"}) {
		t.Fatalf("plan = %+v", result.Plan)
	}
	assertChunkIDs(t, result.Chunks, []string{"a", "b"})
}

func TestRetrieveWithPlanFusesChunksAcrossQueries(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, false)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"a": {ID: "a", DocumentID: "doc", Active: true},
			"b": {ID: "b", DocumentID: "doc", Active: true},
			"c": {ID: "c", DocumentID: "doc", Active: true},
		},
		lexicalByQuery: map[string][]manifest.LexicalHit{
			"original": {{ChunkID: "a", Score: -1}, {ChunkID: "c", Score: -2}},
			"sub":      {{ChunkID: "c", Score: -1}, {ChunkID: "b", Score: -2}},
		},
	}
	result, err := RetrieveWithPlan(ctx, store, RetrieveOptions{
		Question:          "original",
		TopK:              3,
		DisableDiversify:  true,
		Paths:             paths,
		Mode:              RetrievalModeLexical,
		QueryPlannerMode:  QueryPlannerModeHeuristic,
		QueryPlanner:      staticQueryPlanner{queries: []string{"sub"}},
		MaxSubqueries:     2,
		LexicalCandidateK: 3,
	})
	if err != nil {
		t.Fatalf("RetrieveWithPlan: %v", err)
	}
	assertChunkIDs(t, result.Chunks, []string{"c", "a", "b"})
}

func TestRetrieveWithPlanReranksAgainstOriginalQuestion(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, false)
	reranker := &recordingReranker{order: []string{"b", "a"}}
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"a": {ID: "a", DocumentID: "doc", Active: true},
			"b": {ID: "b", DocumentID: "doc", Active: true},
		},
		lexicalByQuery: map[string][]manifest.LexicalHit{
			"original": {{ChunkID: "a", Score: -1}},
			"sub":      {{ChunkID: "b", Score: -1}},
		},
	}
	result, err := RetrieveWithPlan(ctx, store, RetrieveOptions{
		Question:          "original",
		TopK:              2,
		DisableDiversify:  true,
		Paths:             paths,
		Mode:              RetrievalModeLexical,
		QueryPlannerMode:  QueryPlannerModeHeuristic,
		QueryPlanner:      staticQueryPlanner{queries: []string{"sub"}},
		MaxSubqueries:     2,
		Reranker:          reranker,
		LexicalCandidateK: 2,
	})
	if err != nil {
		t.Fatalf("RetrieveWithPlan: %v", err)
	}
	if reranker.question != "original" {
		t.Fatalf("reranker question = %q, want original", reranker.question)
	}
	assertChunkIDs(t, result.Chunks, []string{"b", "a"})
}

func TestRetrieveWithPlanCapsTotalCandidates(t *testing.T) {
	candidates := make([]candidateHit, 0, maxTotalCandidates+25)
	for i := 0; i < maxTotalCandidates+25; i++ {
		candidates = append(candidates, candidateHit{ChunkID: fmt.Sprintf("chunk-%03d", i), Rank: i + 1, BestRank: i + 1, QueryIndex: 1})
	}
	fused := fuseCandidatesAcrossQueries(candidates)
	if len(fused) <= maxTotalCandidates {
		t.Fatalf("pre-trim fused candidates = %d, want above cap", len(fused))
	}
	if len(fused) > maxTotalCandidates {
		fused = fused[:maxTotalCandidates]
	}
	if len(fused) != maxTotalCandidates {
		t.Fatalf("fused len = %d, want cap %d", len(fused), maxTotalCandidates)
	}
}

func TestRetrievePreservesCandidateOrderAfterLoadingChunks(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"a": {ID: "a", DocumentID: "doc", Active: true},
			"b": {ID: "b", DocumentID: "doc", Active: true},
			"c": {ID: "c", DocumentID: "doc", Active: true},
		},
		chunkOrder: []string{"a", "b", "c"},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:          "hello",
		TopK:              3,
		Paths:             paths,
		Embedder:          fakeRAGEmbedder{},
		DisableDiversify:  true,
		VectorCandidateK:  3,
		LexicalCandidateK: 3,
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "c", Score: 0.9},
			{ChunkID: "a", Score: 0.8},
			{ChunkID: "b", Score: 0.7},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"c", "a", "b"})
}

func TestRetrieveFiltersBeforeTrimming(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"bad-doc":  {ID: "bad-doc", Path: "bad.md", Status: manifest.DocumentStatusError},
			"good-doc": {ID: "good-doc", Path: "good.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"bad-doc":  {ID: "bad-doc", DocumentID: "bad-doc", Active: true},
			"inactive": {ID: "inactive", DocumentID: "good-doc", Active: false},
			"good-1":   {ID: "good-1", DocumentID: "good-doc", Active: true},
			"good-2":   {ID: "good-2", DocumentID: "good-doc", Active: true},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question: "hello",
		TopK:     2,
		Paths:    paths,
		Embedder: fakeRAGEmbedder{},
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "bad-doc", Score: 0.9},
			{ChunkID: "inactive", Score: 0.8},
			{ChunkID: "good-1", Score: 0.7},
			{ChunkID: "good-2", Score: 0.6},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"good-1", "good-2"})
}

func TestRetrieveOverfetchesBeforeFiltering(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"bad-doc":  {ID: "bad-doc", Path: "bad.md", Status: manifest.DocumentStatusError},
			"good-doc": {ID: "good-doc", Path: "good.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"inactive": {ID: "inactive", DocumentID: "good-doc", Active: false},
			"bad-doc":  {ID: "bad-doc", DocumentID: "bad-doc", Active: true},
			"good-1":   {ID: "good-1", DocumentID: "good-doc", Active: true},
			"good-2":   {ID: "good-2", DocumentID: "good-doc", Active: true},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:         "hello",
		TopK:             2,
		VectorCandidateK: 5,
		Paths:            paths,
		Embedder:         fakeRAGEmbedder{},
		Vector: fakeVector{
			wantLimit: 5,
			hits: []vectorstore.Hit{
				{ChunkID: "missing", Score: 0.9},
				{ChunkID: "inactive", Score: 0.8},
				{ChunkID: "bad-doc", Score: 0.7},
				{ChunkID: "good-1", Score: 0.6},
				{ChunkID: "good-2", Score: 0.5},
			},
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"good-1", "good-2"})
}

func TestRetrieveDiversifiesAcrossDocuments(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"docA": {ID: "docA", Path: "a.md", Status: manifest.DocumentStatusIndexed},
			"docB": {ID: "docB", Path: "b.md", Status: manifest.DocumentStatusIndexed},
			"docC": {ID: "docC", Path: "c.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"docA-1": {ID: "docA-1", DocumentID: "docA", Active: true},
			"docA-2": {ID: "docA-2", DocumentID: "docA", Active: true},
			"docA-3": {ID: "docA-3", DocumentID: "docA", Active: true},
			"docB-1": {ID: "docB-1", DocumentID: "docB", Active: true},
			"docC-1": {ID: "docC-1", DocumentID: "docC", Active: true},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:       "hello",
		TopK:           4,
		MaxPerDocument: 2,
		Paths:          paths,
		Embedder:       fakeRAGEmbedder{},
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "docA-1", Score: 0.9},
			{ChunkID: "docA-2", Score: 0.8},
			{ChunkID: "docA-3", Score: 0.7},
			{ChunkID: "docB-1", Score: 0.6},
			{ChunkID: "docC-1", Score: 0.5},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"docA-1", "docA-2", "docB-1", "docC-1"})
}

func TestRetrieveDiversifyBackfillsWhenNeeded(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"docA": {ID: "docA", Path: "a.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"docA-1": {ID: "docA-1", DocumentID: "docA", Active: true},
			"docA-2": {ID: "docA-2", DocumentID: "docA", Active: true},
			"docA-3": {ID: "docA-3", DocumentID: "docA", Active: true},
			"docA-4": {ID: "docA-4", DocumentID: "docA", Active: true},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:       "hello",
		TopK:           4,
		MaxPerDocument: 2,
		Paths:          paths,
		Embedder:       fakeRAGEmbedder{},
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "docA-1", Score: 0.9},
			{ChunkID: "docA-2", Score: 0.8},
			{ChunkID: "docA-3", Score: 0.7},
			{ChunkID: "docA-4", Score: 0.6},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"docA-1", "docA-2", "docA-3", "docA-4"})
}

func TestRetrieveCanDisableDiversify(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"docA": {ID: "docA", Path: "a.md", Status: manifest.DocumentStatusIndexed},
			"docB": {ID: "docB", Path: "b.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"docA-1": {ID: "docA-1", DocumentID: "docA", Active: true},
			"docA-2": {ID: "docA-2", DocumentID: "docA", Active: true},
			"docA-3": {ID: "docA-3", DocumentID: "docA", Active: true},
			"docB-1": {ID: "docB-1", DocumentID: "docB", Active: true},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:         "hello",
		TopK:             4,
		DisableDiversify: true,
		Paths:            paths,
		Embedder:         fakeRAGEmbedder{},
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "docA-1", Score: 0.9},
			{ChunkID: "docA-2", Score: 0.8},
			{ChunkID: "docA-3", Score: 0.7},
			{ChunkID: "docB-1", Score: 0.6},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"docA-1", "docA-2", "docA-3", "docB-1"})
}

func TestRetrieveAppliesSourceFilterBeforeTrim(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"skip": {ID: "skip", Path: "internal/vendor/lib.go", Status: manifest.DocumentStatusIndexed},
			"keep": {ID: "keep", Path: "internal/rag/retrieve.go", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"skip-1": {ID: "skip-1", DocumentID: "skip", Active: true},
			"skip-2": {ID: "skip-2", DocumentID: "skip", Active: true},
			"keep-1": {ID: "keep-1", DocumentID: "keep", Active: true},
			"keep-2": {ID: "keep-2", DocumentID: "keep", Active: true},
		},
	}
	filter, err := ParseSourceFilter([]string{"internal/rag/**"}, nil, nil, "/repo")
	if err != nil {
		t.Fatal(err)
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:         "hello",
		TopK:             2,
		DisableDiversify: true,
		Paths:            paths,
		Embedder:         fakeRAGEmbedder{},
		SourceFilter:     filter,
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "skip-1", Score: 0.9},
			{ChunkID: "skip-2", Score: 0.8},
			{ChunkID: "keep-1", Score: 0.7},
			{ChunkID: "keep-2", Score: 0.6},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"keep-1", "keep-2"})
}

func TestRetrieveSourceFilterIncreasesDefaultCandidateK(t *testing.T) {
	filter, err := ParseSourceFilter([]string{"internal/**"}, nil, nil, "/repo")
	if err != nil {
		t.Fatal(err)
	}
	opts, err := NormalizeRetrieveOptions(RetrieveOptions{TopK: 3, SourceFilter: filter})
	if err != nil {
		t.Fatalf("NormalizeRetrieveOptions: %v", err)
	}
	if opts.CandidateK != 100 || opts.VectorCandidateK != 100 || opts.LexicalCandidateK != 100 {
		t.Fatalf("candidate defaults = %d/%d/%d, want 100 with source filter", opts.CandidateK, opts.VectorCandidateK, opts.LexicalCandidateK)
	}
	opts, err = NormalizeRetrieveOptions(RetrieveOptions{TopK: 3, CandidateK: 7, SourceFilter: filter})
	if err != nil {
		t.Fatalf("NormalizeRetrieveOptions explicit: %v", err)
	}
	if opts.CandidateK != 7 {
		t.Fatalf("explicit CandidateK = %d, want 7", opts.CandidateK)
	}
}

func TestRetrieveHybridOrderPreservedAfterOutOfOrderManifestLoad(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"a": {ID: "a", DocumentID: "doc", Active: true},
			"b": {ID: "b", DocumentID: "doc", Active: true},
			"c": {ID: "c", DocumentID: "doc", Active: true},
		},
		chunkOrder:  []string{"b", "c", "a"},
		lexicalHits: []manifest.LexicalHit{{ChunkID: "b", Score: -1}, {ChunkID: "a", Score: -2}},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:         "hello",
		TopK:             3,
		DisableDiversify: true,
		Paths:            paths,
		Mode:             RetrievalModeHybrid,
		Embedder:         fakeRAGEmbedder{},
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "a", Score: 0.9},
			{ChunkID: "c", Score: 0.8},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"a", "b", "c"})
}

func TestRetrieveRerankerNonePreservesCandidateOrder(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"a": {ID: "a", DocumentID: "doc", Active: true},
			"b": {ID: "b", DocumentID: "doc", Active: true},
			"c": {ID: "c", DocumentID: "doc", Active: true},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:         "hello",
		TopK:             3,
		DisableDiversify: true,
		RerankerMode:     RerankerModeNone,
		Paths:            paths,
		Embedder:         fakeRAGEmbedder{},
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "b", Score: 0.9},
			{ChunkID: "a", Score: 0.8},
			{ChunkID: "c", Score: 0.7},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"b", "a", "c"})
	if results[0].Reranker != "" {
		t.Fatalf("Reranker = %q, want empty for none", results[0].Reranker)
	}
}

func TestRetrieveHeuristicRerankerReordersAfterFiltering(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"bad-doc":  {ID: "bad-doc", Path: "bad.md", Status: manifest.DocumentStatusError},
			"good-doc": {ID: "good-doc", Path: "good.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"bad":       {ID: "bad", DocumentID: "bad-doc", ChunkText: "write lock acquired", Active: true},
			"good-low":  {ID: "good-low", DocumentID: "good-doc", ChunkText: "unrelated content", Active: true},
			"good-high": {ID: "good-high", DocumentID: "good-doc", ChunkText: "write lock acquired in repair", Active: true},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:         "write lock acquired",
		TopK:             2,
		DisableDiversify: true,
		RerankerMode:     RerankerModeHeuristic,
		Paths:            paths,
		Embedder:         fakeRAGEmbedder{},
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "bad", Score: 0.5},
			{ChunkID: "good-low", Score: 0.5},
			{ChunkID: "good-high", Score: 0.5},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"good-high", "good-low"})
	if results[0].Reranker != string(RerankerModeHeuristic) || results[0].RerankScore <= 0 {
		t.Fatalf("rerank metadata = %+v, want heuristic score", results[0])
	}
}

func TestRetrieveRerankHappensBeforeDiversify(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"docA": {ID: "docA", Path: "a.md", Status: manifest.DocumentStatusIndexed},
			"docB": {ID: "docB", Path: "b.md", Status: manifest.DocumentStatusIndexed},
			"docC": {ID: "docC", Path: "c.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"docA-low":  {ID: "docA-low", DocumentID: "docA", Active: true},
			"docA-high": {ID: "docA-high", DocumentID: "docA", Active: true},
			"docB-mid":  {ID: "docB-mid", DocumentID: "docB", Active: true},
			"docC-mid":  {ID: "docC-mid", DocumentID: "docC", Active: true},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:       "hello",
		TopK:           3,
		MaxPerDocument: 1,
		Paths:          paths,
		Embedder:       fakeRAGEmbedder{},
		Reranker:       orderReranker{order: []string{"docA-high", "docB-mid", "docC-mid", "docA-low"}},
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "docA-low", Score: 0.9},
			{ChunkID: "docA-high", Score: 0.8},
			{ChunkID: "docB-mid", Score: 0.7},
			{ChunkID: "docC-mid", Score: 0.6},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"docA-high", "docB-mid", "docC-mid"})
}

func TestRetrieveCustomRerankerCanBeInjected(t *testing.T) {
	ctx := context.Background()
	paths := testPaths(t, true)
	store := &fakeManifest{
		docs: map[string]*manifest.Document{
			"doc": {ID: "doc", Path: "README.md", Status: manifest.DocumentStatusIndexed},
		},
		chunks: map[string]manifest.Chunk{
			"a": {ID: "a", DocumentID: "doc", Active: true},
			"b": {ID: "b", DocumentID: "doc", Active: true},
		},
	}
	results, err := Retrieve(ctx, store, RetrieveOptions{
		Question:         "hello",
		TopK:             2,
		DisableDiversify: true,
		Paths:            paths,
		Embedder:         fakeRAGEmbedder{},
		Reranker:         orderReranker{order: []string{"b", "a"}},
		Vector: fakeVector{hits: []vectorstore.Hit{
			{ChunkID: "a", Score: 0.9},
			{ChunkID: "b", Score: 0.8},
		}},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	assertChunkIDs(t, results, []string{"b", "a"})
	if results[0].Reranker != "custom" {
		t.Fatalf("Reranker = %q, want custom", results[0].Reranker)
	}
}

func TestRetrieveRejectsInvalidRerankerMode(t *testing.T) {
	_, err := Retrieve(context.Background(), &fakeManifest{}, RetrieveOptions{
		RerankerMode: RerankerMode("bad"),
	})
	if err == nil || !strings.Contains(err.Error(), "--reranker must be none or heuristic") {
		t.Fatalf("Retrieve error = %v, want invalid reranker mode", err)
	}
}

func TestRetrieveRejectsInvalidMode(t *testing.T) {
	_, err := Retrieve(context.Background(), &fakeManifest{}, RetrieveOptions{
		Mode: RetrievalMode("bad"),
	})
	if err == nil || !strings.Contains(err.Error(), "--retrieval must be vector, lexical, or hybrid") {
		t.Fatalf("Retrieve error = %v, want invalid mode", err)
	}
}

func TestBuildFTSQueryHandlesCodeSymbolsAndFlags(t *testing.T) {
	for _, input := range []string{"ErrPendingRunDetected", "--state-dir", "FALKENGO_EMBEDDING_MODEL", "internal/rag/retrieve.go", "what does c.active = 1 mean?"} {
		if got := BuildFTSQuery(input); got == "" {
			t.Fatalf("BuildFTSQuery(%q) returned empty", input)
		}
	}
}

type fakeRAGEmbedder struct{}

func (fakeRAGEmbedder) EmbedText(context.Context, string) (llm.Embedding, error) {
	return llm.Embedding{Model: "fake", Vector: []float32{1, 0}}, nil
}

type fakeVector struct {
	hits      []vectorstore.Hit
	wantLimit int
}

func (f fakeVector) Close() error                                             { return nil }
func (f fakeVector) Insert(context.Context, []float32, string) (int64, error) { return 0, nil }
func (f fakeVector) Search(_ context.Context, _ []float32, limit int) ([]vectorstore.Hit, error) {
	if f.wantLimit != 0 && limit != f.wantLimit {
		return nil, fmt.Errorf("limit = %d, want %d", limit, f.wantLimit)
	}
	return f.hits, nil
}
func (f fakeVector) Commit(context.Context) error { return nil }

type fakeManifest struct {
	manifest.Store
	docs           map[string]*manifest.Document
	chunks         map[string]manifest.Chunk
	chunkOrder     []string
	lexicalHits    []manifest.LexicalHit
	lexicalByQuery map[string][]manifest.LexicalHit
}

func (f *fakeManifest) GetDocumentByID(_ context.Context, id string) (*manifest.Document, error) {
	doc, ok := f.docs[id]
	if !ok {
		return nil, manifest.ErrNotFound
	}
	return doc, nil
}

func (f *fakeManifest) GetActiveChunksByIDs(_ context.Context, ids []string) ([]manifest.Chunk, error) {
	var chunks []manifest.Chunk
	if len(f.chunkOrder) != 0 {
		requested := make(map[string]struct{}, len(ids))
		for _, id := range ids {
			requested[id] = struct{}{}
		}
		for _, id := range f.chunkOrder {
			if _, ok := requested[id]; !ok {
				continue
			}
			chunk, ok := f.chunks[id]
			if ok && chunk.Active {
				chunks = append(chunks, chunk)
			}
		}
		return chunks, nil
	}
	for _, id := range ids {
		chunk, ok := f.chunks[id]
		if ok && chunk.Active {
			chunks = append(chunks, chunk)
		}
	}
	return chunks, nil
}

func (f *fakeManifest) SearchLexicalChunks(_ context.Context, query string, _ int) ([]manifest.LexicalHit, error) {
	if f.lexicalByQuery != nil {
		return f.lexicalByQuery[query], nil
	}
	return f.lexicalHits, nil
}

func (f *fakeManifest) RebuildLexicalIndex(context.Context) error {
	return nil
}

func (f *fakeManifest) EnsureLexicalIndex(context.Context) (bool, error) {
	return false, nil
}

type orderReranker struct {
	order []string
}

func (r orderReranker) Rerank(_ context.Context, _ string, candidates []RerankCandidate, _ int) ([]RerankResult, error) {
	scoreByID := make(map[string]float64, len(r.order))
	for i, id := range r.order {
		scoreByID[id] = float64(len(r.order) - i)
	}
	results := make([]RerankResult, 0, len(candidates))
	for _, id := range r.order {
		for _, candidate := range candidates {
			if candidate.ChunkID == id {
				results = append(results, RerankResult{ChunkID: id, Score: scoreByID[id]})
				break
			}
		}
	}
	return results, nil
}

type staticQueryPlanner struct {
	queries []string
	err     error
}

func (p staticQueryPlanner) Plan(_ context.Context, question string, _ QueryPlanOptions) (QueryPlan, error) {
	if p.err != nil {
		return QueryPlan{}, p.err
	}
	return QueryPlan{
		OriginalQuestion: question,
		Queries:          append([]string(nil), p.queries...),
		Mode:             string(QueryPlannerModeHeuristic),
	}, nil
}

type recordingReranker struct {
	order    []string
	question string
}

func (r *recordingReranker) Rerank(_ context.Context, question string, candidates []RerankCandidate, _ int) ([]RerankResult, error) {
	r.question = question
	scoreByID := make(map[string]float64, len(r.order))
	for i, id := range r.order {
		scoreByID[id] = float64(len(r.order) - i)
	}
	results := make([]RerankResult, 0, len(candidates))
	for _, id := range r.order {
		for _, candidate := range candidates {
			if candidate.ChunkID == id {
				results = append(results, RerankResult{ChunkID: id, Score: scoreByID[id]})
				break
			}
		}
	}
	return results, nil
}

func touchDir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
}

func testPaths(t *testing.T, withVecgo bool) config.Paths {
	t.Helper()
	tmp := t.TempDir()
	paths, err := config.ResolvePaths(filepath.Join(tmp, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatalf("EnsureStateDirs: %v", err)
	}
	if err := os.WriteFile(paths.ManifestPath, []byte("test"), 0o644); err != nil {
		t.Fatalf("write manifest sentinel: %v", err)
	}
	if withVecgo {
		touchDir(t, paths.VecgoPath)
	}
	return paths
}

func sameSources(got []string, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func assertChunkIDs(t *testing.T, results []RetrievedChunk, want []string) {
	t.Helper()
	if len(results) != len(want) {
		t.Fatalf("result count = %d, want %d; results = %+v", len(results), len(want), results)
	}
	for i, result := range results {
		if result.Chunk.ID != want[i] {
			t.Fatalf("result[%d] chunk = %q, want %q; results = %+v", i, result.Chunk.ID, want[i], results)
		}
	}
}
