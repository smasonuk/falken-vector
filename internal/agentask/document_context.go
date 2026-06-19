package agentask

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/smasonuk/falken-vector/internal/rag"
)

const ReadIndexDocumentToolName = "read_index_document"

type Range struct {
	StartLine int `json:"start_line"`
	EndLine   int `json:"end_line"`
}

type DocumentMatchSummary struct {
	Path                  string   `json:"path"`
	SourceNumbers         []int    `json:"source_numbers"`
	HitCount              int      `json:"hit_count"`
	NewHitCount           int      `json:"new_hit_count"`
	TopKShare             float64  `json:"top_k_share"`
	MinStartLine          int      `json:"min_start_line"`
	MaxEndLine            int      `json:"max_end_line"`
	RetrievedLineRanges   []Range  `json:"retrieved_line_ranges"`
	MergedRetrievedRanges []Range  `json:"merged_retrieved_line_ranges"`
	CoveredLineCount      int      `json:"covered_line_count"`
	FileLineCount         int      `json:"file_line_count"`
	FileByteCount         int64    `json:"file_byte_count"`
	EstimatedFileTokens   int      `json:"estimated_file_tokens"`
	ScatteredHitCount     int      `json:"scattered_hit_count"`
	FirstRank             int      `json:"first_rank"`
	BestRank              int      `json:"best_rank"`
	Queries               []string `json:"queries"`
}

type DocumentPromotionOptions struct {
	Enabled                 bool
	MaxDocumentReadLines    int
	MaxDocumentReadTokens   int
	MaxDocumentReads        int
	SmallFileLineLimit      int
	SmallFileTokenLimit     int
	DominantDocShare        float64
	MinHitsForDominantDoc   int
	MinHitsForSmallFile     int
	MaxTotalDocumentTokens  int
	ScatteredHitThreshold   int
	AllowOverlappingSources bool
}

type DocumentPromotion struct {
	SourceNumber int    `json:"source_number"`
	Path         string `json:"path"`
	Mode         string `json:"mode"`
	StartLine    int    `json:"start_line,omitempty"`
	EndLine      int    `json:"end_line,omitempty"`
	Reason       string `json:"reason"`
}

type DocumentPromotionDecision struct {
	SourceNumber    int    `json:"source_number,omitempty"`
	Path            string `json:"path"`
	Status          string `json:"status"`
	Reason          string `json:"reason"`
	StartLine       int    `json:"start_line,omitempty"`
	EndLine         int    `json:"end_line,omitempty"`
	Lines           int    `json:"lines,omitempty"`
	EstimatedTokens int    `json:"estimated_tokens,omitempty"`
	MaxLines        int    `json:"max_lines,omitempty"`
	MaxTokens       int    `json:"max_tokens,omitempty"`
}

const (
	DefaultMaxDocumentReadLines    = 500
	DefaultMaxDocumentReadTokens   = 25000
	DefaultMaxDocumentReads        = 2
	DefaultSmallFileLineLimit      = 400
	DefaultSmallFileTokenLimit     = 25000
	DefaultDominantDocShare        = 0.45
	DefaultMinHitsForDominantDoc   = 4
	DefaultMinHitsForSmallFile     = 2
	DefaultMaxTotalDocumentTokens  = 40000
	DefaultScatteredHitThreshold   = 3
)

func defaultDocumentPromotionOptions() DocumentPromotionOptions {
	return DocumentPromotionOptions{
		Enabled:                true,
		MaxDocumentReadLines:   DefaultMaxDocumentReadLines,
		MaxDocumentReadTokens:  DefaultMaxDocumentReadTokens,
		MaxDocumentReads:       DefaultMaxDocumentReads,
		SmallFileLineLimit:     DefaultSmallFileLineLimit,
		SmallFileTokenLimit:    DefaultSmallFileTokenLimit,
		DominantDocShare:       DefaultDominantDocShare,
		MinHitsForDominantDoc:  DefaultMinHitsForDominantDoc,
		MinHitsForSmallFile:    DefaultMinHitsForSmallFile,
		MaxTotalDocumentTokens: DefaultMaxTotalDocumentTokens,
		ScatteredHitThreshold:  DefaultScatteredHitThreshold,
	}
}

func normalizeDocumentPromotionOptions(opts DocumentPromotionOptions) DocumentPromotionOptions {
	if opts.MaxDocumentReadLines <= 0 {
		opts.MaxDocumentReadLines = DefaultMaxDocumentReadLines
	}
	if opts.MaxDocumentReadTokens <= 0 {
		opts.MaxDocumentReadTokens = DefaultMaxDocumentReadTokens
	}
	if opts.MaxDocumentReads <= 0 {
		opts.MaxDocumentReads = DefaultMaxDocumentReads
	}
	if opts.SmallFileLineLimit <= 0 {
		opts.SmallFileLineLimit = DefaultSmallFileLineLimit
	}
	if opts.SmallFileTokenLimit <= 0 {
		opts.SmallFileTokenLimit = DefaultSmallFileTokenLimit
	}
	if opts.DominantDocShare <= 0 {
		opts.DominantDocShare = DefaultDominantDocShare
	}
	if opts.MinHitsForDominantDoc <= 0 {
		opts.MinHitsForDominantDoc = DefaultMinHitsForDominantDoc
	}
	if opts.MinHitsForSmallFile <= 0 {
		opts.MinHitsForSmallFile = DefaultMinHitsForSmallFile
	}
	if opts.MaxTotalDocumentTokens <= 0 {
		opts.MaxTotalDocumentTokens = DefaultMaxTotalDocumentTokens
	}
	if opts.ScatteredHitThreshold <= 0 {
		opts.ScatteredHitThreshold = DefaultScatteredHitThreshold
	}
	return opts
}

func BuildDocumentMatchSummaries(sources []rag.SourceChunk, beforeSourceCount int, topK int) []DocumentMatchSummary {
	type docAccumulator struct {
		summary     DocumentMatchSummary
		queries     map[string]struct{}
		seenSources map[int]struct{}
	}
	docs := map[string]*docAccumulator{}
	for i, source := range sources {
		if strings.TrimSpace(source.Path) == "" {
			continue
		}
		displayPath := rag.DisplayPath(source.Path, source.SourceRoot)
		acc := docs[source.Path]
		if acc == nil {
			acc = &docAccumulator{
				summary: DocumentMatchSummary{
					Path:      displayPath,
					FirstRank: i + 1,
					BestRank:  i + 1,
				},
				queries:     map[string]struct{}{},
				seenSources: map[int]struct{}{},
			}
			docs[source.Path] = acc
		}
		acc.summary.HitCount++
		if source.SourceNumber > beforeSourceCount {
			acc.summary.NewHitCount++
		}
		if source.SourceNumber > 0 {
			if _, seen := acc.seenSources[source.SourceNumber]; !seen {
				acc.summary.SourceNumbers = append(acc.summary.SourceNumbers, source.SourceNumber)
				acc.seenSources[source.SourceNumber] = struct{}{}
			}
		}
		if source.StartLine > 0 {
			if acc.summary.MinStartLine == 0 || source.StartLine < acc.summary.MinStartLine {
				acc.summary.MinStartLine = source.StartLine
			}
			endLine := source.EndLine
			if endLine < source.StartLine {
				endLine = source.StartLine
			}
			if endLine > acc.summary.MaxEndLine {
				acc.summary.MaxEndLine = endLine
			}
			acc.summary.RetrievedLineRanges = append(acc.summary.RetrievedLineRanges, Range{StartLine: source.StartLine, EndLine: endLine})
		}
		rank := i + 1
		if acc.summary.FirstRank == 0 {
			acc.summary.FirstRank = rank
		}
		if acc.summary.BestRank == 0 || rank < acc.summary.BestRank {
			acc.summary.BestRank = rank
		}
		if source.Provenance != nil && strings.TrimSpace(source.Provenance.Query) != "" {
			acc.queries[source.Provenance.Query] = struct{}{}
		}
	}

	out := make([]DocumentMatchSummary, 0, len(docs))
	for path, acc := range docs {
		summary := acc.summary
		sort.Ints(summary.SourceNumbers)
		summary.MergedRetrievedRanges = mergeRanges(summary.RetrievedLineRanges)
		for _, r := range summary.MergedRetrievedRanges {
			summary.CoveredLineCount += rangeLen(r)
		}
		summary.ScatteredHitCount = len(summary.MergedRetrievedRanges)
		if topK <= 0 {
			topK = len(sources)
		}
		if topK > 0 {
			summary.TopKShare = float64(summary.HitCount) / float64(topK)
		}
		if data, err := os.ReadFile(path); err == nil {
			stats := documentTextStats(data)
			summary.FileLineCount = stats.LineCount
			summary.FileByteCount = stats.ByteCount
			summary.EstimatedFileTokens = stats.EstimatedTokens
		}
		for query := range acc.queries {
			summary.Queries = append(summary.Queries, query)
		}
		sort.Strings(summary.Queries)
		out = append(out, summary)
	}
	sortDocumentMatchSummaries(out)
	return out
}

func sortDocumentMatchSummaries(values []DocumentMatchSummary) {
	sort.SliceStable(values, func(i, j int) bool {
		if values[i].HitCount != values[j].HitCount {
			return values[i].HitCount > values[j].HitCount
		}
		leftRank := values[i].BestRank
		rightRank := values[j].BestRank
		if leftRank == 0 {
			leftRank = int(^uint(0) >> 1)
		}
		if rightRank == 0 {
			rightRank = int(^uint(0) >> 1)
		}
		if leftRank != rightRank {
			return leftRank < rightRank
		}
		if values[i].FileByteCount != values[j].FileByteCount {
			return values[i].FileByteCount < values[j].FileByteCount
		}
		return values[i].Path < values[j].Path
	})
}

func SelectDocumentPromotions(question string, matches []DocumentMatchSummary, opts DocumentPromotionOptions) []DocumentPromotion {
	opts = normalizeDocumentPromotionOptions(opts)
	if !opts.Enabled {
		return nil
	}
	broad := isBroadCoverageQuestion(question)
	narrow := isNarrowLookupQuestion(question)
	remainingReads := opts.MaxDocumentReads
	remainingTokens := opts.MaxTotalDocumentTokens
	var promotions []DocumentPromotion
	for _, match := range matches {
		if remainingReads <= 0 {
			break
		}
		if len(match.SourceNumbers) == 0 || unsupportedDocumentExtension(match.Path) {
			continue
		}
		if match.FileLineCount <= 0 || match.EstimatedFileTokens <= 0 {
			continue
		}
		if match.FileLineCount > opts.MaxDocumentReadLines || match.EstimatedFileTokens > opts.MaxDocumentReadTokens || match.EstimatedFileTokens > remainingTokens {
			continue
		}
		reason := ""
		small := match.FileLineCount <= opts.SmallFileLineLimit && match.EstimatedFileTokens <= opts.SmallFileTokenLimit
		switch {
		case small && match.HitCount >= opts.MinHitsForSmallFile && !narrow:
			reason = fmt.Sprintf("small file; %d hits; file %d lines <= %d-line cap", match.HitCount, match.FileLineCount, opts.SmallFileLineLimit)
		case broad && match.HitCount >= opts.MinHitsForSmallFile && match.TopKShare >= opts.DominantDocShare:
			reason = fmt.Sprintf("broad question; %d sources from same document; share %.2f >= %.2f", match.HitCount, match.TopKShare, opts.DominantDocShare)
		case broad && match.HitCount >= opts.MinHitsForDominantDoc:
			reason = fmt.Sprintf("broad question; %d hits >= dominant hit threshold %d", match.HitCount, opts.MinHitsForDominantDoc)
		case match.ScatteredHitCount >= opts.ScatteredHitThreshold && small && !narrow:
			reason = fmt.Sprintf("scattered hits; %d non-adjacent ranges; file within budget", match.ScatteredHitCount)
		default:
			continue
		}
		promotions = append(promotions, DocumentPromotion{
			SourceNumber: match.SourceNumbers[0],
			Path:         match.Path,
			Mode:         "whole",
			Reason:       reason,
		})
		remainingReads--
		remainingTokens -= match.EstimatedFileTokens
	}
	return promotions
}

func documentTextStats(data []byte) documentStats {
	text := string(data)
	return documentStats{
		LineCount:       countDocumentLines(text),
		ByteCount:       int64(len(data)),
		EstimatedTokens: estimateDocumentTokens(text),
	}
}

type documentStats struct {
	LineCount       int
	ByteCount       int64
	EstimatedTokens int
}

func countDocumentLines(text string) int {
	if text == "" {
		return 0
	}
	lines := strings.Count(text, "\n")
	if !strings.HasSuffix(text, "\n") {
		lines++
	}
	return lines
}

func estimateDocumentTokens(text string) int {
	if text == "" {
		return 0
	}
	tokens := len(text) / 4
	if tokens == 0 {
		return 1
	}
	return tokens
}

func mergeRanges(ranges []Range) []Range {
	if len(ranges) == 0 {
		return nil
	}
	sorted := append([]Range(nil), ranges...)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].StartLine == sorted[j].StartLine {
			return sorted[i].EndLine < sorted[j].EndLine
		}
		return sorted[i].StartLine < sorted[j].StartLine
	})
	merged := []Range{sorted[0]}
	for _, r := range sorted[1:] {
		last := &merged[len(merged)-1]
		if r.StartLine <= last.EndLine+1 {
			if r.EndLine > last.EndLine {
				last.EndLine = r.EndLine
			}
			continue
		}
		merged = append(merged, r)
	}
	return merged
}

func rangeLen(r Range) int {
	if r.EndLine < r.StartLine {
		return 0
	}
	return r.EndLine - r.StartLine + 1
}

func isNarrowLookupQuestion(question string) bool {
	normalized := strings.ToLower(strings.Join(strings.Fields(question), " "))
	if normalized == "" {
		return false
	}
	phrases := []string{"where is ", "where are ", "implemented", "definition", "function", "class", "exact", "line "}
	for _, phrase := range phrases {
		if strings.Contains(normalized, phrase) {
			return true
		}
	}
	return false
}

func unsupportedDocumentExtension(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz", ".tar", ".bin", ".exe", ".dylib", ".so", ".sqlite", ".db":
		return true
	default:
		return false
	}
}
