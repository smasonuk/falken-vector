package agentask

import "testing"

func TestSelectDocumentPromotionsBroadSmallDominant(t *testing.T) {
	opts := defaultDocumentPromotionOptions()
	matches := []DocumentMatchSummary{{
		Path:                "meetings/sdb.md",
		SourceNumbers:       []int{12, 13},
		HitCount:            6,
		TopKShare:           0.6,
		FileLineCount:       206,
		EstimatedFileTokens: 5200,
	}}
	promotions := SelectDocumentPromotions("summarize anything related to AlphaFold", matches, opts)
	if len(promotions) != 1 || promotions[0].SourceNumber != 12 || promotions[0].Mode != "whole" {
		t.Fatalf("promotions = %+v, want one whole-document promotion", promotions)
	}
}

func TestSelectDocumentPromotionsBroadLargeDominantSkipped(t *testing.T) {
	opts := defaultDocumentPromotionOptions()
	matches := []DocumentMatchSummary{{
		Path:                "big.md",
		SourceNumbers:       []int{1, 2, 3, 4},
		HitCount:            9,
		TopKShare:           0.75,
		FileLineCount:       4200,
		EstimatedFileTokens: 97000,
	}}
	if promotions := SelectDocumentPromotions("summarize anything related to AlphaFold", matches, opts); len(promotions) != 0 {
		t.Fatalf("promotions = %+v, want large document skipped", promotions)
	}
}

func TestSelectDocumentPromotionsNarrowSmallDocumentSkipped(t *testing.T) {
	opts := defaultDocumentPromotionOptions()
	matches := []DocumentMatchSummary{{
		Path:                "internal/agentask/read_source_tool.go",
		SourceNumbers:       []int{1, 2},
		HitCount:            2,
		TopKShare:           1,
		FileLineCount:       120,
		EstimatedFileTokens: 6000,
	}}
	if promotions := SelectDocumentPromotions("Where is read_index_source implemented?", matches, opts); len(promotions) != 0 {
		t.Fatalf("promotions = %+v, want narrow lookup skipped", promotions)
	}
}

func TestSelectDocumentPromotionsScatteredHits(t *testing.T) {
	opts := defaultDocumentPromotionOptions()
	matches := []DocumentMatchSummary{{
		Path:                "notes.md",
		SourceNumbers:       []int{1, 2, 3},
		HitCount:            3,
		TopKShare:           0.3,
		FileLineCount:       100,
		EstimatedFileTokens: 4000,
		ScatteredHitCount:   3,
	}}
	promotions := SelectDocumentPromotions("find AlphaFold details", matches, opts)
	if len(promotions) != 1 {
		t.Fatalf("promotions = %+v, want scattered small document promoted", promotions)
	}
}

func TestSelectDocumentPromotionsRespectsReadCap(t *testing.T) {
	opts := defaultDocumentPromotionOptions()
	opts.MaxDocumentReads = 1
	matches := []DocumentMatchSummary{
		{Path: "one.md", SourceNumbers: []int{1}, HitCount: 4, TopKShare: 0.5, FileLineCount: 10, EstimatedFileTokens: 100},
		{Path: "two.md", SourceNumbers: []int{2}, HitCount: 4, TopKShare: 0.5, FileLineCount: 10, EstimatedFileTokens: 100},
	}
	promotions := SelectDocumentPromotions("summarize anything related", matches, opts)
	if len(promotions) != 1 {
		t.Fatalf("promotions = %+v, want cap to one promotion", promotions)
	}
}
