package agentask

import (
	"reflect"
	"testing"

	"github.com/smasonuk/falken-vector/internal/rag"
)

func TestThinCitedSourcesDeduplicatesOverlappingProjectedWindows(t *testing.T) {
	sources := []rag.SourceChunk{
		{SourceNumber: 1, Path: "alphafold/meetings/sdb.md", StartLine: 11, EndLine: 11},
		{SourceNumber: 2, Path: "alphafold/meetings/sdb.md", StartLine: 13, EndLine: 13},
	}

	got := thinCitedSources("See [source 1] and [source 2].", sources, 2, 20, nil)
	want := []int{1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("thinCitedSources = %+v, want %+v", got, want)
	}
}

func TestThinCitedSourcesKeepsNonOverlappingProjectedWindows(t *testing.T) {
	sources := []rag.SourceChunk{
		{SourceNumber: 1, Path: "alphafold/meetings/sdb.md", StartLine: 1, EndLine: 2},
		{SourceNumber: 2, Path: "alphafold/meetings/sdb.md", StartLine: 100, EndLine: 101},
	}

	got := thinCitedSources("See [source 1] and [source 2].", sources, 2, 20, nil)
	want := []int{1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("thinCitedSources = %+v, want %+v", got, want)
	}
}
