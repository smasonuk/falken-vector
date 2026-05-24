package agentask

import (
	"strings"
	"testing"

	"github.com/smasonuk/falken-vector/internal/rag"
)

func TestSuggestFollowupQueriesUsesRetrievedSourceTerms(t *testing.T) {
	queries := SuggestFollowupQueries("summarize alphafold folding", []rag.SourceChunk{{
		Path: "alphafold/meetings/sdb.md",
		Text: `AlphaFold outputs include FASTA, A3M, PDB, and error JSON.
UniProt metadata tracks monomer, dimer, and ligand workflow notes.`,
	}}, 3)
	got := strings.Join(queries, "\n")
	for _, want := range []string{"alphafold", "A3M", "PDB", "JSON", "UniProt"} {
		if !strings.Contains(got, want) {
			t.Fatalf("queries = %+v, want term %q", queries, want)
		}
	}
	if len(queries) > 3 {
		t.Fatalf("queries = %+v, want cap at 3", queries)
	}
}

func TestSuggestFollowupQueriesEmptyWhenNoSourceTerms(t *testing.T) {
	if got := SuggestFollowupQueries("hello", nil, 3); len(got) != 0 {
		t.Fatalf("queries = %+v, want none", got)
	}
}
