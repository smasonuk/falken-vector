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

func TestSuggestFollowupQueriesIgnoresAbsolutePathNoise(t *testing.T) {
	queries := SuggestFollowupQueries("summarize alphafold folding", []rag.SourceChunk{{
		Path: "/Volumes/mac_mini_ex/code/falken2/test_vec/source/alphafold/meetings/sdb - May 18 2026.md",
		Text: `AlphaFold outputs include FASTA, A3M, PDB, and error JSON.
UniProt metadata tracks monomer, dimer, and ligand workflow notes.`,
	}}, 3)

	got := strings.ToLower(strings.Join(queries, "\n"))
	for _, bad := range []string{
		"volumes", "mac mini", "falken2", "test vec", "source alphafold", "meetings sdb may",
	} {
		if strings.Contains(got, bad) {
			t.Fatalf("queries = %+v, leaked path noise %q", queries, bad)
		}
	}
	for _, want := range []string{"a3m", "pdb", "uniprot", "monomer", "ligand"} {
		if !strings.Contains(got, want) {
			t.Fatalf("queries = %+v, want %q", queries, want)
		}
	}
}

func TestSuggestFollowupQueriesAnchorsTermsToOriginalTopic(t *testing.T) {
	queries := SuggestFollowupQueries("AlphaFold folding", []rag.SourceChunk{{
		Path: "alphafold/meetings/sdb.md",
		Text: `A3M, PDB, error JSON, UniProt, monomer, dimer, and ligand are mentioned.`,
	}}, 3)
	if len(queries) == 0 {
		t.Fatal("queries empty, want anchored suggestions")
	}
	for _, query := range queries {
		if !strings.Contains(strings.ToLower(query), "alphafold") || !strings.Contains(strings.ToLower(query), "folding") {
			t.Fatalf("query = %q, want original topic anchor", query)
		}
	}
}

func TestSuggestFollowupQueriesEmptyWhenNoSourceTerms(t *testing.T) {
	if got := SuggestFollowupQueries("hello", nil, 3); len(got) != 0 {
		t.Fatalf("queries = %+v, want none", got)
	}
}
