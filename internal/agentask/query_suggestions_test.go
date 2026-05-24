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

func TestSuggestFollowupQueriesRejectsTranscriptFragments(t *testing.T) {
	queries := SuggestFollowupQueries("AlphaFold", []rag.SourceChunk{{
		Path: "alphafold/meetings/sdb.md",
		Text: `So these two folders here which are shared completed and in progress.
For example amaranthus is species. Outputs include FASTA, A3M, PDB, error JSON and NCBI tax IDs.`,
	}}, 3)

	got := strings.ToLower(strings.Join(queries, "\n"))
	for _, bad := range []string{
		"these two folders here",
		"for example",
		"amaranthus",
		"species i",
	} {
		if strings.Contains(got, bad) {
			t.Fatalf("queries = %+v, leaked transcript fragment %q", queries, bad)
		}
	}
	for _, want := range []string{"a3m", "pdb", "json", "ncbi"} {
		if !strings.Contains(got, want) {
			t.Fatalf("queries = %+v, want %q", queries, want)
		}
	}
}

func TestSuggestFollowupQueriesRejectsNumericOCRNoise(t *testing.T) {
	queries := SuggestFollowupQueries("protein folding", []rag.SourceChunk{{
		Path: "alphafold/meetings/sdb.md",
		Text: `The notes mention DVI 000 150 and PDB output files. Error JSON and A3M are important.`,
	}}, 3)

	got := strings.Join(queries, "\n")
	if strings.Contains(got, "DVI") || strings.Contains(got, "000") || strings.Contains(got, "150") {
		t.Fatalf("queries = %+v, leaked numeric/OCR noise", queries)
	}
	for _, want := range []string{"PDB", "JSON", "A3M"} {
		if !strings.Contains(got, want) {
			t.Fatalf("queries = %+v, want %q", queries, want)
		}
	}
}

func TestSuggestFollowupQueriesRejectsMetadataTranscriptFragment(t *testing.T) {
	queries := SuggestFollowupQueries("protein folding", []rag.SourceChunk{{
		Path: "alphafold/meetings/sdb.md",
		Text: `We haven't got any metadata what was used to run that.
PDB and error JSON are available for the folded structures.`,
	}}, 3)

	got := strings.ToLower(strings.Join(queries, "\n"))
	if strings.Contains(got, "got any metadata what was used run") ||
		strings.Contains(got, "got any metadata") ||
		strings.Contains(got, "what was used") {
		t.Fatalf("queries = %+v, leaked transcript-style metadata fragment", queries)
	}
	if !strings.Contains(got, "metadata") {
		t.Fatalf("queries = %+v, want metadata preserved", queries)
	}
	if !strings.Contains(got, "pdb") || !strings.Contains(got, "json") {
		t.Fatalf("queries = %+v, want technical terms preserved", queries)
	}
}

func TestCleanExpansionGroupKeepsKnownTermsAndDropsOCRNoise(t *testing.T) {
	cleaned := cleanExpansionGroup([]string{"DVI", "000", "150", "PDB", "MMseqs"})
	got := strings.Join(cleaned, " ")
	for _, bad := range []string{"DVI", "000", "150"} {
		if strings.Contains(got, bad) {
			t.Fatalf("cleaned = %+v, leaked %q", cleaned, bad)
		}
	}
	for _, want := range []string{"PDB", "MMseqs"} {
		if !strings.Contains(got, want) {
			t.Fatalf("cleaned = %+v, want %q", cleaned, want)
		}
	}
}

func TestCleanBroadExpansionGroupDropsTranscriptFillerAroundMetadata(t *testing.T) {
	cleaned := cleanBroadExpansionGroup([]string{"got any metadata what was used run"})
	got := strings.Join(cleaned, " ")
	if got != "metadata" {
		t.Fatalf("cleaned = %+v, want only metadata", cleaned)
	}
}

func TestTooSimilarExpansionQueryRejectsTokenReorder(t *testing.T) {
	existing := []string{"AlphaFold A3M PDB JSON NCBI"}
	if !tooSimilarExpansionQuery("AlphaFold NCBI A3M JSON", existing, 0.75) {
		t.Fatal("tooSimilarExpansionQuery = false, want reordered near duplicate rejected")
	}
	if tooSimilarExpansionQuery("AlphaFold metadata method version", existing, 0.75) {
		t.Fatal("metadata query was treated as too similar to artifact query")
	}
}

func TestSelectExpansionQueriesPrefersDiverseCleanGroups(t *testing.T) {
	queries := selectExpansionQueries([]expansionQueryCandidate{
		{Query: "AlphaFold A3M PDB JSON NCBI", Category: CategoryArtifact},
		{Query: "AlphaFold NCBI A3M JSON", Category: CategoryArtifact},
		{Query: "AlphaFold metadata method version", Category: CategoryMetadata},
		{Query: "AlphaFold monomer dimer ligand", Category: CategoryModality},
	}, 3)
	got := strings.Join(queries, "\n")
	for _, want := range []string{
		"AlphaFold A3M PDB JSON NCBI",
		"AlphaFold metadata method version",
		"AlphaFold monomer dimer ligand",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("queries = %+v, want %q", queries, want)
		}
	}
	if strings.Contains(got, "AlphaFold NCBI A3M JSON") {
		t.Fatalf("queries = %+v, near-duplicate artifact query should not fill budget", queries)
	}
}

func TestUsefulExpansionQueryQualityGate(t *testing.T) {
	accepted := []string{
		"AlphaFold A3M PDB JSON",
		"protein folding PDB error JSON",
	}
	for _, query := range accepted {
		if !usefulExpansionQuery(query, "AlphaFold") {
			t.Fatalf("usefulExpansionQuery(%q) = false, want true", query)
		}
	}

	rejected := []string{
		"AlphaFold s these two folders here which are shared completed in progress Don several things",
		"protein folding s going to it probably about 10 species",
		"AlphaFold folders species amaranthus several project meeting notes unclear transcript ordinary tokens",
	}
	for _, query := range rejected {
		if usefulExpansionQuery(query, "AlphaFold") {
			t.Fatalf("usefulExpansionQuery(%q) = true, want false", query)
		}
	}
}

func TestSuggestAgentFollowupHintsAllowsUsefulNonTechnicalPhrases(t *testing.T) {
	queries := SuggestAgentFollowupHints("deployment config", []rag.SourceChunk{{
		Path: "ops/notes.md",
		Text: `The rollout notes mention retry policy, timeout settings, cache invalidation, and header handling.`,
	}}, 2)
	got := strings.ToLower(strings.Join(queries, "\n"))
	if !strings.Contains(got, "retry") || !strings.Contains(got, "timeout") {
		t.Fatalf("queries = %+v, want useful ordinary hints", queries)
	}
}

func TestSuggestFollowupQueriesEmptyWhenNoSourceTerms(t *testing.T) {
	if got := SuggestFollowupQueries("hello", nil, 3); len(got) != 0 {
		t.Fatalf("queries = %+v, want none", got)
	}
}
