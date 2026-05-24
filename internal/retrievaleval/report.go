package retrievaleval

import (
	"encoding/json"
	"fmt"
	"io"
)

func WriteText(w io.Writer, datasetPath string, summary Summary, details bool) error {
	if w == nil {
		return nil
	}
	if _, err := fmt.Fprintln(w, "Retrieval evaluation"); err != nil {
		return err
	}
	if _, err := fmt.Fprintln(w); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "dataset: %s\n", datasetPath); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "top-k: %d\n", summary.TopK); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "cases: %d\n", summary.Cases); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "evaluated: %d\n", summary.EvaluatedCases); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "failed: %d\n\n", summary.FailedCases); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "hit@%d:       %.4f\n", summary.TopK, summary.HitAtK); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "recall@%d:    %.4f\n", summary.TopK, summary.RecallAtK); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "precision@%d: %.4f\n", summary.TopK, summary.PrecisionAtK); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "mrr@%d:       %.4f\n", summary.TopK, summary.MRRAtK); err != nil {
		return err
	}
	if details {
		for _, result := range summary.CaseResults {
			if err := writeCaseDetails(w, result); err != nil {
				return err
			}
		}
	}
	return nil
}

func WriteJSON(w io.Writer, summary Summary) error {
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return encoder.Encode(summary)
}

func writeCaseDetails(w io.Writer, result CaseResult) error {
	if _, err := fmt.Fprintln(w); err != nil {
		return err
	}
	status := "FAIL"
	if result.Hit && result.Error == "" {
		status = "PASS"
	}
	label := result.CaseID
	if label == "" {
		label = result.Question
	}
	if _, err := fmt.Fprintf(w, "[%s] %s\n", status, label); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "question: %s\n", result.Question); err != nil {
		return err
	}
	if result.Error != "" {
		if _, err := fmt.Fprintf(w, "error: %s\n", result.Error); err != nil {
			return err
		}
		return nil
	}
	if _, err := fmt.Fprintf(w, "hit: %t recall: %.4f precision: %.4f rr: %.4f\n", result.Hit, result.Recall, result.Precision, result.ReciprocalRank); err != nil {
		return err
	}
	if len(result.QueryPlan) != 0 {
		if _, err := fmt.Fprintln(w, "query plan:"); err != nil {
			return err
		}
		for i, query := range result.QueryPlan {
			if _, err := fmt.Fprintf(w, "  %d. %s\n", i+1, query); err != nil {
				return err
			}
		}
	}
	relevant := relevantMatches(result.Retrieved)
	if len(relevant) != 0 {
		if _, err := fmt.Fprintln(w, "relevant:"); err != nil {
			return err
		}
		for _, match := range relevant {
			if err := writeMatch(w, match); err != nil {
				return err
			}
		}
		return nil
	}
	if _, err := fmt.Fprintln(w, "top results:"); err != nil {
		return err
	}
	for _, match := range result.Retrieved {
		if err := writeMatch(w, match); err != nil {
			return err
		}
	}
	return nil
}

func relevantMatches(matches []RetrievedMatch) []RetrievedMatch {
	out := make([]RetrievedMatch, 0)
	for _, match := range matches {
		if match.Relevant {
			out = append(out, match)
		}
	}
	return out
}

func writeMatch(w io.Writer, match RetrievedMatch) error {
	_, err := fmt.Fprintf(w, "  %d. %s:%d-%d score=%.4f chunk=%s\n", match.Rank, match.Path, match.StartLine, match.EndLine, match.Score, match.ChunkID)
	return err
}
