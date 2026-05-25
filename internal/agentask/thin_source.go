package agentask

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/smasonuk/falken-vector/internal/rag"
)

type thinSourcePolicy struct {
	Enabled       bool
	MaxRetries    int
	MaxSources    int
	LineThreshold int
	ContextLines  int
}

type thinSourceDecision struct {
	Nudge         bool
	Reason        string
	SourceNumbers []int
	ContextLines  int
}

func normalizeThinSourcePolicy(opts Options) thinSourcePolicy {
	enabled := true
	if opts.ThinSourceNudge != nil {
		enabled = *opts.ThinSourceNudge
	}
	maxRetries := opts.MaxThinSourceRetries
	if maxRetries <= 0 {
		maxRetries = 1
	}
	maxSources := opts.MaxThinSourcesToExpand
	if maxSources <= 0 {
		maxSources = 3
	}
	lineThreshold := opts.ThinSourceLineThreshold
	if lineThreshold <= 0 {
		lineThreshold = 2
	}
	contextLines := opts.ThinSourceContextLines
	if contextLines <= 0 {
		contextLines = 20
	}
	return thinSourcePolicy{
		Enabled:       enabled,
		MaxRetries:    maxRetries,
		MaxSources:    maxSources,
		LineThreshold: lineThreshold,
		ContextLines:  contextLines,
	}
}

func thinSourceNudgeDecision(question string, readSourceEnabled bool, policy thinSourcePolicy, answer string, sources []rag.SourceChunk, trace AgentTrace, retries int) thinSourceDecision {
	if !policy.Enabled {
		return thinSourceDecision{Reason: "disabled"}
	}
	if !isBroadCoverageQuestion(question) {
		return thinSourceDecision{Reason: "not broad"}
	}
	if !readSourceEnabled {
		return thinSourceDecision{Reason: "read_index_source disabled"}
	}
	if retries >= policy.MaxRetries {
		return thinSourceDecision{Reason: "retry limit reached"}
	}
	readSources := readSourceNumbers(trace)
	candidates := thinCitedSources(answer, sources, policy.LineThreshold, policy.ContextLines, readSources)
	if len(candidates) == 0 {
		return thinSourceDecision{Reason: "no cited spans under threshold"}
	}
	if len(candidates) > policy.MaxSources {
		candidates = candidates[:policy.MaxSources]
	}
	return thinSourceDecision{
		Nudge:         true,
		Reason:        "thin cited spans",
		SourceNumbers: candidates,
		ContextLines:  policy.ContextLines,
	}
}

func thinCitedSources(answer string, sources []rag.SourceChunk, lineThreshold, contextLines int, alreadyRead map[int]struct{}) []int {
	cited := rag.ExtractCitedSourceNumbers(answer)
	if len(cited) == 0 {
		return nil
	}
	citedSet := make(map[int]struct{}, len(cited))
	for _, number := range cited {
		citedSet[number] = struct{}{}
	}
	out := make([]int, 0, len(cited))
	seen := map[int]struct{}{}
	selectedRanges := make([]sourceLineRange, 0, len(cited))
	for _, source := range sources {
		if _, ok := citedSet[source.SourceNumber]; !ok {
			continue
		}
		if _, ok := alreadyRead[source.SourceNumber]; ok {
			continue
		}
		if _, ok := seen[source.SourceNumber]; ok {
			continue
		}
		if sourceLineSpan(source) > lineThreshold {
			continue
		}
		projected := projectedSourceRange(source, contextLines)
		if overlapsSelectedSourceRange(projected, selectedRanges) {
			continue
		}
		seen[source.SourceNumber] = struct{}{}
		selectedRanges = append(selectedRanges, projected)
		out = append(out, source.SourceNumber)
	}
	return out
}

type sourceLineRange struct {
	Key   string
	Start int
	End   int
}

func projectedSourceRange(source rag.SourceChunk, contextLines int) sourceLineRange {
	if contextLines < 0 {
		contextLines = 0
	}
	start := source.StartLine - contextLines
	if start < 1 {
		start = 1
	}
	end := source.EndLine + contextLines
	if end < start {
		end = start
	}
	return sourceLineRange{
		Key:   source.Path,
		Start: start,
		End:   end,
	}
}

func overlapsSelectedSourceRange(candidate sourceLineRange, selected []sourceLineRange) bool {
	for _, existing := range selected {
		if rangesOverlapEnough(candidate, existing) {
			return true
		}
	}
	return false
}

func rangesOverlapEnough(a, b sourceLineRange) bool {
	if a.Key == "" || b.Key == "" || a.Key != b.Key {
		return false
	}
	overlap := minInt(a.End, b.End) - maxInt(a.Start, b.Start) + 1
	if overlap <= 0 {
		return false
	}
	smaller := minInt(lineRangeLen(a), lineRangeLen(b))
	return smaller > 0 && float64(overlap)/float64(smaller) >= 0.75
}

func lineRangeLen(r sourceLineRange) int {
	if r.End < r.Start {
		return 0
	}
	return r.End - r.Start + 1
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func sourceLineSpan(source rag.SourceChunk) int {
	if source.StartLine <= 0 || source.EndLine <= 0 || source.EndLine < source.StartLine {
		return 1
	}
	return source.EndLine - source.StartLine + 1
}

func readSourceNumbers(trace AgentTrace) map[int]struct{} {
	out := map[int]struct{}{}
	for _, call := range trace.ToolCalls {
		if call.Name != ReadIndexSourceToolName {
			continue
		}
		var args readIndexSourceArgs
		if err := json.Unmarshal(call.Arguments, &args); err != nil {
			continue
		}
		if args.SourceNumber > 0 {
			out[args.SourceNumber] = struct{}{}
		}
	}
	return out
}

func thinSourceNudgePrompt(question, previousAnswer string, sourceNumbers []int, contextLines int) string {
	sourceRefs := make([]string, 0, len(sourceNumbers))
	for _, number := range sourceNumbers {
		sourceRefs = append(sourceRefs, fmt.Sprintf("[source %d]", number))
	}
	return fmt.Sprintf(`Your answer cites some very short source spans. Before finalizing this broad summary, read nearby context for the most important thin sources.

You are expanding source context, not restarting the answer.

The user's question:
%s

Previous answer:
%s

Call read_index_source for these source numbers:
%s

Use around %d context lines. Then revise only where the expanded context improves accuracy, support, or detail. Preserve relevant points from the previous answer if they remain supported by cited or available sources. Do not drop useful supported sections merely because they were not part of the newly expanded sources. When preserving points from the previous answer, preserve or rewrite their citations using the exact one-source-per-bracket format. Use one bracket per cited source: [source 2] [source 3]. Never write [sources 2, 3], [source 2, 3], [source 2 and 3], or multiple source numbers inside one bracket. If the answer is already supported, keep it concise and cite the same source numbers.`, question, previousAnswer, strings.Join(sourceRefs, ", "), contextLines)
}

func thinSourceNudgeWarning(sourceNumbers []int) string {
	sourceRefs := make([]string, 0, len(sourceNumbers))
	for _, number := range sourceNumbers {
		sourceRefs = append(sourceRefs, fmt.Sprintf("[source %d]", number))
	}
	return "thin-source nudge: expanding " + strings.Join(sourceRefs, ", ")
}

func thinSourceSkippedWarning(reason string) string {
	switch reason {
	case "read_index_source disabled":
		return "thin-source nudge skipped: read_index_source disabled"
	case "no cited spans under threshold":
		return "thin-source nudge skipped: no cited spans under threshold"
	case "retry limit reached":
		return "thin-source nudge skipped: retry limit reached"
	default:
		return ""
	}
}
