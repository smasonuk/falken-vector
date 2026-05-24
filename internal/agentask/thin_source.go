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
	candidates := thinCitedSources(answer, sources, policy.LineThreshold, readSources)
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

func thinCitedSources(answer string, sources []rag.SourceChunk, lineThreshold int, alreadyRead map[int]struct{}) []int {
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
		seen[source.SourceNumber] = struct{}{}
		out = append(out, source.SourceNumber)
	}
	return out
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

func thinSourceNudgePrompt(sourceNumbers []int, contextLines int) string {
	sourceRefs := make([]string, 0, len(sourceNumbers))
	for _, number := range sourceNumbers {
		sourceRefs = append(sourceRefs, fmt.Sprintf("[source %d]", number))
	}
	return fmt.Sprintf(`Your answer cites some very short source spans. Before finalizing this broad summary, read nearby context for the most important thin sources.

Call read_index_source for these source numbers:
%s

Use around %d context lines. Then revise the answer if the expanded context changes or improves it. If the answer is already supported, keep it concise and cite the same source numbers.`, strings.Join(sourceRefs, ", "), contextLines)
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
