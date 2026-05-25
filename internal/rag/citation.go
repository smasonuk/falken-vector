package rag

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type CitationPolicy string

const (
	CitationPolicyValidateAndRetry CitationPolicy = "validate-and-retry"
	CitationPolicyValidateOnly     CitationPolicy = "validate-only"
	CitationPolicyOff              CitationPolicy = "off"
)

type CitationValidation struct {
	Valid              bool
	CitedSources       []int
	MissingCitation    bool
	OutOfRangeSources  []int
	MalformedCitations []string
	Warnings           []string
}

var (
	citationPattern          = regexp.MustCompile(`(?i)\[source\s+(\d+)\]`)
	canonicalCitationPattern = regexp.MustCompile(`(?i)^\[source\s+\d+\]$`)
	sourceBracketPattern     = regexp.MustCompile(`(?i)\[sources?[^\]]*\]`)
	integerPattern           = regexp.MustCompile(`\d+`)
	citationRangePattern     = regexp.MustCompile(`\d+\s*-\s*\d+`)
)

func ValidateAnswerCitations(answer string, sourceCount int) CitationValidation {
	validation := CitationValidation{}
	outOfRange := make(map[int]struct{})
	matches := citationPattern.FindAllStringSubmatch(answer, -1)
	validation.CitedSources = ExtractCitedSourceNumbers(answer)
	validation.MalformedCitations = MalformedCitations(answer)
	for _, sourceNumber := range validation.CitedSources {
		if sourceNumber < 1 || sourceNumber > sourceCount {
			outOfRange[sourceNumber] = struct{}{}
		}
	}
	if len(matches) == 0 && sourceCount > 0 {
		validation.MissingCitation = true
		validation.Warnings = append(validation.Warnings, "answer did not cite any source")
	}
	for _, citation := range validation.MalformedCitations {
		validation.Warnings = append(validation.Warnings, fmt.Sprintf("malformed citation %s; use one bracket per source, e.g. [source 2] [source 3]", citation))
	}
	for sourceNumber := range outOfRange {
		validation.OutOfRangeSources = append(validation.OutOfRangeSources, sourceNumber)
	}
	sort.Ints(validation.OutOfRangeSources)
	for _, sourceNumber := range validation.OutOfRangeSources {
		validation.Warnings = append(validation.Warnings, fmt.Sprintf("answer cited unavailable source [source %d]; available sources are 1-%d", sourceNumber, sourceCount))
	}
	validation.Valid = !validation.MissingCitation && len(validation.OutOfRangeSources) == 0 && len(validation.MalformedCitations) == 0
	return validation
}

func MalformedCitations(answer string) []string {
	matches := sourceBracketPattern.FindAllString(answer, -1)
	out := make([]string, 0)
	seen := map[string]struct{}{}
	for _, match := range matches {
		if canonicalCitationPattern.MatchString(match) {
			continue
		}
		if _, ok := seen[match]; ok {
			continue
		}
		seen[match] = struct{}{}
		out = append(out, match)
	}
	return out
}

func NormalizeGroupedCitations(answer string) (string, []string) {
	var warnings []string
	normalized := sourceBracketPattern.ReplaceAllStringFunc(answer, func(match string) string {
		if canonicalCitationPattern.MatchString(match) {
			return match
		}
		if citationRangePattern.MatchString(match) {
			return match
		}
		numbers := extractCitationInts(match)
		if len(numbers) == 0 {
			return match
		}
		refs := make([]string, 0, len(numbers))
		seen := map[int]struct{}{}
		for _, number := range numbers {
			if _, ok := seen[number]; ok {
				continue
			}
			seen[number] = struct{}{}
			refs = append(refs, fmt.Sprintf("[source %d]", number))
		}
		replacement := strings.Join(refs, " ")
		warnings = append(warnings, fmt.Sprintf("normalized %s -> %s", match, replacement))
		return replacement
	})
	return normalized, warnings
}

func extractCitationInts(value string) []int {
	matches := integerPattern.FindAllString(value, -1)
	numbers := make([]int, 0, len(matches))
	for _, match := range matches {
		number, err := strconv.Atoi(match)
		if err != nil {
			continue
		}
		numbers = append(numbers, number)
	}
	return numbers
}

func ExtractCitedSourceNumbers(answer string) []int {
	matches := citationPattern.FindAllStringSubmatch(answer, -1)
	seen := make(map[int]struct{})
	numbers := make([]int, 0, len(matches))
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		sourceNumber, err := strconv.Atoi(match[1])
		if err != nil {
			continue
		}
		if _, ok := seen[sourceNumber]; ok {
			continue
		}
		seen[sourceNumber] = struct{}{}
		numbers = append(numbers, sourceNumber)
	}
	sort.Ints(numbers)
	return numbers
}

func normalizeCitationPolicy(policy CitationPolicy) CitationPolicy {
	switch policy {
	case CitationPolicyValidateOnly, CitationPolicyOff:
		return policy
	case "", CitationPolicyValidateAndRetry:
		return CitationPolicyValidateAndRetry
	default:
		return CitationPolicyValidateAndRetry
	}
}

func joinCitationWarnings(warnings []string) string {
	return strings.Join(warnings, "; ")
}
