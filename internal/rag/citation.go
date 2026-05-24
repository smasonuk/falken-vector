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
	Valid             bool
	CitedSources      []int
	MissingCitation   bool
	OutOfRangeSources []int
	Warnings          []string
}

var citationPattern = regexp.MustCompile(`(?i)\[source\s+(\d+)\]`)

func ValidateAnswerCitations(answer string, sourceCount int) CitationValidation {
	validation := CitationValidation{}
	outOfRange := make(map[int]struct{})
	matches := citationPattern.FindAllStringSubmatch(answer, -1)
	validation.CitedSources = ExtractCitedSourceNumbers(answer)
	for _, sourceNumber := range validation.CitedSources {
		if sourceNumber < 1 || sourceNumber > sourceCount {
			outOfRange[sourceNumber] = struct{}{}
		}
	}
	if len(matches) == 0 && sourceCount > 0 {
		validation.MissingCitation = true
		validation.Warnings = append(validation.Warnings, "answer did not cite any source")
	}
	for sourceNumber := range outOfRange {
		validation.OutOfRangeSources = append(validation.OutOfRangeSources, sourceNumber)
	}
	sort.Ints(validation.OutOfRangeSources)
	for _, sourceNumber := range validation.OutOfRangeSources {
		validation.Warnings = append(validation.Warnings, fmt.Sprintf("answer cited unavailable source [source %d]; available sources are 1-%d", sourceNumber, sourceCount))
	}
	validation.Valid = !validation.MissingCitation && len(validation.OutOfRangeSources) == 0
	return validation
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
