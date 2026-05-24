package ingest

import (
	"os"
	"path/filepath"
	"strings"
)

const CurrentIndexTextVersion = 1

func BuildIndexedChunkText(path string, chunk TextChunk) string {
	var lines []string
	displayPath := contextualDisplayPath(path)
	if chunk.Chunker == string(ChunkerModeCode) || chunk.Language != "" || chunk.SymbolName != "" || chunk.SymbolKind != "" {
		if displayPath != "" {
			lines = append(lines, "File: "+displayPath)
		}
		if chunk.Language != "" {
			lines = append(lines, "Language: "+chunk.Language)
		}
		if symbol := formatSymbol(chunk.SymbolKind, chunk.SymbolName); symbol != "" {
			lines = append(lines, "Symbol: "+symbol)
		}
	} else {
		if displayPath != "" {
			lines = append(lines, "Document: "+displayPath)
		}
		if len(chunk.HeadingPath) != 0 {
			lines = append(lines, "Section: "+strings.Join(chunk.HeadingPath, " > "))
		}
	}
	lines = append(lines, "Chunk:", chunk.Text)
	return strings.Join(lines, "\n")
}

func formatSymbol(kind string, name string) string {
	kind = strings.TrimSpace(kind)
	name = strings.TrimSpace(name)
	switch {
	case kind != "" && name != "":
		return kind + " " + name
	case name != "":
		return name
	default:
		return ""
	}
}

func contextualDisplayPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	clean := filepath.Clean(path)
	if filepath.IsAbs(clean) {
		if wd, err := os.Getwd(); err == nil {
			if rel, err := filepath.Rel(wd, clean); err == nil && rel != "." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && rel != ".." {
				clean = rel
			}
		}
	}
	return filepath.ToSlash(clean)
}
