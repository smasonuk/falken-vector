package ingest

import (
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

type ChunkerMode string

const (
	ChunkerModeAuto     ChunkerMode = "auto"
	ChunkerModeFixed    ChunkerMode = "fixed"
	ChunkerModeMarkdown ChunkerMode = "markdown"
	ChunkerModeText     ChunkerMode = "text"
	ChunkerModeCode     ChunkerMode = "code"
)

type ChunkOptions struct {
	ChunkSize    int
	ChunkOverlap int
	Mode         ChunkerMode
	Path         string
}

type TextChunk struct {
	Index       int
	Text        string
	StartLine   int
	EndLine     int
	Hash        string
	Chunker     string
	Language    string
	HeadingPath []string
	SymbolName  string
	SymbolKind  string
}

type chunkMeta struct {
	Chunker     ChunkerMode
	Language    string
	HeadingPath []string
	SymbolName  string
	SymbolKind  string
}

type textBlock struct {
	Text      string
	StartLine int
	EndLine   int
}

var markdownHeadingPattern = regexp.MustCompile(`^(#{1,6})\s+(.+?)\s*#*\s*$`)

func ParseChunkerMode(value string) (ChunkerMode, error) {
	switch ChunkerMode(strings.TrimSpace(value)) {
	case "", ChunkerModeAuto:
		return ChunkerModeAuto, nil
	case ChunkerModeFixed:
		return ChunkerModeFixed, nil
	case ChunkerModeMarkdown:
		return ChunkerModeMarkdown, nil
	case ChunkerModeText:
		return ChunkerModeText, nil
	case ChunkerModeCode:
		return ChunkerModeCode, nil
	default:
		return "", fmt.Errorf("--chunker must be auto, fixed, markdown, text, or code")
	}
}

func DetectChunkerMode(path string) ChunkerMode {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".md", ".markdown", ".mdx":
		return ChunkerModeMarkdown
	case ".txt", ".text":
		return ChunkerModeText
	case ".go", ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".bash", ".zsh", ".sql":
		return ChunkerModeCode
	default:
		return ChunkerModeText
	}
}

func EffectiveChunkerMode(path string, mode ChunkerMode) ChunkerMode {
	if mode == "" || mode == ChunkerModeAuto {
		return DetectChunkerMode(path)
	}
	return mode
}

func DetectLanguage(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go":
		return "go"
	case ".py":
		return "python"
	case ".js", ".jsx":
		return "javascript"
	case ".ts", ".tsx":
		return "typescript"
	case ".java":
		return "java"
	case ".rs":
		return "rust"
	case ".sql":
		return "sql"
	case ".sh", ".bash", ".zsh":
		return "shell"
	case ".rb":
		return "ruby"
	case ".php":
		return "php"
	case ".swift":
		return "swift"
	case ".kt":
		return "kotlin"
	case ".scala":
		return "scala"
	case ".c", ".h":
		return "c"
	case ".cpp", ".hpp":
		return "cpp"
	case ".cs":
		return "csharp"
	default:
		return ""
	}
}

func ChunkText(input string, opts ChunkOptions) ([]TextChunk, error) {
	if err := validateChunkOptions(opts); err != nil {
		return nil, err
	}
	mode := EffectiveChunkerMode(opts.Path, opts.Mode)
	switch mode {
	case ChunkerModeFixed:
		return chunkFixed(input, opts, chunkMeta{Chunker: ChunkerModeFixed})
	case ChunkerModeMarkdown:
		return chunkMarkdown(input, opts)
	case ChunkerModeText:
		return chunkPlainText(input, opts)
	case ChunkerModeCode:
		return chunkCode(input, opts)
	default:
		return nil, fmt.Errorf("unsupported chunker mode %q", mode)
	}
}

func validateChunkOptions(opts ChunkOptions) error {
	if opts.ChunkSize <= 0 {
		return errors.New("chunk_size must be > 0")
	}
	if opts.ChunkOverlap < 0 {
		return errors.New("chunk_overlap must be >= 0")
	}
	if opts.ChunkOverlap >= opts.ChunkSize {
		return errors.New("chunk_overlap must be smaller than chunk_size")
	}
	return nil
}

func chunkFixed(input string, opts ChunkOptions, meta chunkMeta) ([]TextChunk, error) {
	return chunkFixedWithBase(input, opts, 1, meta)
}

func chunkFixedWithBase(input string, opts ChunkOptions, baseLine int, meta chunkMeta) ([]TextChunk, error) {
	runes := []rune(input)
	if len(runes) == 0 || strings.TrimSpace(input) == "" {
		return nil, nil
	}
	chunks := make([]TextChunk, 0, 1+(len(runes)/opts.ChunkSize))
	step := opts.ChunkSize - opts.ChunkOverlap
	for start := 0; start < len(runes); start += step {
		end := start + opts.ChunkSize
		if end > len(runes) {
			end = len(runes)
		}
		text := strings.TrimSpace(string(runes[start:end]))
		if text != "" {
			chunks = append(chunks, newTextChunk(len(chunks), text, baseLine+lineNumberAt(runes, start)-1, baseLine+lineNumberAt(runes, end)-1, meta))
		}
		if end == len(runes) {
			break
		}
	}
	return chunks, nil
}

func chunkMarkdown(input string, opts ChunkOptions) ([]TextChunk, error) {
	lines := splitLines(input)
	type section struct {
		lines       []string
		startLine   int
		headingPath []string
	}
	var sections []section
	var current section
	var headings [6]string
	inFence := false
	flush := func() {
		if len(current.lines) == 0 || strings.TrimSpace(strings.Join(current.lines, "")) == "" {
			current = section{}
			return
		}
		sections = append(sections, current)
		current = section{}
	}
	for i, line := range lines {
		lineNo := i + 1
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
		}
		if !inFence {
			if match := markdownHeadingPattern.FindStringSubmatch(trimmed); match != nil {
				flush()
				level := len(match[1])
				title := strings.TrimSpace(match[2])
				headings[level-1] = title
				for j := level; j < len(headings); j++ {
					headings[j] = ""
				}
				path := make([]string, 0, level)
				for _, heading := range headings[:level] {
					if heading != "" {
						path = append(path, heading)
					}
				}
				current = section{startLine: lineNo, headingPath: path}
			}
		}
		if len(current.lines) == 0 {
			current.startLine = lineNo
		}
		current.lines = append(current.lines, line)
	}
	flush()
	var chunks []TextChunk
	for _, sec := range sections {
		meta := chunkMeta{Chunker: ChunkerModeMarkdown, HeadingPath: sec.headingPath}
		secText := strings.Join(sec.lines, "")
		if len([]rune(secText)) <= opts.ChunkSize {
			text := strings.TrimSpace(secText)
			if text != "" {
				chunks = append(chunks, newTextChunk(len(chunks), text, sec.startLine, sec.startLine+len(sec.lines)-1, meta))
			}
			continue
		}
		blocks := paragraphBlocks(sec.lines, sec.startLine, true)
		packed, err := packBlocks(blocks, opts, meta)
		if err != nil {
			return nil, err
		}
		chunks = appendReindexed(chunks, packed)
	}
	return chunks, nil
}

func chunkPlainText(input string, opts ChunkOptions) ([]TextChunk, error) {
	lines := splitLines(input)
	blocks := paragraphBlocks(lines, 1, false)
	return packBlocks(blocks, opts, chunkMeta{Chunker: ChunkerModeText})
}

func chunkCode(input string, opts ChunkOptions) ([]TextChunk, error) {
	lines := splitLines(input)
	language := DetectLanguage(opts.Path)
	symbols := detectCodeSymbols(lines, language)
	if len(symbols) == 0 {
		return packBlocks([]textBlock{{Text: input, StartLine: 1, EndLine: len(lines)}}, opts, chunkMeta{Chunker: ChunkerModeCode, Language: language, SymbolKind: "section"})
	}
	commentStart := func(symbol codeSymbol, floor int) int {
		start := symbol.start
		for start > floor && isCommentLine(strings.TrimSpace(lines[start-1]), language) {
			start--
		}
		return start
	}
	type codeSection struct {
		block textBlock
		meta  chunkMeta
	}
	var sections []codeSection
	addSection := func(start int, end int, meta chunkMeta) {
		if start >= end {
			return
		}
		text := strings.Join(lines[start:end], "")
		if strings.TrimSpace(text) == "" {
			return
		}
		sections = append(sections, codeSection{
			block: textBlock{Text: text, StartLine: start + 1, EndLine: end},
			meta:  meta,
		})
	}
	firstStart := commentStart(symbols[0], 0)
	addSection(0, firstStart, chunkMeta{Chunker: ChunkerModeCode, Language: language, SymbolKind: "preamble"})
	for i, sym := range symbols {
		start := commentStart(sym, 0)
		end := len(lines)
		if i+1 < len(symbols) {
			end = commentStart(symbols[i+1], start)
		}
		addSection(start, end, chunkMeta{Chunker: ChunkerModeCode, Language: language, SymbolName: sym.name, SymbolKind: sym.kind})
	}

	var chunks []TextChunk
	for _, section := range sections {
		block := section.block
		if strings.TrimSpace(block.Text) == "" {
			continue
		}
		meta := section.meta
		if len([]rune(block.Text)) <= opts.ChunkSize {
			chunks = append(chunks, newTextChunk(len(chunks), strings.TrimSpace(block.Text), block.StartLine, block.EndLine, meta))
			continue
		}
		split, err := chunkFixedWithBase(block.Text, opts, block.StartLine, meta)
		if err != nil {
			return nil, err
		}
		chunks = appendReindexed(chunks, split)
	}
	return chunks, nil
}

func packBlocks(blocks []textBlock, opts ChunkOptions, meta chunkMeta) ([]TextChunk, error) {
	var chunks []TextChunk
	var current []textBlock
	currentLen := 0
	flush := func() {
		if len(current) == 0 {
			return
		}
		text := joinBlocks(current)
		chunks = append(chunks, newTextChunk(len(chunks), strings.TrimSpace(text), current[0].StartLine, current[len(current)-1].EndLine, meta))
		if opts.ChunkOverlap > 0 && len(current) > 1 {
			last := current[len(current)-1]
			current = []textBlock{last}
			currentLen = len([]rune(last.Text))
		} else {
			current = nil
			currentLen = 0
		}
	}
	for _, block := range blocks {
		if strings.TrimSpace(block.Text) == "" {
			continue
		}
		blockLen := len([]rune(block.Text))
		if blockLen > opts.ChunkSize {
			flush()
			split, err := chunkFixedWithBase(block.Text, opts, block.StartLine, meta)
			if err != nil {
				return nil, err
			}
			chunks = appendReindexed(chunks, split)
			current = nil
			currentLen = 0
			continue
		}
		nextLen := func() int {
			if len(current) == 0 {
				return blockLen
			}
			return currentLen + 2 + blockLen
		}
		if len(current) != 0 && nextLen() > opts.ChunkSize {
			flush()
			if len(current) != 0 && nextLen() > opts.ChunkSize {
				current = nil
				currentLen = 0
			}
		}
		current = append(current, block)
		if len(current) == 1 {
			currentLen = blockLen
		} else {
			currentLen += 2 + blockLen
		}
	}
	flush()
	return chunks, nil
}

func paragraphBlocks(lines []string, baseLine int, markdown bool) []textBlock {
	var blocks []textBlock
	var current []string
	startLine := 0
	inFence := false
	flush := func(endLine int) {
		if len(current) == 0 {
			return
		}
		text := strings.Join(current, "")
		if strings.TrimSpace(text) != "" {
			blocks = append(blocks, textBlock{Text: text, StartLine: startLine, EndLine: endLine})
		}
		current = nil
		startLine = 0
	}
	for i, line := range lines {
		lineNo := baseLine + i
		trimmed := strings.TrimSpace(line)
		if markdown && strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
		}
		if !inFence && trimmed == "" {
			flush(lineNo - 1)
			continue
		}
		if len(current) == 0 {
			startLine = lineNo
		}
		current = append(current, line)
	}
	flush(baseLine + len(lines) - 1)
	return blocks
}

func splitLines(input string) []string {
	if input == "" {
		return nil
	}
	parts := strings.SplitAfter(input, "\n")
	if parts[len(parts)-1] == "" {
		parts = parts[:len(parts)-1]
	}
	return parts
}

func joinBlocks(blocks []textBlock) string {
	parts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		parts = append(parts, strings.TrimSpace(block.Text))
	}
	return strings.Join(parts, "\n\n")
}

func appendReindexed(dst []TextChunk, src []TextChunk) []TextChunk {
	for _, chunk := range src {
		chunk.Index = len(dst)
		dst = append(dst, chunk)
	}
	return dst
}

func newTextChunk(index int, text string, startLine int, endLine int, meta chunkMeta) TextChunk {
	return TextChunk{
		Index:       index,
		Text:        text,
		StartLine:   startLine,
		EndLine:     endLine,
		Hash:        HashText(text),
		Chunker:     string(meta.Chunker),
		Language:    meta.Language,
		HeadingPath: append([]string(nil), meta.HeadingPath...),
		SymbolName:  meta.SymbolName,
		SymbolKind:  meta.SymbolKind,
	}
}

func lineNumberAt(runes []rune, pos int) int {
	if pos < 0 {
		pos = 0
	}
	if pos > len(runes) {
		pos = len(runes)
	}
	line := 1
	for i := 0; i < pos; i++ {
		if runes[i] == '\n' {
			line++
		}
	}
	return line
}

type codeSymbol struct {
	start int
	name  string
	kind  string
}

func detectCodeSymbols(lines []string, language string) []codeSymbol {
	var symbols []codeSymbol
	for i, line := range lines {
		if strings.TrimLeft(line, " \t") != line {
			continue
		}
		if symbol, ok := detectCodeSymbol(strings.TrimSpace(line), language); ok {
			symbol.start = i
			symbols = append(symbols, symbol)
		}
	}
	return symbols
}

var (
	goFuncPattern  = regexp.MustCompile(`^func\s+(\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	goTypePattern  = regexp.MustCompile(`^type\s+([A-Za-z_][A-Za-z0-9_]*)\s+`)
	pyDefPattern   = regexp.MustCompile(`^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	pyClassPattern = regexp.MustCompile(`^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]`)
	jsFuncPattern  = regexp.MustCompile(`^(export\s+)?(async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(`)
	jsClassPattern = regexp.MustCompile(`^(export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*`)
	jsConstPattern = regexp.MustCompile(`^(export\s+)?const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(async\s*)?\(`)
	genericPattern = regexp.MustCompile(`(function|func|def|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)`)
)

func detectCodeSymbol(line string, language string) (codeSymbol, bool) {
	switch language {
	case "go":
		if match := goFuncPattern.FindStringSubmatch(line); match != nil {
			return codeSymbol{name: match[2], kind: "function"}, true
		}
		if match := goTypePattern.FindStringSubmatch(line); match != nil {
			return codeSymbol{name: match[1], kind: "type"}, true
		}
	case "python":
		if match := pyDefPattern.FindStringSubmatch(line); match != nil {
			return codeSymbol{name: match[1], kind: "function"}, true
		}
		if match := pyClassPattern.FindStringSubmatch(line); match != nil {
			return codeSymbol{name: match[1], kind: "class"}, true
		}
	case "javascript", "typescript":
		if match := jsFuncPattern.FindStringSubmatch(line); match != nil {
			return codeSymbol{name: match[3], kind: "function"}, true
		}
		if match := jsClassPattern.FindStringSubmatch(line); match != nil {
			return codeSymbol{name: match[2], kind: "class"}, true
		}
		if match := jsConstPattern.FindStringSubmatch(line); match != nil {
			return codeSymbol{name: match[2], kind: "const"}, true
		}
	}
	if strings.HasSuffix(line, "{") {
		if match := genericPattern.FindStringSubmatch(line); match != nil {
			kind := match[1]
			if kind == "def" || kind == "func" {
				kind = "function"
			}
			return codeSymbol{name: match[2], kind: kind}, true
		}
	}
	return codeSymbol{}, false
}

func isCommentLine(line string, language string) bool {
	if line == "" {
		return false
	}
	switch language {
	case "python", "ruby", "shell":
		return strings.HasPrefix(line, "#")
	case "sql":
		return strings.HasPrefix(line, "--")
	default:
		return strings.HasPrefix(line, "//") || strings.HasPrefix(line, "/*") || strings.HasPrefix(line, "*")
	}
}
