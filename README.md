# falkengo

`falkengo` is a pure-Go, project-local RAG CLI for indexing text-like files, retrieving relevant chunks, and asking questions over those chunks with an OpenAI-compatible LLM.

It stores all state under `./.falkengo/` in the project you run it from:

- Manifest metadata: `./.falkengo/manifest.sqlite`
- Vector data: `./.falkengo/vecgo-data`
- Write lock: `./.falkengo/locks/write.lock`

No background daemon, external database server, CGO, or home-directory state is required.

## Environment

Embeddings use Portkey/OpenAI-compatible APIs.

```bash
export PK=...
```

Optional embedding overrides:

```bash
export FALKENGO_EMBEDDING_MODEL=text-embedding-3-small
export FALKENGO_BASE_URL=https://portkey.syngenta.com/v1
```

Optional LLM overrides for `ask`:

```bash
export FALKENGO_LLM_API_KEY=...
export FALKENGO_LLM_BASE_URL=https://portkey.syngenta.com/v1
export FALKENGO_LLM_MODEL=gpt-5.2
```

If `FALKENGO_LLM_API_KEY` is not set, `ask` uses `PK`.

## Development checkout

This branch currently uses sibling local modules for Falken agent support. The expected development checkout is:

```text
falken2/
  go.work
  falken-core/
  falken-extra/
  falken-vector/
```

`falken-vector/go.mod` contains local replacements for `github.com/smasonuk/falken-core` and `github.com/smasonuk/falken-extra`, so `go test ./...` requires those sibling repositories or equivalent module replacements. Prefer keeping local workspace wiring in the parent `go.work` when developing across all three modules.

## Usage

```bash
falkengo ingest .
falkengo ingest . --chunker auto
falkengo ingest ./docs --sync-source
falkengo query "what does this project do?"
falkengo query "ErrPendingRunDetected" --retrieval lexical
falkengo query "where is the write lock acquired?" --retrieval hybrid --reranker heuristic
falkengo query "how does compact avoid stale vectors?" --retrieval hybrid --reranker heuristic --query-planner heuristic --show-query-plan
falkengo query "citation validation" --include "internal/rag/**" --json
falkengo ask "how is the embedding client configured?"
falkengo ask "how does retrieval work?" --source-root ./internal/rag --open-source 1
falkengo ask --agent "where is RetrieveWithPlan implemented?" --retrieval hybrid --show-agent-tools
falkengo eval retrieval --dataset examples/retrieval-eval.jsonl --retrieval hybrid --reranker heuristic --query-planner heuristic
falkengo status
falkengo compact
falkengo repair
falkengo reset
```

Default indexed extensions are:

```text
.txt,.md,.go,.py,.js,.ts,.tsx,.jsx,.json,.yaml,.yml,.toml
```

Use `--state-dir` to choose another project-local state directory, `--timeout` to control command timeouts, and `--verbose` for detailed progress.

## Commands

`falkengo ingest <directory>` scans files, skips unchanged indexed files, chunks changed files, embeds chunks, stores vectors in Vecgo, and stores metadata in SQLite. Use `--chunker auto|fixed|markdown|text|code` to choose the chunking strategy; `auto` is the default and selects Markdown, prose, or code chunking from the file extension. Ingest embeds a contextual indexed form of each chunk that includes short path/heading/symbol metadata, while raw chunk text is still kept for display and prompts. Add `--sync-source` to treat that directory as the source of truth and logically delete previously indexed files from that same source root when they are no longer present on disk.

`falkengo query <question>` retrieves relevant chunks and prints editor-friendly source references like `[source 1] internal/rag/retrieve.go:35-73`. It does not call the LLM unless `--query-planner llm` is explicitly selected. Use `--retrieval vector`, `--retrieval lexical`, or `--retrieval hybrid`; vector remains the default. Lexical mode searches SQLite FTS over paths and contextual indexed text, while hybrid mode fuses vector and lexical ranks. Add `--reranker heuristic` to locally reorder filtered candidates before diversification and final trimming; reranking defaults to `none`. Add `--query-planner heuristic` to expand one question into multiple deterministic retrieval queries, or `--query-planner llm` to use the configured chat client for JSON query planning. Use `--max-subqueries` to cap planned queries and `--show-query-plan` to print them. Use `--include`, `--exclude`, and `--source-root` to restrict sources, `--json` for machine-readable chunks, `--show-retrieval-debug` for retrieval settings, and `--open-source N` to open a retrieved source in `FALKENGO_EDITOR`, `EDITOR`, or `VISUAL`.

`falkengo ask <question>` retrieves chunks, builds a context-only prompt, sends only those chunks to the LLM, prints the answer, and lists editor-friendly sources. It supports the same `--retrieval`, `--reranker`, `--query-planner`, `--include`, `--exclude`, `--source-root`, and `--open-source` controls as `query`. Ask validates model citations such as `[source 1]` by default and retries once with a stricter citation prompt; use `--no-citation-validation` or `--no-citation-retry` to relax that behavior.

### Agentic ask

`falkengo ask --agent "question"` uses a Falken agent with a `search_index` tool. The agent decides when and how often to search the local index, and it must cite sources returned by the tool using `[source N]`. The normal `falkengo ask "question"` path remains the legacy one-shot RAG flow: retrieve chunks first, then send them to the LLM. `falkengo query "question"` remains deterministic retrieval without an answer-generating LLM.

Existing retrieval flags become defaults or restrictions for the agent's `search_index` calls:

```text
--retrieval
--reranker
--query-planner
--max-subqueries
--top-k
--include
--exclude
--source-root
```

Examples:

```bash
falkengo ingest . --extensions .go,.md

falkengo query "Where is RetrieveWithPlan implemented?" \
  --retrieval hybrid \
  --top-k 5

falkengo ask --agent "Where is RetrieveWithPlan implemented?" \
  --retrieval hybrid \
  --show-agent-tools

falkengo ask --agent "Where is citation validation implemented?" \
  --retrieval lexical \
  --open-source 1
```

Agentic ask uses the same embedding environment variables as retrieval and the OpenAI-compatible agent LLM variables:

```text
PK
FALKENGO_EMBEDDING_MODEL
FALKENGO_BASE_URL
FALKENGO_LLM_API_KEY
FALKENGO_LLM_BASE_URL
FALKENGO_LLM_MODEL
```

Smoke-test checklist:

```bash
go test ./...
falkengo query "citation validation" --retrieval lexical --top-k 3
falkengo ask --agent "citation validation" --retrieval lexical --show-agent-tools
```

`falkengo eval retrieval --dataset <path>` runs retrieval-only golden questions against the current index and reports `hit@k`, `recall@k`, `precision@k`, and `mrr@k`. Use `--retrieval vector|lexical|hybrid`, `--reranker none|heuristic`, and `--query-planner none|heuristic|llm` to compare modes, `--details` for per-case results and query plans, `--format json` for CI-friendly output, and `--fail-under-hit` / `--fail-under-recall` to make the command fail below a threshold. Datasets are strict JSONL with one case per line; see `examples/retrieval-eval.jsonl`.

`falkengo status` prints manifest and vector DB paths plus document/chunk counts.

`falkengo compact` rebuilds `./.falkengo/vecgo-data` from active chunks whose documents are still indexed. It re-embeds those chunks with the currently configured embedding model, updates vector references in SQLite, and can recreate a missing vector DB from the manifest. Use `--dry-run` to report the rebuild without embedding or writing, and `--keep-backup` to retain the previous vector DB directory. A real compact refuses to run while pending ingest runs exist; dry-run reports the counts and prints a repair warning.

`falkengo repair` recovers a pending ingest run if Vecgo committed but manifest activation did not complete. Use `--discard-pending` to remove staged manifest rows instead of activating them. Use `--rebuild-lexical` to rebuild the SQLite lexical index from active indexed chunks; lexical and hybrid reads only self-heal when the lexical table is missing or empty.

`falkengo reset` deletes `./.falkengo/` after confirmation. Use `--yes` to skip the prompt. For safety, reset refuses to delete a state directory whose basename is not `.falkengo` unless `--force-state-dir` is passed.

## Limitations

- Only text-like files are indexed.
- Binary files are skipped by extension filtering.
- Chunking is structure-aware for Markdown, prose, and common code files, with fixed character windows still available via `--chunker fixed`.
- Vecgo is wrapped internally so it can be replaced later if needed.
- The tool stores local state in `./.falkengo`.
- `--sync-source` performs logical deletion only; old vectors remain in Vecgo until `falkengo compact` rebuilds the vector DB.
