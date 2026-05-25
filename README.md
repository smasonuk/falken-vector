# falkengo

`falkengo` is a pure-Go, project-local RAG CLI for indexing text-like files, retrieving relevant chunks, and asking questions over those chunks with an OpenAI-compatible LLM.

It stores all state under `./.falkengo/` in the project you run it from:

- Manifest metadata: `./.falkengo/manifest.sqlite`
- Vector data: `./.falkengo/vecgo-data`
- Write lock: `./.falkengo/locks/write.lock`

No background daemon, external database server, CGO, or home-directory state is required.

## Environment

Falken Vector expects OpenAI-compatible `/embeddings` and `/chat/completions` endpoints. Providers with different wire formats should be integrated by implementing the provider/embedder interfaces.

```bash
export FALKENGO_EMBEDDING_MODEL_URL="https://api.example.com/v1"
export FALKENGO_EMBEDDING_MODEL="embedding-model"
export FALKENGO_EMBEDDING_MODEL_API_KEY="optional-token"
export FALKENGO_EMBEDDING_MODEL_HEADERS='{"Header-Name":"value"}'

export FALKENGO_LLM_BASE_URL="https://api.example.com/v1"
export FALKENGO_LLM_MODEL="chat-model"
export FALKENGO_LLM_API_KEY="optional-token"
export FALKENGO_LLM_HEADERS='{"Header-Name":"value"}'
```

The API key variables may be omitted when the provider does not need bearer auth. Header variables are JSON objects and are useful for provider routing, organization IDs, or non-bearer authentication.

Hosted OpenAI-compatible provider with bearer token:

```bash
export FALKENGO_EMBEDDING_MODEL_URL="https://api.example.com/v1"
export FALKENGO_EMBEDDING_MODEL="text-embedding-3-small"
export FALKENGO_EMBEDDING_MODEL_API_KEY="sk-..."

export FALKENGO_LLM_BASE_URL="https://api.example.com/v1"
export FALKENGO_LLM_MODEL="gpt-compatible-model"
export FALKENGO_LLM_API_KEY="sk-..."
```

Local OpenAI-compatible provider with no API key:

```bash
export FALKENGO_EMBEDDING_MODEL_URL="http://127.0.0.1:11434/v1"
export FALKENGO_EMBEDDING_MODEL="nomic-embed-text"

export FALKENGO_LLM_BASE_URL="http://127.0.0.1:11434/v1"
export FALKENGO_LLM_MODEL="llama3.1"
```

Portkey can be used by supplying its URL and routing header explicitly:

```bash
export FALKENGO_EMBEDDING_MODEL_URL="https://portkey.syngenta.com/v1"
export FALKENGO_EMBEDDING_MODEL="text-embedding-3-small"
export FALKENGO_EMBEDDING_MODEL_API_KEY="..."
export FALKENGO_EMBEDDING_MODEL_HEADERS='{"X-Portkey-Provider":"@openai-aifoundry-swc-001"}'

export FALKENGO_LLM_BASE_URL="https://portkey.syngenta.com/v1"
export FALKENGO_LLM_MODEL="gpt-5.2"
export FALKENGO_LLM_API_KEY="..."
export FALKENGO_LLM_HEADERS='{"X-Portkey-Provider":"@openai-aifoundry-swc-001"}'
```

If `FALKENGO_LLM_API_KEY` is not set, `ask` uses `FALKENGO_EMBEDDING_MODEL_API_KEY` when present.

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

By default, ingest indexes text-like files discovered by content sniffing. Use `--extensions .md,.go` to restrict indexing to specific file extensions.

Use `--state-dir` to choose another project-local state directory, `--timeout` to control command timeouts, and `--verbose` for detailed progress. `ingest` and `compact` do not apply the default timeout because they can run for a long time; pass `--timeout` explicitly to bound those commands.

## Commands

`falkengo ingest <directory>` scans text-like files, skips unchanged indexed files, chunks changed files, embeds chunks, stores vectors in Vecgo, and stores metadata in SQLite. It prints progress as it scans, embeds, commits the vector database, and updates the manifest; add `--verbose` for per-file skip and error details. Use `--extensions .md,.go` to restrict discovery to specific extensions. Use `--chunker auto|fixed|markdown|text|code` to choose the chunking strategy; `auto` is the default and selects Markdown, prose, or code chunking from the file extension, with unknown text files chunked as plain text. Ingest embeds a contextual indexed form of each chunk that includes short path/heading/symbol metadata, while raw chunk text is still kept for display and prompts. Add `--sync-source` to treat that directory as the source of truth and logically delete previously indexed files from that same source root when they are no longer present on disk.

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

To inspect how an agent searched, run with debug output and the agent source view:

```bash
falkengo ask --agent "summarize information on alphafold or anything related to folding" \
  --retrieval hybrid \
  --top-k 12 \
  --max-agent-searches 8 \
  --sources both \
  --show-agent-tools
```

`--show-agent-tools` prints each `agent tool call` and compact `agent tool result` without dumping source text. Search results include `new` sources added by that call, `duplicates` already seen, `docs` touched, and actual `retrievals` performed. Broad searches may also show `agent seed query plan`, `agent broad expansion queries`, and `agent suggested follow-up queries`; if the tool normalizes a repeated-term query, the result shows the normalized query and original text. A `agent thin-source nudge` line means the agent is expanding very short cited spans with `read_index_source` before finalizing.

Agent answers default to `--sources both`: `Sources cited` lists sources referenced by the final answer, while `Other sources available to the agent` shows retrieved but uncited evidence. Use `--sources cited` for the cited-only view, `--sources all` for every available source in one list, and `--show-source-provenance` to print first-introduced query details per source. If an uncited source is fully contained by an expanded cited range, it is marked as covered by that cited source.

`Source audit` summarizes the evidence set:

```text
Source audit:
- available to agent: 20
- cited in answer: 4
- uncited: 16
- search tool calls: 2
- retrieval calls: 5
- read source calls: 1
- read source overlap policy: merge
- introduced by query:
  - AlphaFold: 12
  - protein folding: 8
```

`available to agent` is the unique source set registered from tool searches. `cited in answer` is the subset referenced in the final text, and `uncited` is the difference. `search tool calls` counts outer `search_index` calls; `retrieval calls` can be higher because broad search may perform internal expansion retrievals. `introduced by query` counts the first query that introduced each source and appears when provenance is available under `--show-agent-tools`.

`read_index_source` is an optional agent tool for expanding context around an already returned source number. It cannot read arbitrary paths: the agent must pass a registered `source_number`, and `context_lines` defaults to 20 and caps at 100. Broad summary questions auto-enable the tool unless `--no-agent-read-source-tool` is set; `--agent-read-source-tool` enables it explicitly. Expanded ranges broaden the existing source number so final citations keep pointing at `[source N]`.

Use `--read-source-overlap-policy skip|merge|allow` to control overlapping reads. Broad agent summaries default to `merge` unless explicitly overridden; other agent questions default to `skip`. `skip` avoids exact or mostly covered duplicate reads, `merge` broadens the first expanded source when a later overlapping read adds useful lines, and `allow` always performs the read and returns separate expanded context. `--max-merged-read-source-lines` caps merged source ranges and defaults to 300 lines; if a merge would exceed the cap, the tool returns `merge_too_large` and asks the agent to choose a narrower read. Debug output reports `already covered`, `merged`, or `merge skipped` decisions, for example when `[source 2]` is already covered by expanded `[source 11]`.

```bash
falkengo ask --agent "summarize anything related to AlphaFold" \
  --read-source-overlap-policy merge \
  --max-merged-read-source-lines 300 \
  --show-agent-tools
```

Agentic ask uses the same embedding environment variables as retrieval and the OpenAI-compatible agent LLM variables:

```text
FALKENGO_EMBEDDING_MODEL_API_KEY
FALKENGO_EMBEDDING_MODEL
FALKENGO_EMBEDDING_MODEL_URL
FALKENGO_EMBEDDING_MODEL_HEADERS
FALKENGO_LLM_API_KEY
FALKENGO_LLM_BASE_URL
FALKENGO_LLM_MODEL
FALKENGO_LLM_HEADERS
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
