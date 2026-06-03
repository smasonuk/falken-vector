## The mental model

This program builds a **local RAG index**. It stores your source files as small text chunks, turns each chunk into a numeric vector, saves those vectors in a vector database, and later searches that database with a vector made from the user’s question.

```
INGEST:
file → chunks → indexed text → embedding vector → vector DB
                         ↘ metadata/text rows → SQLite manifest

RETRIEVE (vector):
question → question vector → vector DB search → chunk IDs
                                      ↘ load chunk text from manifest → ranked results

RETRIEVE (lexical/hybrid):
question → SQLite FTS and/or vector DB → fused chunk IDs
                                 ↘ load chunk text from manifest → ranked results
```

A key point: the **vector database is not the main source of truth for text**. The manifest/SQLite side stores documents, raw chunk text, indexed text, line numbers, vector IDs, active/deleted status, and metadata. The vector store interface only exposes `Insert(vector, chunkID)`, `Search(vector, limit)`, and `Commit()`.

## What “indexing” means here

When you run ingest, the program scans files, decides whether each file is new, changed, or unchanged, and only indexes files that need work. It chooses an effective chunker for the file, checks the file/chunk config, skips unchanged files, and calls `indexFile` for new or changed files. 

Inside `indexFile`, the file is read, split into chunks, and each chunk is converted into an “indexed text” string. That indexed text is what gets embedded, not just the raw chunk body. 

The indexed text deliberately adds useful context. For code, it can add `File:`, `Language:`, and `Symbol:`. For prose/Markdown, it can add `Document:` and `Section:`. Then it appends `Chunk:` plus the actual chunk text. This helps the embedding model understand where the chunk came from. 

## What gets stored in the vector database

For each chunk:

1. The embedder sends the indexed text to an OpenAI-compatible `/embeddings` endpoint.
2. The embedding response is converted into a `[]float32`.
3. The vector database is opened using the vector’s dimensionality.
4. The vector is inserted with the chunk ID as payload.
5. The returned vector ID is stored back in the manifest row for that chunk.

The code builds the chunk ID, inserts the vector with `Insert(ctx, embedding.Vector, chunkID)`, and stores both the `VectorID` and `EmbeddingModel` in the manifest chunk row. 

The actual vector store is `vecgo`. When a DB is first created, it uses the vector dimensionality and the cosine metric. Cosine similarity is a common vector-search metric: it measures whether two vectors “point in the same direction,” which usually corresponds to semantic similarity. 

The inserted payload is just JSON containing the `chunk_id`. On search, vecgo returns candidates with payloads, and this wrapper turns them back into `{ChunkID, Score}` hits. 

## The manifest is the lookup table

The manifest has `documents` and `chunks` tables. The chunk rows include raw `chunk_text`, `indexed_text`, start/end lines, `vector_id`, `embedding_model`, active status, chunker/language/heading/symbol metadata, and a foreign key back to the document. 

So the vector DB answers: **“which chunk IDs are closest to this vector?”**

The manifest answers: **“what text, file path, line numbers, and metadata belong to those chunk IDs?”**

That separation is important. Vector stores are good at nearest-neighbour search, but not necessarily ideal as the authoritative database for full document/chunk metadata.

## Commit order and consistency

After all changed chunks have been embedded and inserted, ingest commits the vector database. Only after that does it update or activate the manifest rows. 

That order matters: the manifest should not point to vector IDs unless the vector DB has successfully committed them.

There is also a compaction path. Compaction reads active indexed chunks from the manifest, re-embeds them, builds a temporary vector DB, inserts each active chunk, commits it, swaps it into place, and then updates the manifest vector references. This is basically “rebuild the vector DB from the manifest.” 

## Retrieval: how a question becomes results

The query command opens the manifest, prepares lexical search when needed, configures an embedder if vector retrieval is being used, fills in `Question`, `TopK`, paths, and embedder, then calls `RetrieveWithPlan`. 

Retrieval supports three modes:

* `vector`: semantic search through the vector database.
* `lexical`: keyword-style search through SQLite FTS.
* `hybrid`: does both and fuses the results.

Those modes are defined directly in `retrieve.go`. 

For vector retrieval, the question is embedded using the same embedder abstraction. Then the vector DB is opened, searched with the question vector, and the returned hits are converted into candidate hits with chunk IDs, ranks, scores, and source `"vector"`. 

For lexical retrieval, the manifest’s full-text index is searched instead. Each lexical hit is converted into a candidate with a reciprocal-rank score. 

For hybrid retrieval, the code runs vector search and lexical search, then fuses the candidates. The fusion code combines scores by chunk ID, keeps source labels like `"vector"` and `"lexical"`, and sorts by fused score and rank.  

## Final result filtering

Vector search only gives candidate chunk IDs. The final step loads those chunk IDs from the manifest, skips inactive chunks, skips non-indexed documents, applies source filters, optionally reranks, optionally diversifies results across documents, trims to `TopK`, and returns `RetrievedChunk` objects containing the chunk text and file path. 

So the vector retrieval process is:

```
question
  ↓
embed question
  ↓
search vector DB
  ↓
get chunk IDs + similarity scores
  ↓
load real chunk text/metadata from SQLite manifest
  ↓
filter/rerank/diversify
  ↓
return top chunks
```

Lexical retrieval skips the embedding/vector-search steps and starts from the
SQLite FTS index. Hybrid retrieval runs both candidate paths and fuses them
before the shared manifest lookup, filtering, reranking, diversification, and
trimming steps.

## Summary

A vector database is like a **semantic map**. Each chunk of text becomes a point on that map. Similar chunks land near each other. When the user asks a question, the question is also turned into a point. The vector DB then finds the chunk-points nearest to the question-point.

In this codebase, the vector DB is deliberately small in responsibility: it stores vectors plus a `chunk_id` payload. The richer information lives in SQLite. This lets vecgo do fast similarity search while SQLite handles document state, chunk text, metadata, active/deleted filtering, and lexical search.
