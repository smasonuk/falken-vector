package manifest

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

var ErrNotFound = errors.New("manifest record not found")

type Store interface {
	Close() error
	Init(ctx context.Context) error

	GetDocumentByPath(ctx context.Context, path string) (*Document, error)
	GetDocumentByID(ctx context.Context, id string) (*Document, error)
	UpsertDocument(ctx context.Context, doc Document) error
	MarkDocumentError(ctx context.Context, path string, errText string) error
	ListIndexedDocumentsUnderRoot(ctx context.Context, sourceRoot string) ([]Document, error)
	MarkDocumentDeleted(ctx context.Context, documentID string, deletedAt time.Time) error
	MarkDocumentsDeleted(ctx context.Context, documentIDs []string, deletedAt time.Time) error

	MarkChunksInactive(ctx context.Context, documentID string) error
	InsertChunk(ctx context.Context, chunk Chunk) error
	GetActiveChunksByIDs(ctx context.Context, ids []string) ([]Chunk, error)

	Stats(ctx context.Context) (Stats, error)
}

type PendingRunStore interface {
	ListPendingRuns(ctx context.Context) ([]PendingRun, error)
	ActivatePendingRun(ctx context.Context, runID string) error
	ClearPendingRun(ctx context.Context, runID string) error
}

type CompactStore interface {
	ListActiveIndexedChunks(ctx context.Context, limit int, offset int) ([]Chunk, error)
	UpdateChunkVectorRefs(ctx context.Context, updates []ChunkVectorUpdate) error
}

type LexicalSearchStore interface {
	SearchLexicalChunks(ctx context.Context, query string, limit int) ([]LexicalHit, error)
	RebuildLexicalIndex(ctx context.Context) error
	EnsureLexicalIndex(ctx context.Context) (bool, error)
}

type SQLiteStore struct {
	db        *sql.DB
	columnsMu sync.Mutex
	columns   map[string]map[string]bool
}

func Open(path string) (*SQLiteStore, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(`pragma foreign_keys = on`); err != nil {
		_ = db.Close()
		return nil, err
	}
	return &SQLiteStore{db: db, columns: make(map[string]map[string]bool)}, nil
}

func DocumentID(path string) string {
	sum := sha256.Sum256([]byte(path))
	return hex.EncodeToString(sum[:])
}

func ChunkID(documentID string, chunkIndex int, contentHash string) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s:%d:%s", documentID, chunkIndex, contentHash)))
	return hex.EncodeToString(sum[:])
}

var ftsTermPattern = regexp.MustCompile(`[A-Za-z0-9_./-]+`)

func BuildFTSQuery(question string) string {
	matches := ftsTermPattern.FindAllString(question, -1)
	seen := make(map[string]struct{})
	terms := make([]string, 0, len(matches))
	add := func(term string) {
		term = strings.Trim(term, "_-./")
		if !usableFTSTerm(term) {
			return
		}
		key := strings.ToLower(term)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		terms = append(terms, `"`+strings.ReplaceAll(term, `"`, `""`)+`"`)
	}
	for _, match := range matches {
		add(match)
		for _, part := range strings.FieldsFunc(match, func(r rune) bool {
			return r == '_' || r == '-' || r == '.' || r == '/'
		}) {
			add(part)
		}
	}
	return strings.Join(terms, " OR ")
}

func usableFTSTerm(term string) bool {
	if len(term) < 2 {
		return false
	}
	switch strings.ToLower(term) {
	case "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what", "where", "with":
		return false
	default:
		return true
	}
}

func (s *SQLiteStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *SQLiteStore) Init(ctx context.Context) error {
	defer s.clearColumnCache()
	if _, err := s.db.ExecContext(ctx, schemaSQL); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "documents", "deleted_at", "text"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "documents", "source_root", "text"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "documents", "chunker", "text not null default ''"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "documents", "chunk_size", "integer not null default 0"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "documents", "chunk_overlap", "integer not null default 0"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "documents", "index_text_version", "integer not null default 0"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "pending_documents", "source_root", "text"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "pending_documents", "chunker", "text not null default ''"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "pending_documents", "chunk_size", "integer not null default 0"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "pending_documents", "chunk_overlap", "integer not null default 0"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, s.db, "pending_documents", "index_text_version", "integer not null default 0"); err != nil {
		return err
	}
	for _, table := range []string{"chunks", "pending_chunks"} {
		if err := ensureColumn(ctx, s.db, table, "indexed_text", "text not null default ''"); err != nil {
			return err
		}
		if err := ensureColumn(ctx, s.db, table, "chunker", "text not null default ''"); err != nil {
			return err
		}
		if err := ensureColumn(ctx, s.db, table, "language", "text not null default ''"); err != nil {
			return err
		}
		if err := ensureColumn(ctx, s.db, table, "heading_path", "text not null default ''"); err != nil {
			return err
		}
		if err := ensureColumn(ctx, s.db, table, "symbol_name", "text not null default ''"); err != nil {
			return err
		}
		if err := ensureColumn(ctx, s.db, table, "symbol_kind", "text not null default ''"); err != nil {
			return err
		}
	}
	_, err := s.db.ExecContext(ctx, `create index if not exists documents_source_root on documents(source_root)`)
	return err
}

func (s *SQLiteStore) GetDocumentByPath(ctx context.Context, path string) (*Document, error) {
	selectSQL, err := s.documentSelectSQL(ctx)
	if err != nil {
		return nil, err
	}
	return scanDocument(s.db.QueryRowContext(ctx, selectSQL+` where path = ?`, path))
}

func (s *SQLiteStore) GetDocumentByID(ctx context.Context, id string) (*Document, error) {
	selectSQL, err := s.documentSelectSQL(ctx)
	if err != nil {
		return nil, err
	}
	return scanDocument(s.db.QueryRowContext(ctx, selectSQL+` where id = ?`, id))
}

func (s *SQLiteStore) UpsertDocument(ctx context.Context, doc Document) error {
	return upsertDocument(ctx, s.db, doc)
}

func (s *SQLiteStore) MarkDocumentError(ctx context.Context, path string, errText string) error {
	now := time.Now().UTC()
	existing, err := s.GetDocumentByPath(ctx, path)
	if err == nil && existing.Status == DocumentStatusIndexed {
		_, err := s.db.ExecContext(ctx, `update documents set error = ? where path = ?`, errText, path)
		return err
	}
	if err != nil && !errors.Is(err, ErrNotFound) {
		return err
	}
	doc := Document{
		ID:          DocumentID(path),
		Path:        path,
		ContentHash: "",
		SizeBytes:   0,
		ModifiedAt:  now,
		Status:      DocumentStatusError,
		Error:       &errText,
	}
	if existing != nil {
		doc.ID = existing.ID
		doc.ContentHash = existing.ContentHash
		doc.SizeBytes = existing.SizeBytes
		doc.ModifiedAt = existing.ModifiedAt
		doc.IndexedAt = existing.IndexedAt
		doc.DeletedAt = existing.DeletedAt
		doc.SourceRoot = existing.SourceRoot
		doc.Chunker = existing.Chunker
		doc.ChunkSize = existing.ChunkSize
		doc.ChunkOverlap = existing.ChunkOverlap
		doc.IndexTextVersion = existing.IndexTextVersion
	}
	return upsertDocument(ctx, s.db, doc)
}

func (s *SQLiteStore) ListIndexedDocumentsUnderRoot(ctx context.Context, sourceRoot string) ([]Document, error) {
	selectSQL, err := s.documentSelectSQL(ctx)
	if err != nil {
		return nil, err
	}
	sourceRootExpr, err := s.selectColumnExpr(ctx, "documents", "", "source_root", "''")
	if err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(ctx, selectSQL+` where `+sourceRootExpr+` = ? and status = ? order by path`, sourceRoot, DocumentStatusIndexed)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	docs := make([]Document, 0)
	for rows.Next() {
		doc, err := scanDocument(rows)
		if err != nil {
			return nil, err
		}
		docs = append(docs, *doc)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return docs, nil
}

func (s *SQLiteStore) ListIndexedDocuments(ctx context.Context) ([]Document, error) {
	selectSQL, err := s.documentSelectSQL(ctx)
	if err != nil {
		return nil, err
	}
	sourceRootExpr, err := s.selectColumnExpr(ctx, "documents", "", "source_root", "''")
	if err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(ctx, selectSQL+` where status = ? order by `+sourceRootExpr+`, path`, DocumentStatusIndexed)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	docs := make([]Document, 0)
	for rows.Next() {
		doc, err := scanDocument(rows)
		if err != nil {
			return nil, err
		}
		docs = append(docs, *doc)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return docs, nil
}

func (s *SQLiteStore) MarkDocumentDeleted(ctx context.Context, documentID string, deletedAt time.Time) error {
	return s.MarkDocumentsDeleted(ctx, []string{documentID}, deletedAt)
}

func (s *SQLiteStore) MarkDocumentsDeleted(ctx context.Context, documentIDs []string, deletedAt time.Time) error {
	if len(documentIDs) == 0 {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, documentID := range documentIDs {
		if err := markDocumentDeletedTx(ctx, tx, documentID, deletedAt); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (s *SQLiteStore) MarkChunksInactive(ctx context.Context, documentID string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx, `update chunks set active = 0 where document_id = ?`, documentID); err != nil {
		return err
	}
	if err := deleteFTSForDocument(ctx, tx, documentID); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *SQLiteStore) InsertChunk(ctx context.Context, chunk Chunk) error {
	return insertChunk(ctx, s.db, chunk)
}

func (s *SQLiteStore) ReplaceDocumentChunks(ctx context.Context, doc Document, chunks []Chunk) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if err := upsertDocument(ctx, tx, doc); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `update chunks set active = 0 where document_id = ?`, doc.ID); err != nil {
		return err
	}
	if err := deleteFTSForDocument(ctx, tx, doc.ID); err != nil {
		return err
	}
	for _, chunk := range chunks {
		if err := insertChunk(ctx, tx, chunk); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (s *SQLiteStore) StagePendingDocument(ctx context.Context, runID string, doc Document, chunks []Chunk) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.ExecContext(ctx, `delete from pending_chunks where run_id = ? and document_id = ?`, runID, doc.ID); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `delete from pending_documents where run_id = ? and document_id = ?`, runID, doc.ID); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `
insert into pending_documents(run_id, document_id, path, content_hash, size_bytes, modified_at, indexed_at, source_root, chunker, chunk_size, chunk_overlap, index_text_version, created_at)
values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
`, runID, doc.ID, doc.Path, doc.ContentHash, doc.SizeBytes, formatTime(doc.ModifiedAt), nullableTime(doc.IndexedAt), doc.SourceRoot, doc.Chunker, doc.ChunkSize, doc.ChunkOverlap, doc.IndexTextVersion, formatTime(time.Now().UTC())); err != nil {
		return err
	}
	for _, chunk := range chunks {
		headingPath, err := headingPathJSON(chunk.HeadingPath)
		if err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, `
insert into pending_chunks(run_id, id, document_id, chunk_index, content_hash, chunk_text, indexed_text, start_line, end_line, vector_id, embedding_model, created_at, chunker, language, heading_path, symbol_name, symbol_kind)
values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
`, runID, chunk.ID, chunk.DocumentID, chunk.ChunkIndex, chunk.ContentHash, chunk.ChunkText, chunk.IndexedText, chunk.StartLine, chunk.EndLine, nullableInt64(chunk.VectorID), chunk.EmbeddingModel, formatTime(chunk.CreatedAt), chunk.Chunker, chunk.Language, headingPath, chunk.SymbolName, chunk.SymbolKind); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (s *SQLiteStore) ActivatePendingRun(ctx context.Context, runID string) error {
	sourceRootExpr, err := s.selectColumnExpr(ctx, "pending_documents", "", "source_root", "''")
	if err != nil {
		return err
	}
	chunkerExpr, err := s.selectColumnExpr(ctx, "pending_documents", "", "chunker", "''")
	if err != nil {
		return err
	}
	chunkSizeExpr, err := s.selectColumnExpr(ctx, "pending_documents", "", "chunk_size", "0")
	if err != nil {
		return err
	}
	chunkOverlapExpr, err := s.selectColumnExpr(ctx, "pending_documents", "", "chunk_overlap", "0")
	if err != nil {
		return err
	}
	indexTextVersionExpr, err := s.selectColumnExpr(ctx, "pending_documents", "", "index_text_version", "0")
	if err != nil {
		return err
	}
	pendingIndexedTextExpr, err := s.selectColumnExpr(ctx, "pending_chunks", "", "indexed_text", "''")
	if err != nil {
		return err
	}
	pendingChunkerExpr, err := s.selectColumnExpr(ctx, "pending_chunks", "", "chunker", "''")
	if err != nil {
		return err
	}
	pendingLanguageExpr, err := s.selectColumnExpr(ctx, "pending_chunks", "", "language", "''")
	if err != nil {
		return err
	}
	pendingHeadingPathExpr, err := s.selectColumnExpr(ctx, "pending_chunks", "", "heading_path", "''")
	if err != nil {
		return err
	}
	pendingSymbolNameExpr, err := s.selectColumnExpr(ctx, "pending_chunks", "", "symbol_name", "''")
	if err != nil {
		return err
	}
	pendingSymbolKindExpr, err := s.selectColumnExpr(ctx, "pending_chunks", "", "symbol_kind", "''")
	if err != nil {
		return err
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	rows, err := tx.QueryContext(ctx, `select document_id, path, content_hash, size_bytes, modified_at, indexed_at, `+sourceRootExpr+`, `+chunkerExpr+`, `+chunkSizeExpr+`, `+chunkOverlapExpr+`, `+indexTextVersionExpr+` from pending_documents where run_id = ? order by path`, runID)
	if err != nil {
		return err
	}
	var docs []Document
	for rows.Next() {
		var doc Document
		var modifiedAt string
		var indexedAt string
		if err := rows.Scan(&doc.ID, &doc.Path, &doc.ContentHash, &doc.SizeBytes, &modifiedAt, &indexedAt, &doc.SourceRoot, &doc.Chunker, &doc.ChunkSize, &doc.ChunkOverlap, &doc.IndexTextVersion); err != nil {
			rows.Close()
			return err
		}
		t, err := parseTime(modifiedAt)
		if err != nil {
			rows.Close()
			return err
		}
		doc.ModifiedAt = t
		indexed, err := parseTime(indexedAt)
		if err != nil {
			rows.Close()
			return err
		}
		doc.IndexedAt = &indexed
		doc.Status = DocumentStatusIndexed
		docs = append(docs, doc)
	}
	if err := rows.Close(); err != nil {
		return err
	}
	if err := rows.Err(); err != nil {
		return err
	}

	for _, doc := range docs {
		if err := upsertDocument(ctx, tx, doc); err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, `update chunks set active = 0 where document_id = ?`, doc.ID); err != nil {
			return err
		}
		if err := deleteFTSForDocument(ctx, tx, doc.ID); err != nil {
			return err
		}
		chunkRows, err := tx.QueryContext(ctx, `
select id, document_id, chunk_index, content_hash, chunk_text, `+pendingIndexedTextExpr+`, start_line, end_line, vector_id, embedding_model, 1, created_at, `+pendingChunkerExpr+`, `+pendingLanguageExpr+`, `+pendingHeadingPathExpr+`, `+pendingSymbolNameExpr+`, `+pendingSymbolKindExpr+`
from pending_chunks
where run_id = ? and document_id = ?
order by chunk_index
`, runID, doc.ID)
		if err != nil {
			return err
		}
		for chunkRows.Next() {
			chunk, err := scanChunk(chunkRows)
			if err != nil {
				chunkRows.Close()
				return err
			}
			if err := insertChunk(ctx, tx, *chunk); err != nil {
				chunkRows.Close()
				return err
			}
		}
		if err := chunkRows.Close(); err != nil {
			return err
		}
		if err := chunkRows.Err(); err != nil {
			return err
		}
	}
	if _, err := tx.ExecContext(ctx, `delete from pending_documents where run_id = ?`, runID); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *SQLiteStore) ListPendingRuns(ctx context.Context) ([]PendingRun, error) {
	rows, err := s.db.QueryContext(ctx, `
select pd.run_id, count(distinct pd.document_id), count(pc.id), min(pd.created_at)
from pending_documents pd
left join pending_chunks pc on pc.run_id = pd.run_id and pc.document_id = pd.document_id
group by pd.run_id
order by min(pd.created_at), pd.run_id
`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	runs := make([]PendingRun, 0)
	for rows.Next() {
		var run PendingRun
		var createdAt string
		if err := rows.Scan(&run.ID, &run.DocumentCount, &run.ChunkCount, &createdAt); err != nil {
			return nil, err
		}
		t, err := parseTime(createdAt)
		if err != nil {
			return nil, err
		}
		run.CreatedAt = t
		runs = append(runs, run)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return runs, nil
}

func (s *SQLiteStore) ClearPendingRun(ctx context.Context, runID string) error {
	if _, err := s.db.ExecContext(ctx, `delete from pending_chunks where run_id = ?`, runID); err != nil {
		return err
	}
	_, err := s.db.ExecContext(ctx, `delete from pending_documents where run_id = ?`, runID)
	return err
}

func (s *SQLiteStore) GetActiveChunksByIDs(ctx context.Context, ids []string) ([]Chunk, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	chunks := make([]Chunk, 0, len(ids))
	selectSQL, err := s.chunkSelectSQL(ctx, "c")
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		row := s.db.QueryRowContext(ctx, `
`+selectSQL+`
from chunks c
join documents d on d.id = c.document_id
where c.id = ? and c.active = 1 and d.status = ?
`, id, DocumentStatusIndexed)
		chunk, err := scanChunk(row)
		if errors.Is(err, ErrNotFound) {
			continue
		}
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, *chunk)
	}
	return chunks, nil
}

func (s *SQLiteStore) SearchLexicalChunks(ctx context.Context, query string, limit int) ([]LexicalHit, error) {
	if limit <= 0 {
		return nil, nil
	}
	ftsQuery := BuildFTSQuery(query)
	if ftsQuery == "" {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx, `
select chunk_fts.chunk_id, bm25(chunk_fts) as score
from chunk_fts
join chunks c on c.id = chunk_fts.chunk_id
join documents d on d.id = c.document_id
where chunk_fts match ?
  and c.active = 1
  and d.status = ?
order by score asc, chunk_fts.chunk_id asc
limit ?
`, ftsQuery, DocumentStatusIndexed, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	hits := make([]LexicalHit, 0)
	for rows.Next() {
		var hit LexicalHit
		if err := rows.Scan(&hit.ChunkID, &hit.Score); err != nil {
			return nil, err
		}
		hits = append(hits, hit)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return hits, nil
}

func (s *SQLiteStore) RebuildLexicalIndex(ctx context.Context) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx, `delete from chunk_fts`); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `
insert into chunk_fts(chunk_id, document_id, path, chunk_text)
select c.id, c.document_id, d.path, coalesce(nullif(c.indexed_text, ''), c.chunk_text)
from chunks c
join documents d on d.id = c.document_id
where c.active = 1 and d.status = ?
`, DocumentStatusIndexed); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *SQLiteStore) EnsureLexicalIndex(ctx context.Context) (bool, error) {
	var activeChunks int
	if err := s.db.QueryRowContext(ctx, `
select count(*)
from chunks c
join documents d on d.id = c.document_id
where c.active = 1 and d.status = ?
`, DocumentStatusIndexed).Scan(&activeChunks); err != nil {
		return false, err
	}
	if activeChunks == 0 {
		return false, nil
	}
	var indexedChunks int
	if err := s.db.QueryRowContext(ctx, `select count(*) from chunk_fts`).Scan(&indexedChunks); err != nil {
		return false, err
	}
	if indexedChunks != 0 {
		return false, nil
	}
	if err := s.RebuildLexicalIndex(ctx); err != nil {
		return false, err
	}
	return true, nil
}

func (s *SQLiteStore) ListActiveIndexedChunks(ctx context.Context, limit int, offset int) ([]Chunk, error) {
	if limit <= 0 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}
	selectSQL, err := s.chunkSelectSQL(ctx, "c")
	if err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(ctx, `
`+selectSQL+`
from chunks c
join documents d on d.id = c.document_id
where c.active = 1 and d.status = ?
order by d.path, c.chunk_index
limit ? offset ?
`, DocumentStatusIndexed, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	chunks := make([]Chunk, 0)
	for rows.Next() {
		chunk, err := scanChunk(rows)
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, *chunk)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return chunks, nil
}

func (s *SQLiteStore) UpdateChunkVectorRefs(ctx context.Context, updates []ChunkVectorUpdate) error {
	if len(updates) == 0 {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, update := range updates {
		result, err := tx.ExecContext(ctx, `
update chunks
set vector_id = ?, embedding_model = ?
where id = ? and active = 1
`, update.VectorID, update.EmbeddingModel, update.ChunkID)
		if err != nil {
			return err
		}
		affected, err := result.RowsAffected()
		if err != nil {
			return fmt.Errorf("check vector ref update for chunk %s: %w", update.ChunkID, err)
		}
		if affected != 1 {
			return fmt.Errorf("update vector ref for chunk %s: affected %d rows, want 1", update.ChunkID, affected)
		}
	}
	return tx.Commit()
}

func (s *SQLiteStore) Stats(ctx context.Context) (Stats, error) {
	var stats Stats
	if err := s.db.QueryRowContext(ctx, `select count(*) from documents where status = ?`, DocumentStatusIndexed).Scan(&stats.IndexedDocuments); err != nil {
		return Stats{}, err
	}
	if err := s.db.QueryRowContext(ctx, `select count(*) from documents where status = ?`, DocumentStatusError).Scan(&stats.ErrorDocuments); err != nil {
		return Stats{}, err
	}
	if err := s.db.QueryRowContext(ctx, `select count(*) from documents where status = ?`, DocumentStatusDeleted).Scan(&stats.DeletedDocuments); err != nil {
		return Stats{}, err
	}
	if err := s.db.QueryRowContext(ctx, `select count(*) from chunks where active = 1`).Scan(&stats.ActiveChunks); err != nil {
		return Stats{}, err
	}
	if err := s.db.QueryRowContext(ctx, `select count(*) from chunks where active = 0`).Scan(&stats.InactiveChunks); err != nil {
		return Stats{}, err
	}
	var last sql.NullString
	if err := s.db.QueryRowContext(ctx, `select max(indexed_at) from documents where indexed_at is not null`).Scan(&last); err != nil {
		return Stats{}, err
	}
	if last.Valid && strings.TrimSpace(last.String) != "" {
		t, err := parseTime(last.String)
		if err != nil {
			return Stats{}, err
		}
		stats.LastIndexedAt = &t
	}
	return stats, nil
}

type rowScanner interface {
	Scan(dest ...any) error
}

type queryer interface {
	ExecContext(context.Context, string, ...any) (sql.Result, error)
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func (s *SQLiteStore) documentSelectSQL(ctx context.Context) (string, error) {
	deletedAt, err := s.selectColumnExpr(ctx, "documents", "", "deleted_at", "''")
	if err != nil {
		return "", err
	}
	sourceRoot, err := s.selectColumnExpr(ctx, "documents", "", "source_root", "''")
	if err != nil {
		return "", err
	}
	chunker, err := s.selectColumnExpr(ctx, "documents", "", "chunker", "''")
	if err != nil {
		return "", err
	}
	chunkSize, err := s.selectColumnExpr(ctx, "documents", "", "chunk_size", "0")
	if err != nil {
		return "", err
	}
	chunkOverlap, err := s.selectColumnExpr(ctx, "documents", "", "chunk_overlap", "0")
	if err != nil {
		return "", err
	}
	indexTextVersion, err := s.selectColumnExpr(ctx, "documents", "", "index_text_version", "0")
	if err != nil {
		return "", err
	}
	return `select id, path, content_hash, size_bytes, modified_at, indexed_at, ` + deletedAt + `, ` + sourceRoot + `, ` + chunker + `, ` + chunkSize + `, ` + chunkOverlap + `, ` + indexTextVersion + `, status, error from documents`, nil
}

func (s *SQLiteStore) chunkSelectSQL(ctx context.Context, alias string) (string, error) {
	indexedText, err := s.selectColumnExpr(ctx, "chunks", alias, "indexed_text", "''")
	if err != nil {
		return "", err
	}
	chunker, err := s.selectColumnExpr(ctx, "chunks", alias, "chunker", "''")
	if err != nil {
		return "", err
	}
	language, err := s.selectColumnExpr(ctx, "chunks", alias, "language", "''")
	if err != nil {
		return "", err
	}
	headingPath, err := s.selectColumnExpr(ctx, "chunks", alias, "heading_path", "''")
	if err != nil {
		return "", err
	}
	symbolName, err := s.selectColumnExpr(ctx, "chunks", alias, "symbol_name", "''")
	if err != nil {
		return "", err
	}
	symbolKind, err := s.selectColumnExpr(ctx, "chunks", alias, "symbol_kind", "''")
	if err != nil {
		return "", err
	}
	return `select ` + qualifiedColumn(alias, "id") + `, ` + qualifiedColumn(alias, "document_id") + `, ` + qualifiedColumn(alias, "chunk_index") + `, ` + qualifiedColumn(alias, "content_hash") + `, ` + qualifiedColumn(alias, "chunk_text") + `, ` + indexedText + `, ` + qualifiedColumn(alias, "start_line") + `, ` + qualifiedColumn(alias, "end_line") + `, ` + qualifiedColumn(alias, "vector_id") + `, ` + qualifiedColumn(alias, "embedding_model") + `, ` + qualifiedColumn(alias, "active") + `, ` + qualifiedColumn(alias, "created_at") + `, ` + chunker + `, ` + language + `, ` + headingPath + `, ` + symbolName + `, ` + symbolKind, nil
}

func (s *SQLiteStore) selectColumnExpr(ctx context.Context, table string, alias string, column string, defaultExpr string) (string, error) {
	exists, err := s.hasColumn(ctx, table, column)
	if err != nil {
		return "", err
	}
	if !exists {
		return defaultExpr, nil
	}
	return `coalesce(` + qualifiedColumn(alias, column) + `, ` + defaultExpr + `)`, nil
}

func (s *SQLiteStore) hasColumn(ctx context.Context, table string, column string) (bool, error) {
	columns, err := s.tableColumns(ctx, table)
	if err != nil {
		return false, err
	}
	return columns[column], nil
}

func (s *SQLiteStore) tableColumns(ctx context.Context, table string) (map[string]bool, error) {
	s.columnsMu.Lock()
	if columns, ok := s.columns[table]; ok {
		s.columnsMu.Unlock()
		return columns, nil
	}
	s.columnsMu.Unlock()

	rows, err := s.db.QueryContext(ctx, `pragma table_info(`+table+`)`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	columns := make(map[string]bool)
	for rows.Next() {
		var cid int
		var name string
		var columnType string
		var notNull int
		var defaultValue sql.NullString
		var primaryKey int
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			return nil, err
		}
		columns[name] = true
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	s.columnsMu.Lock()
	s.columns[table] = columns
	s.columnsMu.Unlock()
	return columns, nil
}

func (s *SQLiteStore) clearColumnCache() {
	s.columnsMu.Lock()
	defer s.columnsMu.Unlock()
	s.columns = make(map[string]map[string]bool)
}

func qualifiedColumn(alias string, column string) string {
	if alias == "" {
		return column
	}
	return alias + "." + column
}

func scanDocument(row rowScanner) (*Document, error) {
	var doc Document
	var modifiedAt string
	var indexedAt sql.NullString
	var deletedAt sql.NullString
	var errText sql.NullString
	if err := row.Scan(&doc.ID, &doc.Path, &doc.ContentHash, &doc.SizeBytes, &modifiedAt, &indexedAt, &deletedAt, &doc.SourceRoot, &doc.Chunker, &doc.ChunkSize, &doc.ChunkOverlap, &doc.IndexTextVersion, &doc.Status, &errText); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	t, err := parseTime(modifiedAt)
	if err != nil {
		return nil, err
	}
	doc.ModifiedAt = t
	if indexedAt.Valid && indexedAt.String != "" {
		t, err := parseTime(indexedAt.String)
		if err != nil {
			return nil, err
		}
		doc.IndexedAt = &t
	}
	if deletedAt.Valid && deletedAt.String != "" {
		t, err := parseTime(deletedAt.String)
		if err != nil {
			return nil, err
		}
		doc.DeletedAt = &t
	}
	if errText.Valid {
		doc.Error = &errText.String
	}
	return &doc, nil
}

func scanChunk(row rowScanner) (*Chunk, error) {
	var chunk Chunk
	var vectorID sql.NullInt64
	var active int
	var createdAt string
	var headingPath string
	if err := row.Scan(&chunk.ID, &chunk.DocumentID, &chunk.ChunkIndex, &chunk.ContentHash, &chunk.ChunkText, &chunk.IndexedText, &chunk.StartLine, &chunk.EndLine, &vectorID, &chunk.EmbeddingModel, &active, &createdAt, &chunk.Chunker, &chunk.Language, &headingPath, &chunk.SymbolName, &chunk.SymbolKind); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	if headingPath != "" {
		if err := json.Unmarshal([]byte(headingPath), &chunk.HeadingPath); err != nil {
			return nil, err
		}
	}
	if vectorID.Valid {
		chunk.VectorID = &vectorID.Int64
	}
	chunk.Active = active != 0
	t, err := parseTime(createdAt)
	if err != nil {
		return nil, err
	}
	chunk.CreatedAt = t
	return &chunk, nil
}

func upsertDocument(ctx context.Context, q queryer, doc Document) error {
	if doc.ID == "" {
		doc.ID = DocumentID(doc.Path)
	}
	_, err := q.ExecContext(ctx, `
insert into documents(id, path, content_hash, size_bytes, modified_at, indexed_at, deleted_at, source_root, chunker, chunk_size, chunk_overlap, index_text_version, status, error)
values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
on conflict(path) do update set
  id = excluded.id,
  content_hash = excluded.content_hash,
  size_bytes = excluded.size_bytes,
  modified_at = excluded.modified_at,
  indexed_at = excluded.indexed_at,
  deleted_at = excluded.deleted_at,
  source_root = excluded.source_root,
  chunker = excluded.chunker,
  chunk_size = excluded.chunk_size,
  chunk_overlap = excluded.chunk_overlap,
  index_text_version = excluded.index_text_version,
  status = excluded.status,
  error = excluded.error
`, doc.ID, doc.Path, doc.ContentHash, doc.SizeBytes, formatTime(doc.ModifiedAt), nullableTime(doc.IndexedAt), nullableTime(doc.DeletedAt), doc.SourceRoot, doc.Chunker, doc.ChunkSize, doc.ChunkOverlap, doc.IndexTextVersion, doc.Status, nullableString(doc.Error))
	return err
}

func markDocumentDeletedTx(ctx context.Context, tx *sql.Tx, documentID string, deletedAt time.Time) error {
	if _, err := tx.ExecContext(ctx, `
update documents
set status = ?, indexed_at = null, deleted_at = ?, error = null
where id = ?
`, DocumentStatusDeleted, formatTime(deletedAt), documentID); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `update chunks set active = 0 where document_id = ?`, documentID); err != nil {
		return err
	}
	return deleteFTSForDocument(ctx, tx, documentID)
}

func insertChunk(ctx context.Context, q queryer, chunk Chunk) error {
	active := 0
	if chunk.Active {
		active = 1
	}
	var existingID string
	err := q.QueryRowContext(ctx, `select id from chunks where document_id = ? and chunk_index = ?`, chunk.DocumentID, chunk.ChunkIndex).Scan(&existingID)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return err
	}
	if existingID != "" {
		if err := deleteFTSForChunk(ctx, q, existingID); err != nil {
			return err
		}
	}
	if err := deleteFTSForChunk(ctx, q, chunk.ID); err != nil {
		return err
	}
	headingPath, err := headingPathJSON(chunk.HeadingPath)
	if err != nil {
		return err
	}
	_, err = q.ExecContext(ctx, `
insert into chunks(id, document_id, chunk_index, content_hash, chunk_text, indexed_text, start_line, end_line, vector_id, embedding_model, active, created_at, chunker, language, heading_path, symbol_name, symbol_kind)
values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
on conflict(document_id, chunk_index) do update set
  id = excluded.id,
  content_hash = excluded.content_hash,
  chunk_text = excluded.chunk_text,
  indexed_text = excluded.indexed_text,
  start_line = excluded.start_line,
  end_line = excluded.end_line,
  vector_id = excluded.vector_id,
  embedding_model = excluded.embedding_model,
  active = excluded.active,
  created_at = excluded.created_at,
  chunker = excluded.chunker,
  language = excluded.language,
  heading_path = excluded.heading_path,
  symbol_name = excluded.symbol_name,
  symbol_kind = excluded.symbol_kind
`, chunk.ID, chunk.DocumentID, chunk.ChunkIndex, chunk.ContentHash, chunk.ChunkText, chunk.IndexedText, chunk.StartLine, chunk.EndLine, nullableInt64(chunk.VectorID), chunk.EmbeddingModel, active, formatTime(chunk.CreatedAt), chunk.Chunker, chunk.Language, headingPath, chunk.SymbolName, chunk.SymbolKind)
	if err != nil {
		return err
	}
	if chunk.Active {
		if err := insertFTSForChunk(ctx, q, chunk.ID); err != nil {
			return err
		}
	}
	return nil
}

func deleteFTSForDocument(ctx context.Context, q queryer, documentID string) error {
	_, err := q.ExecContext(ctx, `delete from chunk_fts where document_id = ?`, documentID)
	return err
}

func deleteFTSForChunk(ctx context.Context, q queryer, chunkID string) error {
	if chunkID == "" {
		return nil
	}
	_, err := q.ExecContext(ctx, `delete from chunk_fts where chunk_id = ?`, chunkID)
	return err
}

func insertFTSForChunk(ctx context.Context, q queryer, chunkID string) error {
	_, err := q.ExecContext(ctx, `
insert into chunk_fts(chunk_id, document_id, path, chunk_text)
select c.id, c.document_id, d.path, coalesce(nullif(c.indexed_text, ''), c.chunk_text)
from chunks c
join documents d on d.id = c.document_id
where c.id = ? and c.active = 1 and d.status = ?
`, chunkID, DocumentStatusIndexed)
	return err
}

func formatTime(t time.Time) string {
	return t.UTC().Format(time.RFC3339Nano)
}

func parseTime(value string) (time.Time, error) {
	return time.Parse(time.RFC3339Nano, value)
}

func nullableTime(t *time.Time) any {
	if t == nil {
		return nil
	}
	return formatTime(*t)
}

func nullableString(value *string) any {
	if value == nil {
		return nil
	}
	return *value
}

func nullableInt64(value *int64) any {
	if value == nil {
		return nil
	}
	return *value
}

func headingPathJSON(path []string) (string, error) {
	if len(path) == 0 {
		return "", nil
	}
	bytes, err := json.Marshal(path)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

func ensureColumn(ctx context.Context, db *sql.DB, table string, column string, columnType string) error {
	rows, err := db.QueryContext(ctx, `pragma table_info(`+table+`)`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var cid int
		var name string
		var typ string
		var notNull int
		var defaultValue any
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notNull, &defaultValue, &pk); err != nil {
			return err
		}
		if name == column {
			return rows.Err()
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}
	_, err = db.ExecContext(ctx, fmt.Sprintf("alter table %s add column %s %s", table, column, columnType))
	return err
}
