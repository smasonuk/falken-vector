package manifest

const schemaSQL = `
create table if not exists schema_migrations (
  version integer primary key,
  applied_at text not null
);

create table if not exists documents (
  id text primary key,
  path text not null unique,
  content_hash text not null,
  size_bytes integer not null,
  modified_at text not null,
  indexed_at text,
  deleted_at text,
  source_root text,
  chunker text not null default '',
  chunk_size integer not null default 0,
  chunk_overlap integer not null default 0,
  index_text_version integer not null default 0,
  status text not null,
  error text
);

create table if not exists chunks (
  id text primary key,
  document_id text not null,
  chunk_index integer not null,
  content_hash text not null,
  chunk_text text not null,
  indexed_text text not null default '',
  start_line integer,
  end_line integer,
  vector_id integer,
  embedding_model text not null,
  active integer not null default 1,
  created_at text not null,
  chunker text not null default '',
  language text not null default '',
  heading_path text not null default '',
  symbol_name text not null default '',
  symbol_kind text not null default '',
  foreign key(document_id) references documents(id)
);

create table if not exists pending_documents (
  run_id text not null,
  document_id text not null,
  path text not null,
  content_hash text not null,
  size_bytes integer not null,
  modified_at text not null,
  indexed_at text not null,
  source_root text not null,
  chunker text not null default '',
  chunk_size integer not null default 0,
  chunk_overlap integer not null default 0,
  index_text_version integer not null default 0,
  created_at text not null,
  primary key(run_id, document_id)
);

create table if not exists pending_chunks (
  run_id text not null,
  id text not null,
  document_id text not null,
  chunk_index integer not null,
  content_hash text not null,
  chunk_text text not null,
  indexed_text text not null default '',
  start_line integer,
  end_line integer,
  vector_id integer,
  embedding_model text not null,
  created_at text not null,
  chunker text not null default '',
  language text not null default '',
  heading_path text not null default '',
  symbol_name text not null default '',
  symbol_kind text not null default '',
  primary key(run_id, id),
  foreign key(run_id, document_id) references pending_documents(run_id, document_id) on delete cascade
);

create virtual table if not exists chunk_fts using fts5(
  chunk_id unindexed,
  document_id unindexed,
  path,
  chunk_text,
  tokenize = 'unicode61'
);

create unique index if not exists chunks_document_index
on chunks(document_id, chunk_index);

create index if not exists chunks_vector_id
on chunks(vector_id);

create index if not exists chunks_active
on chunks(active);

create index if not exists documents_status
on documents(status);

create index if not exists pending_chunks_run_document
on pending_chunks(run_id, document_id);

insert or ignore into schema_migrations(version, applied_at) values (1, strftime('%Y-%m-%dT%H:%M:%fZ','now'));
`
