create table if not exists public.sales_collection (
  id uuid primary key default gen_random_uuid(),
  content text,
  metadata jsonb,
  embedding vector(3072)
);

-- If the table existed with a different dimension, force the column to 3072
do $$
begin
  if exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'sales_collection'
      and column_name = 'embedding'
  ) then
    begin
      alter table public.sales_collection
      alter column embedding type vector(3072);
    exception
      when others then null; -- ignore if already 3072 or incompatible without manual steps
    end;
  end if;
end$$;

-- FAQ table
create table if not exists public.faq_collection (
  id uuid primary key default gen_random_uuid(),
  content text,
  metadata jsonb,
  embedding vector(3072)
);

do $$
begin
  if exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'faq_collection'
      and column_name = 'embedding'
  ) then
    begin
      alter table public.faq_collection
      alter column embedding type vector(3072);
    exception
      when others then null;
    end;
  end if;
end$$;


-- 3) Generic match function (cosine similarity; similarity = 1 - distance)
create or replace function public.match_documents(
  query_embedding vector(3072),
  match_count int default 8,
  similarity_threshold float default 0.0,
  table_name text default 'sales_collection'
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
declare
  sql text;
begin
  sql := format($f$
    select
      id,
      content,
      metadata,
      1 - (embedding <=> $1) as similarity
    from %I
    where (1 - (embedding <=> $1)) >= $3
    order by embedding <=> $1
    limit $2
  $f$, table_name);

  return query execute sql using query_embedding, match_count, similarity_threshold;
end;
$$;

-- 4) Per-collection wrappers (optional but convenient)
create or replace function public.match_documents_sales(
  query_embedding vector(3072),
  match_count int default 8,
  similarity_threshold float default 0.0
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql
as $$
  select id, content, metadata, similarity
  from public.match_documents(query_embedding, match_count, similarity_threshold, 'sales_collection');
$$;

create or replace function public.match_documents_faq(
  query_embedding vector(3072),
  match_count int default 8,
  similarity_threshold float default 0.0
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql
as $$
  select id, content, metadata, similarity
  from public.match_documents(query_embedding, match_count, similarity_threshold, 'faq_collection');
$$;
