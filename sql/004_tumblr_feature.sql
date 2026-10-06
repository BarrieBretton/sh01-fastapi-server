create table if not exists public.tumblr_publish_jobs (
    id bigserial primary key,
    account_key text not null,
    idempotency_key text not null,
    media_type text not null check (media_type in ('IMAGE', 'VIDEO')),
    media_url text not null,
    text_body text not null default '',
    state text not null default 'published' check (state in ('published', 'draft', 'queue', 'private')),
    post_id text,
    permalink text,
    status text not null,
    last_error text,
    published_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (account_key, idempotency_key)
);
create index if not exists tumblr_publish_jobs_status_idx on public.tumblr_publish_jobs (status, updated_at);
alter table public.tumblr_publish_jobs enable row level security;
