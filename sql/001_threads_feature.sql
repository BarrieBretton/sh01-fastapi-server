create table if not exists public.threads_auth (
    account_key text primary key,
    threads_user_id text not null,
    access_token_encrypted text not null,
    expires_at timestamptz not null,
    refreshed_at timestamptz,
    verified_at timestamptz,
    last_refresh_error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.threads_publish_jobs (
    id bigserial primary key,
    account_key text not null,
    idempotency_key text not null,
    media_type text not null check (media_type in ('IMAGE', 'VIDEO')),
    media_url text not null,
    text_body text not null default '',
    container_id text,
    post_id text,
    permalink text,
    status text not null,
    last_error text,
    published_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (account_key, idempotency_key)
);

create index if not exists threads_publish_jobs_status_idx
    on public.threads_publish_jobs (status, updated_at);

alter table public.threads_auth enable row level security;
alter table public.threads_publish_jobs enable row level security;
