-- Threads token refresh observability and scheduler hardening.
-- Additive/idempotent: safe to run before deploying the matching application code.

alter table public.threads_auth
    add column if not exists last_refresh_attempt_at timestamptz,
    add column if not exists last_refresh_success_at timestamptz,
    add column if not exists next_refresh_after timestamptz,
    add column if not exists consecutive_refresh_failures integer not null default 0,
    add column if not exists refresh_error_code text,
    add column if not exists refresh_error_status integer;

update public.threads_auth
   set last_refresh_success_at = coalesce(last_refresh_success_at, refreshed_at),
       next_refresh_after = coalesce(next_refresh_after, expires_at - interval '21 days'),
       consecutive_refresh_failures = 0,
       last_refresh_error = case
           when expires_at > now() + interval '21 days' then null
           else last_refresh_error
       end,
       refresh_error_code = case
           when expires_at > now() + interval '21 days' then null
           else refresh_error_code
       end,
       refresh_error_status = case
           when expires_at > now() + interval '21 days' then null
           else refresh_error_status
       end;

create index if not exists threads_auth_expiry_idx
    on public.threads_auth (expires_at);
