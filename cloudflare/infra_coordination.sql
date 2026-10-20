-- Apply to MEDIA_QUEUE_DB once. Separate from jobs and router KV.
CREATE TABLE IF NOT EXISTS infra_coordination (
  name TEXT PRIMARY KEY CHECK (name = 'global'),
  generation INTEGER NOT NULL DEFAULT 0,
  owner TEXT,
  token TEXT,
  lease_until INTEGER NOT NULL DEFAULT 0,
  active_render TEXT NOT NULL,
  active_postgres TEXT NOT NULL,
  active_b2 TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
-- Seed ONE TIME after confirming the live routers and actual active database.
-- INSERT INTO infra_coordination(name,active_render,active_postgres,active_b2,updated_at)
-- VALUES('global','<current-render>','<current-postgres>','<current-b2>',unixepoch()*1000);
