# Repository onboarding for coding assistants

Read `docs/PROJECT_HANDBOOK.md` first, then `README.md`, the relevant source modules, and the linked subsystem documentation. This repository is only part of the deployment: n8n workflow graphs, Cloudflare dashboard-managed infra-control/D1 configuration, Render environment and external monitors require their own inventories.

When reporting deployment behavior, separate **code-confirmed**, **operator-observed**, **operator-reported**, and **unverified** facts. Verify current `main`, active registry/router state and environment before claiming a Render/Supabase/B2 slot is current. Never infer current health from historical READMEs, Render exports or a single proxy response.

API keys have different scopes: `CONTROL_PLANE_API_KEY` for `/infra/*`, `MEDIA_JOB_STORE_API_KEY` for the D1 queue, `MEDIA_CLIP_API_KEY` for media editing, and router-specific `ROUTER_CONTROL_KEY` for `/__control/*`. Never print or commit their values, browser cookies, credential exports, or signed URLs.

For local and Kaggle testing, the working sample notebook is `notebooks/sh01_kaggle_branding_endpoint_test.ipynb`. Its `source_path` is accessible to the Kaggle consumer only. For Render/n8n use a reachable `source_url` or `telegram_file_id`; use `unlimited_resources=false` on small Render instances. An endpoint's availability is not proof that the complete media job fits in memory.

Keep `docs/PROJECT_HANDBOOK.md` current in the same change as code, workflows, bindings or schedules. Do not present unknown external UI settings as implemented source. The handbook contains the evidence register and a checklist for a new AI or engineer.
