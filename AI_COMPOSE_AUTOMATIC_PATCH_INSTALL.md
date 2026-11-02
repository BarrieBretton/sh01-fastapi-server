# Apply the automatic-editor patch

Prepared against `feature/ai-compose-kaggle` at
`3aa0007024f364ca16ef927b416a8d286b0366da`. GitHub rejected the integration's
write request with HTTP 403. This bundle has **not** been pushed or merged.

The ZIP contains repository-relative paths. Preserve any local modifications to
these files before extracting over your repository root. An accompanying `.patch`
is available if you prefer `git apply --check` followed by `git apply`, which
refuses conflicting changes instead of overwriting them.

In PowerShell, from your repository root, after extraction/application:

```powershell
python -m py_compile ai_compose.py ai_compose_engine.py ai_compose_schema.py ai_compose_ir.py ai_compose_editor.py ai_compose_motion.py ai_compose_sound.py ai_compose_vision.py ai_compose_local_store.py ai_compose_local_app.py
python -m unittest discover -s tests -p "test_ai_compose*.py" -v
git add -- AI_COMPOSE_CATALOG_COVERAGE.md AI_COMPOSE_README.md AUTOMATIC_EDITOR_README.md AI_COMPOSE_AUTOMATIC_PATCH_INSTALL.md ai_compose.py ai_compose_engine.py ai_compose_ir.py ai_compose_schema.py ai_compose_editor.py ai_compose_local_app.py ai_compose_local_store.py ai_compose_motion.py ai_compose_sound.py ai_compose_vision.py requirements-ai-compose.txt docs/PROJECT_HANDBOOK.md notebooks/sh01_automatic_editor.ipynb tests/test_ai_compose_automatic_editor.py scripts/build_editor_notebook.py scripts/ai_compose_showcase.py
git diff --cached --check
git commit -m "feat: add automatic editor design and visual review workflow"
git push origin feature/ai-compose-kaggle
```

Do not add `.env`, generated videos, provider keys or unrelated untracked files.
Offline tests require Python/Pydantic and FFmpeg for render cases; they do not
download Hugging Face models. If FFmpeg is absent, render tests are skipped, so
check the test summary rather than treating skipped tests as verification.

Import `notebooks/sh01_automatic_editor.ipynb` into Kaggle after pushing. Run its
15 cells in order. It preserves `.env`, starts a nonblocking server, and performs
one integrated edit. The default requires local VLM review and enables automatic
design, captions, camera and SFX; B-roll needs keys or a mounted asset library.
Read `AUTOMATIC_EDITOR_README.md` for controls and implementation boundaries.

Development verification: 23 AI Compose tests passed, including real CPU FFmpeg
renders. Notebook Python/Bash syntax and JSON structure checked. Kaggle GPU,
Piper, live providers and HTTP startup require testing in the target environment.
No production deployment or merge to `main` has been performed.
