# Agentic AI-compose refinement patch

This is a delta against the already-installed feature/ai-compose-kaggle branch. It does not replace or modify the existing caption endpoints.

Included files:

- ai_compose_schema.py
- ai_compose_engine.py
- ai_compose_refinement.py

The patch adds:

- Transformers 5.x BatchEncoding compatibility;
- a bounded draft -> quality review -> model revision -> rerender loop;
- repeated asset resolution after each revised plan;
- output probing and FFmpeg decode checks;
- refinement_iterations in the request schema, defaulting to 2.

Copy these three Python files over the corresponding files in the repository root. Keep the existing app.py and media_dispatcher.py AI-compose integration from the previous patch.

Validate:

    python -m py_compile ai_compose_schema.py ai_compose_engine.py ai_compose_refinement.py
    python verify_ai_compose.py
    git diff --check

Commit only these files:

    git add ai_compose_schema.py ai_compose_engine.py ai_compose_refinement.py
    git commit -m "feat: add bounded AI compose refinement loop"
    git push

Use refinement_iterations: 0 for a one-pass smoke test. Omit it or use 2 for the bounded agentic refinement workflow.

