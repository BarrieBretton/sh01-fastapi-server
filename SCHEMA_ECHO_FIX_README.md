# Local director schema-echo fix

This delta is applied on top of the existing AI-compose branch and includes
schema-echo, small-model type-normalisation, and deterministic-recovery
stabilisation layers.

Copy `ai_compose_engine.py` over the repository-root file. It removes the huge
Pydantic JSON Schema from the Transformers prompt, rejects schema echoes,
retries once with a compact concrete-plan contract, normalises harmless
scalar/list differences, and can produce a valid deterministic plan when the
small model still fails.

Set `AI_COMPOSE_ALLOW_DETERMINISTIC_RECOVERY=true` on the Kaggle test host if
you want a failed model response to produce a completed render with a warning
receipt instead of failing the job. The model is still attempted first.

The `tests/` directory contains regression fixtures for the failures observed
in Kaggle. The existing `json-repair` dependency remains required.

Validate locally:

    python -m py_compile ai_compose_engine.py
    python -m unittest discover -s tests -v
    git diff --check

Then commit and push the engine file, restart the Kaggle Uvicorn process, and
rerun the agentic request.
