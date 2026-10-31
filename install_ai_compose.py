"""Idempotently install the AI composition routes into an SH01 checkout.

Run from the repository root after unzipping this package:

    python install_ai_compose.py

The script edits only app.py and media_dispatcher.py, creates a timestamped
backup of each file, and refuses to continue if the expected upstream markers
are missing. Running it again is safe.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
_backed_up: set[Path] = set()
def backup(path: Path) -> None:
    if path in _backed_up:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.pre_ai_compose_{stamp}.bak")
    shutil.copy2(path, target)
    _backed_up.add(path)
    print(f"backup: {target.name}")


def replace_once(path: Path, old: str, new: str, label: str, installed_marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if installed_marker in text:
        print(f"already installed: {label}")
        return False
    if old not in text:
        raise RuntimeError(f"Could not find expected upstream marker for {label} in {path}")
    backup(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"installed: {label}")
    return True


def main() -> None:
    app = ROOT / "app.py"
    dispatcher = ROOT / "media_dispatcher.py"
    required = [ROOT / "ai_compose.py", ROOT / "ai_compose_engine.py", ROOT / "ai_compose_schema.py"]
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Package is incomplete; missing: {', '.join(missing)}")
    if not app.exists() or not dispatcher.exists():
        raise SystemExit("Run this installer from the SH01 repository root")

    app_text = app.read_text(encoding="utf-8")
    dispatcher_text = dispatcher.read_text(encoding="utf-8")
    if "from ai_compose import router as ai_compose_router" not in app_text and "from auto_editor import router as auto_editor_router\n" not in app_text:
        raise SystemExit("app.py does not contain the expected auto_editor import")
    if "app.include_router(ai_compose_router)" not in app_text and "app.include_router(auto_editor_router)\n" not in app_text:
        raise SystemExit("app.py does not contain the expected auto_editor router registration")
    if '        elif kind == "ai_compose":' not in dispatcher_text and '        elif kind == "clip_caption":\n' not in dispatcher_text:
        raise SystemExit("media_dispatcher.py does not contain the expected clip_caption branch")

    replace_once(
        app,
        "from auto_editor import router as auto_editor_router\n",
        "from auto_editor import router as auto_editor_router\nfrom ai_compose import router as ai_compose_router  # SH01_AI_COMPOSE_PATCH_v1\n",
        "AI compose import",
        "from ai_compose import router as ai_compose_router",
    )
    replace_once(
        app,
        "app.include_router(auto_editor_router)\n",
        "app.include_router(auto_editor_router)\napp.include_router(ai_compose_router)  # SH01_AI_COMPOSE_PATCH_v1\n",
        "AI compose router",
        "app.include_router(ai_compose_router)",
    )

    dispatcher_branch = '''        elif kind == "ai_compose":
            import ai_compose as m
            request = m.AIComposeRequest.model_validate(payload["request"])
            validate_high_resources(request.unlimited_resources)
            job = m.make_dispatch_job(job_id, request, row["attempts"], lease_id)
            await m.register_dispatch_job(job)
            await spawn_media_task(m._execute(job_id, request), request.unlimited_resources)
            result = m._job_view(job)
'''
    replace_once(
        dispatcher,
        '        elif kind == "clip_caption":\n',
        dispatcher_branch + '        elif kind == "clip_caption":\n',
        "AI compose queue branch",
        '        elif kind == "ai_compose":',
    )
    print("AI compose installation complete")
    print("Start the server normally, then use POST /media/ai-compose.")


if __name__ == "__main__":
    main()

