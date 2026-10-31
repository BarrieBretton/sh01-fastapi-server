"""Cheap post-install verification for the SH01 AI compose patch."""
from __future__ import annotations

import ast
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    required = ["ai_compose.py", "ai_compose_engine.py", "ai_compose_schema.py", "install_ai_compose.py"]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"missing patch files: {', '.join(missing)}")
    for name in required:
        py_compile.compile(str(ROOT / name), doraise=True)
    app = ROOT / "app.py"
    dispatcher = ROOT / "media_dispatcher.py"
    if app.exists():
        app_text = app.read_text(encoding="utf-8")
        if "from ai_compose import router as ai_compose_router" not in app_text or "app.include_router(ai_compose_router)" not in app_text:
            raise SystemExit("app.py is not installed; run python install_ai_compose.py")
    if dispatcher.exists():
        dispatcher_text = dispatcher.read_text(encoding="utf-8")
        if 'elif kind == "ai_compose":' not in dispatcher_text:
            raise SystemExit("media_dispatcher.py is not installed; run python install_ai_compose.py")
        ast.parse(dispatcher_text)
    print("AI compose patch verification passed")


if __name__ == "__main__":
    main()

