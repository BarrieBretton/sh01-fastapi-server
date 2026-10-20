"""A selection endpoint must not declare a failover without moving traffic."""

import ast
import asyncio
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


class Conflict(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


def selection_functions():
    source = Path(__file__).resolve().parents[1] / "infra" / "routes.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in {"select_slot", "select_render"}]
    for node in functions:
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    state = Mock()
    state.get_active.side_effect = lambda kind: {
        "render": "render-b", "postgres": "supabase-b", "b2": "b2-a"
    }[kind]
    registry = Mock()
    namespace = {
        "SlotSelectionRequest": object,
        "RenderSlotSelectionRequest": object,
        "_require_slot": Mock(),
        "state": state,
        "registry": registry,
        "HTTPException": Conflict,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace, state


class SelectionGuardTests(unittest.TestCase):
    def test_postgres_and_b2_changes_rejected_without_persisting(self):
        funcs, state = selection_functions()
        for ring, candidate in (("postgres", "supabase-a"), ("b2", "b2-b")):
            with self.subTest(ring=ring), self.assertRaises(Conflict) as caught:
                funcs["select_slot"](ring, SimpleNamespace(slot=candidate, dry_run=False))
            self.assertEqual(caught.exception.status_code, 409)
        state.set_active.assert_not_called()

    def test_same_slot_is_noop_and_dry_run_is_retained(self):
        funcs, state = selection_functions()
        noop = funcs["select_slot"]("b2", SimpleNamespace(slot="b2-a", dry_run=False))
        preview = funcs["select_slot"]("b2", SimpleNamespace(slot="b2-b", dry_run=True))
        self.assertFalse(noop["changed"])
        self.assertFalse(preview["changed"])
        self.assertEqual(preview["target"], "b2-b")
        state.set_active.assert_not_called()

    def test_render_metadata_selection_cannot_bypass_paired_switch(self):
        funcs, state = selection_functions()
        with self.assertRaises(Conflict) as caught:
            asyncio.run(funcs["select_render"](SimpleNamespace(slot="render-a", dry_run=False)))
        self.assertEqual(caught.exception.status_code, 409)
        state.set_active.assert_not_called()

    def test_render_preview_and_same_slot_are_safe(self):
        funcs, state = selection_functions()
        preview = asyncio.run(funcs["select_render"](SimpleNamespace(slot="render-a", dry_run=True)))
        noop = asyncio.run(funcs["select_render"](SimpleNamespace(slot="render-b", dry_run=False)))
        self.assertEqual(preview["target"], "render-a")
        self.assertTrue(preview["dry_run"])
        self.assertFalse(noop["changed"])
        state.set_active.assert_not_called()


if __name__ == "__main__":
    unittest.main()
