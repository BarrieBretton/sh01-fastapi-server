from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path


PATCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH_ROOT))

from ai_compose_engine import LocalDirector  # noqa: E402
from ai_compose_schema import AIComposeRequest  # noqa: E402


FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "director_outputs.json").read_text(encoding="utf-8")
)


class DirectorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = AIComposeRequest(
            source_path="/tmp/source.mp4",
            brief="Make a concise informative short",
            llm_required=True,
            refinement_iterations=0,
        )
        self.director = LocalDirector(self.request)

    def test_small_model_scalar_fields_are_normalised(self) -> None:
        payload = self.director._normalise_plan_payload(FIXTURES["small_model_scalar_fields"])
        plan = self.director._normalise_plan_payload(payload)
        self.assertEqual(plan["schema_version"], "sh01/ai-compose@1")
        self.assertEqual(plan["director_notes"], ["Keep the opening concise."])

    def test_schema_echo_is_not_a_plan(self) -> None:
        payload = self.director._normalise_plan_payload(FIXTURES["schema_echo"])
        self.assertNotIn("base_segments", payload)

    def test_fallback_is_valid_and_explicitly_warned(self) -> None:
        old_model = os.environ.pop("AI_COMPOSE_HF_MODEL", None)
        old_url = os.environ.pop("AI_COMPOSE_LLM_URL", None)
        old_recovery = os.environ.get("AI_COMPOSE_ALLOW_DETERMINISTIC_RECOVERY")
        os.environ["AI_COMPOSE_ALLOW_DETERMINISTIC_RECOVERY"] = "true"
        try:
            plan, warnings = self.director.plan(
                {"duration": 12.0, "width": 1920, "height": 1080},
                [{"word": "important", "start": 0.0, "end": 1.0, "end": 1.0}],
                [{"time": 0.0}],
            )
            self.assertTrue(plan.base_segments)
            self.assertTrue(any("recovery" in warning for warning in warnings))
        finally:
            if old_model is not None:
                os.environ["AI_COMPOSE_HF_MODEL"] = old_model
            if old_url is not None:
                os.environ["AI_COMPOSE_LLM_URL"] = old_url
            if old_recovery is None:
                os.environ.pop("AI_COMPOSE_ALLOW_DETERMINISTIC_RECOVERY", None)
            else:
                os.environ["AI_COMPOSE_ALLOW_DETERMINISTIC_RECOVERY"] = old_recovery


if __name__ == "__main__":
    unittest.main()
