import unittest
from unittest.mock import patch

from ai_compose_ir import SourceClip, compile_edit_plan
from ai_compose_policy import resolve_editorial_policy
from ai_compose_quality import inspect_edit_plan
from ai_compose_engine import _search_remote
from ai_compose_schema import (
    AIComposeRequest,
    AssetRequest,
    BaseSegment,
    CompositionPlan,
    ResolvedLayer,
)


class PolicyIrQualityTests(unittest.TestCase):
    def request(self, **overrides):
        payload = {
            "source_path": "/media/source.mp4",
            "brief": "Create a polished informative short.",
            "target_seconds": 12,
            "fetch_assets": True,
            "refinement_iterations": 0,
        }
        payload.update(overrides)
        return AIComposeRequest.model_validate(payload)

    def plan(self):
        return CompositionPlan(
            title="test",
            base_segments=[
                BaseSegment(source_start=0, source_end=5, reason="hook"),
                BaseSegment(source_start=8, source_end=15, reason="proof"),
            ],
            asset_requests=[
                AssetRequest(
                    query="laboratory research",
                    start=2,
                    end=4,
                    confidence=0.8,
                    evidence_refs=["utterance_2"],
                )
            ],
        )

    def test_dialogue_policy_preserves_audio_and_disables_unrequested_voiceover(self):
        request = self.request(fetch_assets=False)
        words = [
            {"word": "This", "start": 0.0, "end": 0.4},
            {"word": "is", "start": 0.4, "end": 0.6},
            {"word": "important", "start": 0.6, "end": 1.1},
        ]
        policy = resolve_editorial_policy(request, {"has_audio": True}, words)
        self.assertTrue(policy.speech_present)
        self.assertEqual(policy.source_audio, "preserve")
        self.assertEqual(policy.voiceover, "off")
        self.assertEqual(policy.supplementary_visuals, "off")
        self.assertTrue(any("dialogue" in item.lower() for item in policy.constraints))

    def test_explicit_policy_can_enable_supporting_visuals(self):
        request = self.request(
            editorial_policy={
                "content_type": "documentary",
                "supplementary_visuals": "prefer",
                "camera_motion": "punch_ins",
            }
        )
        policy = resolve_editorial_policy(request, {"has_audio": True}, [])
        self.assertEqual(policy.content_type, "documentary")
        self.assertEqual(policy.supplementary_visuals, "prefer")
        self.assertEqual(policy.camera_motion, "punch_ins")

    def test_edit_plan_maps_source_time_to_contiguous_program_time(self):
        request = self.request()
        policy = resolve_editorial_policy(request, {"has_audio": True}, [])
        layers = [
            ResolvedLayer(
                asset_id="asset_1",
                local_path="/media/asset.mp4",
                source="local",
                query="laboratory research",
                kind="video",
                start=2,
                end=4,
                x=0.1,
                y=0.55,
                width=0.8,
                height=0.35,
            )
        ]
        edit_plan = compile_edit_plan(
            request,
            self.plan(),
            layers,
            [],
            policy,
            {"has_audio": True},
            variant=2,
        )
        self.assertEqual(edit_plan.schema_version, "sh01/ai-edit@2")
        self.assertAlmostEqual(edit_plan.duration, 12.0)
        self.assertEqual(edit_plan.source_clips[0].program_start, 0.0)
        self.assertEqual(edit_plan.source_clips[1].program_start, 5.0)
        self.assertEqual(edit_plan.variant_strategy, "dynamic_punch_in")
        self.assertGreaterEqual(len(edit_plan.camera_tracks), 1)
        self.assertEqual(edit_plan.visual_layers[0].asset_id, "asset_1")

    def test_quality_rejects_timeline_gap(self):
        request = self.request()
        policy = resolve_editorial_policy(request, {"has_audio": True}, [])
        edit_plan = compile_edit_plan(
            request,
            self.plan(),
            [],
            [],
            policy,
            {"has_audio": True},
        ).model_copy(update={"source_clips": [
            SourceClip(source_start=0, source_end=5, program_start=0, program_end=5, reason="hook"),
            SourceClip(source_start=8, source_end=15, program_start=6, program_end=13, reason="proof"),
        ], "duration": 13})
        report = inspect_edit_plan(edit_plan, request)
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.code == "program_gap" for issue in report.issues))

    def test_remote_search_without_provider_keys_is_a_noop(self):
        request = self.request(fetch_assets=True)
        with patch.dict("os.environ", {"PEXELS_API_KEY": "", "PIXABAY_API_KEY": "", "UNSPLASH_ACCESS_KEY": ""}, clear=False):
            self.assertEqual(_search_remote("irrelevant", "video", request), [])


if __name__ == "__main__":
    unittest.main()
