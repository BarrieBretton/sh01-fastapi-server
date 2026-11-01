import tempfile
import unittest
from pathlib import Path

from ai_compose_engine import _make_ass
from ai_compose_ir import compile_edit_plan
from ai_compose_policy import resolve_editorial_policy
from ai_compose_quality import inspect_edit_plan, inspect_render
from ai_compose_schema import AIComposeRequest, BaseSegment, CompositionPlan, GraphicPlan


class RenderProgramTests(unittest.TestCase):
    def request(self, **overrides):
        payload = {
            "source_path": "/media/source.mp4",
            "brief": "Create a cinematic short with readable graphics.",
            "preset": "documentary",
            "target_seconds": 10,
            "fetch_assets": False,
            "refinement_iterations": 0,
            "editorial_policy": {
                "content_type": "documentary",
                "editing_intensity": "cinematic",
                "motion_graphics": "hero",
                "camera_motion": "dynamic",
                "supplementary_visuals": "off",
            },
        }
        payload.update(overrides)
        return AIComposeRequest.model_validate(payload)

    def test_explicit_graphics_compile_and_ass_render(self):
        request = self.request()
        plan = CompositionPlan(
            base_segments=[
                BaseSegment(source_start=0, source_end=5, reason="opening"),
                BaseSegment(source_start=5, source_end=10, reason="payoff"),
            ],
            graphics=[
                GraphicPlan(
                    kind="callout",
                    start=1,
                    end=3,
                    text="The important detail",
                    placement="upper_left",
                    animation="slide_down",
                    style="accent",
                )
            ],
        )
        policy = resolve_editorial_policy(request, {"has_audio": True}, [])
        edit_plan = compile_edit_plan(request, plan, [], [], policy, {"has_audio": True})

        self.assertEqual(len(edit_plan.graphics), 1)
        self.assertEqual(edit_plan.graphics[0].placement, "upper_left")
        self.assertEqual(edit_plan.graphics[0].animation, "slide_down")
        self.assertEqual(len(edit_plan.camera_tracks), 2)
        self.assertEqual(len(edit_plan.transitions), 1)
        self.assertGreater(edit_plan.transitions[0].duration, 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "captions.ass"
            _make_ass([], plan.base_segments, edit_plan.duration, plan, destination, edit_plan)
            content = destination.read_text(encoding="utf-8")
            self.assertIn("GraphicAccent", content)
            self.assertIn("\\move(90,170,90,260,0,280)", content)
            self.assertIn("The important detail", content)

    def test_render_review_rejects_unreported_planned_features(self):
        request = self.request()
        plan = CompositionPlan(
            base_segments=[BaseSegment(source_start=0, source_end=5)],
            graphics=[GraphicPlan(start=0, end=2, text="Hook", kind="hook")],
        )
        policy = resolve_editorial_policy(request, {"has_audio": True}, [])
        edit_plan = compile_edit_plan(request, plan, [], [], policy, {"has_audio": True})
        report = inspect_render(
            Path("/tmp/rendered.mp4"),
            edit_plan.duration,
            request,
            lambda _path: {"duration": 5, "width": 1080, "height": 1920, "has_audio": True, "video_codec": "h264"},
            run_decode=False,
            edit_plan=edit_plan,
            render_features={"graphics_rendered": 0, "camera_tracks_rendered": 0, "transitions_rendered": 0},
        )
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.code == "graphics_rendered_not_rendered" for issue in report.issues))

    def test_edit_plan_quality_reports_graphics_and_camera_counts(self):
        request = self.request()
        plan = CompositionPlan(base_segments=[BaseSegment(source_start=0, source_end=5)])
        policy = resolve_editorial_policy(request, {"has_audio": True}, [])
        edit_plan = compile_edit_plan(request, plan, [], [], policy, {"has_audio": True})
        report = inspect_edit_plan(edit_plan, request)
        self.assertTrue(report.ok)
        self.assertEqual(report.checks["graphic_count"], 0)
        self.assertEqual(report.checks["camera_track_count"], 1)


if __name__ == "__main__":
    unittest.main()
