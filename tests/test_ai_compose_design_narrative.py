"""The visual design pass must not invalidate an approved spoken narrative."""
import copy
import json
import unittest

from ai_compose_engine import ComposeError, LocalDirector
from ai_compose_schema import AIComposeRequest, CompositionPlan


class DesignNarrativeTests(unittest.TestCase):
    def setUp(self):
        self.request = AIComposeRequest(source_path='/tmp/source.mp4', llm_required=True)
        self.baseline = CompositionPlan.model_validate({
            'base_segments': [{'source_start': 0, 'source_end': 8},
                              {'source_start': 12, 'source_end': 20}],
            'narration_sections': [
                {'position': 'intro', 'text': 'Listen to the explanation.'},
                {'position': 'bridge', 'after_segment': 0, 'text': 'Here is the next point.'},
                {'position': 'outro', 'text': 'Consider the speaker’s conclusion.'}],
        })
        self.proposed = self.baseline.model_dump(mode='json')
        del self.proposed['narration_sections'][1]['text']
        self.proposed['graphics'] = [{'kind': 'callout', 'start': 2, 'end': 4,
                                     'text': 'The next point'}]
        self.proposed['asset_requests'] = [{'query': 'relevant visual', 'start': 3, 'end': 5}]

    def test_missing_bridge_text_uses_validated_narrative_and_keeps_new_design(self):
        director = LocalDirector(self.request, narrative_plan=self.baseline)
        calls = []
        def generate(prompt):
            calls.append(prompt)
            return json.dumps(self.proposed)
        result = director._generate_valid_plan(generate, 'Design this narrative')
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.narration_sections, self.baseline.narration_sections)
        self.assertEqual(result.graphics[0].text, 'The next point')
        self.assertEqual(result.asset_requests[0].start, 3)
        self.assertNotIn('text', self.proposed['narration_sections'][1])

    def test_design_cannot_rewrite_cuts_or_insert_unapproved_speech(self):
        director = LocalDirector(self.request, narrative_plan=self.baseline)
        self.proposed['base_segments'] = [{'source_start': 1, 'source_end': 2}]
        self.proposed['narration_sections'] = [{'position': 'intro', 'text': 'Unapproved claim'}]
        self.proposed['voiceovers'] = [{'text': 'Unapproved overlay', 'start': 0}]
        result = director._generate_valid_plan(lambda _: json.dumps(self.proposed), 'Design')
        self.assertEqual(result.base_segments, self.baseline.base_segments)
        self.assertEqual(result.narration_sections, self.baseline.narration_sections)
        self.assertEqual(result.voiceovers, [])

    def test_missing_narration_without_baseline_still_fails(self):
        with self.assertRaisesRegex(ComposeError, 'narration_sections'):
            LocalDirector(self.request)._generate_valid_plan(
                lambda _: json.dumps(self.proposed), 'Initial narrative')

    def test_invalid_visual_fields_still_require_correction(self):
        director = LocalDirector(self.request, narrative_plan=self.baseline)
        del self.proposed['asset_requests'][0]['end']
        with self.assertRaisesRegex(ComposeError, 'asset_requests'):
            director._generate_valid_plan(lambda _: json.dumps(self.proposed), 'Design')

    def test_baseline_is_a_snapshot_and_returned_plans_do_not_mutate_it(self):
        original = copy.deepcopy(self.baseline.narration_sections)
        director = LocalDirector(self.request, narrative_plan=self.baseline)
        self.baseline.narration_sections[1].text = 'Later mutation'
        first = director._generate_valid_plan(lambda _: json.dumps(self.proposed), 'Design')
        self.assertEqual(first.narration_sections, original)
        first.narration_sections[1].text = 'Rendered plan changed'
        second = director._generate_valid_plan(lambda _: json.dumps(self.proposed), 'Design')
        self.assertEqual(second.narration_sections, original)

    def test_schema_echo_is_not_accepted_with_a_baseline(self):
        director = LocalDirector(self.request, narrative_plan=self.baseline)
        with self.assertRaisesRegex(ComposeError, 'base_segments'):
            director._generate_valid_plan(lambda _: '{"$defs": {}}', 'Design')

    def test_required_narration_is_completed_before_design_can_start(self):
        request = self.request.model_copy(deep=True)
        request.editor.narration = 'intro_bridge_outro'
        director = LocalDirector(request)
        incomplete = self.baseline.model_dump(mode='json')
        incomplete['narration_sections'] = incomplete['narration_sections'][:1]
        prompts = []
        def generate(prompt):
            prompts.append(prompt)
            return json.dumps(incomplete) if len(prompts) == 1 else self.baseline.model_dump_json()
        result = director._generate_valid_plan(generate, 'Narrative')
        self.assertEqual(len(prompts), 2)
        self.assertIn('Missing requested narration_sections: bridge, outro', prompts[1])
        self.assertEqual(result.narration_sections, self.baseline.narration_sections)


if __name__ == '__main__':
    unittest.main()
