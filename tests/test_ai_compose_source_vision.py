"""Source observation regressions, including the uploaded placeholder response."""
import copy
import os
import unittest
from unittest.mock import patch

from ai_compose_editor import vision_review
from ai_compose_editorial import parse_source_observations
from ai_compose_schema import AIComposeRequest


class SourceVisionTests(unittest.TestCase):
    def setUp(self):
        self.frames = [{'time': 0.0}, {'time': 3.341}]
        self.valid = {'observations': [
            {'time': 0.0, 'description': 'A seated speaker occupies the center of the portrait frame.'},
            {'time': 3.341, 'description': 'The speaker raises one hand beside the microphone.'},
        ]}
        # Exact model output in evidence.json, not an invented failure fixture.
        self.echo = {
            'score': 0.9,
            'issues': ["The speaker's face should be more centered in the frame to maintain focus on his speech."],
            'revision': 'timestamped visible subjects, actions, readable text and suggested safe overlay regions',
        }
        self.request = AIComposeRequest(source_path='/tmp/source.mp4',
                                       editor={'enabled': True, 'visual_review_required': True})

    def test_uploaded_placeholder_response_is_not_source_evidence(self):
        with self.assertRaisesRegex(ValueError, 'timestamped observations'):
            parse_source_observations(self.echo, self.frames)

    def test_valid_observations_supply_director_context_without_quality_score(self):
        result = parse_source_observations(dict(self.valid, score=0.99), self.frames)
        self.assertTrue(result['performed'])
        self.assertEqual(result['observations'], self.valid['observations'])
        self.assertIn('3.34s: The speaker raises', result['revision'])
        self.assertNotIn('score', result)

    def test_missing_frame_invalid_time_and_audio_claims_are_rejected(self):
        invalid = []
        invalid.append({'observations': self.valid['observations'][:1]})
        for time in (30.0, float('nan'), float('inf'), True, None, 'unknown', '1-3s'):
            item = copy.deepcopy(self.valid)
            item['observations'][0]['time'] = time
            invalid.append(item)
        for description in (None, '', 'Speaker is inaudible.', self.echo['revision']):
            item = copy.deepcopy(self.valid)
            item['observations'][0]['description'] = description
            invalid.append(item)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_source_observations(value, self.frames)

    def test_explicit_timestamp_formats_preserve_frame_evidence(self):
        for time in ('3.341', '3.341s', '3.341 seconds', '00:03.341', '00:00:03.341'):
            value = copy.deepcopy(self.valid)
            value['observations'][0]['time'] = '0'
            value['observations'][1]['time'] = time
            with self.subTest(time=time):
                result = parse_source_observations(value, self.frames)
                self.assertEqual(result['observations'][1]['time'], 3.341)

    def test_parseable_but_unobserved_times_still_fail(self):
        for time in ('30s', '00:30', '00:63', 'NaN', '-1', 'frame 2'):
            value = copy.deepcopy(self.valid)
            value['observations'][1]['time'] = time
            with self.subTest(time=time), self.assertRaises(ValueError):
                parse_source_observations(value, self.frames)

    def test_echo_gets_one_correction_and_valid_result_reaches_director(self):
        with patch.dict(os.environ, {'AI_COMPOSE_VISION_HF_MODEL': 'fixture'}), \
             patch('ai_compose_editor.visual_json', side_effect=[self.echo, self.valid]) as model:
            result = vision_review(self.request, self.frames, 'Describe the source', source=True)
        self.assertEqual(model.call_count, 2)
        first = model.call_args_list[0].args[2]
        second = model.call_args_list[1].args[2]
        self.assertIn('EVERY supplied frame', first)
        self.assertNotIn(self.echo['revision'], first)
        self.assertIn('timestamped observations', second)
        self.assertEqual(result['observations'], self.valid['observations'])

    def test_repeated_echo_fails_required_vision_instead_of_claiming_success(self):
        with patch.dict(os.environ, {'AI_COMPOSE_VISION_HF_MODEL': 'fixture'}), \
             patch('ai_compose_editor.visual_json', return_value=self.echo) as model, \
             self.assertRaisesRegex(ValueError, 'Visual review evidence contract failed'):
            vision_review(self.request, self.frames, 'Describe the source', source=True)
        self.assertEqual(model.call_count, 2)

    def test_frame_sampling_contract_matches_frames_sent_to_model(self):
        frames = [{'time': float(i)} for i in range(12)]

        def response(request, selected, instruction):
            self.assertEqual(len(selected), 8)
            self.assertEqual(selected[0]['time'], 0.0)
            self.assertEqual(selected[-1]['time'], 11.0)
            return {'observations': [{'time': f['time'], 'description': 'A speaker is seated beside a microphone.'}
                                     for f in selected]}

        with patch.dict(os.environ, {'AI_COMPOSE_VISION_HF_MODEL': 'fixture'}), \
             patch('ai_compose_editor.visual_json', side_effect=response):
            result = vision_review(self.request, frames, 'Describe the source', source=True)
        self.assertEqual(len(result['observations']), 8)

    def test_draft_review_still_uses_its_separate_score_and_evidence_contract(self):
        draft = {'score': 0.8, 'observations': [
            {'time': f['time'], 'category': 'framing', 'observation': 'The speaker remains fully visible.',
             'problem': False, 'action': ''} for f in self.frames]}
        with patch.dict(os.environ, {'AI_COMPOSE_VISION_HF_MODEL': 'fixture'}), \
             patch('ai_compose_editor.visual_json', return_value=draft):
            result = vision_review(self.request, self.frames, 'Review the draft')
        self.assertEqual(result['score'], 0.8)
        self.assertTrue(result['ok'])

    def test_model_runtime_failure_is_not_retried_as_malformed_json(self):
        with patch.dict(os.environ, {'AI_COMPOSE_VISION_HF_MODEL': 'fixture'}), \
             patch('ai_compose_editor.visual_json', side_effect=RuntimeError('model failed')) as model, \
             self.assertRaisesRegex(RuntimeError, 'model failed'):
            vision_review(self.request, self.frames, 'Describe the source', source=True)
        self.assertEqual(model.call_count, 1)


if __name__ == '__main__':
    unittest.main()
