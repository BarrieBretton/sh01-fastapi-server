"""Exercise real parsing/validation with scripted model responses; no GPU needed."""
import copy
import json
import unittest

from ai_compose_engine import ComposeError, LocalDirector
from ai_compose_schema import AIComposeRequest


class DirectorRetryTests(unittest.TestCase):
    def setUp(self):
        self.director = LocalDirector(AIComposeRequest(source_path='/tmp/video.mp4', llm_required=True))
        self.plan = {
            'base_segments': [{'source_start': 12, 'source_end': 27}],
            'asset_requests': [{'query': 'pensioners protest', 'start': 2, 'end': 5}],
        }

    def test_missing_asset_times_retry_even_when_prompts_match(self):
        broken = copy.deepcopy(self.plan)
        del broken['asset_requests'][0]['start']
        del broken['asset_requests'][0]['end']
        prompts = []

        def generate(prompt):
            prompts.append(prompt)
            return json.dumps(broken if len(prompts) == 1 else self.plan)

        plan = self.director._generate_valid_plan(generate, 'same prompt', 'same prompt')
        self.assertEqual(len(prompts), 2)
        self.assertIn('["asset_requests", 0, "start"]', prompts[1])
        self.assertIn('["asset_requests", 0, "end"]', prompts[1])
        self.assertIn('pensioners protest', prompts[1])
        self.assertNotEqual(prompts[0], prompts[1])
        self.assertEqual(plan.asset_requests[0].start, 2)
        self.assertEqual(plan.asset_requests[0].end, 5)

    def test_exhausted_retries_fail_without_inventing_times_or_dropping_assets(self):
        calls = []
        broken = {'base_segments': self.plan['base_segments'],
                  'asset_requests': [{'query': 'pensioners protest'}]}

        def generate(prompt):
            calls.append(prompt)
            return json.dumps(broken)

        with self.assertRaisesRegex(ComposeError, 'after 3 validation attempts') as raised:
            self.director._generate_valid_plan(generate, 'brief')
        self.assertEqual(len(calls), 3)
        self.assertIn('start', str(raised.exception))
        self.assertIn('end', str(raised.exception))
        self.assertEqual(broken['asset_requests'], [{'query': 'pensioners protest'}])

    def test_valid_response_needs_one_generation(self):
        calls = []
        def generate(prompt):
            calls.append(prompt)
            return json.dumps(self.plan)
        self.director._generate_valid_plan(generate, 'brief')
        self.assertEqual(len(calls), 1)

    def test_invalid_range_is_returned_to_model_for_correction(self):
        broken = copy.deepcopy(self.plan)
        broken['asset_requests'][0]['end'] = 1
        prompts = []
        def generate(prompt):
            prompts.append(prompt)
            return json.dumps(broken if len(prompts) == 1 else self.plan)
        self.director._generate_valid_plan(generate, 'brief')
        self.assertIn('end must be greater than start', prompts[1])

    def test_model_runtime_error_is_not_retried_as_json(self):
        calls = []
        def generate(prompt):
            calls.append(prompt)
            raise RuntimeError('simulated CUDA failure')
        with self.assertRaisesRegex(RuntimeError, 'CUDA failure'):
            self.director._generate_valid_plan(generate, 'brief')
        self.assertEqual(len(calls), 1)

    def test_prompt_explains_required_asset_timing_in_both_modes(self):
        for compact in (False, True):
            prompt = self.director._prompt(self.director.request,
                {'duration': 36.8, 'has_audio': True}, [], [], compact=compact)
            self.assertIn('REQUIRED query, start, and end', prompt)
            self.assertIn('BODY', prompt)
            self.assertIn('"start":2.0,"end":5.0', prompt)


if __name__ == '__main__':
    unittest.main()
