"""Run the Transformers adapter with lightweight fakes; no GPU or downloads."""
import json
import os
import sys
import types
import unittest
import weakref
from unittest.mock import Mock, patch

from ai_compose_engine import LocalDirector
from ai_compose_schema import AIComposeRequest


class TransformerBackendTests(unittest.TestCase):
    def exercise(self, mode, fail=False):
        references = []
        calls = []
        decoded = []
        response = json.dumps({'base_segments': [{'source_start': 0, 'source_end': 5}]})

        class Tensor:
            shape = (1, 2)

            def to(self, device):
                return self

        class Batch(dict):
            def __init__(self):
                self.input_ids = Tensor()
                super().__init__(input_ids=self.input_ids)

            def to(self, device):
                return self

        class Tokenizer:
            def __call__(self, prompt, **kwargs):
                return Batch()

            def decode(self, tokens, **kwargs):
                decoded.append(tokens)
                return response

        class ChatTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return Batch() if mode == 'batch' else Tensor()

        class Model:
            device = 'cpu'

            def generate(self, *args, **kwargs):
                calls.append(kwargs)
                if fail:
                    raise RuntimeError('generation failed')
                return [[10, 11, 12, 13]]

        def tokenizer_factory(*args, **kwargs):
            obj = Tokenizer() if mode == 'plain' else ChatTokenizer()
            references.append(weakref.ref(obj))
            return obj

        def model_factory(*args, **kwargs):
            obj = Model()
            references.append(weakref.ref(obj))
            return obj

        cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=Mock())
        torch = types.SimpleNamespace(cuda=cuda, float16='float16', float32='float32')
        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer_factory),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_factory))
        director = LocalDirector(AIComposeRequest(source_path='/tmp/test.mp4', llm_required=True))
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), \
             patch.dict(os.environ, {'AI_COMPOSE_HF_MODEL': 'offline-fixture',
                                     'AI_COMPOSE_HF_LOAD_4BIT': 'false'}), \
             patch('ai_compose_engine.gc.collect') as collect:
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                    director._plan_transformers('Make a short.')
            else:
                plan = director._plan_transformers('Make a short.')
                self.assertEqual(plan.base_segments[0].source_end, 5)
                self.assertEqual(decoded, [[12, 13]])
            collect.assert_called_once()
            cuda.empty_cache.assert_called_once()
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]['do_sample'])
        self.assertTrue(all(ref() is None for ref in references))

    def test_generation_and_cleanup_for_all_tokenizer_outputs(self):
        for mode in ('tensor', 'batch', 'plain'):
            with self.subTest(mode=mode):
                self.exercise(mode)

    def test_generation_failure_still_cleans_up(self):
        self.exercise('batch', fail=True)


if __name__ == '__main__':
    unittest.main()
