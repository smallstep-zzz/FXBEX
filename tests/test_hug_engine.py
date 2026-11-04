import unittest
from types import SimpleNamespace

from fxbex.hug_engine import generate


class Logits:
    def __init__(self, token):
        self.token = token

    def argmax(self):
        return self.token


class Runtime:
    capacity = 20

    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.resets = 0
        self.reset()

    def reset(self):
        self.consumed_ids = []
        self.resets += 1

    def step(self, token, return_logits=True):
        assert len(self.consumed_ids) < self.capacity
        self.consumed_ids.append(token)
        return Logits(next(self.predictions)) if return_logits else None


class Tokenizer:
    eos_token_id = 9

    def encode(self, text):
        return [int(token) for token in text.split()]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        return [1, 2]

    def decode(self, ids, **kwargs):
        return ''.join({3: 'a', 4: 'b', 5: '\ufffd', 6: 'é'}.get(i, '') for i in ids).replace('\ufffdé', 'é')


class HugEngineTest(unittest.TestCase):
    def adapter(self, predictions):
        return SimpleNamespace(runtime=Runtime(predictions), tokenizer=Tokenizer())

    def test_incremental_eos_and_state(self):
        adapter = self.adapter([3, 4, 9])
        events = []
        result = generate(adapter, '1 2', 4, lambda text, ids: events.append((text, ids)))
        self.assertEqual(events, [('a', [3]), ('ab', [3, 4])])
        self.assertEqual(result['token_ids'], [3, 4])
        self.assertEqual(result['prompt_token_ids'], [1, 2])
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertEqual(adapter.runtime.consumed_ids, [1, 2, 3, 4])

    def test_prefix_reuse_and_final_token_consumed(self):
        adapter = self.adapter([3, 4])
        generate(adapter, '1 2', 1)
        result = generate(adapter, '1 2 3 7', 1)
        self.assertEqual(result['num_cached_tokens'], 3)
        self.assertEqual(result['finish_reason'], 'length')
        self.assertEqual(adapter.runtime.consumed_ids, [1, 2, 3, 7, 4])

    def test_exact_or_changed_prompt_resets(self):
        for prompt in ('1 2 3', '8 2'):
            adapter = self.adapter([3, 4])
            generate(adapter, '1 2', 1)
            result = generate(adapter, prompt, 1)
            self.assertEqual(result['num_cached_tokens'], 0)
            self.assertEqual(adapter.runtime.consumed_ids, [int(i) for i in prompt.split()] + [4])

    def test_unicode_stream_waits_for_complete_character(self):
        adapter = self.adapter([5, 6])
        events = []
        result = generate(adapter, [{'role': 'user', 'content': 'hi'}], 2,
                          lambda text, ids: events.append((text, ids)))
        self.assertEqual(events, [('é', [5, 6])])
        self.assertEqual(result['text'], 'é')

    def test_validation_does_not_mutate_runtime(self):
        adapter = self.adapter([])
        for prompt, count in [('1', -1), ('1', True), ('1', 20), ('', 1)]:
            with self.assertRaises(ValueError):
                generate(adapter, prompt, count)
        self.assertEqual(adapter.runtime.consumed_ids, [])

    def test_zero_tokens_and_immediate_eos(self):
        for count, reason in [(0, 'length'), (1, 'stop')]:
            result = generate(self.adapter([9]), '1', count)
            self.assertEqual(result['text'], '')
            self.assertEqual(result['token_ids'], [])
            self.assertEqual(result['finish_reason'], reason)
