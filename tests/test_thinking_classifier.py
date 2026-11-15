import unittest

import torch

from src import llm_runtime
from src.thinking_classifier import ThinkingClassifier


class Tokenizer:
    eos_token_id = 2
    unk_token_id = None

    def apply_chat_template(self, *args, **kwargs):
        return "대화 시작\n"

    def __call__(self, *args, **kwargs):
        class Batch(dict):
            def to(self, device):
                return self
        return Batch(input_ids=torch.tensor([[1, 3, 4]]), attention_mask=torch.ones((1, 3), dtype=torch.long))

    def convert_tokens_to_ids(self, text):
        return 9

    def encode(self, text, **kwargs):
        return [5]

    def decode(self, tokens, **kwargs):
        if tokens.tolist() != [5, 2]:
            raise AssertionError("최종 분류 외의 내부 검토를 복호화했습니다.")
        return "현장 작업"


class Model:
    def __init__(self, completed, trailing=False):
        self.completed, self.calls, self.trailing = completed, 0, trailing

    def generate(self, input_ids, **kwargs):
        self.calls += 1
        suffix = [8, 8, 9 if self.completed else 8] if self.calls == 1 else [5, 2]
        if self.calls == 1 and self.completed and self.trailing:
            suffix.append(7)
        return torch.cat((input_ids, torch.tensor([suffix])), dim=1)


class TestThinkingClassifier(unittest.TestCase):
    def generator(self, completed):
        generator = ThinkingClassifier.__new__(ThinkingClassifier)
        generator.tokenizer, generator.model = Tokenizer(), Model(completed)
        generator.torch, generator.runtime, generator.device = torch, llm_runtime, torch.device("cpu")
        return generator

    def test_only_final_label_is_decoded_and_returned(self):
        generator = self.generator(True)
        result = generator.choose([], {"현장 작업"})
        self.assertEqual(result["answer"], "현장 작업")
        self.assertTrue(result["reasoning_completed"])
        self.assertEqual(generator.model.calls, 2)
        self.assertNotIn("reasoning_text", result)

    def test_unfinished_review_cannot_produce_a_score_label(self):
        generator = self.generator(False)
        result = generator.choose([], {"현장 작업"})
        self.assertEqual(result["answer"], "")
        self.assertFalse(result["reasoning_completed"])
        self.assertEqual(generator.model.calls, 1)

    def test_tokens_after_review_end_are_discarded_before_final_classification(self):
        generator = self.generator(True)
        generator.model.trailing = True
        result = generator.choose([], {"현장 작업"})
        self.assertTrue(result["reasoning_completed"])
        self.assertEqual(result["discarded_after_review_tokens"], 1)
        self.assertEqual(result["answer"], "현장 작업")


if __name__ == "__main__":
    unittest.main()
