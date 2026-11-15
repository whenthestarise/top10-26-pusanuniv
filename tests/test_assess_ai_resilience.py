from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from src.ai_resilience_cases import build_cases
from src.assess_ai_resilience import assess, checked_evidence, read_rubric, scoring_messages
from src.extract_ai_evidence import choice_trie, extract_evidence, next_choice_tokens, prepare_bundle


class TestSingleCriterionAssessment(unittest.TestCase):
    def test_failed_model_validation_blocks_score_without_calling_model(self):
        result = assess(self.extracted(), lambda _: self.fail("검증 실패 뒤 모델을 호출했습니다."), calibration_passed=False)
        self.assertEqual(result["status"], "model_validation_failed")
        self.assertIsNone(result["proposed_score"])
        self.assertEqual(result["model_calls"], [])

    @patch("src.data_policy.enforce_evaluation_population")
    @patch("src.data_policy.validate_user_sources")
    def test_real_input_requires_automatic_model_validation(self, _source_policy, _population):
        extraction = self.extracted()
        extraction["dataset_type"] = "real"
        result = assess(extraction, lambda _: self.fail("모델 검증 없이 실제 자료를 평가했습니다."))
        self.assertEqual(result["status"], "model_validation_required")
        self.assertIsNone(result["proposed_score"])
        result = assess(extraction, self.responses("현장 작업", "예"), calibration_passed=True)
        self.assertEqual(result["proposed_score"], 5)
        self.assertFalse(result["ready_for_ranking"])

    def test_real_generation_stays_in_choices_and_stops_at_explicit_end_token(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=16, max_position_embeddings=64, pad_token_id=0,
                             bos_token_id=1, eos_token_id=2)
        model = Qwen3ForCausalLM(config).eval()
        trie = choice_trie([[5, 6], [7]], 2)
        with torch.inference_mode():
            result = model.generate(torch.tensor([[1, 3, 4]]),
                attention_mask=torch.ones((1, 3), dtype=torch.long), max_new_tokens=8,
                do_sample=False, eos_token_id=2, pad_token_id=0,
                prefix_allowed_tokens_fn=lambda _, sequence: next_choice_tokens(trie, sequence[3:].tolist()))
        self.assertIn(result[0, 3:].tolist(), ([5, 6, 2], [7, 2]))

    def responses(self, *answers):
        values = iter(answers)
        return lambda _: {"answer": next(values)}

    def setUp(self):
        self.cases = {item["case_id"]: item for item in build_cases()}

    def extracted(self, key="physical"):
        return extract_evidence(prepare_bundle(self.cases[key]["bundle"]), lambda _: {"answer": "업무 수행"})

    def test_missing_evidence_never_calls_model_or_defaults_to_middle_score(self):
        for key in ("empty", "other_job"):
            result = assess(self.extracted(key), lambda _: self.fail("모델이 호출됐습니다."))
            self.assertIsNone(result["proposed_score"])
            self.assertEqual(result["status"], "insufficient_evidence")

    def test_low_and_high_scores_both_require_evidence(self):
        for key, label, score in (("physical", "현장 작업", 5), ("digital", "반복 사무", 1)):
            with self.subTest(key=key):
                result = assess(self.extracted(key), self.responses(label, "예"))
                self.assertEqual(result["proposed_score"], score)
                self.assertEqual(result["anchor"], read_rubric()["anchors"][score])
                self.assertTrue(result["reviewed_evidence"])
                self.assertFalse(result["ready_for_ranking"])

    def test_conflict_or_uncertainty_blocks_scoring_call(self):
        for label, status in (("모순", "conflicting_evidence"), ("보류", "insufficient_evidence")):
            calls = []
            def generate(prompt):
                calls.append(prompt)
                return {"answer": label}
            result = assess(self.extracted("conflicting"), generate)
            self.assertIsNone(result["proposed_score"])
            self.assertEqual(result["status"], status)
            self.assertEqual(len(calls), 1)

    def test_malformed_conflict_output_cannot_fall_through_to_score(self):
        result = assess(self.extracted("conflicting"), lambda _: {"answer": "대체로 양립"})
        self.assertEqual(result["status"], "invalid_output")
        self.assertIsNone(result["proposed_score"])
        self.assertEqual(len(result["model_calls"]), 1)

    def test_uncertain_scoring_and_invalid_numeric_response_never_become_scores(self):
        for answer, expected in (("정보 부족", "insufficient_evidence"), ("근거 충돌", "conflicting_evidence"),
                                 ("6", "invalid_output"), ("5점이다. 확실하다", "invalid_output"),
                                 ('{"score": 5}', "invalid_output"), (None, "invalid_output")):
            result = assess(self.extracted(), lambda _: {"answer": answer})
            self.assertEqual(result["status"], expected)
            self.assertIsNone(result["proposed_score"])

    def test_tampered_quote_hash_source_or_scope_is_rejected_before_model(self):
        changes = [lambda r: r["evidence"][0].update(quote="새로 지어낸 문장"),
                   lambda r: r["evidence"][0].update(source_text_sha256="잘못된 해시"),
                   lambda r: r["sources"][0].update(text="달라진 원문"),
                   lambda r: r["sources"][0]["scope"].update(target_job="다른 직무"),
                   lambda r: r["context"].update(company_id="다른 기업")]
        for change in changes:
            result = deepcopy(self.extracted())
            change(result)
            with self.assertRaises(ValueError):
                checked_evidence(result)

    def test_unknown_duplicate_or_additional_evidence_is_rejected(self):
        for key in ("unknown", "duplicate", "extra_field"):
            result = self.extracted()
            if key == "unknown":
                result["evidence"][0]["sentence_id"] = "문장9999"
            elif key == "duplicate":
                result["evidence"].append(deepcopy(result["evidence"][0]))
            else:
                result["evidence"][0]["invented_reason"] = "새로운 설명"
            with self.assertRaises(ValueError):
                checked_evidence(result)

    def test_failed_upstream_and_inconsistent_status_are_rejected(self):
        for updates in ({"validation_errors": ["추출 오류"]}, {"status": "invalid_output"},
                        {"status": "insufficient_evidence"}, {"extraction_version": "다른 버전"}):
            extraction = self.extracted()
            extraction.update(updates)
            with self.assertRaises(ValueError):
                checked_evidence(extraction)

    def test_model_receives_evidence_and_rubric_without_expected_answer_or_company_fame(self):
        extraction = self.extracted()
        prompt = scoring_messages(extraction["evidence"], read_rubric())
        user_data = json.loads(prompt[1]["content"])
        self.assertEqual(user_data, [extraction["evidence"][0]["quote"]])
        self.assertNotIn("expected_score", json.dumps(prompt))
        self.assertNotIn(extraction["context"]["company_name"], prompt[1]["content"])


if __name__ == "__main__":
    unittest.main()
