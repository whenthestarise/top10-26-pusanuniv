from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.build_learning_standard import make_case
from src.read_selected50 import evaluate_one, messages, run, summarize


class TestSelectedReading(unittest.TestCase):
    def setUp(self):
        self.record = {"company_id": "selected50:001", "company_name": "가상기업", "region": "부산",
            "location": "가상 위치", "main_hiring_roles": "가상 직군", "ai_exposure_label": "낮음",
            "ai_exposure_explanation": "가상 업무 설명", "annual_salary_manwon": None,
            "raw_fields": {"B": "가상기업", "C": "부산", "D": "가상 위치", "G": "평균 미확인",
                           "I": "가상 직군", "J": "낮음", "K": "가상 업무 설명"},
            "source_sheet": "가상 명단", "source_row": 2}
        self.case = make_case(self.record, {"raw_sources": {"selected50": {"sha256": "가상해시"}}},
                              {"rules": ["제공 원문만 사용"]})

    def generator(self, answer, finish="stop"):
        return lambda *args, **kwargs: {"answer": answer, "finish_reason": finish}

    def test_expected_answers_are_not_passed_to_model_and_original_markers_survive(self):
        payload = json.loads(messages(self.case)[1]["content"])
        self.assertNotIn("expected_facts", payload)
        self.assertEqual(payload["provided_fields"]["annual_salary_manwon"], "평균 미확인")
        self.assertFalse(isinstance(payload["output_fields"], dict))

    def test_unchanged_facts_pass_without_enabling_scoring(self):
        answer = json.dumps(self.case["expected_facts"], ensure_ascii=False)
        result = evaluate_one(self.case, self.generator(answer), 768)
        self.assertTrue(result["passed"])
        self.assertFalse(result["company_scoring_validated"])
        self.assertFalse(result["ready_for_ranking"])

    def test_invented_score_and_salary_are_blocked_with_field_diagnostics(self):
        response = deepcopy(self.case["expected_facts"])
        response.update(annual_salary_manwon=7000, ai_resilience_score=5)
        result = evaluate_one(self.case, self.generator(json.dumps(response)), 768)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["facts"])
        self.assertEqual(set(result["mismatched_fields"]), {"annual_salary_manwon", "ai_resilience_score"})

    def test_duplicate_keys_incomplete_output_and_non_json_are_blocked(self):
        valid = json.dumps(self.case["expected_facts"])
        for answer, finish in [(valid[:-1] + ', "region": "부산"}', "stop"), (valid, "length"), ("설명만 있음", "stop")]:
            with self.subTest(answer=answer, finish=finish):
                result = evaluate_one(self.case, self.generator(answer, finish), 768)
                self.assertFalse(result["passed"])
                self.assertIsNone(result["facts"])

    def test_generation_error_is_recorded_as_failure_not_a_default_answer(self):
        def broken(*args, **kwargs):
            raise RuntimeError("가상 실행 오류")
        result = evaluate_one(self.case, broken, 768)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["facts"])
        summary = summarize([result], 50, "running", 1)
        self.assertEqual(summary["completed_count"], 1)
        self.assertFalse(summary["all_cases_completed"])
        self.assertFalse(summary["scoring_performed"])

    def test_invalid_population_is_rejected_before_loading_model_or_creating_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "configs").mkdir()
            (root / "configs/provided_data_learning.json").write_text(json.dumps({"rules": []}))
            (root / "configs/selected_model.json").write_text(json.dumps({
                "status": "candidate_for_provided_fact_reading", "candidate": {"model_id": "가상", "revision": "가상"}}))
            output = root / "report"
            with patch("src.read_selected50.load_user_records", return_value=({}, {})), \
                 patch("src.read_selected50.load_selected", return_value={"companies": []}):
                with self.assertRaises(ValueError):
                    run(root, output, generator_factory=lambda *args: self.fail("모델을 불러오면 안 됩니다."))
            self.assertFalse(output.exists())
