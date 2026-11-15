from copy import deepcopy
import unittest

from src.build_learning_standard import make_case, validate_fact_response


class TestLearningStandard(unittest.TestCase):
    def setUp(self):
        self.record = {"company_id": "selected50:001", "company_name": "가상기업", "region": "부산",
            "location": "가상 위치", "main_hiring_roles": "가상 직군", "ai_exposure_label": "낮음",
            "ai_exposure_explanation": "가상 업무 설명", "annual_salary_manwon": None,
            "raw_fields": {"B": "가상기업", "C": "부산", "D": "가상 위치", "G": "평균 미확인",
                           "I": "가상 직군", "J": "낮음", "K": "가상 업무 설명"},
            "source_sheet": "가상 명단", "source_row": 2}
        self.policy = {"raw_sources": {"selected50": {"sha256": "가상해시"}}}
        self.case = make_case(self.record, self.policy, {"rules": ["제공 원문만 사용"]})

    def test_model_input_has_original_missing_marker_and_no_answer_object(self):
        payload = self.case["model_input"]
        self.assertNotIn("expected_facts", payload)
        self.assertEqual(payload["provided_fields"]["annual_salary_manwon"], "평균 미확인")
        self.assertIn("cell=G2", payload["source_links"]["annual_salary_manwon"])
        self.assertIsNone(self.case["expected_facts"]["annual_salary_manwon"])

    def test_copied_facts_are_not_claimed_as_validated_company_scores(self):
        result = validate_fact_response(self.case, deepcopy(self.case["expected_facts"]))
        self.assertTrue(result["fact_fidelity_passed"])
        self.assertFalse(result["company_scoring_validated"])
        self.assertFalse(result["ready_for_ranking"])

    def test_invented_salary_score_explanation_or_extra_fields_are_rejected(self):
        for key, value in [("annual_salary_manwon", 7000), ("work_life_balance_score", 3),
                           ("ai_resilience_score", 5), ("ai_exposure_explanation", "외부 설명"),
                           ("extra", "새 필드")]:
            response = deepcopy(self.case["expected_facts"])
            response[key] = value
            with self.assertRaises(ValueError):
                validate_fact_response(self.case, response)
