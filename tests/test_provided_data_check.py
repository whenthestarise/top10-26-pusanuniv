import json
import unittest

from src.provided_data_check import messages, validate


class TestProvidedReading(unittest.TestCase):
    def setUp(self):
        self.record = {"company_name": "가상 기업", "region": "가상 지역", "evaluated_job": "가상 직무",
                       "annual_salary_manwon": 1000, "raw_fields": {"K": "기능 검사 가상 문장"},
                       "ai_scope_limitations": "기능 검사 전용"}
        self.answer = {"company_name": "가상 기업", "region": "가상 지역", "target_job": "가상 직무",
                       "annual_salary_manwon": 1000, "work_hours": None}

    def test_exact_copy_and_unknown_hours_are_accepted(self):
        errors, _ = validate(json.dumps(self.answer), self.record)
        self.assertEqual(errors, [])

    def test_invented_hours_and_modified_facts_are_rejected(self):
        for key, value in (("work_hours", "주 40시간"), ("annual_salary_manwon", 1100),
                           ("company_name", "외부 기업"), ("target_job", "다른 직무")):
            answer = {**self.answer, key: value}
            errors, _ = validate(json.dumps(answer), self.record)
            self.assertTrue(errors)

    def test_extra_scores_duplicate_keys_or_non_json_are_rejected(self):
        for answer in (json.dumps({**self.answer, "score": 5}), '{"company_name":"가상", "company_name":"다름"}', "설명"):
            self.assertTrue(validate(answer, self.record)[0])

    def test_model_input_contains_only_provided_fields_and_no_default_answer(self):
        prompt = messages(self.record)
        self.assertNotIn("work_hours", prompt[1]["content"])
        self.assertNotIn('"company_name"', prompt[1]["content"])
        self.assertIn("가상 기업", prompt[1]["content"])
