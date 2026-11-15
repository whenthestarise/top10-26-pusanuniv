from copy import deepcopy
import json
import unittest

from src.model_company_ratings import company_fields, messages, parse_rating, rate_fields


def fields_fixture():
    return {"D": "가상 도시", "E": "가상 업종", "F": "가상 기술 제품", "I": "가상 연구·생산 직군",
            "J": "중간", "K": "가상 직군의 현장 업무와 사무 업무가 혼재함"}


def response_fixture():
    return {"w": 3, "i": 4, "o": 5, "a": 4}


class TestModelCompanyRatings(unittest.TestCase):
    def test_scores_quote_the_original_fields_and_do_not_claim_empirical_verification(self):
        fields = fields_fixture()
        parsed = parse_rating(json.dumps(response_fixture(), ensure_ascii=False), fields)
        self.assertEqual(parsed["work_life_balance"]["score"], 3)
        self.assertEqual(parsed["job_outlook"]["basis_quotes"]["F"], fields["F"])
        self.assertTrue(all(not value["empirically_verified"] for value in parsed.values()))
        self.assertTrue(all(value["score_origin"] == "model_interpretation" for value in parsed.values()))

    def test_bad_scores_never_default_to_three(self):
        for score in (None, True, "4", 0, 5.1, 3.14, float("nan"), float("inf")):
            response = response_fixture()
            response["w"] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                parse_rating(json.dumps(response), fields_fixture())

    def test_missing_original_fields_are_not_replaced_by_invented_evidence(self):
        fields = fields_fixture()
        fields.pop("D")
        with self.assertRaises(ValueError):
            parse_rating(json.dumps(response_fixture()), fields)

    def test_extra_narrative_fields_and_duplicate_keys_are_rejected(self):
        response = response_fixture()
        response["실제야근시간"] = 10
        with self.assertRaises(ValueError):
            parse_rating(json.dumps(response), fields_fixture())
        encoded = json.dumps(response_fixture())
        with self.assertRaises(ValueError):
            parse_rating(encoded[:-1] + ', "w": 5}', fields_fixture())

    def test_company_name_salary_and_expected_answer_are_not_sent_to_model(self):
        record = {"company_name": "가상기업", "annual_salary_manwon": 9999, "raw_fields": fields_fixture()}
        fields = company_fields(record)
        payload = json.loads(messages(fields)[1]["content"])
        self.assertEqual(set(payload), {"D", "E", "F", "I", "J", "K"})
        self.assertNotIn("가상기업", messages(fields)[1]["content"])
        self.assertNotIn("9999", messages(fields)[1]["content"])

    def test_partial_pdf_job_scope_is_preserved_and_does_not_replace_primary_roles(self):
        record = {"raw_fields": fields_fixture(), "pdf_reference": {
            "evaluated_job": "가상 일부 직무", "raw_fields": {"K": "일부 직무 원문"}}}
        original = deepcopy(record)
        fields = company_fields(record)
        self.assertIn("일부 직무: 가상 일부 직무", fields["P"])
        self.assertEqual(fields["I"], record["raw_fields"]["I"])
        self.assertEqual(record, original)

    def test_truncated_generation_is_blocked(self):
        def generator(*args, **kwargs):
            return {"answer": json.dumps(response_fixture()), "finish_reason": "length"}
        with self.assertRaises(ValueError):
            rate_fields(fields_fixture(), generator)
