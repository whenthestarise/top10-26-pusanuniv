from copy import deepcopy
import unittest

from src.classify_selected_evidence import check_claim, claim_units, classify, LABELS, split_claims


class FakeGenerator:
    def __init__(self, answer, finish="stop"):
        self.answer, self.finish, self.calls = answer, finish, []

    def choose(self, messages, allowed):
        self.calls.append((messages, allowed))
        return {"answer": self.answer, "finish_reason": self.finish}


class TestEvidenceContent(unittest.TestCase):
    def setUp(self):
        self.record = {"company_id": "selected50:001", "source_sheet": "가상 명단", "source_row": 2,
                       "raw_fields": {"K": " 생산 로봇 자동화 중간, 사무직 높음\n기술직 낮음 "}}

    def test_offsets_preserve_whitespace_decimal_and_raw_source(self):
        text = "  연구 6.1% 낮음, 사무직 높음; 장비 자동화\n끝 "
        items = list(split_claims(text))
        self.assertEqual([i["quote"] for i in items], ["연구 6.1% 낮음", "사무직 높음", "장비 자동화", "끝"])
        for item in items:
            self.assertEqual(text[item["start"]:item["end"]], item["quote"])

    def test_forged_quote_offset_owner_scope_and_document_are_rejected(self):
        claim = claim_units(self.record, "가상해시")[0]
        check_claim(claim, self.record, "가상해시")
        for key, value in [("quote", "새 문장"), ("start", 0), ("owner_id", "selected50:002"),
                           ("source_scope", "general_job_reference"), ("source_document_sha256", "다른해시")]:
            changed = deepcopy(claim)
            changed[key] = value
            with self.assertRaises(ValueError):
                check_claim(changed, self.record, "가상해시")

    def test_physical_automation_does_not_become_digital_or_company_score(self):
        result = classify({"quote": "생산 로봇 자동화 진행"}, FakeGenerator("물리 자동화"))
        self.assertEqual(result["effective_label"], "물리 자동화")
        self.assertFalse(result["verified_company_fact"])
        self.assertIsNone(result["score"])
        wrong = classify({"quote": "생산 로봇 자동화 진행"}, FakeGenerator("디지털 자동화"))
        self.assertEqual(wrong["effective_label"], "판단 보류")
        self.assertTrue(wrong["validation_errors"])

    def test_rating_is_not_treated_as_deployed_automation(self):
        result = classify({"quote": "사무직 높음"}, FakeGenerator("디지털 자동화"))
        self.assertEqual(result["effective_label"], "판단 보류")
        self.assertIsNone(result["score"])

    def test_prompt_has_no_company_name_expected_label_or_score(self):
        generator = FakeGenerator("정성 등급·직군 설명")
        result = classify({"quote": "기술직 낮음"}, generator)
        messages, allowed = generator.calls[0]
        self.assertEqual(messages[1]["content"], "기술직 낮음")
        self.assertEqual(allowed, {"업무 내용 서술", "정성 등급·직군 설명", "판단 보류"})
        self.assertEqual(result["effective_label"], "정성 등급·직군 설명")
        self.assertIsNone(result["score"])

    def test_human_computer_input_cannot_select_ai_automation_without_source_marker(self):
        generator = FakeGenerator("업무 내용 서술")
        result = classify({"quote": "직원이 전산 화면에 주문 번호를 반복 입력한다."}, generator)
        self.assertNotIn("디지털 자동화", generator.calls[0][1])
        self.assertEqual(result["effective_label"], "업무 내용 서술")
        self.assertFalse(result["generative_ai_explicit"])

    def test_invalid_or_truncated_label_is_blocked(self):
        for generator in (FakeGenerator("5점"), FakeGenerator("물리 자동화", "length")):
            result = classify({"quote": "생산 로봇 자동화"}, generator)
            self.assertTrue(result["validation_errors"])
            self.assertEqual(result["effective_label"], "판단 보류")
