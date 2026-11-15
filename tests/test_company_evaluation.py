from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.build_company_evaluation import (
    ACTIVE_IDS, WIDE_HEADERS, build_table, check_join, csv_rows, extract_pdf_notes, write_outputs,
)


def fixtures():
    policy = {"policy_version": "기능검사", "raw_sources": {
        "pdf": {"stored_name": "가상문서.pdf", "sha256": "가상해시"},
        "companies": {"stored_name": "가상목록.csv", "sha256": "가상해시"}}}
    raw = {"G": "8000", "H": "가상 연봉 자료", "N": "가상 기간", "O": "80000000", "P": "0.0001",
           "I": "가상 직무", "J": "높음(정성 추정)", "K": "기능 검사용 가상 설명", "L": "직무 자료 확인",
           "AA": "기능 검사 전용", "D": "가상 위치", "E": "가상 업종", "F": "가상 제품"}
    record = {"annual_salary_manwon": 8000, "raw_fields": raw, "source_page": 1, "location": "가상 위치",
              "evaluated_job": "가상 직무", "ai_exposure_label": raw["J"], "salary_data_type": raw["H"],
              "salary_reference_period": raw["N"], "included_in_salary_statistics": True,
              "duplicate_of": None, "exclusion_reason": None}
    company = {"company_id": "SYNTHETIC:001", "original_id": 1, "company_name": "가상 기업",
               "region": "부산", "industry": "가상 업종", "products_business": "가상 제품",
               "catalog_notes": "기능 검사 전용", "research": record}
    weights = {"criteria": [{"criterion_id": key, "criterion_name": key, "active": True,
                              "weight": 1, "normalized_active_weight": 0.2} for key in ACTIVE_IDS],
               "active_weight_sum": 5, "response_count": 1, "important_total": 1,
               "unimportant_total": 1, "counts_match_document_reference": True}
    config = {"salary": {"baseline": 7150}, "model_id": "가상기능검사"}
    return [company], weights, config, policy


def table(companies=None, exclusions=None):
    base, weights, config, policy = fixtures()
    return build_table(companies or base, weights, config, policy, exclusions or {}, {"quote": "가상 자료 한계"})


class TestCompanyEvaluation(unittest.TestCase):
    def test_salary_uses_already_converted_g_without_applying_conversion_again(self):
        result = table()
        salary = result["companies"][0]["criteria"]["salary"]
        self.assertEqual(salary["numeric_value"], 8000)
        self.assertAlmostEqual(salary["index"], 8000 / 7150)
        self.assertIsNone(salary["raw_score"])

    def test_grades_addresses_and_industries_do_not_become_non_salary_scores(self):
        result = table()
        for key in ACTIVE_IDS[1:]:
            row = result["companies"][0]["criteria"][key]
            self.assertIsNone(row["raw_score"])
            self.assertIsNone(row["index"])
        self.assertEqual(result["companies"][0]["criteria"]["ai_resilience"]["reference_text"], "높음(정성 추정)")
        self.assertFalse(result["summary"]["ready_for_ranking"])
        self.assertIsNone(result["summary"]["final_top10"])

    def test_missing_salary_is_kept_with_original_exclusion_reason(self):
        companies, _, _, _ = fixtures()
        companies[0]["research"] = None
        excluded = {1: {"company_name": "가상 기업", "region": "부산", "reason": "가상 연봉 미확인",
                        "source": {"quote": "가상 연봉 미확인", "page": 1, "field": "제외 사유"}}}
        result = table(companies, excluded)
        self.assertEqual(len(result["companies"]), 1)
        self.assertIsNone(result["companies"][0]["criteria"]["salary"]["numeric_value"])
        self.assertEqual(result["summary"]["salary_missing_count"], 1)
        self.assertEqual(len(result["companies"][0]["missing_criteria"]), 5)

    def test_duplicate_record_is_preserved_but_not_counted_as_salary_unit(self):
        companies, _, _, _ = fixtures()
        duplicate = deepcopy(companies[0])
        duplicate.update(company_id="SYNTHETIC:002", original_id=2, company_name="가상 중복")
        duplicate["research"].update(included_in_salary_statistics=False, duplicate_of=1, exclusion_reason="가상 중복")
        result = table(companies + [duplicate])
        self.assertEqual(result["summary"]["salary_available_count"], 2)
        self.assertEqual(result["summary"]["salary_calculation_unit_count"], 1)
        self.assertEqual(result["summary"]["duplicate_excluded_count"], 1)
        self.assertEqual(len(result["companies"]), 2)

    def test_mismatched_exclusion_identity_is_rejected(self):
        companies, _, _, _ = fixtures()
        companies[0]["research"] = None
        with self.assertRaises(ValueError):
            check_join(companies, {})
        with self.assertRaises(ValueError):
            check_join(companies, {1: {"company_name": "다른 기업", "region": "부산"}})

    def test_pdf_exclusion_reason_keeps_exact_page_span(self):
        text = ("제외 항목 1 | 가상 기업\n원본순번: 1\n기업명(원본): 가상 기업\n지역: 부산\n업종: 가상\n"
                "확인하지 못한 내용·제외 사유: 가상 사유\n두 번째 줄.\n처리: 비교표 제외\n"
                "현재 자료에는 회사별 가상 값이 없다.")
        _, _, _, policy = fixtures()
        exclusions, limitation = extract_pdf_notes([text], policy)
        ref = exclusions[1]["source"]
        self.assertEqual(ref["quote"], text[ref["start"]:ref["end"]])
        self.assertEqual(limitation["quote"], text[limitation["start"]:limitation["end"]])

    def test_csv_preserves_unicode_and_missing_score_as_blank(self):
        result = table()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "새표"
            write_outputs(result, output)
            with (output / "기업별_평가표.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["기업명"], "가상 기업")
            self.assertEqual(rows[0]["워라밸점수"], "")
            self.assertEqual(len(csv_rows(result)[1]), 5)
            comparison = json.loads((output / "document_comparison_inputs.json").read_text())
            self.assertEqual(comparison["companies"], [])
            with self.assertRaises(FileExistsError):
                write_outputs(result, output)

    def test_invalid_salary_and_unexpected_criterion_are_rejected(self):
        companies, weights, config, policy = fixtures()
        companies[0]["research"]["annual_salary_manwon"] = float("nan")
        with self.assertRaises(ValueError):
            build_table(companies, weights, config, policy, {}, {})
        companies, weights, config, policy = fixtures()
        weights["criteria"][-1]["criterion_id"] = "다른 기준"
        with self.assertRaises(ValueError):
            build_table(companies, weights, config, policy, {}, {})


if __name__ == "__main__":
    unittest.main()
