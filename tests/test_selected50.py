from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.data_policy import build_selected_bundle, enforce_evaluation_population, validate_user_sources
from src.prepare_selected50 import match_catalog, salary_value
from src.selected50_table import selected_table_companies
from src.build_company_evaluation import build_table, ACTIVE_IDS


class TestSelectedFifty(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data/processed").mkdir(parents=True)
        (self.root / "configs").mkdir()
        self.policy = {"policy_version": "기능검사", "raw_sources": {
            "selected50": {"stored_name": "가상.xlsx", "original_name": "가상.xlsx", "sha256": "가상해시"},
            "pdf": {"stored_name": "가상.pdf", "sha256": "문서해시"}}}
        self.rows = []
        for i in range(1, 51):
            self.rows.append({"company_id": f"selected50:{i:03d}", "selection_order": i, "company_name": f"가상{i}",
                "region": "부산", "location": "가상 위치", "industry": "가상 업종", "products_business": "가상 제품",
                "annual_salary_manwon": 7000, "salary_source_text": "가상 연봉 근거", "main_hiring_roles": "가상 직군",
                "ai_exposure_label": "중간~높음", "review_result": "기능 검사", "source_sheet": "부울경 중견기업 50선",
                "source_row": i + 1, "raw_fields": {"A": i, "B": f"가상{i}", "C": "부산", "D": "가상 위치",
                    "E": "가상 업종", "F": "가상 제품", "G": 7000, "H": "가상 근거", "I": "가상 직군",
                    "J": "중간~높음", "K": "가상 직군 설명", "L": "기능 검사"},
                "catalog_link": {"status": "원본 대응 미확정", "company_id": None, "original_id": None, "company_name": None},
                "pdf_reference": None, "salary_difference_manwon": None, "salary_comparison_status": "대응 PDF 없음"})
        self.write_cache()

    def tearDown(self):
        self.temp.cleanup()

    def write_cache(self):
        path = self.root / "data/processed/selected50_companies.json"
        path.write_text(json.dumps({"source_sha256": "가상해시", "companies": self.rows,
            "selection_policy": {"final_selection_target": 10}}, ensure_ascii=False))
        self.policy["processed_selected50_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.root / "configs/data_policy.json").write_text(json.dumps(self.policy))

    def test_catalog_link_uses_name_and_region_not_filtered_order_or_substring(self):
        catalog = [{"company_name": "가상", "region": "부산", "original_id": 90},
                   {"company_name": "다른(울산)", "region": "울산", "original_id": 70}]
        self.assertEqual(match_catalog("가상", "부산", catalog)[1]["original_id"], 90)
        self.assertEqual(match_catalog("다른", "울산", catalog)[0], "지역 괄호 표기 차이")
        self.assertIsNone(match_catalog("가상추가", "부산", catalog)[1])
        self.assertIsNone(match_catalog("가상", "경남", catalog)[1])

    def test_missing_salary_is_not_zero_and_bad_numeric_values_are_rejected(self):
        self.assertIsNone(salary_value("(평균 미확인)"))
        self.assertEqual(salary_value("7,000"), 7000)
        for value in (0, -1, True, float("nan"), "약 7000"):
            with self.assertRaises(ValueError):
                salary_value(value)

    def test_selected_rows_survive_without_parent_mapping(self):
        companies, _ = selected_table_companies(self.root, self.policy, [], {})
        self.assertEqual(len(companies), 50)
        self.assertIsNone(companies[0]["original_id"])
        self.assertEqual(companies[0]["selection_order"], 1)
        self.assertEqual(companies[0]["company_id"], "selected50:001")

    def test_primary_workbook_salary_is_not_replaced_by_pdf_or_filled_when_missing(self):
        ref = {"annual_salary_manwon": 8000, "ai_exposure_label": "낮음", "evaluated_job": "문서 직무",
               "location": "문서 위치", "comparison_notes": "문서 주석", "source_page": 1,
               "raw_fields": {"G": "8000"}}
        self.rows[0]["pdf_reference"] = ref
        self.rows[1]["pdf_reference"] = ref
        self.rows[1]["annual_salary_manwon"] = None
        self.rows[1]["raw_fields"]["G"] = "(평균 미확인)"
        self.write_cache()
        companies, _ = selected_table_companies(self.root, self.policy, [], {})
        weights = {"criteria": [{"criterion_id": k, "criterion_name": k, "active": True, "weight": 1,
                                  "normalized_active_weight": .2} for k in ACTIVE_IDS], "active_weight_sum": 5,
                   "response_count": 1, "important_total": 1, "unimportant_total": 1, "counts_match_document_reference": True}
        result = build_table(companies, weights, {"salary": {"baseline": 7150}, "model_id": "기능검사"}, self.policy, {}, {})
        self.assertEqual(result["companies"][0]["criteria"]["salary"]["numeric_value"], 7000)
        self.assertEqual(result["companies"][0]["pdf_reference_salary_manwon"], 8000)
        self.assertIsNone(result["companies"][1]["criteria"]["salary"]["numeric_value"])
        self.assertEqual(result["summary"]["evaluation_row_count"], 250)

    def test_primary_xlsx_source_is_verified_by_unchanged_cell(self):
        bundle = build_selected_bundle(self.rows[0], self.policy)
        with patch("src.data_policy.load_user_records", return_value=(self.policy, {})):
            validate_user_sources(bundle, self.root)
            changed = deepcopy(bundle)
            changed["sources"][0]["text"] = "다른 기업의 자료"
            with self.assertRaises(ValueError):
                validate_user_sources(changed, self.root)

    def test_actual_evaluation_rejects_reference_population_ids(self):
        with self.assertRaises(ValueError):
            enforce_evaluation_population({"dataset_type": "real", "context": {"company_id": "document:001"}}, self.root)
        enforce_evaluation_population({"dataset_type": "real", "context": {"company_id": "selected50:001"}}, self.root)
        enforce_evaluation_population({"dataset_type": "synthetic"}, self.root)
