import csv
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from src import prepare_project_data as preparation


class TestProjectDataImport(unittest.TestCase):
    def write_csv(self, path, headers, rows, encoding="utf-8-sig"):
        with path.open("w", encoding=encoding, newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def test_cp949_company_catalog_preserves_order_and_region(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "companies.csv"
            row = dict(zip(preparation.COMPANY_COLUMNS,
                           ("부산", "가상기업", "정비", "설비", "원본 기록")))
            self.write_csv(path, preparation.COMPANY_COLUMNS, [row], "cp949")
            companies, encoding = preparation.load_catalog(path)
            self.assertEqual(encoding, "cp949")
            self.assertEqual(companies[0]["company_id"], "document:001")
            self.assertEqual(companies[0]["company_name"], "가상기업")

    def test_survey_uses_actual_counts_and_keeps_all_nonblank_selections(self):
        config = yaml.safe_load((preparation.ROOT / "configs/document_model.yaml").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "survey.csv"
            columns = [preparation.QUESTION, preparation.UNIMPORTANT_QUESTION]
            self.write_csv(path, columns, [
                {columns[0]: "높은 급여, 워라밸, 높은 급여", columns[1]: "수평적 조직문화"},
                {columns[0]: "확보된 근무지역 인프라", columns[1]: ""},
            ])
            weights = preparation.load_survey_weights(path, config)
        self.assertEqual(weights["response_count"], 2)
        self.assertEqual(weights["important_total"], 3)
        self.assertEqual(weights["unimportant_total"], 1)
        self.assertEqual(weights["active_criterion_count"], 5)
        self.assertFalse(weights["counts_match_document_reference"])
        self.assertEqual(weights["source_audit"]["important"]["rows_with_repeated_items"], 1)
        self.assertEqual(weights["source_audit"]["unimportant"]["blank_answer_rows"], 1)
        salary = next(item for item in weights["criteria"] if item["criterion_id"] == "salary")
        self.assertAlmostEqual(salary["weight"], 1 + 0.1 / 3)

    def test_unknown_survey_answer_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "알 수 없는 선택지"):
            preparation.survey_counts([{"question": "미등록 선택"}], "question")

    def test_catalog_pdf_mismatch_fails_instead_of_joining_wrong_company(self):
        catalog = [{"original_id": 1, "company_name": "A", "region": "부산"}]
        with self.assertRaisesRegex(ValueError, "불일치"):
            preparation.join_companies(catalog, [{"original_id": 1, "company_name": "B", "region": "부산"}])

    def test_missing_salary_record_is_preserved_and_duplicate_ids_rejected(self):
        catalog = [{"original_id": 1, "company_name": "A", "region": "부산"}]
        self.assertIsNone(preparation.join_companies(catalog, [])[0]["research"])
        record = {"original_id": 1, "company_name": "A", "region": "부산"}
        with self.assertRaisesRegex(ValueError, "중복"):
            preparation.join_companies(catalog, [record, deepcopy(record)])

    def test_incomplete_company_never_receives_default_scores(self):
        config = yaml.safe_load((preparation.ROOT / "configs/document_model.yaml").read_text())
        company = {"company_id": "document:001", "company_name": "A", "region": "부산",
                   "industry": "정비", "products_business": "설비", "research": None}
        payload = preparation.analysis_inputs([company], config)
        result = payload["companies"][0]
        self.assertEqual(result["ranking_status"], "평가 보류")
        self.assertTrue(all(item["score"] is None for item in result["criteria"].values()))
        self.assertNotIn("career_capital", result["criteria"])
        self.assertIn("career_capital", payload["inactive_criteria"])
        self.assertEqual(len(result["criteria"]), 5)
        self.assertEqual(result["criteria"]["salary"]["status"], "자료 부족")

    def test_duplicate_csv_headers_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            path.write_text("기업명,기업명\nA,B\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "중복"):
                preparation.read_csv(path)


if __name__ == "__main__":
    unittest.main()
