from copy import deepcopy
import json
import unittest

from src.export_model_top10 import apply_refinement, build_payload, render_markdown, verify_calculation
from src.model_company_ratings import company_fields, parse_rating
from src.model_role_balance import aggregate_roles
from src.relative_selection import CRITERIA, calculate_selection


def fixtures():
    records, results = [], []
    for i in range(1, 51):
        roles = "연구, 생산" if i % 2 else "연구, 설계"
        fields = {"D": "가상 위치", "E": "가상 업종", "F": "가상 제품", "I": roles, "J": "중간", "K": "가상 직군 설명"}
        record = {"company_id": f"selected50:{i:03d}", "selection_order": i, "company_name": f"가상기업 {i}",
                  "region": ("부산", "울산", "경남")[(i-1) % 3], "annual_salary_manwon": 20000-i*300 if i < 49 else None,
                  "raw_fields": fields, "location": fields["D"], "main_hiring_roles": roles,
                  "products_business": fields["F"], "industry": fields["E"]}
        raw = json.dumps({"w": 3, "i": 2, "o": 4, "a": 3})
        records.append(record)
        results.append({"company_id": record["company_id"], "company_name": record["company_name"],
                        "provided_fields": company_fields(record), "generation": {"answer": raw, "finish_reason": "stop"},
                        "ratings": parse_rating(raw, company_fields(record))})
    roles = {r: {"score": s, "generation": {"answer": str(s), "finish_reason": "stop"}}
             for r, s in (("연구", 4), ("생산", 2), ("설계", 4))}
    locations = {"가상 위치": {"score": 3.5, "generation": {"answer": "3.5", "finish_reason": "stop"}}}
    balances = [aggregate_roles(record, roles) for record in records]
    return records, results, roles, locations, balances


class TestExportModelTop10(unittest.TestCase):
    def prepared(self):
        records, results, roles, locations, balances = fixtures()
        payload, ratings = build_payload(records, results, "가상해시")
        apply_refinement(records, payload, ratings, roles, locations, balances)
        return records, payload, ratings

    def test_all_five_values_are_connected_and_missing_salaries_are_preserved(self):
        records, payload, ratings = self.prepared()
        self.assertEqual(len(payload["companies"]), 50)
        self.assertEqual(len(payload["comparison_cohort"]["company_ids"]), 48)
        self.assertIsNone(payload["companies"][-1]["values"]["salary"])
        self.assertEqual(payload["companies"][0]["values"]["work_life_balance"], 3)
        self.assertEqual(payload["companies"][1]["values"]["work_life_balance"], 4)
        self.assertEqual({c["values"]["infrastructure"] for c in payload["companies"]}, {3.5})

    def test_tampered_source_and_score_are_rejected(self):
        for field in ("provided_fields", "ratings"):
            records, results, *_ = fixtures()
            if field == "provided_fields":
                results[0][field]["D"] = "원문과 다른 위치"
            else:
                results[0][field]["job_outlook"]["score"] = 5
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_payload(records, results, "가상해시")

    def test_missing_duplicate_or_foreign_company_is_rejected(self):
        records, results, *_ = fixtures()
        for altered in (results[:-1], results[:-1] + [results[0]]):
            with self.assertRaises(ValueError):
                build_payload(records, altered, "가상해시")

    def test_refinement_is_recomputed_from_model_responses(self):
        records, results, roles, locations, balances = fixtures()
        payload, ratings = build_payload(records, results, "가상해시")
        roles["연구"]["score"] = 5
        with self.assertRaises(ValueError):
            apply_refinement(records, payload, ratings, roles, locations, balances)
        roles["연구"]["score"] = 4
        balances[0]["score"] = 4
        with self.assertRaises(ValueError):
            apply_refinement(records, payload, ratings, roles, locations, balances)

    def test_calculation_and_document_trace_source_model_score_and_selection(self):
        records, payload, ratings = self.prepared()
        original = deepcopy(records)
        calculated = calculate_selection(payload, {key: 1 for key in CRITERIA})
        checks = verify_calculation(calculated)
        self.assertEqual(checks["selected_count_checked"], 10)
        self.assertFalse(calculated["real_selection_confirmed"])
        document = render_markdown(calculated, records, ratings, {"sanity_passed": 6, "sanity_count": 6, "role_count": 3, "location_count": 1})
        for expected in ("기업별 선정 근거", "전체 50개 기업 평가표", "동일 가중 평균", "모델이 부여한 평가값", "가상 제품"):
            self.assertIn(expected, document)
        self.assertEqual(records, original)

    def test_calculation_tampering_fails_verification(self):
        _, payload, _ = self.prepared()
        calculated = calculate_selection(payload, {key: 1 for key in CRITERIA})
        eligible = next(row for row in calculated["company_results"] if row["display_index_P"] is not None)
        eligible["weighted_sum_T"] += 1
        with self.assertRaises(ValueError):
            verify_calculation(calculated)
