from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.relative_selection import CRITERIA, calculate_selection
from src.run_mvp_selection import prepare_payload, public_result, run, validate_scenario


def scenario_fixture():
    return {"scenario_id": "기능검사", "result_label": "시연용 결과 · 검증 전",
            "neutral_values": {"work_life_balance": 2, "infrastructure": 4, "job_outlook": 3},
            "reference_ai_values": {"높음": 1, "중간~높음": 2, "중간": 3, "낮음~중간": 4, "낮음": 5},
            "comparison_scope": "가상 시연 범위", "user_authorization": "가상 기능 검사 전용"}


def records_fixture():
    records = []
    for i in range(1, 51):
        records.append({"company_id": f"selected50:{i:03d}", "company_name": f"가상 기능검사 기업 {i}",
            "region": ("부산", "울산", "경남")[(i - 1) % 3], "location": "가상 위치",
            "annual_salary_manwon": 12000 - i * 100 if i < 50 else None,
            "main_hiring_roles": "가상 직군", "ai_exposure_label": "중간", "ai_exposure_explanation": "가상 정성 설명",
            "industry": "가상 업종"})
    facts = {r["company_id"]: deepcopy(r) for r in records}
    return records, facts


class TestMvpSelection(unittest.TestCase):
    def test_demo_selects_ten_but_never_marks_verified_real_selection(self):
        records, facts = records_fixture()
        payload, provenance = prepare_payload(records, facts, scenario_fixture(), "가상해시")
        original = deepcopy(records)
        result = calculate_selection(payload, {k: 1 for k in CRITERIA})
        self.assertEqual(len(result["selected_companies"]), 10)
        self.assertTrue(result["selection_is_demo"])
        self.assertTrue(result["assumed_values_used"])
        self.assertFalse(result["real_selection_confirmed"])
        self.assertFalse(result["ready_for_ranking"])
        self.assertIsNone(result["final_top10"])
        self.assertEqual(records, original)
        self.assertEqual(len(provenance), 50)

    def test_missing_salary_stays_unknown_and_outside_demo_comparison(self):
        records, facts = records_fixture()
        payload, provenance = prepare_payload(records, facts, scenario_fixture(), "가상해시")
        self.assertIsNone(payload["companies"][-1]["values"]["salary"])
        self.assertNotIn("selected50:050", payload["comparison_cohort"]["company_ids"])
        self.assertFalse(provenance[-1]["eligible_for_demo_calculation"])

    def test_public_result_has_demo_label_and_excludes_private_detail_fields(self):
        records, facts = records_fixture()
        scenario = scenario_fixture()
        payload, _ = prepare_payload(records, facts, scenario, "가상해시")
        calculated = calculate_selection(payload, {k: 1 for k in CRITERIA})
        published = public_result(calculated, records, scenario["result_label"])
        self.assertEqual(published["selected_count"], 10)
        self.assertFalse(published["verified_research_result"])
        self.assertIn("시연용", published["result_label"])
        text = json.dumps(published, ensure_ascii=False)
        for key in ("neutral_values", "reference_ai_values", "demo_values", "work_life_balance", "scenario_sha256", "limitations"):
            self.assertNotIn(key, text)
        with self.assertRaises(ValueError):
            public_result(calculated, records, "최종 확정 결과")

    def test_original_real_gates_cannot_be_enabled_by_demo_metadata(self):
        records, facts = records_fixture()
        payload, _ = prepare_payload(records, facts, scenario_fixture(), "가상해시")
        payload["dataset_type"] = "real"
        result = calculate_selection(payload, {k: 1 for k in CRITERIA})
        self.assertFalse(result["calculation_completed"])
        self.assertFalse(result["real_selection_confirmed"])
        self.assertIsNone(result["final_top10"])

    def test_demo_requires_explicit_label_and_internal_record(self):
        records, facts = records_fixture()
        payload, _ = prepare_payload(records, facts, scenario_fixture(), "가상해시")
        payload.pop("demo_metadata")
        with self.assertRaises(ValueError):
            calculate_selection(payload, {k: 1 for k in CRITERIA})
        scenario = scenario_fixture()
        scenario["reference_ai_values"]["낮음"] = 1
        with self.assertRaises(ValueError):
            validate_scenario(scenario)

    def test_changed_company_facts_cannot_enter_demo_as_provided_data(self):
        records, facts = records_fixture()
        facts["selected50:001"]["annual_salary_manwon"] = 99999
        with self.assertRaises(ValueError):
            prepare_payload(records, facts, scenario_fixture(), "가상해시")

    def test_private_audit_cannot_be_written_under_public_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                run(root, root / "missing.json", root / "public", root / "public/internal")
            self.assertFalse((root / "public").exists())
