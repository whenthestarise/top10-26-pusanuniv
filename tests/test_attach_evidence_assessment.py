from copy import deepcopy
import unittest

from src.attach_evidence_assessment import audit_content, connect_table, selection_readiness, verify_claim_results
from src.build_company_evaluation import ACTIVE_IDS
from src.classify_selected_evidence import claim_units, classify


class FakeGenerator:
    def choose(self, messages, allowed):
        return {"answer": "정성 등급·직군 설명", "finish_reason": "stop"}


def table_fixture():
    weights = [{"criterion_id": k, "weight": 1.0} for k in ACTIVE_IDS]
    companies = []
    for i in range(1, 11):
        companies.append({"company_id": f"selected50:{i:03d}", "company_name": f"가상{i}",
            "region": ("부산", "울산", "경남")[i % 3], "criteria": {k: {
                "name": k, "raw_score": None, "index": 1.0 if k == "salary" else None,
                "missing_information": ["제공 근거 부족"]} for k in ACTIVE_IDS}})
    return {"weights": weights, "companies": companies, "summary": {"ready_for_ranking": False}}


class TestAttachedAssessment(unittest.TestCase):
    def test_rating_only_classified_as_task_is_deferred_without_rewriting_model_output(self):
        item = {"claim": {"quote": "현장 생산 중간"}, "effective_label": "업무 내용 서술"}
        audited = audit_content(item)
        self.assertEqual(item["effective_label"], "업무 내용 서술")
        self.assertNotIn("assessment_effective_label", item)
        self.assertEqual(audited["assessment_effective_label"], "판단 보류")
        self.assertTrue(audited["content_audit_flags"])

    def test_missing_criteria_do_not_change_weights_or_enable_top10(self):
        table = table_fixture()
        original = deepcopy(table)
        result = selection_readiness(table, False)
        self.assertEqual(table, original)
        self.assertEqual(result["active_weight_sum"], 5)
        self.assertEqual(result["usable_value_counts"]["salary"], 10)
        self.assertEqual(result["all_five_complete_count"], 0)
        self.assertFalse(result["can_confirm_top10"])
        self.assertIsNone(result["final_top10"])
        self.assertFalse(result["reduced_criteria_scenario_used"])
        self.assertFalse(result["missing_values_imputed"])

    def test_reduced_criteria_weights_are_rejected(self):
        table = table_fixture()
        table["weights"] = [r for r in table["weights"] if r["criterion_id"] in {"salary", "ai_resilience"}]
        with self.assertRaises(ValueError):
            selection_readiness(table, False)

    def test_model_gate_or_missing_normalization_blocks_complete_raw_scores(self):
        table = table_fixture()
        for company in table["companies"]:
            for k in ACTIVE_IDS[1:]:
                company["criteria"][k]["raw_score"] = 4
        table["summary"]["ready_for_ranking"] = True
        result = selection_readiness(table, False)
        self.assertEqual(result["all_five_complete_count"], 10)
        self.assertEqual(result["all_five_normalized_count"], 0)
        self.assertFalse(result["can_confirm_top10"])

    def test_all_labels_cannot_turn_qualitative_notes_into_scores(self):
        table = table_fixture()
        results = [{"company_id": c["company_id"], "company_name": c["company_name"],
                    "claims": [], "ai_resilience_score": None} for c in table["companies"]]
        connected = connect_table(table, results, False)
        self.assertTrue(all(c["criteria"]["ai_resilience"]["raw_score"] is None for c in connected["companies"]))
        self.assertEqual(connected["summary"]["usable_value_counts"]["ai_resilience"], 0)
        self.assertFalse(connected["selection_feasibility"]["can_confirm_top10"])
        self.assertTrue(all(c["composite_index"] is None for c in connected["companies"]))

    def test_modified_classification_label_or_missing_claim_is_rejected(self):
        record = {"company_id": "selected50:001", "source_sheet": "가상", "source_row": 2,
                  "raw_fields": {"K": "기술직 낮음, 사무직 높음"}}
        items = [classify(c, FakeGenerator()) for c in claim_units(record, "가상해시")]
        verify_claim_results(items, record, "가상해시")
        changed = deepcopy(items)
        changed[0]["effective_label"] = "업무 내용 서술"
        with self.assertRaises(ValueError):
            verify_claim_results(changed, record, "가상해시")
        with self.assertRaises(ValueError):
            verify_claim_results(items[:1], record, "가상해시")
