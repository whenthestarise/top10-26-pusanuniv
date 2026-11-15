from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import math
import random
import unittest

from src.relative_selection import CRITERIA, REGIONS, calculate_selection
from src.selection_function_cases import regional_case


WEIGHTS = {k: 1 for k in CRITERIA}


def fixture(count=3):
    rows = [{"company_id": f"synthetic:{i:03d}", "company_name": f"가상기업 {i}", "region": REGIONS[(i - 1) % 3],
             "comparison_scope_id": "같은범위", "target_job": "가상 직무",
             "values": {k: 7150 if k == "salary" else 3 for k in CRITERIA}, "evidence": {}}
            for i in range(1, count + 1)]
    return {"dataset_type": "synthetic", "companies": rows,
            "comparison_cohort": {"scope_id": "같은범위", "target_job": "가상 직무", "scope_definition": "동일 가상 조건",
                                  "company_ids": [r["company_id"] for r in rows]}}


class TestRelativeSelection(unittest.TestCase):
    def test_relative_formula_shared_means_and_values_above_100(self):
        payload = fixture()
        for company, value in zip(payload["companies"], [1, 3, 5]):
            for k in CRITERIA[1:]:
                company["values"][k] = value
        payload["companies"][1]["values"]["salary"] = 14300
        result = calculate_selection(payload, WEIGHTS, top_n=3)
        self.assertEqual(set(result["normalization"]["non_salary_means"].values()), {3.0})
        rows = result["company_results"]
        self.assertAlmostEqual(rows[0]["relative_index_I"], 7 / 15)
        self.assertAlmostEqual(rows[1]["weighted_sum_T"], 6)
        self.assertAlmostEqual(rows[1]["relative_index_I"], 1.2)
        self.assertAlmostEqual(rows[2]["relative_index_I"], 23 / 15)
        self.assertGreater(rows[2]["display_index_P"], 100)
        self.assertAlmostEqual(rows[1]["salary_equivalent_M"], 8580)
        for row in rows:
            self.assertAlmostEqual(sum(v["relative_index_contribution"] for v in row["criterion_explanations"].values()), row["relative_index_I"])
        self.assertFalse(result["real_selection_confirmed"])
        self.assertIsNone(result["final_top10"])

    def test_missing_value_blocks_whole_declared_cohort_without_per_criterion_mean(self):
        payload = fixture()
        payload["companies"][0]["values"]["ai_resilience"] = None
        result = calculate_selection(payload, WEIGHTS, top_n=3)
        self.assertFalse(result["calculation_completed"])
        self.assertIsNone(result["normalization"]["non_salary_means"])
        self.assertTrue(all(row["relative_index_I"] is None for row in result["company_results"]))
        self.assertEqual(result["active_weight_sum"], 5)
        self.assertEqual(result["company_results"][0]["values"]["ai_resilience"], None)

    def test_salary_missing_is_not_filled(self):
        payload = fixture()
        payload["companies"][0]["values"]["salary"] = None
        result = calculate_selection(payload, WEIGHTS, top_n=3)
        self.assertIsNone(result["company_results"][0]["salary_index"])
        self.assertEqual(result["usable_value_counts"]["salary"], 2)
        self.assertFalse(result["missing_values_imputed"])

    def test_explicit_complete_cohort_excludes_missing_outside_rows_consistently(self):
        result = calculate_selection(regional_case(), WEIGHTS)
        self.assertTrue(result["calculation_completed"])
        self.assertEqual(result["normalization"]["cohort_size"], 14)
        self.assertEqual(result["eligible_count"], 14)
        self.assertEqual(result["deferred_company_ids"], ["synthetic:015"])
        self.assertEqual(set(result["normalization"]["non_salary_means"].values()), {3.0})

    def test_regional_selection_maximizes_sum_and_replaces_lowest_avoidable_company(self):
        result = calculate_selection(regional_case(), WEIGHTS)
        selected = {r["company_id"] for r in result["selected_companies"]}
        self.assertEqual(selected, {f"synthetic:{i:03d}" for i in range(1, 9)} | {"synthetic:009", "synthetic:012"})
        self.assertEqual(result["selected_region_counts"], {"부산": 8, "울산": 1, "경남": 1})
        self.assertGreater(result["regional_constraint_cost_I"], 0)
        self.assertEqual(result["optimal_combination_count"], 1)
        chosen = next(r for r in result["company_results"] if r["company_id"] == "synthetic:012")
        self.assertTrue(chosen["selection_reason"]["regional_constraint_affected_selection"])

    def test_all_optimal_ties_keep_candidate_union_and_mandatory_region(self):
        payload = fixture(11)
        for i, row in enumerate(payload["companies"]):
            row["region"] = "부산" if i < 8 else "울산" if i < 10 else "경남"
        result = calculate_selection(payload, WEIGHTS)
        self.assertTrue(result["boundary_tie"])
        self.assertEqual(result["optimal_combination_count"], 10)
        self.assertEqual(len(result["candidate_companies"]), 11)
        self.assertEqual([c["company_id"] for c in result["mandatory_companies"]], ["synthetic:011"])
        self.assertFalse(result["selected_companies"])
        self.assertIsNone(result["final_top10"])

    def test_ties_within_exactly_ten_do_not_create_false_boundary_tie(self):
        result = calculate_selection(fixture(10), WEIGHTS)
        self.assertEqual(result["optimal_combination_count"], 1)
        self.assertEqual(len(result["selected_companies"]), 10)
        self.assertFalse(result["boundary_tie"])
        self.assertEqual({r["rank"] for r in result["ranked_companies"]}, {1})

    def test_fifty_company_tie_count_matches_independent_combination_formula(self):
        result = calculate_selection(fixture(50), WEIGHTS)
        counts = [17, 17, 16]
        expected = math.comb(50, 10) - sum(math.comb(50 - count, 10) for count in counts) + sum(math.comb(count, 10) for count in counts)
        self.assertEqual(result["optimal_combination_count"], expected)
        self.assertEqual(len(result["candidate_companies"]), 50)
        self.assertEqual(len(result["mandatory_companies"]), 0)
        self.assertFalse(result["selected_companies"])

    def test_equal_regional_alternatives_do_not_use_id_as_selection_tiebreaker(self):
        payload = fixture(11)
        for i, row in enumerate(payload["companies"]):
            row["region"] = "부산" if i < 8 else "울산" if i == 8 else "경남"
            row["values"]["salary"] = 10000 - i * 100 if i < 9 else 5000
        result = calculate_selection(payload, WEIGHTS)
        self.assertEqual(result["optimal_combination_count"], 2)
        self.assertEqual(len(result["mandatory_companies"]), 9)
        self.assertEqual(len(result["candidate_companies"]), 11)
        self.assertFalse(result["selected_companies"])

    def test_near_scores_are_not_rounded_into_ties(self):
        payload = fixture(4)
        for row in payload["companies"]:
            row["values"]["salary"] = 7150
        payload["companies"][3]["values"]["salary"] += 1e-8
        result = calculate_selection(payload, WEIGHTS, top_n=3)
        self.assertEqual(result["optimal_combination_count"], 1)
        self.assertIn("synthetic:004", {r["company_id"] for r in result["selected_companies"]})
        self.assertNotIn("synthetic:001", {r["company_id"] for r in result["selected_companies"]})

    def test_missing_region_and_company_shortfall_do_not_produce_partial_final_list(self):
        payload = fixture(12)
        for row in payload["companies"]:
            row["region"] = "부산"
        result = calculate_selection(payload, WEIGHTS)
        self.assertTrue(result["calculation_completed"])
        self.assertFalse(result["selected_companies"])
        self.assertTrue(result["selection_block_reasons"])
        result = calculate_selection(fixture(9), WEIGHTS)
        self.assertEqual(result["shortfall"], 1)
        self.assertFalse(result["selected_companies"])

    def test_mixed_job_scope_blocks_normalization(self):
        payload = fixture()
        payload["companies"][1]["target_job"] = "다른 가상 직무"
        result = calculate_selection(payload, WEIGHTS, top_n=3)
        self.assertFalse(result["calculation_completed"])
        self.assertIsNone(result["normalization"]["non_salary_means"])

    def test_input_order_does_not_change_selection_or_tie_pool(self):
        payload = fixture(11)
        first = calculate_selection(payload, WEIGHTS)
        payload["companies"].reverse()
        payload["comparison_cohort"]["company_ids"].reverse()
        second = calculate_selection(payload, WEIGHTS)
        for key in ("candidate_companies", "mandatory_companies", "optimal_combination_count", "objective_sum_I_exact"):
            self.assertEqual(first[key], second[key])

    def test_invalid_values_duplicate_ids_and_incomplete_weight_sets_are_rejected(self):
        for bad in (True, "3", 0, 6, float("inf"), float("nan")):
            with self.subTest(value=bad):
                payload = fixture()
                payload["companies"][0]["values"]["ai_resilience"] = bad
                with self.assertRaises(ValueError):
                    calculate_selection(payload, WEIGHTS, top_n=3)
        payload = fixture()
        payload["companies"][0]["company_id"] = payload["companies"][1]["company_id"]
        with self.assertRaises(ValueError):
            calculate_selection(payload, WEIGHTS, top_n=3)
        with self.assertRaises(ValueError):
            calculate_selection(fixture(), {"salary": 1, "ai_resilience": 1}, top_n=3)
        bad_weights = dict(WEIGHTS, career_capital=1, horizontal_culture=1)
        with self.assertRaises(ValueError):
            calculate_selection(fixture(), bad_weights, top_n=3)

    def test_unrepresentable_results_are_rejected_instead_of_saving_infinity(self):
        payload = fixture()
        payload["companies"][0]["values"]["salary"] = 1e308
        with self.assertRaises(ValueError):
            calculate_selection(payload, WEIGHTS, salary_baseline=1e-300, top_n=3)

    def test_unknown_or_duplicate_cohort_members_are_rejected(self):
        for ids in (["synthetic:001", "synthetic:001"], ["synthetic:999"]):
            payload = fixture()
            payload["comparison_cohort"]["company_ids"] = ids
            with self.assertRaises(ValueError):
                calculate_selection(payload, WEIGHTS, top_n=3)

    def test_real_payload_requires_selected_fifty_and_automatic_validation_gates(self):
        payload = fixture(50)
        payload["dataset_type"] = "real"
        with self.assertRaises(ValueError):
            calculate_selection(payload, WEIGHTS)
        for row in payload["companies"]:
            row["company_id"] = row["company_id"].replace("synthetic:", "selected50:")
        payload["comparison_cohort"]["company_ids"] = [c["company_id"] for c in payload["companies"]]
        result = calculate_selection(payload, WEIGHTS)
        self.assertFalse(result["calculation_completed"])
        self.assertFalse(result["real_selection_confirmed"])
        self.assertIsNone(result["final_top10"])
        self.assertTrue(all(row["relative_index_I"] is None for row in result["company_results"]))

    def test_input_values_and_denominator_are_unchanged(self):
        payload = regional_case()
        original, w = deepcopy(payload), deepcopy(WEIGHTS)
        calculate_selection(payload, WEIGHTS)
        self.assertEqual(payload, original)
        self.assertEqual(WEIGHTS, w)

    def test_dynamic_programming_matches_exhaustive_search_on_small_functional_cases(self):
        rng = random.Random(82)
        for iteration in range(15):
            payload = fixture(8)
            for row in payload["companies"]:
                row["values"]["salary"] = rng.randrange(40, 100) * 100
                for key in CRITERIA[1:]:
                    row["values"][key] = rng.randint(1, 5)
            result = calculate_selection(payload, WEIGHTS, top_n=4)
            companies = payload["companies"]
            means = {k: sum(Fraction(c["values"][k]) for c in companies) / 8 for k in CRITERIA[1:]}
            score = {c["company_id"]: (Fraction(c["values"]["salary"], 7150) +
                sum(Fraction(c["values"][k]) / means[k] for k in CRITERIA[1:])) / 5 for c in companies}
            choices = [(sum(score[c["company_id"]] for c in group), {c["company_id"] for c in group})
                       for group in combinations(companies, 4) if {c["region"] for c in group} == set(REGIONS)]
            best = max(total for total, _ in choices)
            best_sets = [ids for total, ids in choices if total == best]
            with self.subTest(iteration=iteration):
                self.assertEqual(result["objective_sum_I_exact"], str(best))
                self.assertEqual(result["optimal_combination_count"], len(best_sets))
                self.assertEqual({c["company_id"] for c in result["candidate_companies"]}, set.union(*best_sets))
                self.assertEqual({c["company_id"] for c in result["mandatory_companies"]}, set.intersection(*best_sets))
