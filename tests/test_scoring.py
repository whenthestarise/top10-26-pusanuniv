import unittest

from src.scoring import (
    CRITERION_IDS,
    rank_companies,
    score_company,
)


def make_company(company_id, score):
    return {
        "company_id": str(company_id),
        "company_name": f"가상기업 {company_id}",
        "scores": {key: score for key in CRITERION_IDS},
    }


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.weights = {key: 1.0 for key in CRITERION_IDS}

    def test_equal_weights_match_average(self):
        company = make_company("A", 0)
        values = [10, 20, 30, 40, 50, 60, 70]
        company["scores"] = dict(zip(CRITERION_IDS, values))

        result = score_company(company, self.weights)

        self.assertAlmostEqual(result["final_score"], 40.0)

    def test_weighted_score_and_monotonicity(self):
        weights = self.weights.copy()
        weights["salary"] = 1.1
        weights["career_capital"] = 0.9

        company = make_company("A", 0)
        company["scores"]["salary"] = 100

        result = score_company(company, weights)
        self.assertAlmostEqual(result["final_score"], 110 / 7)

        company["scores"]["infrastructure"] = 70
        increased = score_company(company, weights)
        self.assertGreater(
            increased["final_score"], result["final_score"]
        )

    def test_missing_score_is_deferred(self):
        company = make_company("A", 70)
        company["scores"]["salary"] = None

        result = score_company(company, self.weights)

        self.assertEqual(result["status"], "평가 보류")
        self.assertIsNone(result["final_score"])
        self.assertIn("salary", result["missing_criteria"])

    def test_zero_score_is_valid(self):
        result = score_company(make_company("A", 0), self.weights)

        self.assertEqual(result["status"], "계산 완료")
        self.assertEqual(result["final_score"], 0)

    def test_boundary_tie_includes_all_tied_companies(self):
        companies = [
            make_company(index, 100 - index)
            for index in range(1, 10)
        ]
        companies += [
            make_company("10", 80),
            make_company("11", 80),
            make_company("12", 70),
        ]

        result = rank_companies(companies, self.weights)

        self.assertTrue(result["boundary_tie"])
        self.assertEqual(result["candidate_count"], 11)
        self.assertEqual(
            [item["rank"] for item in result["top_candidates"][-2:]],
            [10, 10],
        )

    def test_shortfall_and_deferral(self):
        companies = [make_company("A", 80), make_company("B", 60)]
        companies[1]["scores"]["salary"] = None

        result = rank_companies(companies, self.weights)

        self.assertEqual(result["eligible_count"], 1)
        self.assertEqual(result["deferred_count"], 1)
        self.assertEqual(result["shortfall"], 9)

    def test_invalid_input_is_rejected(self):
        for invalid in [-1, 101, float("nan"), True, "80"]:
            with self.subTest(invalid=invalid):
                company = make_company("A", 70)
                company["scores"]["salary"] = invalid
                with self.assertRaises(ValueError):
                    score_company(company, self.weights)

    def test_duplicate_ids_are_rejected(self):
        companies = [make_company("A", 80), make_company("A", 70)]

        with self.assertRaises(ValueError):
            rank_companies(companies, self.weights)

    def test_invalid_weights_are_rejected(self):
        weights = self.weights.copy()
        weights["salary"] = 1.2

        with self.assertRaises(ValueError):
            score_company(make_company("A", 70), weights)


if __name__ == "__main__":
    unittest.main()