import unittest

from src.model_role_balance import aggregate_roles, parse_score, roles_of


class TestModelRoleBalance(unittest.TestCase):
    def record(self):
        return {"company_id": "가상기업", "main_hiring_roles": "연구, 생산, 설계",
                "raw_fields": {"E": "가상 산업", "I": "연구, 생산, 설계"}}

    def test_only_complete_half_step_model_answers_are_accepted(self):
        self.assertEqual(parse_score(" 3.5 "), 3.5)
        for value in (None, True, "3점", "3.3", "0", "6", "NaN", "3\n4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_score(value)

    def test_role_average_is_equal_weighted_and_rounded_once(self):
        result = aggregate_roles(self.record(), {"연구": {"score": 4}, "생산": {"score": 2}, "설계": {"score": 4.5}})
        self.assertEqual(result["score"], 3.5)
        self.assertEqual(len(result["role_scores"]), 3)
        self.assertFalse(result["empirically_verified"])
        self.assertEqual(result["basis_quotes"]["I"], self.record()["raw_fields"]["I"])

    def test_missing_or_invalid_role_does_not_get_neutral_score(self):
        with self.assertRaises(ValueError):
            aggregate_roles(self.record(), {})
        for invalid in (0, 6, 3.1, float("nan")):
            ratings = {r: {"score": invalid} for r in roles_of(self.record())}
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                aggregate_roles(self.record(), ratings)

    def test_duplicated_or_empty_roles_are_rejected(self):
        for value in ("", "연구, 연구"):
            record = self.record()
            record["main_hiring_roles"] = value
            with self.assertRaises(ValueError):
                roles_of(record)
