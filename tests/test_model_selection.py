import unittest

from src.model_selection import select_for_pilot


def report(name, passed=19, reading=6, memory=2.0):
    return {"candidate": {"candidate_id": name}, "functional_count": 19, "functional_passed": passed,
            "provided_data_count": 6, "provided_data_passed": reading, "model_peak_gib": memory,
            "wall_seconds": 10}


class TestModelSelection(unittest.TestCase):
    def test_functional_success_cannot_replace_source_reading_fidelity(self):
        result = select_for_pilot([report("오류 후보", reading=5), report("읽기 후보", passed=18)])
        self.assertEqual(result["candidate"]["candidate_id"], "읽기 후보")
        self.assertFalse(result["functional_scoring_checks_passed"])
        self.assertFalse(result["ready_for_ranking"])

    def test_more_correct_functional_cases_take_priority_over_memory(self):
        result = select_for_pilot([report("작은 후보", passed=17, memory=1), report("정확한 후보", memory=3)])
        self.assertEqual(result["candidate"]["candidate_id"], "정확한 후보")
        self.assertFalse(result["real_company_evaluation_accuracy_validated"])

    def test_incomplete_or_failed_candidates_cannot_be_selected(self):
        self.assertIsNone(select_for_pilot([{"status": "runtime_error"}])["candidate"])
        partial = report("미완료")
        partial["functional_count"] = 18
        self.assertIsNone(select_for_pilot([partial])["candidate"])
