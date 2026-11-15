import unittest

from src.ai_task_cases import development_cases
from src.ai_task_profile import combine_profiles, share_mentions
from src.assess_ai_resilience import assess
from src.extract_ai_evidence import extract_evidence, prepare_bundle


def profile(kind, share="비중 미상"):
    return {"task_type": kind, "share": share}


class TestTaskProfile(unittest.TestCase):
    def test_all_five_scores_have_distinct_grounded_paths(self):
        for kind, share, expected in (("반복 사무", "대부분", 1), ("반복 사무", "상당 부분", 2),
                                      ("혼합 업무", "비중 미상", 3), ("현장 작업", "주로", 4),
                                      ("사람 판단", "대부분", 5)):
            with self.subTest(kind=kind, share=share):
                self.assertEqual(combine_profiles([profile(kind, share)])["score"], expected)

    def test_single_task_without_share_is_not_middle_score(self):
        for kind in ("반복 사무", "현장 작업", "사람 판단", "정보 부족"):
            self.assertIsNone(combine_profiles([profile(kind)])["score"])

    def test_mixed_work_does_not_require_invented_time_percentages(self):
        result = combine_profiles([profile("반복 사무"), profile("사람 판단")])
        self.assertEqual(result["score"], 3)

    def test_explicit_dominant_task_takes_precedence_over_incidental_task(self):
        result = combine_profiles([profile("현장 작업", "대부분"), profile("반복 사무")])
        self.assertEqual(result["score"], 5)

    def test_opposing_dominant_claims_are_not_averaged(self):
        result = combine_profiles([profile("현장 작업", "대부분"), profile("반복 사무", "대부분")])
        self.assertEqual(result["status"], "conflicting_evidence")
        self.assertIsNone(result["score"])

    def test_ambiguous_share_of_mixed_task_cannot_be_attached_to_either_type(self):
        result = combine_profiles([profile("혼합 업무", "대부분")])
        self.assertIsNone(result["score"])

    def test_share_evidence_is_an_exact_slice_including_spaces(self):
        text = "업무의 상당  부분은 기록 처리다. 나머지는 미상이다."
        mentions = share_mentions(text)
        self.assertEqual(mentions[0]["label"], "상당 부분")
        for item in mentions:
            self.assertEqual(text[item["start"]:item["end"]], item["quote"])
        self.assertEqual(share_mentions("업무 시간의 구십 퍼센트다."), [])

    def extraction(self, key):
        case = next(item for item in development_cases() if item["case_id"] == key)
        return extract_evidence(prepare_bundle(case["bundle"]), lambda _: {"answer": "업무 수행"})

    def test_model_cannot_invent_a_different_share(self):
        answers = iter(["현장 작업", "대부분"])
        result = assess(self.extraction("mainly"), lambda _: {"answer": next(answers)})
        self.assertEqual(result["status"], "invalid_output")
        self.assertIsNone(result["proposed_score"])

    def test_share_model_is_skipped_when_source_has_no_share(self):
        calls = []
        def generate(prompt):
            calls.append(prompt)
            return {"answer": "현장 작업"}
        result = assess(self.extraction("unknown_share"), generate)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(result["proposed_score"])
        self.assertEqual(result["task_profiles"][0]["share"], "비중 미상")

    def test_model_can_abstain_even_when_a_share_word_exists(self):
        answers = iter(["현장 작업", "아니오"])
        result = assess(self.extraction("physical"), lambda _: {"answer": next(answers)})
        self.assertIsNone(result["proposed_score"])
        self.assertEqual(result["task_profiles"][0]["share_evidence"], [])

    def test_employee_share_is_not_job_share(self):
        self.assertEqual(share_mentions("대부분의 직원이 직접 수리한다."), [])
        self.assertEqual(share_mentions("직원의 대부분이 직접 수리한다."), [])
        self.assertEqual(share_mentions("직원의 업무 대부분은 직접 수리다.")[0]["label"], "대부분")


if __name__ == "__main__":
    unittest.main()
