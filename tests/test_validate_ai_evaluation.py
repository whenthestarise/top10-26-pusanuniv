import unittest

from src.validate_ai_evaluation import (
    CRITERIA,
    validate_evaluation,
)


class TestAIEvaluation(unittest.TestCase):
    def setUp(self):
        self.config = {
            "rubric_version": "test_v1",
            "scope": {
                "required": [
                    "company_id",
                    "company_name",
                    "workplace",
                    "target_job",
                ]
            },
            "rating_scale": {
                "allowed_values": [1, 2, 3, 4, 5]
            },
        }

        context = {
            "company_id": "TEST-001",
            "company_name": "가상기업",
            "workplace": "가상 사업장",
            "target_job": "생산기술",
        }

        self.payload = {
            "context": context,
            "sources": [
                {
                    "source_id": "TEST-SOURCE",
                    "title": "형식 검사 전용 가상 자료",
                    "locator": "가상 문서 1쪽",
                    "retrieved_at": "2026-11-09",
                    "text": "이 문장은 인용 일치 검사 전용입니다.",
                }
            ],
            "evaluation": {
                **context,
                "rubric_version": "test_v1",
                "criteria": {
                    key: {
                        "status": "assessed",
                        "score": 3,
                        "reason": "형식 검사 전용 평가입니다.",
                        "evidence": [
                            {
                                "source_id": "TEST-SOURCE",
                                "quote": (
                                    "이 문장은 인용 일치 검사 전용입니다."
                                ),
                            }
                        ],
                    }
                    for key in CRITERIA
                },
            },
        }

    def assert_blocked(self, payload):
        errors, scores = validate_evaluation(
            payload, self.config
        )
        self.assertTrue(errors)
        self.assertTrue(
            all(value is None for value in scores.values())
        )

    def test_valid_structure_and_quote(self):
        """형식·인용 일치만 검사하며 평가 타당성을 검증하지 않습니다."""
        errors, scores = validate_evaluation(
            self.payload, self.config
        )
        self.assertEqual(errors, [])
        self.assertTrue(
            all(value == 3 for value in scores.values())
        )

    def test_list_status_is_rejected_without_crash(self):
        result = self.payload["evaluation"]["criteria"]
        result["work_life_balance"]["status"] = []

        self.assert_blocked(self.payload)

    def test_invalid_quote_blocks_all_scores(self):
        result = self.payload["evaluation"]["criteria"]
        result["work_life_balance"]["evidence"][0]["quote"] = (
            "원문에 존재하지 않는 문장입니다."
        )

        self.assert_blocked(self.payload)

    def test_unknown_source_blocks_all_scores(self):
        result = self.payload["evaluation"]["criteria"]
        result["infrastructure"]["evidence"][0]["source_id"] = (
            "UNKNOWN"
        )

        self.assert_blocked(self.payload)

    def test_context_mismatch_blocks_all_scores(self):
        self.payload["evaluation"]["company_id"] = "OTHER"

        self.assert_blocked(self.payload)

    def test_missing_evidence_stays_unscored(self):
        self.payload["sources"] = []

        for result in self.payload["evaluation"]["criteria"].values():
            result["status"] = "insufficient_evidence"
            result["score"] = None
            result["reason"] = "자료가 없습니다."
            result["evidence"] = []

        errors, scores = validate_evaluation(
            self.payload, self.config
        )

        self.assertEqual(errors, [])
        self.assertTrue(
            all(value is None for value in scores.values())
        )


if __name__ == "__main__":
    unittest.main()