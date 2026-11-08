import unittest

from src.prepare_scoring_input import get_verified_score


class TestResearchEvaluation(unittest.TestCase):
    def setUp(self):
        self.config = {
            "rating_scale": {
                "minimum": 0,
                "maximum": 100,
                "allowed_values": [0, 25, 50, 75, 100],
            }
        }

        self.criterion = {
            "id": "work_life_balance",
            "submetrics": [
                {
                    "id": "working_hours",
                    "aggregation_weight": 0.34,
                }
            ],
        }

        # 실제 기업의 평가 자료가 아닌 테스트용 입력입니다.
        self.row = {
            "company_id": "TEST-001",
            "submetric_id": "working_hours",
            "score": "75",
            "source": "테스트용 문서 1쪽",
            "reference_date": "2026-11-01",
            "reason": "테스트용 평가 근거",
            "review_status": "검증 완료",
            "reviewer": "테스트 검토자",
        }

    def evaluate(self, row, criterion=None):
        return get_verified_score(
            row,
            self.criterion if criterion is None else criterion,
            self.config,
        )

    def test_verified_score_is_used(self):
        """필수 정보가 있고 검증 완료이면 점수를 사용합니다."""
        self.assertEqual(self.evaluate(self.row), 75.0)

    def test_unreviewed_score_is_excluded(self):
        """점수가 있어도 미검토 상태이면 사용하지 않습니다."""
        row = dict(self.row, review_status="미검토")
        self.assertIsNone(self.evaluate(row))

    def test_missing_evidence_is_excluded(self):
        """필수 정보가 하나라도 없으면 사용하지 않습니다."""
        for field in (
            "score",
            "source",
            "reference_date",
            "reason",
            "reviewer",
        ):
            with self.subTest(field=field):
                row = dict(self.row)
                row[field] = ""
                self.assertIsNone(self.evaluate(row))

        self.assertIsNone(self.evaluate(None))

    def test_invalid_score_is_rejected(self):
        """잘못된 숫자는 조용히 계산하지 않고 오류를 냅니다."""
        for value in ("-1", "101", "nan", "inf", "점수 없음"):
            with self.subTest(value=value):
                row = dict(self.row, score=value)
                with self.assertRaises(ValueError):
                    self.evaluate(row)

    def test_manual_score_requires_allowed_value(self):
        """수동 세부항목 평가에 60점 등 임의 점수를 허용하지 않습니다."""
        row = dict(self.row, score="60")
        with self.assertRaises(ValueError):
            self.evaluate(row)

    def test_zero_is_valid(self):
        """근거를 갖춘 0점은 결측값과 구분합니다."""
        row = dict(self.row, score="0")
        self.assertEqual(self.evaluate(row), 0.0)

    def test_direct_score_allows_decimal(self):
        """연봉 백분위 같은 직접 입력 점수는 소수점을 허용합니다."""
        criterion = {"id": "salary"}
        row = dict(self.row, submetric_id="direct", score="72.5")
        self.assertEqual(self.evaluate(row, criterion), 72.5)


if __name__ == "__main__":
    unittest.main()