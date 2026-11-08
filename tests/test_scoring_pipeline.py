from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src import prepare_scoring_input as preparation
from src import scoring


class TestScoringPipeline(unittest.TestCase):
    def test_research_csv_to_ranking(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)

            config_path = folder / "criteria.yaml"
            company_path = folder / "companies.json"
            csv_path = folder / "research_evaluations.csv"
            scores_path = folder / "company_scores.json"
            weights_path = folder / "weights.yaml"
            ranking_path = folder / "ranking.json"

            # 독립적인 테스트용 설정입니다.
            criterion_ids = (
                "salary",
                "infrastructure",
                "work_life_balance",
                "ai_resilience",
                "job_outlook",
                "horizontal_culture",
                "career_capital",
            )

            criteria = []

            for criterion_id in criterion_ids:
                criterion = {
                    "id": criterion_id,
                    "name": criterion_id,
                }

                if criterion_id not in ("salary", "ai_resilience"):
                    criterion["submetrics"] = [
                        {
                            "id": "item_a",
                            "name": "세부항목 A",
                            "aggregation_weight": 0.34,
                        },
                        {
                            "id": "item_b",
                            "name": "세부항목 B",
                            "aggregation_weight": 0.33,
                        },
                        {
                            "id": "item_c",
                            "name": "세부항목 C",
                            "aggregation_weight": 0.33,
                        },
                    ]

                criteria.append(criterion)

            config = {
                "rating_scale": {
                    "minimum": 0,
                    "maximum": 100,
                    "allowed_values": [0, 25, 50, 75, 100],
                },
                "criteria": criteria,
            }

            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True),
                encoding="utf-8",
            )

            company_path.write_text(
                json.dumps(
                    {
                        "companies": [
                            {
                                "company_id": "TEST-001",
                                "company_name": "가상기업 A",
                            },
                            {
                                "company_id": "TEST-002",
                                "company_name": "가상기업 B",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            # 급여만 1.1, 경력 확보만 0.9, 나머지는 1.0입니다.
            weights = {
                criterion_id: 1.0
                for criterion_id in criterion_ids
            }
            weights["salary"] = 1.1
            weights["career_capital"] = 0.9

            weights_path.write_text(
                yaml.safe_dump({"weights": weights}),
                encoding="utf-8",
            )

            # 변환기의 파일 경로만 임시 폴더로 바꿉니다.
            with patch.multiple(
                preparation,
                CONFIG_PATH=config_path,
                COMPANY_PATH=company_path,
                CSV_PATH=csv_path,
                OUTPUT_PATH=scores_path,
            ):
                loaded_config, companies, loaded_criteria = (
                    preparation.load_settings()
                )

                preparation.make_template(companies, loaded_criteria)

                with csv_path.open(
                    encoding="utf-8-sig", newline=""
                ) as file:
                    rows = list(csv.DictReader(file))

                # 기업당 17개 평가 행 × 2개 기업
                self.assertEqual(len(rows), 34)

                for row in rows:
                    if row["criterion_id"] == "salary":
                        row["score"] = "80"
                    elif row["criterion_id"] == "ai_resilience":
                        row["score"] = "60"
                    else:
                        row["score"] = {
                            "item_a": "100",
                            "item_b": "75",
                            "item_c": "25",
                        }[row["submetric_id"]]

                    row["source"] = "가상 평가 문서 1쪽"
                    row["reference_date"] = "2026-11-01"
                    row["reason"] = "자동 테스트용 근거"
                    row["review_status"] = "검증 완료"
                    row["reviewer"] = "테스트 검토자"

                    # B 기업은 워라밸 세부항목 하나가 미검토입니다.
                    if (
                        row["company_id"] == "TEST-002"
                        and row["criterion_id"] == "work_life_balance"
                        and row["submetric_id"] == "item_a"
                    ):
                        row["review_status"] = "미검토"

                with csv_path.open(
                    "w", encoding="utf-8-sig", newline=""
                ) as file:
                    writer = csv.DictWriter(
                        file, fieldnames=preparation.FIELDS
                    )
                    writer.writeheader()
                    writer.writerows(rows)

                preparation.build_scores(
                    loaded_config,
                    companies,
                    loaded_criteria,
                    dataset_type="synthetic",
                )

            converted = json.loads(
                scores_path.read_text(encoding="utf-8")
            )
            company_a, company_b = converted["companies"]

            # 100×0.34 + 75×0.33 + 25×0.33 = 67점
            self.assertAlmostEqual(
                company_a["scores"]["work_life_balance"], 67.0
            )
            self.assertEqual(company_a["scores"]["salary"], 80.0)
            self.assertEqual(
                company_a["scores"]["ai_resilience"], 60.0
            )

            # 일부 세부항목만으로 워라밸 점수를 계산하면 안 됩니다.
            self.assertIsNone(
                company_b["scores"]["work_life_balance"]
            )

            # 사용한 근거가 변환 결과에 남아 있는지도 확인합니다.
            self.assertEqual(
                len(company_a["evidence"]["work_life_balance"]), 3
            )

            # dataset type을 지정합니다.
            self.assertEqual(converted["dataset_type"], "synthetic")

            # 기존 scoring.py의 실행 진입점까지 호출합니다.
            with patch(
                "sys.argv",
                [
                    "scoring.py",
                    "--scores", str(scores_path),
                    "--weights", str(weights_path),
                    "--output", str(ranking_path),
                    "--top-n", "10",
                ],
            ):
                scoring.main()

            ranking = json.loads(
                ranking_path.read_text(encoding="utf-8")
            )

            self.assertEqual(ranking["eligible_count"], 1)
            self.assertEqual(ranking["deferred_count"], 1)
            self.assertEqual(ranking["candidate_count"], 1)
            self.assertEqual(ranking["shortfall"], 9)
            self.assertFalse(ranking["boundary_tie"])
            self.assertEqual(ranking["dataset_type"], "synthetic")

            winner = ranking["top_candidates"][0]
            self.assertEqual(winner["company_id"], "TEST-001")

            # (80×1.1 + 60×1 + 67×4 + 67×0.9) / 7
            self.assertAlmostEqual(
                winner["final_score"], 476.3 / 7
            )

            deferred = ranking["deferred_companies"][0]
            self.assertEqual(deferred["company_id"], "TEST-002")
            self.assertIn(
                "work_life_balance", deferred["missing_criteria"]
            )

    # 잘못된 자료 종류를 주면 파일 처리 전에 오류가 발생하는지를 확인하는 과정입니다.
    def test_invalid_dataset_type_is_rejected(self):
        for invalid_type in ("", "unknown", None):
            with self.subTest(dataset_type=invalid_type):
                with self.assertRaises(ValueError):
                    preparation.build_scores(
                        {},
                        [],
                        [],
                        dataset_type=invalid_type,
                    )

if __name__ == "__main__":
    unittest.main()