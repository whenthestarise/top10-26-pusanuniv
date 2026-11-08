from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CRITERION_IDS = (
    "salary",
    "infrastructure",
    "work_life_balance",
    "ai_resilience",
    "job_outlook",
    "horizontal_culture",
    "career_capital",
)

OUTPUT_PATH = ROOT / "data" / "input" / "demo_company_scores.json"


def make_company(number: int, score: float) -> dict:
    return {
        "company_id": f"DEMO-{number:03d}",
        "company_name": f"가상기업 {number:02d}",
        "scores": {
            criterion_id: score
            for criterion_id in CRITERION_IDS
        },
    }


def main() -> None:
    # 앞의 9개 기업은 95점부터 55점까지입니다.
    base_scores = [
        95, 90, 85, 80, 75, 70, 65, 60, 55,
        50, 50,  # TOP10 경계에서 동점 발생
        40, 30, 0,
    ]

    companies = [
        make_company(number, score)
        for number, score in enumerate(base_scores, start=1)
    ]

    # 마지막 기업은 한 기준이 누락되어 평가 보류되어야 합니다.
    incomplete_company = make_company(15, 85)
    incomplete_company["scores"]["work_life_balance"] = None
    companies.append(incomplete_company)

    payload = {
        "schema_version": "1.0",
        "dataset_type": "synthetic",
        "description": (
            "점수 계산 및 저장 기능을 확인하기 위한 가상 데이터. "
            "실제 기업 평가에 사용하지 않습니다."
        ),
        "companies": companies,
    }

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"이미 파일이 있습니다: {OUTPUT_PATH}\n"
            "기존 내용을 확인한 뒤 필요하면 파일명을 변경하세요."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"가상 기업 수: {len(companies)}")
    print(f"저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()