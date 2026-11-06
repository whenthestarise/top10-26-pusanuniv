from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = ROOT / "configs" / "criteria.yaml"
COMPANY_FILE = ROOT / "data" / "processed" / "companies_clean.csv"
OUTPUT_FILE = ROOT / "data" / "input" / "criteria_evidence.csv"


def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML 내용이 비어 있거나 형식이 잘못되었습니다.")

    criteria = config.get("criteria", [])
    ids = [item["id"] for item in criteria]

    if len(ids) != 7 or len(set(ids)) != 7:
        raise ValueError("서로 다른 7개 평가 기준이 필요합니다.")

    bounds = config["criterion_weight_bounds"]
    lower = float(bounds["minimum"])
    initial = float(bounds["initial"])
    upper = float(bounds["maximum"])

    if not (0 < lower <= initial <= upper):
        raise ValueError("기준 가중치 범위를 확인하세요.")

    for criterion in criteria:
        if not criterion["collection_required"]:
            continue

        metrics = criterion.get("submetrics", [])

        if not metrics:
            raise ValueError(f"{criterion['id']}: 세부항목이 없습니다.")

        metric_ids = [metric["id"] for metric in metrics]

        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError(f"{criterion['id']}: 세부항목 ID 중복")

        weights = [
            float(metric["aggregation_weight"])
            for metric in metrics
        ]

        if any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("세부항목 비중은 유한한 양수여야 합니다.")

        if not math.isclose(sum(weights), 1.0, abs_tol=1e-8):
            raise ValueError(
                f"{criterion['id']}: 세부항목 비중의 합은 1이어야 합니다."
            )

    return config


def load_companies() -> pd.DataFrame:
    companies = pd.read_csv(
        COMPANY_FILE,
        encoding="utf-8-sig",
        dtype={"company_id": "string"},
    )

    required = {"company_id", "company_name", "region", "location"}
    missing = required - set(companies.columns)

    if missing:
        raise ValueError(f"기업 데이터에 필요한 열이 없습니다: {sorted(missing)}")

    if companies.empty:
        raise ValueError("기업 데이터가 비어 있습니다.")

    for column in required:
        blank = (
            companies[column].isna()
            | companies[column].astype("string").str.strip().eq("")
        )
        if blank.any():
            raise ValueError(f"{column}에 빈 값이 있습니다.")

    if companies["company_id"].duplicated().any():
        raise ValueError("company_id가 중복되었습니다.")

    return companies


def build_template(companies: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []

    for company in companies.to_dict(orient="records"):
        for criterion in config["criteria"]:
            if not criterion["collection_required"]:
                continue

            for metric in criterion["submetrics"]:
                rows.append({
                    "company_id": company["company_id"],
                    "company_name": company["company_name"],
                    "region": company["region"],
                    "location": company["location"],
                    "worksite": "",
                    "target_role": "",
                    "criterion_id": criterion["id"],
                    "criterion_name": criterion["name"],
                    "metric_id": metric["id"],
                    "metric_name": metric["name"],
                    "aggregation_weight": metric["aggregation_weight"],
                    "scoring_guide": metric["scoring_guide"],
                    "raw_value": "",
                    "raw_unit": "",
                    "rating_score": "",
                    "source_name": "",
                    "source_url": "",
                    "evidence_date": "",
                    "evidence_text": "",
                    "rating_reason": "",
                    "reviewer": "",
                    "reviewed_at": "",
                })

    return pd.DataFrame(rows)


def main() -> None:
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"기존 수집표가 있어 중단했습니다. 덮어쓰지 않습니다: {OUTPUT_FILE}"
        )

    config = load_config()
    companies = load_companies()
    template = build_template(companies, config)

    collection_criteria = [
        item for item in config["criteria"]
        if item["collection_required"]
    ]

    metrics_per_company = sum(
        len(item["submetrics"])
        for item in collection_criteria
    )

    expected_rows = len(companies) * metrics_per_company

    if len(template) != expected_rows:
        raise ValueError("생성된 행 수가 예상과 다릅니다.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # mode="x": 실행 도중 같은 이름의 파일이 생겨도 덮어쓰지 않습니다.
    template.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    print("2단계 근거 수집표 생성이 완료되었습니다.")
    print(f"기업 수: {len(companies)}")
    print(f"수집 대상 기준 수: {len(collection_criteria)}")
    print(f"기업별 세부항목 수: {metrics_per_company}")
    print(f"전체 수집 행 수: {len(template)}")
    print(f"출력 파일: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()