from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "criteria.yaml"
COMPANY_PATH = ROOT / "data" / "processed" / "company_analysis_inputs.json"

CSV_PATH = ROOT / "data" / "input" / "research_evaluations.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "company_scores.json"

FIELDS = [
    "company_id",
    "company_name",
    "criterion_id",
    "criterion_name",
    "submetric_id",
    "submetric_name",
    "score",
    "source",
    "reference_date",
    "reason",
    "review_status",
    "reviewer",
]


def load_settings():
    with CONFIG_PATH.open(encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    with COMPANY_PATH.open(encoding="utf-8-sig") as file:
        dataset = json.load(file)

    companies = dataset["companies"]
    criteria = config["criteria"]

    company_ids = [str(company["company_id"]) for company in companies]
    criterion_ids = [criterion["id"] for criterion in criteria]

    if len(company_ids) != len(set(company_ids)):
        raise ValueError("기업 데이터에 중복된 company_id가 있습니다.")

    if len(criterion_ids) != len(set(criterion_ids)):
        raise ValueError("criteria.yaml에 중복 기준 ID가 있습니다.")

    # 설정 파일의 세부 가중치도 계산 전에 확인합니다.
    for criterion in criteria:
        submetrics = criterion.get("submetrics", [])
        submetric_ids = [item["id"] for item in submetrics]

        if len(submetric_ids) != len(set(submetric_ids)):
            raise ValueError(f"{criterion['id']}: 세부항목 ID 중복")

        for item in submetrics:
            weight = item["aggregation_weight"]
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight <= 0
            ):
                raise ValueError(
                    f"{criterion['id']}/{item['id']}: "
                    "세부 가중치는 유한한 양수여야 합니다."
                )

    return config, companies, criteria


def get_items(criterion):
    # 연봉·AI 안전성은 현재 세부항목이 없으므로
    # 검증된 기준 점수를 직접 입력하는 행을 만듭니다.
    return criterion.get("submetrics") or [
        {
            "id": "direct",
            "name": "검증된 기준 점수",
            "aggregation_weight": 1.0,
        }
    ]


def make_template(companies, criteria):
    if CSV_PATH.exists():
        raise FileExistsError(f"기존 평가 파일을 보존합니다: {CSV_PATH}")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with CSV_PATH.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for company in companies:
            for criterion in criteria:
                for item in get_items(criterion):
                    writer.writerow({
                        "company_id": str(company["company_id"]),
                        "company_name": company["company_name"],
                        "criterion_id": criterion["id"],
                        "criterion_name": criterion["name"],
                        "submetric_id": item["id"],
                        "submetric_name": item["name"],
                        "score": "",
                        "source": "",
                        "reference_date": "",
                        "reason": "",
                        "review_status": "미검토",
                        "reviewer": "",
                    })
                    row_count += 1

    print(f"평가 양식 생성: {row_count}행")
    print(f"저장: {CSV_PATH}")


def read_evaluations(companies, criteria):
    expected_keys = {
        (str(company["company_id"]), criterion["id"], item["id"])
        for company in companies
        for criterion in criteria
        for item in get_items(criterion)
    }

    rows = {}

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        missing_columns = set(FIELDS) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"CSV 열 누락: {sorted(missing_columns)}")

        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"CSV {line_number}행: 열 개수가 맞지 않습니다.")

            row = {
                key: (value or "").strip()
                for key, value in raw.items()
            }

            key = (
                row["company_id"],
                row["criterion_id"],
                row["submetric_id"],
            )

            if key not in expected_keys:
                raise ValueError(f"CSV {line_number}행: 알 수 없는 ID {key}")

            if key in rows:
                raise ValueError(f"CSV {line_number}행: 중복 평가 {key}")

            rows[key] = row

    return rows


def get_verified_score(row, criterion, config):
    if row is None:
        return None

    # 검토가 끝나지 않은 점수는 계산에 사용하지 않습니다.
    if row["review_status"] != "검증 완료":
        return None

    required = (
        "score",
        "source",
        "reference_date",
        "reason",
        "reviewer",
    )

    if any(not row[field] for field in required):
        return None

    try:
        score = float(row["score"])
    except ValueError as exc:
        raise ValueError(
            f"{row['company_id']}/{criterion['id']}: 점수가 숫자가 아닙니다."
        ) from exc

    scale = config["rating_scale"]

    if (
        not math.isfinite(score)
        or not scale["minimum"] <= score <= scale["maximum"]
    ):
        raise ValueError(
            f"{row['company_id']}/{criterion['id']}: 점수 범위 오류"
        )

    # 사람이 평가하는 세부항목에는 5단계 척도를 적용합니다.
    # 연봉 백분위 등 직접 입력하는 기준 점수는 소수점도 허용합니다.
    if criterion.get("submetrics"):
        if score not in scale["allowed_values"]:
            raise ValueError(
                f"{row['company_id']}/{criterion['id']}/"
                f"{row['submetric_id']}: "
                f"허용 점수는 {scale['allowed_values']}입니다."
            )

    return score


def build_scores(config, companies, criteria, *, dataset_type):
    if dataset_type not in ("real", "synthetic"):
        raise ValueError(
            "dataset_type은 real 또는 synthetic이어야 합니다."
        )
    
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"기존 결과 파일을 보존합니다: {OUTPUT_PATH}")

    evaluations = read_evaluations(companies, criteria)
    results = []

    for company in companies:
        company_id = str(company["company_id"])
        scores = {}
        evidence = {}

        for criterion in criteria:
            criterion_id = criterion["id"]
            weighted_values = []
            weights = []
            used_rows = []
            complete = True

            for item in get_items(criterion):
                key = (company_id, criterion_id, item["id"])
                row = evaluations.get(key)
                score = get_verified_score(row, criterion, config)

                if score is None:
                    complete = False
                    continue

                weight = float(item["aggregation_weight"])
                weighted_values.append(score * weight)
                weights.append(weight)
                used_rows.append(row)

            if complete:
                scores[criterion_id] = (
                    math.fsum(weighted_values) / math.fsum(weights)
                )
                evidence[criterion_id] = used_rows
            else:
                # 일부 세부항목만으로 점수를 추정하지 않습니다.
                scores[criterion_id] = None
                evidence[criterion_id] = []

        results.append({
            "company_id": company_id,
            "company_name": company["company_name"],
            "scores": scores,
            "evidence": evidence,
        })

    payload = {
        "schema_version": "1.0",
        "dataset_type": dataset_type,
        "evaluation_file": str(CSV_PATH),
        "note": (
            "CSV의 검증 완료 표시와 필수 근거 입력 여부를 확인한 결과입니다. "
            "프로그램이 근거의 진실성을 자동 검증한 것은 아닙니다."
        ),
        "companies": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    ready_count = sum(
        all(score is not None for score in company["scores"].values())
        for company in results
    )

    print(f"전체 기업: {len(results)}개")
    print(f"데이터 종류: {dataset_type}")
    print(f"7개 기준 점수 확보: {ready_count}개")
    print(f"평가 보류 대상: {len(results) - ready_count}개")
    print(f"저장: {OUTPUT_PATH}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=["init", "build"],
    )

    parser.add_argument(
        "--dataset-type",
        choices=["real", "synthetic"],
        help="real: 실제 기업 평가 자료, synthetic: 가상 시험 자료",
    )

    args = parser.parse_args()

    # 파일을 읽거나 쓰기 전에 명령을 검사합니다.
    if args.mode == "build" and args.dataset_type is None:
        parser.error(
            "build에는 --dataset-type real 또는 "
            "--dataset-type synthetic을 지정해야 합니다."
        )

    if args.mode == "init" and args.dataset_type is not None:
        parser.error(
            "--dataset-type은 build에서만 사용합니다."
        )

    config, companies, criteria = load_settings()

    if args.mode == "init":
        make_template(companies, criteria)
    else:
        build_scores(
            config,
            companies,
            criteria,
            dataset_type=args.dataset_type,
        )

if __name__ == "__main__":
    main()