from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = ROOT / "configs" / "document_model.yaml"
DEFAULT_OUTPUT = (
    ROOT / "data" / "processed" / "document_survey_weights.json"
)

EXPECTED_IDS = {
    "salary",
    "work_life_balance",
    "infrastructure",
    "job_outlook",
    "career_capital",
    "ai_resilience",
    "horizontal_culture",
}

EXPECTED_ACTIVE_IDS = EXPECTED_IDS - {
    "career_capital",
    "horizontal_culture",
}


def require_integer(value, label, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"{label}: {minimum} 이상의 정수여야 합니다."
        )
    return value


def calculate_weights(config):
    survey = config["survey"]
    criteria = config["criteria"]

    important_total = require_integer(
        survey["important_total"], "중요 선택 총수", minimum=1
    )
    unimportant_total = require_integer(
        survey["unimportant_total"], "비중요 선택 총수", minimum=1
    )

    coefficient = survey["coefficient"]

    if (
        isinstance(coefficient, bool)
        or not isinstance(coefficient, (int, float))
        or not math.isfinite(coefficient)
        or not 0 <= coefficient <= 0.1
    ):
        raise ValueError("가감 계수는 0 이상 0.1 이하의 유한한 숫자여야 합니다.")

    ids = [item["id"] for item in criteria]

    if len(ids) != len(set(ids)):
        raise ValueError("중복된 기준 ID가 있습니다.")

    if set(ids) != EXPECTED_IDS:
        raise ValueError("문서의 7개 기준 ID와 일치하지 않습니다.")

    for item in criteria:
        if not isinstance(item["active"], bool):
            raise ValueError(
                f"{item['id']}: active는 true 또는 false여야 합니다."
            )

        require_integer(
            item["important_count"],
            f"{item['id']} 중요 선택 수",
        )
        require_integer(
            item["unimportant_count"],
            f"{item['id']} 비중요 선택 수",
        )

    active_ids = {
        item["id"] for item in criteria if item["active"]
    }

    if active_ids != EXPECTED_ACTIVE_IDS:
        raise ValueError("활성 기준이 문서의 5개 기준과 다릅니다.")

    important_sum = sum(
        item["important_count"] for item in criteria
    )
    unimportant_sum = sum(
        item["unimportant_count"] for item in criteria
    )

    if important_sum != important_total:
        raise ValueError(
            f"중요 선택 합계 불일치: {important_sum} != {important_total}"
        )

    if unimportant_sum != unimportant_total:
        raise ValueError(
            f"비중요 선택 합계 불일치: "
            f"{unimportant_sum} != {unimportant_total}"
        )

    rows = []

    for item in criteria:
        important_ratio = (
            item["important_count"] / important_total
        )
        unimportant_ratio = (
            item["unimportant_count"] / unimportant_total
        )

        weight = (
            1
            + coefficient * important_ratio
            - coefficient * unimportant_ratio
        )

        rows.append({
            "criterion_id": item["id"],
            "criterion_name": item["name"],
            "active": item["active"],
            "important_count": item["important_count"],
            "unimportant_count": item["unimportant_count"],
            "weight": weight,
        })

    active_weight_sum = math.fsum(
        row["weight"] for row in rows if row["active"]
    )

    for row in rows:
        row["normalized_active_weight"] = (
            row["weight"] / active_weight_sum
            if row["active"]
            else None
        )

    # 7개 항목 전체의 가산 합과 차감 합은 같으므로
    # 전체 가중치 합은 7이어야 합니다.
    all_weight_sum = math.fsum(row["weight"] for row in rows)

    if not math.isclose(
        all_weight_sum, 7.0, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("전체 7개 가중치 합 검산에 실패했습니다.")

    normalized_sum = math.fsum(
        row["normalized_active_weight"]
        for row in rows
        if row["active"]
    )

    if not math.isclose(
        normalized_sum, 1.0, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("활성 기준의 정규화 비중 합 검산에 실패했습니다.")

    return {
        "schema_version": "1.0",
        "model_id": config["model_id"],
        "response_policy": survey["response_policy"],
        "important_total": important_total,
        "unimportant_total": unimportant_total,
        "active_criterion_count": len(active_ids),
        "all_weight_sum": all_weight_sum,
        "active_weight_sum": active_weight_sum,
        "normalized_active_weight_sum": normalized_sum,
        "criteria": rows,
    }


def resolve_path(path):
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )

    args = parser.parse_args()

    config_path = resolve_path(args.config)
    output_path = resolve_path(args.output)

    if output_path.exists():
        parser.error(
            f"기존 결과 파일이 있습니다: {output_path}\n"
            "--output으로 다른 파일명을 지정하세요."
        )

    with config_path.open(encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    result = calculate_weights(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("문서 기준 설문 가중치 계산 완료")
    print(f"중요 선택 합계: {result['important_total']}")
    print(f"비중요 선택 합계: {result['unimportant_total']}")

    for row in result["criteria"]:
        status = "적용" if row["active"] else "보류"
        print(
            f"{row['criterion_name']} | {status} | "
            f"가중치 {row['weight']:.12f}"
        )

    print(
        f"활성 5개 가중치 합: "
        f"{result['active_weight_sum']:.12f}"
    )
    print(
        f"정규화 비중 합: "
        f"{result['normalized_active_weight_sum']:.12f}"
    )
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()