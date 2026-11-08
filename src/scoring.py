from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


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


def checked_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: 숫자가 필요합니다.")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{label}: 유한한 숫자가 필요합니다.")

    return result


def validate_weights(weights: dict) -> dict[str, float]:
    if not isinstance(weights, dict):
        raise ValueError("가중치는 객체 형식이어야 합니다.")

    if set(weights) != set(CRITERION_IDS):
        raise ValueError("가중치는 지정된 7개 기준을 정확히 포함해야 합니다.")

    validated = {}

    for key in CRITERION_IDS:
        value = checked_number(weights[key], f"{key} 가중치")

        if not 0.9 <= value <= 1.1:
            raise ValueError(f"{key}: 가중치는 0.9~1.1이어야 합니다.")

        validated[key] = value

    return validated


def load_weights(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig") as file:
        document = yaml.safe_load(file)

    if not isinstance(document, dict) or "weights" not in document:
        raise ValueError("가중치 YAML에 weights 항목이 없습니다.")

    return validate_weights(document["weights"])


def score_company(company: dict, weights: dict) -> dict:
    """하나의 기업을 계산합니다. 자료의 사실성 검증은 별도입니다."""
    weights = validate_weights(weights)

    if not isinstance(company, dict):
        raise ValueError("기업 데이터는 객체 형식이어야 합니다.")

    company_id = company.get("company_id")
    company_name = company.get("company_name")

    if not isinstance(company_id, str) or not company_id.strip():
        raise ValueError("company_id는 비어 있지 않은 문자열이어야 합니다.")

    if not isinstance(company_name, str) or not company_name.strip():
        raise ValueError("company_name은 비어 있지 않은 문자열이어야 합니다.")

    scores = company.get("scores")

    if not isinstance(scores, dict):
        raise ValueError(f"{company_name}: scores 객체가 필요합니다.")

    unknown = set(scores) - set(CRITERION_IDS)
    if unknown:
        raise ValueError(f"{company_name}: 알 수 없는 평가 기준 {sorted(unknown)}")

    missing = []
    validated_scores = {}

    for key in CRITERION_IDS:
        value = scores.get(key)

        # 실제 0점은 유효한 값입니다. 누락 또는 null만 자료 부족입니다.
        if value is None:
            missing.append(key)
            continue

        value = checked_number(value, f"{company_name}/{key}")

        if not 0 <= value <= 100:
            raise ValueError(f"{company_name}/{key}: 점수는 0~100이어야 합니다.")

        validated_scores[key] = value

    result = {
        "company_id": company_id.strip(),
        "company_name": company_name.strip(),
        "status": "평가 보류" if missing else "계산 완료",
        "missing_criteria": missing,
        "final_score": None,
        "contributions": {},
    }

    if missing:
        return result

    weight_sum = math.fsum(weights.values())

    # 각 기여도의 합계가 최종점수입니다.
    contributions = {
        key: validated_scores[key] * weights[key] / weight_sum
        for key in CRITERION_IDS
    }

    result["final_score"] = math.fsum(contributions.values())
    result["contributions"] = contributions

    return result


def rank_companies(
    companies: list[dict],
    weights: dict,
    top_n: int = 10,
) -> dict:
    if not isinstance(companies, list) or not companies:
        raise ValueError("기업 목록이 비어 있거나 형식이 잘못되었습니다.")

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n은 1 이상의 정수여야 합니다.")

    weights = validate_weights(weights)

    results = []
    seen = set()

    for company in companies:
        result = score_company(company, weights)
        company_id = result["company_id"]

        if company_id in seen:
            raise ValueError(f"중복 기업 ID: {company_id}")

        seen.add(company_id)
        results.append(result)

    eligible = [
        item for item in results
        if item["status"] == "계산 완료"
    ]
    deferred = [
        item for item in results
        if item["status"] == "평가 보류"
    ]

    # ID는 동점 기업의 표시 순서만 결정합니다.
    eligible.sort(
        key=lambda item: (-item["final_score"], item["company_id"])
    )

    previous_score = None
    current_rank = 0

    for position, item in enumerate(eligible, start=1):
        score = item["final_score"]

        # 반올림 전 계산값이 같은 경우 공동 순위: 1, 2, 2, 4
        if previous_score is None or score != previous_score:
            current_rank = position

        item["rank"] = current_rank
        previous_score = score

    candidates = eligible[:top_n]
    boundary_tie = False

    if len(eligible) > top_n:
        cutoff = eligible[top_n - 1]["final_score"]
        candidates = [
            item for item in eligible
            if item["final_score"] >= cutoff
        ]
        boundary_tie = len(candidates) > top_n

    return {
        "requested_top_n": top_n,
        "weights": weights,
        "weight_sum": math.fsum(weights.values()),
        "eligible_count": len(eligible),
        "deferred_count": len(deferred),
        "shortfall": max(0, top_n - len(eligible)),
        "boundary_tie": boundary_tie,
        "candidate_count": len(candidates),
        "top_candidates": candidates,
        "ranked_companies": eligible,
        "deferred_companies": deferred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

    scores_path = resolve(args.scores)
    weights_path = resolve(args.weights)
    output_path = resolve(args.output)

    if output_path.exists():
        raise FileExistsError(
            f"결과 파일이 이미 있습니다. 다른 출력 이름을 지정하세요: {output_path}"
        )

    payload = json.loads(scores_path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict):
        raise ValueError("점수 파일은 JSON 객체여야 합니다.")

    # 가상 자료와 실제 자료를 결과에서도 구분합니다.
    dataset_type = payload.get("dataset_type")
    if dataset_type not in {"synthetic", "real"}:
        raise ValueError("dataset_type은 synthetic 또는 real이어야 합니다.")

    weights = load_weights(weights_path)
    result = rank_companies(
        payload.get("companies"),
        weights,
        args.top_n,
    )

    result["dataset_type"] = dataset_type
    result["scores_file"] = str(args.scores)
    result["weights_file"] = str(args.weights)
    result["note"] = "입력 점수의 사실성·검토 여부는 이 계산기가 검증하지 않음"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"데이터 종류: {dataset_type}")
    print(f"계산 완료: {result['eligible_count']}개")
    print(f"평가 보류: {result['deferred_count']}개")
    print(f"선정 후보: {result['candidate_count']}개")
    print(f"경계 동점: {result['boundary_tie']}")
    print(f"목표 개수 대비 부족: {result['shortfall']}개")

    for item in result["top_candidates"]:
        print(
            f"{item['rank']}위 | {item['company_name']} | "
            f"{item['final_score']:.4f}점"
        )

    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()