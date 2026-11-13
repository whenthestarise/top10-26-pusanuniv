from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ai_evaluation.yaml"

CRITERIA = {
    "work_life_balance",
    "infrastructure",
    "job_outlook",
    "ai_resilience",
}

STATUSES = {
    "assessed",
    "insufficient_evidence",
    "conflicting_evidence",
}


def normalize_text(value):
    # PDF나 웹페이지의 줄바꿈 차이만 정규화합니다.
    return re.sub(r"\s+", " ", value).strip()


def has_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_evaluation(payload, config):
    errors = []
    scores = {criterion_id: None for criterion_id in CRITERIA}

    if not isinstance(payload, dict):
        return ["입력은 JSON 객체여야 합니다."], scores

    context = payload.get("context")
    sources = payload.get("sources")
    evaluation = payload.get("evaluation")

    if not isinstance(context, dict):
        return ["context가 없습니다."], scores

    required_scope = config["scope"]["required"]

    for field in required_scope:
        if not has_text(context.get(field)):
            errors.append(f"context.{field}: 값이 필요합니다.")

    if not isinstance(sources, list):
        return errors + ["sources는 배열이어야 합니다."], scores

    source_map = {}

    for source in sources:
        if not isinstance(source, dict):
            errors.append("출처는 JSON 객체여야 합니다.")
            continue

        required = (
            "source_id",
            "title",
            "locator",
            "retrieved_at",
            "text",
        )

        if any(not has_text(source.get(key)) for key in required):
            errors.append("출처의 필수 정보가 누락됐습니다.")
            continue

        source_id = source["source_id"]

        if source_id in source_map:
            errors.append(f"출처 ID 중복: {source_id}")
            continue

        source_map[source_id] = source

    if not isinstance(evaluation, dict):
        return errors + ["evaluation이 없습니다."], scores

    if evaluation.get("rubric_version") != config["rubric_version"]:
        errors.append("평가표 버전이 일치하지 않습니다.")

    # LLM이 기업명·사업장·직무를 바꾸지 않았는지 확인합니다.
    for field in required_scope:
        if evaluation.get(field) != context.get(field):
            errors.append(f"평가 대상 불일치: {field}")

    results = evaluation.get("criteria")

    if not isinstance(results, dict) or set(results) != CRITERIA:
        return errors + ["비급여 4개 기준이 정확히 필요합니다."], scores

    allowed_values = config["rating_scale"]["allowed_values"]

    for criterion_id, result in results.items():
        prefix = f"{criterion_id}: "

        if not isinstance(result, dict):
            errors.append(prefix + "평가 결과는 객체여야 합니다.")
            continue

        status = result.get("status")
        score = result.get("score")
        evidence = result.get("evidence")

        if not isinstance(status, str) or status not in STATUSES:
            errors.append(prefix + "허용되지 않은 상태입니다.")

        if not has_text(result.get("reason")):
            errors.append(prefix + "평가 또는 보류 이유가 필요합니다.")

        if not isinstance(evidence, list):
            errors.append(prefix + "evidence는 배열이어야 합니다.")
            evidence = []

        if status == "assessed":
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or score not in allowed_values
            ):
                errors.append(prefix + "점수는 1~5의 정수여야 합니다.")
            else:
                scores[criterion_id] = score

            if not evidence:
                errors.append(prefix + "점수에 대한 인용 근거가 없습니다.")

        elif score is not None:
            errors.append(prefix + "평가 보류 상태의 점수는 null이어야 합니다.")

        for citation in evidence:
            if not isinstance(citation, dict):
                errors.append(prefix + "인용은 객체여야 합니다.")
                continue

            source_id = citation.get("source_id")
            quote = citation.get("quote")

            if not isinstance(source_id, str) or source_id not in source_map:
                errors.append(prefix + "등록되지 않은 출처 ID입니다.")
                continue

            if not has_text(quote):
                errors.append(prefix + "인용 문장이 비어 있습니다.")
                continue

            source_text = source_map[source_id]["text"]

            if normalize_text(quote) not in normalize_text(source_text):
                errors.append(prefix + "인용 문장이 수집 원문에 없습니다.")

    # 하나라도 검사에 실패하면 계산에 사용할 점수를 반환하지 않습니다.
    if errors: 
        return errors, {
            criterion_id: None
            for criterion_id in CRITERIA
        }            

    return errors, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    input_path = (
        args.input if args.input.is_absolute()
        else ROOT / args.input
    )

    with CONFIG_PATH.open(encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    with input_path.open(encoding="utf-8-sig") as file:
        payload = json.load(file)

    errors, scores = validate_evaluation(payload, config)

    if errors:
        print("자동 검사: 실패")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("구조·출처 연결 검사: 통과")
    print("평가 방식: AI 평가 / 사람 승인 조건 없음")

    if all(score is not None for score in scores.values()):
        print("비급여 4개 원점수: 확보")
        print("다음 단계: 근거의 대상·내용 적합성 검사")
    else:
        print("비급여 4개 원점수: 미완성")
        print("종합평가: 보류")

    for criterion_id, score in scores.items():
        print(f"- {criterion_id}: {score}")


if __name__ == "__main__":
    main()