"""사용자 제공 50개 명단으로 모델 지침·직군 참고 예시·사실 검증 사례를 만듭니다."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

if __package__:
    from .data_policy import ROOT, load_user_records, selected_locator
    from .selected50_table import load_selected
else:
    from data_policy import ROOT, load_user_records, selected_locator
    from selected50_table import load_selected


FACT_FIELDS = {"company_name": "B", "region": "C", "location": "D", "main_hiring_roles": "I",
               "ai_exposure_label": "J", "ai_exposure_explanation": "K"}


def make_case(record, policy, standard):
    book_hash = policy["raw_sources"]["selected50"]["sha256"]
    fields = dict(FACT_FIELDS, annual_salary_manwon="G")
    inputs = {key: record["raw_fields"][column] for key, column in fields.items()}
    expected = {key: record[key] for key in fields}
    expected["work_life_balance_score"] = None
    expected["infrastructure_score"] = None
    expected["job_outlook_score"] = None
    expected["ai_resilience_score"] = None
    source_links = {key: selected_locator(book_hash, record["source_sheet"], record["source_row"], column)
                    for key, column in fields.items()}
    model_input = {"instruction": "원문 사실만 구조화하세요. 점수는 이 사실 읽기 단계에서 생성하지 마세요. JSON 객체만 반환하세요.",
                   "rules": standard["rules"], "output_fields": list(expected),
                   "company_id": record["company_id"], "provided_fields": inputs, "source_links": source_links}
    return {"case_id": record["company_id"], "purpose": "제공 사실 읽기와 결측 유지 검사; 기업 평가 정답 아님",
            "model_input": model_input, "expected_facts": expected}


def validate_fact_response(case, response):
    """원문 사실 검증이며 정성 평가 점수의 타당성 검증과 구분합니다."""
    expected = case["expected_facts"]
    if not isinstance(response, dict) or set(response) != set(expected):
        raise ValueError("사실 출력 필드가 검증 규격과 다릅니다.")
    for key, value in expected.items():
        actual = response[key]
        if value is None:
            valid = actual is None
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            valid = isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual == value
        else:
            valid = type(actual) is type(value) and actual == value
        if not valid:
            raise ValueError(f"제공 사실 변경 또는 근거 없는 점수: {key}")
    return {"fact_fidelity_passed": True, "company_scoring_validated": False, "ready_for_ranking": False}


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("기존 기준 자료를 보존합니다. 새로운 출력 폴더를 지정하세요.")
    policy, _ = load_user_records(root)
    selected = load_selected(root, policy)
    standard = json.loads((root / "configs/provided_data_learning.json").read_text(encoding="utf-8"))
    rubric = yaml.safe_load((root / standard["scoring_rubric"]).read_text(encoding="utf-8"))
    model = yaml.safe_load((root / standard["relative_index_model"]).read_text(encoding="utf-8"))
    standard = deepcopy(standard)
    standard.update(source_manifest=policy["raw_sources"], source_selected_sha256=policy["processed_selected50_sha256"],
                    non_salary_anchors=rubric["criteria"], salary_baseline_manwon=model["salary"]["baseline"],
                    current_company_scoring_validated=False, ready_for_ranking=False)
    cases = [make_case(row, policy, standard) for row in selected["companies"]]
    examples = [{"purpose": "제공 자료의 직군 일반 참고; 개별 기업 점수 정답이 아님",
                 "job_group": row["job_group"], "tasks": row["tasks"], "reference_grade": row["exposure_label"],
                 "explanation": row["explanation"], "source_file": policy["raw_sources"]["selected50"]["stored_name"],
                 "source_sha256": policy["raw_sources"]["selected50"]["sha256"],
                 "source_sheet": row["source_sheet"], "source_row": row["source_row"]}
                for row in selected["job_guidelines"]]
    output.mkdir(parents=True)
    files = {"평가기준.json": standard, "직군_학습참고8개.json": examples,
             "사실확인_검증50개.json": cases, "모델입력50개.json": [c["model_input"] for c in cases],
             "manifest.json": {"standard_version": standard["standard_version"], "company_count": len(cases),
                 "reference_example_count": len(examples), "source_policy": policy["policy_version"],
                 "model_training_performed": False, "model_evaluation_performed": False,
                 "company_scoring_validated": False, "ready_for_ranking": False}}
    for name, value in files.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return files["manifest.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/learning/provided50")
    args = parser.parse_args()
    try:
        result = run(ROOT, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"평가 기준, 직군 참고 {result['reference_example_count']}개, 사실 검증 {result['company_count']}개를 생성했습니다.")
    print("모델 가중치 학습·추론과 최종 선정은 실행하지 않았습니다.")


if __name__ == "__main__":
    main()
