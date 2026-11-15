"""기존 5개 항목의 선정 계산기를 가상 검사 또는 검증된 실제 자료에 실행합니다."""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml

if __package__:
    from .relative_selection import CRITERIA, CRITERION_NAMES, calculate_selection
    from .calculate_document_weights import calculate_weights
else:
    from relative_selection import CRITERIA, CRITERION_NAMES, calculate_selection
    from calculate_document_weights import calculate_weights


ROOT = Path(__file__).resolve().parents[1]


def adapt_assessment(table, model_selection):
    if table.get("dataset_type") != "real" or len(table.get("companies", [])) != 50:
        raise ValueError("실제 입력은 검증된 선별 50개 평가표여야 합니다.")
    if tuple(w["criterion_id"] for w in table["weights"]) != CRITERIA:
        raise ValueError("실제 평가표의 활성 기준이 기존 5개와 다릅니다.")
    scope = deepcopy(table.get("validated_comparison_cohort"))
    if scope is not None and not isinstance(scope, dict):
        raise ValueError("공통 비교집합의 형식을 확인하세요.")
    companies = []
    for company in table["companies"]:
        criteria = company["criteria"]
        companies.append({"company_id": company["company_id"], "company_name": company["company_name"],
            "region": company["region"], "comparison_scope_id": company.get("validated_comparison_scope_id"),
            "target_job": company.get("validated_target_job"),
            "values": {k: criteria[k]["numeric_value"] if k == "salary" else criteria[k]["raw_score"] for k in CRITERIA},
            "evidence": {k: deepcopy(criteria[k]["sources"]) for k in CRITERIA},
            "deferral_reasons": deepcopy(company["deferral_reasons"])})
    return {"dataset_type": "real", "companies": companies, "comparison_cohort": scope,
        "validation_gates": {"source_linkage": True,
            "score_validation": model_selection.get("functional_scoring_checks_passed") is True
                                and model_selection.get("company_scoring_enabled") is True,
            "comparison_scope": table.get("comparison_scope_validation_passed") is True},
        "note": "현재 평가표의 미상 점수와 직무 범위 미확정을 그대로 유지"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("입력 JSON의 항목 이름이 중복됐습니다.")
        result[key] = value
    return result


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def run(root, output, synthetic_input=None):
    root, output = Path(root), Path(output)
    started = datetime.now(timezone.utc).isoformat()
    if output.exists():
        raise FileExistsError("기존 선정 결과를 보존합니다. 새로운 출력 경로를 지정하세요.")
    source_files = [root / "configs/document_model.yaml"]
    source_files.extend([Path(synthetic_input)] if synthetic_input is not None else
                        [root / "configs/data_policy.json", root / "configs/selected_model.json"])
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    config = yaml.safe_load((root / "configs/document_model.yaml").read_text(encoding="utf-8"))
    calculated = calculate_weights(config)
    weights = {w["criterion_id"]: w["weight"] for w in calculated["criteria"] if w["active"]}
    payload = None
    if synthetic_input is not None:
        payload = json.loads(Path(synthetic_input).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or payload.get("dataset_type") != "synthetic":
            raise ValueError("외부 JSON 입력은 가상 기능 검사 자료만 허용합니다. 실제 입력은 원본 근거를 재검사합니다.")
    result = calculate_selection(payload, weights, config["salary"]["baseline"]) if payload is not None else None
    output.mkdir(parents=True)
    if payload is None:
        if __package__:
            from .attach_evidence_assessment import run as rebuild_assessment
        else:
            from attach_evidence_assessment import run as rebuild_assessment
        report = root / "reports/evidence_assessment_2026-11-15"
        table = rebuild_assessment(root, report / "attempt03", report / "pdf_references", output / "assessment_inputs")
        model = json.loads((root / "configs/selected_model.json").read_text())
        expected = {w["criterion_id"]: w["weight"] for w in table["weights"]}
        if expected != weights:
            raise ValueError("원본 설문으로 검산한 가중치와 모형 설정이 다릅니다.")
        payload = adapt_assessment(table, model)
    if result is None:
        result = calculate_selection(payload, weights, config["salary"]["baseline"])
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != source_hashes[str(p)] for p in source_files):
        raise ValueError("실행 중 입력 또는 설정이 바뀌어 결과를 저장하지 않습니다.")
    write_json(output / "selection_input.json", payload)
    write_json(output / "selection_result.json", result)
    write_json(output / "run_manifest.json", {"started_at_utc": started,
        "dataset_type": payload["dataset_type"], "source_sha256": source_hashes,
        "implementation_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                  for name in ("relative_selection.py", "run_relative_selection.py")},
        "real_input_rebuilt_from_verified_sources": synthetic_input is None,
        "external_company_information_used": False, "model_inference_performed": False})
    with (output / "기업별_선정계산.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["기업ID", "기업명", "지역", "계산상태", "급여지수", "가중합T", "상대지수I", "표시지수P", "모형연봉환산M", "순위", "미상항목", "보류사유"])
        for row in result["company_results"]:
            writer.writerow([row["company_id"], row["company_name"], row["region"], row["status"], row["salary_index"],
                row["weighted_sum_T"], row["relative_index_I"], row["display_index_P"], row["salary_equivalent_M"], row["rank"],
                ", ".join(CRITERION_NAMES[k] for k in row["missing_criteria"]), " / ".join(row.get("deferral_reasons", []))])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--synthetic-input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.project_root, args.output_dir, args.synthetic_input)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print("자료 종류:", "가상 기능 검사" if result["dataset_type"] == "synthetic" else "실제 자료")
    print(f"선정 계산 상태: {result['selection_status']}")
    print(f"수치 계산 대상 {result['eligible_count']}개, 선정 {len(result['selected_companies'])}개")
    print("실제 최종 명단 확정:", "예" if result["real_selection_confirmed"] else "보류")


if __name__ == "__main__":
    main()
