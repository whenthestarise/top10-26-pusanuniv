"""제공 명단과 기존 모델 판독을 재사용해 내부 기록과 분리된 시연용 TOP10을 만듭니다."""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml

if __package__:
    from .build_learning_standard import make_case, validate_fact_response
    from .calculate_document_weights import calculate_weights
    from .data_policy import ROOT, load_user_records
    from .extract_ai_evidence import unique_json_object, unwrap_code_block
    from .relative_selection import CRITERIA, calculate_selection
    from .prepare_project_data import load_survey_weights
    from .selected50_table import load_selected
else:
    from build_learning_standard import make_case, validate_fact_response
    from calculate_document_weights import calculate_weights
    from data_policy import ROOT, load_user_records
    from extract_ai_evidence import unique_json_object, unwrap_code_block
    from relative_selection import CRITERIA, calculate_selection
    from prepare_project_data import load_survey_weights
    from selected50_table import load_selected


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_scenario(scenario):
    if not isinstance(scenario, dict) or not isinstance(scenario.get("scenario_id"), str) or not scenario["scenario_id"]:
        raise ValueError("내부 시연 시나리오 식별자가 필요합니다.")
    label = scenario.get("result_label")
    if not isinstance(label, str) or "시연" not in label or "검증 전" not in label:
        raise ValueError("공개 결과에는 시연용·검증 전 표시가 필요합니다.")
    if not isinstance(scenario.get("neutral_values"), dict) or set(scenario["neutral_values"]) != set(CRITERIA[1:4]):
        raise ValueError("내부 시연 설정의 평가 항목을 확인하세요.")
    if not isinstance(scenario.get("reference_ai_values"), dict):
        raise ValueError("내부 정성 등급 설정은 객체여야 합니다.")
    for value in [*scenario["neutral_values"].values(), *scenario["reference_ai_values"].values()]:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError("내부 시연 값은 1~5의 정수여야 합니다.")
    if set(scenario.get("reference_ai_values", {})) != {"낮음", "낮음~중간", "중간", "중간~높음", "높음"}:
        raise ValueError("제공된 정성 등급에 대한 내부 시연 설정이 필요합니다.")
    ordered = [scenario["reference_ai_values"][k] for k in ("높음", "중간~높음", "중간", "낮음~중간", "낮음")]
    if any(left >= right for left, right in zip(ordered, ordered[1:])):
        raise ValueError("정성 위험이 낮을수록 시연 안전성 값이 높아야 합니다.")
    for key in ("comparison_scope", "user_authorization"):
        if not isinstance(scenario.get(key), str) or not scenario[key].strip():
            raise ValueError("시연 범위와 사용자 요청 기록이 필요합니다.")


def prepare_payload(records, facts, scenario, scenario_hash):
    validate_scenario(scenario)
    expected_ids = {f"selected50:{i:03d}" for i in range(1, 51)}
    if len(records) != 50 or {c["company_id"] for c in records} != expected_ids or set(facts) != expected_ids:
        raise ValueError("시연 후보와 모델 판독 결과는 선별 50개 명단을 그대로 유지해야 합니다.")
    companies, cohort_ids, provenance = [], [], []
    scope_id = f"demo:{scenario['scenario_id']}"
    fact_keys = ("company_name", "region", "location", "annual_salary_manwon", "main_hiring_roles", "ai_exposure_label", "ai_exposure_explanation")
    for record in records:
        cid, read = record["company_id"], facts[record["company_id"]]
        if any(read.get(k) != record[k] for k in fact_keys):
            raise ValueError("모델 판독과 제공 명단의 사실이 다릅니다.")
        values = {"salary": record["annual_salary_manwon"], **deepcopy(scenario["neutral_values"]),
                  "ai_resilience": scenario["reference_ai_values"][read["ai_exposure_label"]]}
        companies.append({"company_id": cid, "company_name": record["company_name"], "region": record["region"],
            "comparison_scope_id": scope_id, "target_job": "시연용 기업 비교", "values": values,
            "evidence": {}, "deferral_reasons": ["제공 자료의 평균연봉 미확인"] if values["salary"] is None else []})
        if values["salary"] is not None:
            cohort_ids.append(cid)
        provenance.append({"company_id": cid, "provided_facts": {k: deepcopy(record[k]) for k in fact_keys},
                           "demo_values": deepcopy(values), "scenario_sha256": scenario_hash,
                           "eligible_for_demo_calculation": values["salary"] is not None})
    return {"dataset_type": "demo", "companies": companies,
        "demo_metadata": {"purpose": "mvp_presentation", "label": scenario["result_label"],
                          "assumptions_recorded": True, "scenario_sha256": scenario_hash},
        "comparison_cohort": {"scope_id": scope_id, "target_job": "시연용 기업 비교",
                              "scope_definition": scenario["comparison_scope"], "company_ids": cohort_ids}}, provenance


def public_result(calculated, records, label):
    if (calculated.get("dataset_type") != "demo" or not calculated.get("selection_is_demo")
            or calculated.get("real_selection_confirmed") or calculated.get("final_top10") is not None
            or len(calculated.get("selected_companies", [])) != 10):
        raise ValueError("공개 출력에는 확정 10개의 시연 결과만 사용할 수 있습니다.")
    if not isinstance(label, str) or "시연" not in label or "검증 전" not in label:
        raise ValueError("시연 표시를 생략할 수 없습니다.")
    source = {c["company_id"]: c for c in records}
    ids = [c["company_id"] for c in calculated["selected_companies"]]
    if len(set(ids)) != 10 or set(ids) - set(source):
        raise ValueError("공개 시연 목록의 식별자가 중복되거나 제공 명단 밖의 기업입니다.")
    rows = []
    for order, item in enumerate(calculated["selected_companies"], 1):
        c = source[item["company_id"]]
        if (c["company_name"] != item["company_name"] or c["region"] != item["region"]
                or c["annual_salary_manwon"] is None):
            raise ValueError("공개 기업 정보가 제공 자료와 다릅니다.")
        rows.append({"rank": order, "company_id": c["company_id"], "company_name": c["company_name"],
            "region": c["region"], "industry": c["industry"], "main_hiring_roles": c["main_hiring_roles"],
            "annual_salary_manwon": c["annual_salary_manwon"], "demo_index": round(item["display_index_P"], 2)})
    regions = {r: sum(c["region"] == r for c in rows) for r in ("부산", "울산", "경남")}
    if regions != calculated["selected_region_counts"] or any(v < 1 for v in regions.values()):
        raise ValueError("공개 시연 결과의 지역 조건이 올바르지 않습니다.")
    return {"result_type": "mvp_demo", "result_label": label, "verified_research_result": False,
            "candidate_count": len(records), "selected_count": 10,
            "region_counts": calculated["selected_region_counts"], "top10": rows}


def load_reading(root, records, policy):
    folder = root / "reports/selected50_reading_2026-11-15/attempt01"
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("status") != "completed" or summary.get("passed_count") != 50:
        raise ValueError("제공 명단 50개의 원문 대조를 통과한 모델 판독 기록이 필요합니다.")
    for name, expected in manifest["source_file_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("기존 모델 판독 이후 제공 자료·설정이 변경됐습니다.")
    for name, expected in manifest["implementation_sha256"].items():
        if digest(root / "src" / name) != expected:
            raise ValueError("기존 모델 판독의 구현 기록과 다릅니다.")
    results = json.loads((folder / "results.json").read_text())
    by_id = {r["case_id"]: r for r in results}
    if len(by_id) != 50 or len(results) != 50:
        raise ValueError("모델 판독의 기업 수 또는 식별자가 올바르지 않습니다.")
    standard = json.loads((root / "configs/provided_data_learning.json").read_text())
    facts = {}
    for record in records:
        result = by_id.get(record["company_id"])
        if result is None or not result["passed"] or result["facts"] is None:
            raise ValueError("원문 대조를 통과하지 않은 기업이 있습니다.")
        case = make_case(record, policy, standard)
        response = json.loads(unwrap_code_block(result["generation"]["answer"]), object_pairs_hook=unique_json_object)
        validate_fact_response(case, response)
        if response != result["facts"]:
            raise ValueError("기록된 모델 원시 응답과 승인된 사실이 다릅니다.")
        facts[record["company_id"]] = response
    return facts, manifest["candidate"], {"source_folder": str(folder),
        "results_sha256": digest(folder / "results.json"), "manifest_sha256": digest(folder / "manifest.json")}


def run(root, scenario_path, public_dir, audit_dir):
    root, scenario_path = Path(root), Path(scenario_path)
    public_dir, audit_dir = Path(public_dir), Path(audit_dir)
    if public_dir.exists() or audit_dir.exists():
        raise FileExistsError("기존 시연 결과와 내부 기록을 보존합니다. 새로운 출력 경로를 사용하세요.")
    if audit_dir.resolve().is_relative_to(public_dir.resolve()) or public_dir.resolve().is_relative_to(audit_dir.resolve()):
        raise ValueError("공개 결과와 내부 기록은 서로 다른 경로로 분리해야 합니다.")
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    validate_scenario(scenario)
    scenario_hash = digest(scenario_path)
    policy, _ = load_user_records(root)
    selected = load_selected(root, policy)
    records = selected["companies"]
    facts, candidate, reading_record = load_reading(root, records, policy)
    payload, provenance = prepare_payload(records, facts, scenario, scenario_hash)
    config_path = root / "configs/document_model.yaml"
    config_hash = digest(config_path)
    config = yaml.safe_load(config_path.read_text())
    weights = {r["criterion_id"]: r["weight"] for r in calculate_weights(config)["criteria"] if r["active"]}
    surveyed = load_survey_weights(root / "data/raw/survey_responses.csv", config)
    if (not surveyed["counts_match_document_reference"]
            or {r["criterion_id"]: r["weight"] for r in surveyed["criteria"] if r["active"]} != weights):
        raise ValueError("제공 설문을 재검산한 기존 가중치와 다릅니다.")
    calculated = calculate_selection(payload, weights, config["salary"]["baseline"])
    published = public_result(calculated, records, scenario["result_label"])
    policy_after, _ = load_user_records(root)
    load_selected(root, policy_after)
    if policy_after != policy or digest(scenario_path) != scenario_hash or digest(config_path) != config_hash:
        raise ValueError("시연 계산 중 원본 또는 설정이 바뀌었습니다.")
    public_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    write_json(audit_dir / "scenario.json", scenario)
    write_json(audit_dir / "calculation_input.json", payload)
    write_json(audit_dir / "calculation_result.json", calculated)
    write_json(audit_dir / "provenance.json", provenance)
    write_json(audit_dir / "manifest.json", {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_type": "mvp_demo", "scenario_sha256": scenario_hash, "provided_source_policy": policy,
        "document_model_sha256": config_hash,
        "candidate_model": candidate, "reused_model_reading": reading_record,
        "model_inference_performed": False, "model_training_performed": False,
        "implementation_sha256": {name: digest(Path(__file__).parent / name) for name in ("run_mvp_selection.py", "relative_selection.py")},
        "real_selection_confirmed": False, "public_label": scenario["result_label"]})
    write_json(public_dir / "top10.json", published)
    with (public_dir / "TOP10_시연용.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["시연순위", "기업명", "지역", "업종", "제공평균연봉(만원/년)", "시연지수", "결과구분"])
        for c in published["top10"]:
            writer.writerow([c["rank"], c["company_name"], c["region"], c["industry"], c["annual_salary_manwon"], c["demo_index"], scenario["result_label"]])
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.project_root, args.scenario_config, args.output_dir, args.audit_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(result["result_label"])
    for c in result["top10"]:
        print(f"{c['rank']}위 | {c['company_name']} | {c['region']} | 시연지수 {c['demo_index']:.2f}")


if __name__ == "__main__":
    main()
