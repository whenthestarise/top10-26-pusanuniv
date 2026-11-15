"""가상 기능 검사와 제공 자료 사실 읽기를 분리해 후보 모델을 비교합니다."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

if __package__:
    from .ai_task_cases import development_cases, confirmation_cases
    from .assess_ai_resilience import assess
    from .data_policy import load_user_records, build_user_bundle, validate_user_sources
    from .extract_ai_evidence import extract_evidence, prepare_bundle
    from .mlx_generator import MlxGenerator
    from .provided_data_check import messages, validate
else:
    from ai_task_cases import development_cases, confirmation_cases
    from assess_ai_resilience import assess
    from data_policy import load_user_records, build_user_bundle, validate_user_sources
    from extract_ai_evidence import extract_evidence, prepare_bundle
    from mlx_generator import MlxGenerator
    from provided_data_check import messages, validate


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def provided_records(records):
    selected = []
    for region in ("부산", "울산", "경남"):
        selected.extend([r for r in records.values() if r["region"] == region
                         and r["evaluated_job"] and r["raw_fields"]["K"]][:2])
    return selected


def run_candidate(candidate, folder, hf_home, cases, records, policy):
    started = time.perf_counter()
    generator = MlxGenerator(candidate["model_id"], candidate["revision"], hf_home)
    functions, reading = [], []
    for case in cases:
        print(f"기능 검사: {case['name']}", flush=True)
        target = folder / "functional" / case["case_id"]
        extraction = extract_evidence(prepare_bundle(case["bundle"]), generator)
        if extraction["validation_errors"]:
            result = {"status": "invalid_extraction", "proposed_score": None,
                      "validation_errors": extraction["validation_errors"], "model_calls": []}
        else:
            result = assess(extraction, generator)
        passed = (not result["validation_errors"] and result["status"] == case["expected_status"]
                  and result["proposed_score"] == case["expected_score"])
        save(target / "input.json", case["bundle"])
        save(target / "extraction.json", extraction)
        save(target / "assessment.json", result)
        functions.append({"case_id": case["case_id"], "passed": passed, "status": result["status"],
                          "proposed_score": result["proposed_score"]})
        print("결과:", "통과" if passed else "실패", flush=True)
    for record in records:
        print(f"제공 자료 읽기 검사: {record['company_name']}", flush=True)
        bundle = build_user_bundle(record, policy)
        validate_user_sources(bundle)
        generation = generator(messages(record), max_tokens=256)
        errors, parsed = validate(generation["answer"], record)
        save(folder / "provided_data" / f"{record['original_id']:03d}.json",
             {"check_type": "provided_facts_only", "company_id": record["company_id"],
              "source": bundle["sources"][0], "generation": generation, "parsed": parsed,
              "errors": errors, "passed": not errors, "scoring_performed": False})
        reading.append({"company_id": record["company_id"], "passed": not errors, "errors": errors})
        print("결과:", "통과" if not errors else "실패", flush=True)
    report = {"candidate": candidate, "functional_count": len(functions),
              "functional_passed": sum(r["passed"] for r in functions),
              "provided_data_count": len(reading), "provided_data_passed": sum(r["passed"] for r in reading),
              "functional_results": functions, "provided_data_results": reading,
              "wall_seconds": round(time.perf_counter() - started, 3),
              "model_peak_gib": generator.max_peak_gib,
              "company_scoring_performed": False, "ready_for_ranking": False}
    save(folder / "summary.json", report)
    mx = generator.mx
    del generator
    gc.collect()
    mx.clear_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="all")
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    registry = json.loads((ROOT / "configs/model_candidates.json").read_text())
    candidates = [c for c in registry["candidates"] if args.candidate in {"all", c["candidate_id"]}]
    if not candidates:
        parser.error("등록된 후보 식별자가 아닙니다.")
    policy, records = load_user_records()
    records = provided_records(records)
    if len(records) != 6:
        parser.error("제공 자료에서 지역별 두 기업씩 확인할 수 없습니다.")
    cases = development_cases() + confirmation_cases()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = args.output_dir or ROOT / "reports" / f"model_comparison_{stamp}"
    if folder.exists():
        parser.error("출력 폴더가 이미 있습니다.")
    folder.mkdir(parents=True)
    save(folder / "functional_expectations.json", cases)
    save(folder / "protocol.json", {"fictional_cases_function_checks_only": True,
         "real_company_sources": "user_provided_only", "model_prompts_fixed": True,
         "model_selection_is_not_real_company_accuracy_validation": True})
    manifest = {}
    for path in sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "configs").glob("*")):
        if path.is_file():
            relative = path.relative_to(ROOT)
            target = folder / "implementation" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(folder / "source_manifest.json", manifest)
    summaries = []
    for candidate in candidates:
        try:
            report = run_candidate(candidate, folder / candidate["candidate_id"], args.hf_home, cases, records, policy)
        except Exception as exc:
            report = {"candidate": candidate, "status": "runtime_error", "error": str(exc), "ready_for_ranking": False}
            save(folder / candidate["candidate_id"] / "runtime_error.json", report)
        summaries.append(report)
        save(folder / f"summary_{candidate['candidate_id']}.json", report)
    save(folder / "summary.json", {"candidates": summaries, "company_scoring_performed": False,
                                   "ready_for_ranking": False})
    print("후보 비교 기록을 저장했습니다. 실제 기업 점수와 순위는 생성하지 않았습니다.")
    return 1 if any(r.get("status") == "runtime_error" for r in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
