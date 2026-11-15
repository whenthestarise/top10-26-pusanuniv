"""가상 사례와 선택적 실제 자료로 근거 추출부터 시험 점수까지 실행합니다."""

import argparse
from datetime import datetime, timezone
import json
import hashlib
import shutil
from pathlib import Path

if __package__:
    from .thinking_classifier import ThinkingClassifier
    from .ai_task_cases import development_cases, confirmation_cases
    from .assess_ai_resilience import ASSESSMENT_VERSION, assess
    from .extract_ai_evidence import LocalGenerator, extract_evidence, prepare_bundle
else:
    from thinking_classifier import ThinkingClassifier
    from ai_task_cases import development_cases, confirmation_cases
    from assess_ai_resilience import ASSESSMENT_VERSION, assess
    from extract_ai_evidence import LocalGenerator, extract_evidence, prepare_bundle


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--real-bundle", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--reasoning", action="store_true", help="검토를 마친 뒤 분류하는 방식과 비교")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = args.output_dir or ROOT / "reports" / f"ai_resilience_{stamp}"
    if folder.exists():
        parser.error("이미 있는 출력 폴더입니다. 새로운 이름을 지정하세요.")
    development = development_cases()
    confirmation = confirmation_cases()
    cases = development + confirmation
    real_bundle = None
    if args.real_bundle:
        real_bundle = json.loads(args.real_bundle.read_text(encoding="utf-8-sig"))
        if real_bundle.get("dataset_type") != "real":
            parser.error("실제 기업 자료의 종류가 잘못됐습니다.")
    bundles = [(case["case_id"], case["bundle"], case) for case in cases]
    confirmation_ids = {case["case_id"] for case in confirmation}
    if real_bundle:
        bundles.append(("real_company", real_bundle, None))
    prepared = [(key, bundle, expected, prepare_bundle(bundle)) for key, bundle, expected in bundles]
    folder.mkdir(parents=True)
    # 고정 기대값과 실제 입력은 모델을 읽기 전에 보존합니다.
    save(folder / "synthetic_expectations.json", {"development": development, "confirmation": confirmation})
    implementation = folder / "implementation"
    manifest = {}
    for name in ("src/assess_ai_resilience.py", "src/ai_task_profile.py", "src/ai_task_cases.py",
                 "src/ai_resilience_cases.py", "src/test_ai_resilience_local.py", "src/extract_ai_evidence.py",
                 "src/thinking_classifier.py",
                 "configs/ai_evaluation.yaml"):
        target = implementation / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        manifest[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    save(folder / "source_manifest.json", manifest)
    generator_class = ThinkingClassifier if args.reasoning else LocalGenerator
    generator = generator_class(args.device, args.hf_home)
    summaries = []
    for key, bundle, expected, ready in prepared:
        if key in confirmation_ids and not all(item["passed"] is True for item in summaries if item["case_id"] not in confirmation_ids):
            summaries.append({"case_id": key, "dataset_type": "synthetic", "passed": False,
                              "skipped": True, "reason": "development_validation_failed", "validation_errors": []})
            continue
        print(f"근거와 점수 시험 시작: {expected['name'] if expected else '실제 기업 직무'}", flush=True)
        target = folder / key
        target.mkdir()
        save(target / "input.json", bundle)
        extraction = extract_evidence(ready, generator)
        save(target / "extraction.json", extraction)
        if extraction["validation_errors"]:
            result = {"status": "invalid_extraction", "proposed_score": None, "ready_for_ranking": False,
                      "validation_errors": extraction["validation_errors"], "model_calls": []}
        else:
            calibration_passed = all(item["passed"] is True for item in summaries) if expected is None else None
            result = assess(extraction, generator, calibration_passed=calibration_passed)
        save(target / "assessment.json", result)
        checks = ({"score": result["proposed_score"] == expected["expected_score"],
                   "status": result["status"] == expected["expected_status"],
                   "no_validation_errors": not result["validation_errors"],
                   "not_final_ranking": result["ready_for_ranking"] is False} if expected else None)
        summary = {"case_id": key, "dataset_type": bundle["dataset_type"],
                   "status": result["status"], "proposed_score": result["proposed_score"],
                   "selected_sentences": len(extraction["evidence"]), "checks": checks,
                   "validation_errors": result["validation_errors"],
                   "passed": all(checks.values()) if checks else None}
        summaries.append(summary)
        print(f"근거 {summary['selected_sentences']}개 / 제안 점수 {result['proposed_score'] if result['proposed_score'] is not None else '판단 보류'}", flush=True)
        for call in result["model_calls"]:
            print(f"모델 판정: {call['answer']}", flush=True)
    report = {"assessment_version": ASSESSMENT_VERSION, "synthetic_case_count": len(cases),
              "reasoning_mode": args.reasoning,
              "synthetic_passed_count": sum(item["passed"] is True for item in summaries),
              "synthetic_executed_count": sum(not item.get("skipped", False) for item in summaries if item["dataset_type"] == "synthetic"),
              "development_case_count": len(development),
              "development_passed_count": sum(item["passed"] is True for item in summaries if item["case_id"] not in confirmation_ids),
              "confirmation_case_count": len(confirmation),
              "confirmation_executed_count": sum(not item.get("skipped", False) for item in summaries if item["case_id"] in confirmation_ids),
              "confirmation_passed_count": sum(item["passed"] is True for item in summaries if item["case_id"] in confirmation_ids),
              "real_company_case_count": int(real_bundle is not None), "cases": summaries,
              "note": "작은 개발 시험 집합이며 실제 기업 점수의 독립적인 정확도 검증은 아님"}
    save(folder / "summary.json", report)
    print(f"실행한 가상 사례 {report['synthetic_executed_count']}개 중 {report['synthetic_passed_count']}개 통과", flush=True)
    return 0 if report["synthetic_passed_count"] == len(cases) and all(not item["validation_errors"] for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
