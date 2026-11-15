"""가상 자료로 근거 선택을 순차 시험합니다. 정답은 모델에 전달하지 않습니다."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

if __package__:
    from .ai_evidence_cases import build_cases
    from .extract_ai_evidence import EXTRACTION_VERSION, LocalGenerator, extract_evidence, prepare_bundle
else:
    from ai_evidence_cases import build_cases
    from extract_ai_evidence import EXTRACTION_VERSION, LocalGenerator, extract_evidence, prepare_bundle


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--hf-home", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = args.output_dir or ROOT / "reports" / f"ai_evidence_{stamp}"
    if folder.exists():
        parser.error("출력 폴더가 이미 있습니다. 새 폴더 이름을 지정하세요.")
    cases = build_cases()
    prepared_cases = [(case, prepare_bundle(case["bundle"])) for case in cases]
    generator = LocalGenerator(args.device, args.hf_home)
    folder.mkdir(parents=True)
    (folder / "inputs").mkdir()
    summaries = []
    for case, prepared in prepared_cases:
        print(f"시험 시작: {case['name']}", flush=True)
        with (folder / "inputs" / f"{case['case_id']}.json").open("x", encoding="utf-8") as stream:
            json.dump(case["bundle"], stream, ensure_ascii=False, indent=2)
        started = time.perf_counter()
        result = extract_evidence(prepared, generator)
        actual_ids = [item["sentence_id"] for item in result["evidence"]]
        checks = {
            "valid_output": result["citation_linkage_passed"],
            "expected_sentences": set(actual_ids) == set(case["expected_sentence_ids"]),
            "expected_model_usage": result["model_called"] != case["expected_model_skipped"],
            "scoring_not_performed": not result["scoring_performed"] and not result["ready_for_ranking"],
        }
        summary = {"case_id": case["case_id"], "name": case["name"], "checks": checks,
                   "passed": all(checks.values()), "model_called": result["model_called"],
                   "sentence_call_count": len(result["sentence_decisions"]),
                   "expected_sentence_ids": case["expected_sentence_ids"],
                   "selected_sentence_ids": actual_ids, "validation_errors": result["validation_errors"],
                   "generation": result["generation"], "wall_seconds": round(time.perf_counter() - started, 3)}
        summaries.append(summary)
        with (folder / f"{case['case_id']}.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"시험 결과: {'통과' if summary['passed'] else '실패'} / 선택 문장 {len(actual_ids)}개", flush=True)
        if not result["model_called"]:
            print("모델을 호출하지 않고 자료 없음 또는 범위 불일치를 처리했습니다.", flush=True)
        elif not summary["passed"]:
            print("문장별 판정:", flush=True)
            for decision in result["sentence_decisions"]:
                print(f"{decision['sentence_id']}: {decision['raw_response']}", flush=True)
    report = {"dataset_type": "synthetic", "extraction_version": EXTRACTION_VERSION,
              "case_count": len(cases), "passed_count": sum(item["passed"] for item in summaries),
              "model_case_count": sum(item["model_called"] for item in summaries),
              "model_sentence_call_count": sum(item["sentence_call_count"] for item in summaries),
              "note": "가상 자료 근거 선택 시험이며 실제 기업 평가나 점수 정확성을 보증하지 않음",
              "cases": summaries}
    with (folder / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"전체 시험: {report['case_count']}개 / 통과: {report['passed_count']}개", flush=True)
    return 0 if report["passed_count"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
