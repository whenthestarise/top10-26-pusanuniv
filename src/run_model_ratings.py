"""선별 50개 기업의 네 비급여 항목을 로컬 모델로 평가하고 원문 연결을 보존합니다."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

if __package__:
    from .data_policy import ROOT, build_selected_bundle, load_user_records, validate_user_sources
    from .model_company_ratings import VERSION, PROMPT, company_fields, rate_fields, sanity_cases
    from .selected50_table import load_selected
else:
    from data_policy import ROOT, build_selected_bundle, load_user_records, validate_user_sources
    from model_company_ratings import VERSION, PROMPT, company_fields, rate_fields, sanity_cases
    from selected50_table import load_selected


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run(root, output, hf_home=None):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("기존 모델 평가 기록을 보존합니다. 새로운 출력 경로를 지정하세요.")
    policy, pdf_records = load_user_records(root)
    selected = load_selected(root, policy)
    records = selected["companies"]
    if len(records) != 50 or {c["company_id"] for c in records} != {f"selected50:{i:03d}" for i in range(1, 51)}:
        raise ValueError("평가 대상은 연구팀 선별 50개 명단이어야 합니다.")
    for record in records:
        validate_user_sources(build_selected_bundle(record, policy), root)
        if record["pdf_reference"] is not None and pdf_records.get(record["catalog_link"]["company_id"]) != record["pdf_reference"]:
            raise ValueError("PDF 일부 직무 참고의 기업 대응이 다릅니다.")
    model_path = root / "configs/selected_model.json"
    model = json.loads(model_path.read_text())["candidate"]
    paths = [root / "configs/data_policy.json", model_path, root / "configs/document_model.yaml",
             root / "data/processed/selected50_companies.json", root / "data/processed/document_companies.json",
             *[root / "data/raw" / source["stored_name"] for source in policy["raw_sources"].values()]]
    before = {str(p.relative_to(root)): digest(p) for p in paths}
    output.mkdir(parents=True)
    (output / "calls").mkdir()
    save(output / "manifest.json", {"version": VERSION, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": model, "source_hashes": before, "system_prompt": PROMPT,
        "implementation_sha256": {name: digest(Path(__file__).parent / name) for name in (
            "model_company_ratings.py", "run_model_ratings.py", "mlx_generator.py", "data_policy.py", "selected50_table.py")},
        "user_authorization": "제공 자료를 해석한 모델의 기업별 4개 비급여 평가 및 5개 기준 TOP10 산출",
        "score_origin": "model_interpretation", "empirical_accuracy_validated": False,
        "model_training_performed": False, "company_names_and_salaries_in_rating_prompt": False,
        "structured_output": "허용 가능한 1~5 정수의 모든 625개 조합; 기업별 기대 답을 출력 후보에 넣지 않음"})
    if __package__:
        from .mlx_generator import MlxGenerator
    else:
        from mlx_generator import MlxGenerator
    started, results, sanity = time.perf_counter(), [], []
    try:
        generator = MlxGenerator(model["model_id"], model["revision"], hf_home)
        def call(fields, call_id):
            return rate_fields(fields, generator, on_generation=lambda generation:
                save(output / "calls" / f"{call_id}.json", {"fields": fields, "generation": generation}))
        for case in sanity_cases():
            criterion = case["criterion_id"]
            high, high_generation = call(case["high"], f"sanity_{criterion}_high")
            low, low_generation = call(case["low"], f"sanity_{criterion}_low")
            passed = high[criterion]["score"] > low[criterion]["score"]
            sanity.append({**case, "high_ratings": high, "low_ratings": low,
                           "high_generation": high_generation, "low_generation": low_generation, "passed": passed})
            save(output / "sanity_checks.json", sanity)
            print(f"방향성 검사 {criterion}: {'통과' if passed else '실패'}", flush=True)
        if not all(s["passed"] for s in sanity):
            raise ValueError("기본 방향성 검사를 통과하지 못해 실제 명단의 모델 평가를 중단합니다.")
        for index, record in enumerate(records, 1):
            fields = company_fields(record)
            ratings, generation = call(fields, f"company_{index:03d}")
            result = {"company_id": record["company_id"], "company_name": record["company_name"],
                      "provided_fields": fields, "ratings": ratings, "generation": generation,
                      "score_origin": "model_interpretation", "empirically_verified": False}
            results.append(result)
            save(output / f"company_{index:03d}.json", result)
            save(output / "progress.json", {"completed_count": index, "planned_count": 50})
            print(f"[{index:02d}/50] {record['company_name']}: " + " / ".join(f"{k}={v['score']:.1f}" for k, v in ratings.items()), flush=True)
        if any(digest(root / name) != expected for name, expected in before.items()):
            raise ValueError("모델 평가 중 제공 자료·설정이 바뀌어 결과를 사용할 수 없습니다.")
        distributions = {key: dict(Counter(str(r["ratings"][key]["score"]) for r in results)) for key in results[0]["ratings"]}
        save(output / "ratings.json", results)
        summary = {"status": "completed", "company_count": len(results), "sanity_passed": sum(c["passed"] for c in sanity),
            "sanity_count": len(sanity), "score_distributions": distributions,
            "all_source_hashes_unchanged": True, "elapsed_seconds": round(time.perf_counter() - started, 3),
            "empirical_accuracy_validated": False, "model_inference_performed": True, "model_training_performed": False}
        save(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        save(output / "summary.json", {"status": "incomplete", "completed_count": len(results), "error": str(exc),
                                       "empirical_accuracy_validated": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path)
    args = parser.parse_args()
    try:
        summary = run(args.project_root, args.output_dir, args.hf_home)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"선별 {summary['company_count']}개 기업의 모델 평가를 완료했습니다.")


if __name__ == "__main__":
    main()
