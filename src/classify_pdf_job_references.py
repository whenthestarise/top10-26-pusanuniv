"""선별 기업과 안전하게 대응된 PDF의 일부 직무 설명을 별도 참고로 분류합니다."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

if __package__:
    from .classify_selected_evidence import classify, digest, save
    from .data_policy import ROOT, build_user_bundle, load_user_records, source_locator, validate_user_sources
    from .selected50_table import load_selected
else:
    from classify_selected_evidence import classify, digest, save
    from data_policy import ROOT, build_user_bundle, load_user_records, source_locator, validate_user_sources
    from selected50_table import load_selected


def reference_claim(selected, pdf_record, policy):
    if selected["catalog_link"]["company_id"] != pdf_record["company_id"] or selected["pdf_reference"] != pdf_record:
        raise ValueError("안전하게 연결된 원본 기업의 PDF 참고 자료가 아닙니다.")
    text = pdf_record["raw_fields"]["K"]
    pdf_hash = policy["raw_sources"]["pdf"]["sha256"]
    return {"claim_id": f"{selected['company_id']}:pdf:K", "owner_id": selected["company_id"],
        "referenced_document_company_id": pdf_record["company_id"],
        "source_scope": "provided_pdf_partial_job_reference", "source_field": "K",
        "source_page": pdf_record["source_page"], "source_document_sha256": pdf_hash,
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "reference_target_job": pdf_record["evaluated_job"], "reference_location": pdf_record["location"],
        "locator": source_locator(pdf_hash, pdf_record["source_page"], "K"), "start": 0, "end": len(text), "quote": text}


def verify_reference_claim(claim, selected, pdf_record, policy):
    if claim != reference_claim(selected, pdf_record, policy):
        raise ValueError("PDF 참고 구절·직무 범위·기업 대응·출처가 다릅니다.")


def run(root, classification_dir, output, hf_home=None):
    root, folder, output = Path(root), Path(classification_dir), Path(output)
    if output.exists():
        raise FileExistsError("기존 참고 분류 결과를 보존합니다.")
    policy, records = load_user_records(root)
    selected = load_selected(root, policy)
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("status") != "completed" or summary["functional_passed"] != summary["functional_count"]:
        raise ValueError("구절 분류 기능 검사를 통과한 완료 기록이 필요합니다.")
    for name, expected in manifest["implementation_sha256"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError("검사한 구절 분류 구현과 다릅니다.")
    for name, expected in manifest["source_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("제공 자료·설정이 분류 기록과 다릅니다.")
    targets = []
    for company in selected["companies"]:
        if company["pdf_reference"] is None:
            continue
        record = records.get(company["catalog_link"]["company_id"])
        if record is None:
            raise ValueError("대응된 PDF 기업 자료가 없습니다.")
        validate_user_sources(build_user_bundle(record, policy), root)
        targets.append((company, reference_claim(company, record, policy)))
    output.mkdir(parents=True)
    save(output / "manifest.json", {"purpose": "선별 기업에 대응된 PDF의 특정 직무 참고 분류",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "candidate": manifest["candidate"],
        "source_hashes": manifest["source_hashes"], "classifier_implementation_sha256": manifest["implementation_sha256"],
        "reference_implementation_sha256": digest(__file__), "company_wide_scoring_performed": False})
    if __package__:
        from .mlx_generator import MlxGenerator
    else:
        from mlx_generator import MlxGenerator
    candidate = manifest["candidate"]
    results, cache, started = [], {}, time.perf_counter()
    try:
        generator = MlxGenerator(candidate["model_id"], candidate["revision"], hf_home)
        for index, (company, claim) in enumerate(targets, 1):
            cached = cache.get(claim["quote"])
            if cached is None:
                result = classify(claim, generator)
                cache[claim["quote"]] = result
                reused = False
            else:
                result = deepcopy(cached)
                result["claim"] = claim
                reused = True
            item = {"company_id": company["company_id"], "company_name": company["company_name"],
                    "reference_classification": result, "identical_quote_result_reused": reused,
                    "usable_as_company_wide_score": False, "reference_job_score": None}
            results.append(item)
            save(output / f"reference_{index:03d}.json", item)
            print(f"PDF 참고 [{index:02d}/{len(targets)}] {company['company_name']}: 직무 범위 별도 보존", flush=True)
        if any(digest(root / name) != expected for name, expected in manifest["source_hashes"].items()):
            raise ValueError("PDF 참고 분류 중 자료·설정이 바뀌었습니다.")
        save(output / "reference_results.json", results)
        summary = {"status": "completed", "reference_company_count": len(results),
            "unique_quote_model_calls": len(cache), "all_source_hashes_unchanged": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "company_wide_scoring_performed": False, "ready_for_ranking": False}
        save(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        save(output / "summary.json", {"status": "incomplete", "completed_count": len(results), "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--classification-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path)
    args = parser.parse_args()
    try:
        summary = run(args.project_root, args.classification_dir, args.output_dir, args.hf_home)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"PDF 특정 직무 참고 {summary['reference_company_count']}개를 별도로 연결했습니다.")


if __name__ == "__main__":
    main()
