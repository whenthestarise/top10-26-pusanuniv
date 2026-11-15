"""선정된 읽기 후보로 제공 문서의 기업 사실만 추출하고 원문과 대조합니다."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

if __package__:
    from .data_policy import ROOT, build_user_bundle, load_user_records, source_locator, validate_user_sources
    from .mlx_generator import MlxGenerator
    from .provided_data_check import messages, validate
else:
    from data_policy import ROOT, build_user_bundle, load_user_records, source_locator, validate_user_sources
    from mlx_generator import MlxGenerator
    from provided_data_check import messages, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-id", required=True)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "reports" / f"provided_reading_{stamp}.json"
    if output.exists():
        parser.error("출력 파일이 이미 있습니다.")
    policy, records = load_user_records()
    record = records.get(args.company_id)
    if record is None:
        parser.error("제공받은 기업 문서에 없는 식별자입니다.")
    bundle = build_user_bundle(record, policy)
    template = bundle["sources"][0]
    sources = []
    for field in ("B", "C", "I", "G", "P", "K", "AA"):
        source = deepcopy(template)
        source.update(source_id=f"{record['company_id']}:{field}", document_field=field,
                      text=record["raw_fields"][field],
                      locator=source_locator(template["source_document_sha256"], record["source_page"], field))
        sources.append(source)
    bundle["sources"] = sources
    validate_user_sources(bundle)
    selected = json.loads((ROOT / "configs/selected_model.json").read_text())
    candidate = selected.get("candidate")
    if candidate is None:
        parser.error("읽기 검사를 통과한 후보가 없습니다.")
    generator = MlxGenerator(candidate["model_id"], candidate["revision"], args.hf_home)
    generation = generator(messages(record), max_tokens=256)
    errors, parsed = validate(generation["answer"], record)
    result = {"dataset_type": "real", "purpose": "provided_fact_reading_only",
              "company_id": record["company_id"], "sources": sources, "generation": generation,
              "facts": parsed if not errors else None, "validation_errors": errors,
              "scoring_performed": False, "ready_for_ranking": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("제공 자료 대조:", "통과" if not errors else "실패")
    print("기업 점수와 순위는 생성하지 않았습니다.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
