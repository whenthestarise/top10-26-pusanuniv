"""사용자가 확정한 선별 50개 엑셀의 원문 셀로 실제 평가 입력을 만듭니다."""

import argparse
import json
from pathlib import Path

if __package__:
    from .data_policy import ROOT, build_selected_bundle, load_user_records, validate_user_sources
    from .selected50_table import load_selected
else:
    from data_policy import ROOT, build_selected_bundle, load_user_records, validate_user_sources
    from selected50_table import load_selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evidence/selected50")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("출력 폴더가 이미 있습니다. 새로운 이름을 지정하세요.")
    policy, _ = load_user_records()
    records = load_selected(ROOT, policy)["companies"]
    selected = [r for r in records if not args.company_id or r["company_id"] == args.company_id]
    if not selected:
        parser.error("연구팀 선별 50개 명단에 없는 기업 식별자입니다.")
    bundles = []
    for record in selected:
        bundle = build_selected_bundle(record, policy)
        validate_user_sources(bundle)
        bundles.append((record, bundle))
    args.output_dir.mkdir(parents=True)
    for record, bundle in bundles:
        path = args.output_dir / f"selected50_{record['selection_order']:03d}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(bundle, stream, ensure_ascii=False, indent=2)
    (args.output_dir / "manifest.json").write_text(json.dumps({"source_policy": policy["policy_version"],
        "bundle_count": len(bundles), "all_sources_user_provided": True, "scoring_performed": False,
        "note": "문서에 기재된 위치·직무의 입력이며 실제 사업장 확인이나 선정 완료를 뜻하지 않음"}, ensure_ascii=False, indent=2))
    print(f"선별 명단 기반 입력 {len(bundles)}개를 생성했습니다. 점수와 순위는 생성하지 않았습니다.")


if __name__ == "__main__":
    main()
