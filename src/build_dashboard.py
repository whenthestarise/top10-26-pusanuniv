"""보존된 기업 평가 결과를 검증하고 대시보드에 연결합니다."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

KEYS = ("salary", "work_life_balance", "infrastructure", "job_outlook", "ai_resilience")
NAMES = ("급여", "워라밸", "지역 인프라", "직무 유망성", "AI 안전성")


def build(root, output):
    root, output = Path(root), Path(output)
    public = root / "data/demo/model_top10_2026-11-15"
    audit = root / "reports/model_evaluation_2026-11-15/internal"
    paths = [public / "top10.json", public / "기업별_모델평가.csv", public / "TOP10_선정결과.md",
             audit / "calculation_result.json", audit / "applied_ratings.json", root / "data/processed/selected50_companies.json"]
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    published, calculated, ratings, selected = [json.loads(p.read_text()) for p in (paths[0], paths[3], paths[4], paths[5])]
    records = {r["company_id"]: r for r in selected["companies"]}
    public_rows = list(csv.DictReader(paths[1].open(encoding="utf-8-sig")))
    if len(records) != 50 or len(public_rows) != 50 or calculated["eligible_count"] != 48:
        raise ValueError("기업 수와 공통 비교집합을 확인하세요.")
    top_ids = [r["company_id"] for r in published["top10"]]
    if top_ids != [r["company_id"] for r in calculated["selected_companies"]] or len(set(top_ids)) != 10:
        raise ValueError("공개 선정 결과와 계산 기록이 다릅니다.")
    csv_rows = {r["기업ID"]: r for r in public_rows}
    companies = []
    for row in calculated["company_results"]:
        cid = row["company_id"]
        record, rated, exported = records[cid], ratings[cid], csv_rows[cid]
        for key, heading in zip(KEYS, ("제공연봉(만원/년)", "워라밸", "지역인프라", "직무유망성", "AI안전성")):
            value = float(exported[heading]) if exported[heading] else None
            if value != row["values"][key] or (key != "salary" and value != rated[key]["score"]):
                raise ValueError("공개 평가표와 계산·모델 평가값이 다릅니다.")
        if row["display_index_P"] is not None:
            if abs(float(exported["종합지수"]) - row["display_index_P"]) > 1e-9:
                raise ValueError("종합지수가 공개 평가표와 다릅니다.")
            if abs(sum(t["relative_index_contribution"] * 100 for t in row["criterion_explanations"].values()) - row["display_index_P"]) > 1e-9:
                raise ValueError("항목별 기여도의 합이 종합지수와 다릅니다.")
        companies.append({"id": cid, "name": record["company_name"], "region": record["region"],
            "location": record["location"], "industry": record["industry"], "products": record["products_business"],
            "roles": record["main_hiring_roles"], "order": record["selection_order"], "values": row["values"],
            "score": row["display_index_P"], "rank": row.get("rank"), "selected": cid in top_ids,
            "selection_rank": top_ids.index(cid) + 1 if cid in top_ids else None,
            "explanations": row["criterion_explanations"], "ratings": rated})
    if len(companies) != 50 or {c["id"] for c in companies} != set(records):
        raise ValueError("대시보드 기업 연결이 완전하지 않습니다.")
    data = {"date": "2026-11-15", "title": "부울경 기업 TOP10", "companies": companies, "top_ids": top_ids,
        "criteria": [{"key": key, "name": name, "weight": calculated["weights"][key],
                      "share": calculated["weights"][key] / calculated["active_weight_sum"] * 100} for key, name in zip(KEYS, NAMES)],
        "baseline": calculated["salary_baseline_manwon"], "means": calculated["normalization"]["non_salary_means"],
        "region_counts": published["region_counts"], "source_hashes": before,
        "model": "Qwen3-4B · 4비트", "score_type": "model_interpretation", "empirically_verified": False}
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != value for name, value in before.items()):
        raise ValueError("화면 데이터를 생성하는 동안 원본이 변경됐습니다.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "downloads").mkdir(exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (output / "data.js").write_text("window.DASHBOARD_DATA = " + encoded + ";\n", encoding="utf-8")
    for path in paths[:3]:
        shutil.copy2(path, output / "downloads" / path.name)
    print(f"대시보드 연결 완료: 기업 {len(companies)}개 · 선정 {len(top_ids)}개 · 공통 비교집합 48개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    build(args.project_root, args.output_dir or args.project_root / "dashboard")
