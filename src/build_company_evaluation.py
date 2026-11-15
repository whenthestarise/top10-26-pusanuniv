"""제공 자료의 값·원문·설문 가중치를 기업별 5개 항목 평가표로 연결합니다."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re

from pypdf import PdfReader
import yaml

if __package__:
    from .data_policy import load_user_records
    from .prepare_project_data import clean_text, join_companies, load_catalog, load_survey_weights
else:
    from data_policy import load_user_records
    from prepare_project_data import clean_text, join_companies, load_catalog, load_survey_weights


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_IDS = ("salary", "work_life_balance", "infrastructure", "job_outlook", "ai_resilience")
REQUIRED = {
    "work_life_balance": ["실제 근로시간·초과근무", "교대 형태", "휴가·유연근무 운영"],
    "infrastructure": ["실제 지원 사업장", "교통·주거·생활편의의 공통 지표"],
    "job_outlook": ["공통 평가 직무·범위", "해당 직무의 고용·채용 수요 전망"],
    "ai_resilience": ["공통 평가 직무·범위", "주요 업무와 비중 근거", "정성 설명의 동일 기준 원점수 평가"],
}
WIDE_HEADERS = ["기업ID", "선별순번", "원본150순번", "원본대응상태", "지역", "기업명", "평가상태", "명단기재위치",
                "주요채용직군", "우선연봉(만원/년)", "PDF참고연봉(만원/년)", "연봉차이(만원)", "연봉지수",
                "워라밸점수", "인프라점수", "직무유망성점수", "AI안전성점수", "명단AI등급", "PDF참고AI등급",
                "급여계산대상", "미상평가항목", "명단연봉출처", "문서참고쪽수", "제외·보류사유"]
DETAIL_HEADERS = ["기업ID", "선별순번", "원본150순번", "지역", "기업명", "평가항목", "가중치", "정규화가중치",
                  "연결수치", "수치단위", "참고문구", "비급여점수", "연봉지수", "연결상태",
                  "근거위치", "원문", "미상사유"]


def pdf_ref(policy, page, field, quote):
    source = policy["raw_sources"]["pdf"]
    return {"kind": "user_provided_pdf", "file": source["stored_name"], "sha256": source["sha256"],
            "page": page, "field": field, "quote": quote}


def record_refs(record, policy, fields):
    if record and record.get("source_kind") == "selected_workbook":
        source = policy["raw_sources"]["selected50"]
        return [{"kind": "user_provided_xlsx", "file": source["stored_name"], "sha256": source["sha256"],
                 "sheet": record["source_sheet"], "row": record["source_row"], "field": field,
                 "address": f"{field}{record['source_row']}", "quote": record["raw_fields"][field]}
                for field in fields if record["raw_fields"].get(field) is not None]
    return [pdf_ref(policy, record["source_page"], field, record["raw_fields"][field])
            for field in fields if record["raw_fields"].get(field)] if record else []


def catalog_ref(company, policy):
    if company.get("selection_source"):
        return company["selection_source"]
    source = policy["raw_sources"]["companies"]
    return {"kind": "user_provided_csv", "file": source["stored_name"], "sha256": source["sha256"],
            "row": company["original_id"] + 1,
            "quote": {k: company[k] for k in ("company_name", "region", "industry", "products_business", "catalog_notes")}}


def extract_pdf_notes(pages, policy):
    """원본 문서의 제외 사유와 자료 부족 설명을 위치와 함께 보존합니다."""
    exclusions, limitation = {}, None
    for page_number, text in enumerate(pages, 1):
        headers = list(re.finditer(r"^제외 항목\s+(\d+)\s*\|[^\n]*", text, re.M))
        for index, header in enumerate(headers):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            block = text[header.end():end]
            name = re.search(r"기업명\(원본\):\s*(.*?)\n지역:", block, re.S)
            region = re.search(r"\n지역:\s*([^\n]+)", block)
            reason = re.search(r"확인하지 못한 내용·제외 사유:\s*(.*?)\n처리:", block, re.S)
            if not all((name, region, reason)):
                raise ValueError(f"제공 문서 {page_number}쪽의 제외 사유 형식을 확인하세요.")
            original_id = int(header.group(1))
            if original_id in exclusions:
                raise ValueError("제외 사유의 기업 순번이 중복됐습니다.")
            quote = reason.group(1)
            ref = pdf_ref(policy, page_number, "연봉 미확인 제외 사유", quote)
            ref.update(start=header.end() + reason.start(1), end=header.end() + reason.end(1))
            exclusions[original_id] = {"company_name": name.group(1).strip(), "region": region.group(1).strip(),
                                       "reason": quote, "source": ref}
        match = re.search(r"현재 자료에는 회사별.*?없다\.", text, re.S)
        if match:
            if limitation is not None:
                raise ValueError("자료 부족 설명이 중복됐습니다.")
            limitation = pdf_ref(policy, page_number, "자료 범위와 한계", match.group())
            limitation.update(start=match.start(), end=match.end())
    if limitation is None:
        raise ValueError("제공 문서의 자료 범위 설명을 찾지 못했습니다.")
    return exclusions, limitation


def check_join(companies, exclusions):
    missing = {c["original_id"] for c in companies if c["research"] is None}
    if missing != set(exclusions):
        raise ValueError("연봉 미확인 기업과 문서의 제외 사유 목록이 다릅니다.")
    for company in companies:
        if company["original_id"] in exclusions:
            excluded = exclusions[company["original_id"]]
            if any(clean_text(company[key]) != clean_text(excluded[key]) for key in ("company_name", "region")):
                raise ValueError("제공 목록과 문서 제외 사유의 기업명·지역이 다릅니다.")


def criterion_row(company, definition, policy, baseline, excluded):
    record, key = company["research"], definition["criterion_id"]
    row = {"criterion_id": key, "name": definition["criterion_name"], "weight": definition["weight"],
           "normalized_weight": definition["normalized_active_weight"], "numeric_value": None,
           "unit": None, "reference_text": None, "raw_score": None, "index": None,
           "status": "자료 미기재", "value_kind": "missing", "sources": [], "missing_information": []}
    if key == "salary":
        if record and record["annual_salary_manwon"] is None:
            row.update(status="새 명단 연봉 미확인", sources=record_refs(record, policy, ("G", "H")),
                       missing_information=["새 엑셀에 숫자 연봉 미기재; PDF 수치는 참고 열에만 병기"])
        elif record:
            value = record["annual_salary_manwon"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("연봉 수치는 유한한 양수여야 합니다.")
            raw_g = float(str(record["raw_fields"]["G"]).replace(",", ""))
            if not math.isclose(value, raw_g, rel_tol=0, abs_tol=1e-8):
                raise ValueError("만원으로 환산된 G열과 연봉 값이 다릅니다.")
            row.update(numeric_value=value, unit="만원/년", index=value / baseline, status="연봉지수 연결",
                       value_kind="provided_numeric", sources=record_refs(record, policy, ("G", "H", "N", "O", "P")))
        else:
            row.update(status="문서 비교표 제외", sources=[excluded["source"]], missing_information=[excluded["reason"]])
    elif key == "work_life_balance":
        row["missing_information"] = REQUIRED[key].copy()
    elif key == "infrastructure":
        row.update(reference_text=record["location"] if record else company["region"],
                   value_kind="location_reference_only", status="위치 참고만 가능" if record else "지역 참고만 가능",
                   sources=record_refs(record, policy, ("D",)) or [catalog_ref(company, policy)],
                   missing_information=REQUIRED[key].copy())
    elif key == "job_outlook":
        row.update(reference_text=record["evaluated_job"] if record else company["industry"],
                   value_kind="job_or_industry_reference_only", status="직무·업종 참고만 가능",
                   sources=record_refs(record, policy, ("I", "E", "F")) or [catalog_ref(company, policy)],
                   missing_information=REQUIRED[key].copy())
    elif key == "ai_resilience":
        row["missing_information"] = REQUIRED[key].copy()
        if record:
            deferred = record["ai_exposure_label"] == "평가 보류"
            row.update(reference_text=record["ai_exposure_label"], value_kind="document_qualitative_estimate",
                       status="문서에서 평가 보류" if deferred else "문서 정성 설명 연결",
                       sources=record_refs(record, policy, ("I", "J", "K", "L", "AA")))
            if deferred:
                row["missing_information"].insert(0, "해당 법인의 구체 직무 근거")
    return row


def build_table(companies, weights, config, policy, exclusions, limitation):
    definitions = [row for row in weights["criteria"] if row["active"]]
    if tuple(row["criterion_id"] for row in definitions) != ACTIVE_IDS:
        raise ValueError("문서 모형의 활성 5개 항목과 순서를 확인하세요.")
    baseline = config["salary"]["baseline"]
    if not isinstance(baseline, (int, float)) or baseline <= 0 or not math.isfinite(baseline):
        raise ValueError("기준연봉은 유한한 양수여야 합니다.")
    check_join(companies, exclusions)
    rows = []
    for company in companies:
        record = company["research"]
        excluded = exclusions.get(company["original_id"])
        criteria = {d["criterion_id"]: criterion_row(company, d, policy, baseline, excluded) for d in definitions}
        duplicate = record.get("duplicate_of") if record else None
        missing = [key for key in ACTIVE_IDS if (criteria[key]["index"] is None if key == "salary" else criteria[key]["raw_score"] is None)]
        reasons = [criteria[key]["name"] + ": " + " / ".join(criteria[key]["missing_information"]) for key in missing]
        if duplicate is not None:
            reasons.insert(0, record["exclusion_reason"])
        rows.append({"company_id": company["company_id"], "original_id": company["original_id"],
            "selection_order": company.get("selection_order"),
            "catalog_link_status": company.get("catalog_link_status", "원본 목록"),
            "company_name": company["company_name"], "region": company["region"],
            "industry": company["industry"], "products_business": company["products_business"],
            "catalog_notes": company["catalog_notes"], "catalog_source": catalog_ref(company, policy),
            "document_location": record["location"] if record else None,
            "document_target_job": record["evaluated_job"] if record else None,
            "document_page": record["source_page"] if record else None,
            "salary_data_type": record["salary_data_type"] if record else None,
            "salary_reference_period": record["salary_reference_period"] if record else None,
            "scope_note": "제공 자료의 기재 위치·채용 직군이며 실제 지원 사업장 또는 회사 전체 대표성은 확인되지 않음",
            "included_in_salary_statistics": bool(record and record["included_in_salary_statistics"]),
            "duplicate_of": duplicate, "criteria": criteria, "missing_criteria": missing,
            "evaluation_status": "중복으로 비교 제외" if duplicate is not None else "자료 부족으로 종합평가 보류",
            "deferral_reasons": reasons, "composite_index": None, "ready_for_ranking": False,
            "pdf_reference_salary_manwon": company.get("pdf_reference_salary_manwon"),
            "pdf_reference_ai_label": company.get("pdf_reference_ai_label"),
            "pdf_reference_job": company.get("pdf_reference_job"),
            "pdf_reference_location": company.get("pdf_reference_location"),
            "pdf_reference_notes": company.get("pdf_reference_notes"),
            "salary_difference_manwon": company.get("salary_difference_manwon"),
            "salary_comparison_status": company.get("salary_comparison_status"),
            "pdf_reference_sources": company.get("pdf_reference_sources", []),
            "selection_review_result": company.get("selection_review_result")})
    coverage = {key: sum(c["criteria"][key]["index"] is not None if key == "salary"
                         else c["criteria"][key]["raw_score"] is not None for c in rows) for key in ACTIVE_IDS}
    counts = {"company_count": len(rows), "region_counts": dict(Counter(r["region"] for r in rows)),
              "evaluation_row_count": len(rows) * len(definitions), "salary_available_count": coverage["salary"],
              "salary_missing_count": len(rows) - coverage["salary"],
              "salary_calculation_unit_count": sum(r["included_in_salary_statistics"] for r in rows),
              "duplicate_excluded_count": sum(r["duplicate_of"] is not None for r in rows),
              "ai_qualitative_reference_count": sum(r["criteria"]["ai_resilience"]["status"] == "문서 정성 설명 연결" for r in rows),
              "ai_document_deferred_count": sum(r["criteria"]["ai_resilience"]["status"] == "문서에서 평가 보류" for r in rows),
              "usable_value_counts": coverage, "common_complete_comparison_count": 0,
              "ready_for_ranking": False, "final_top10": None}
    return {"schema_version": "1.0", "table_version": "company_evaluation_table_v1", "dataset_type": "real",
            "model_id": config["model_id"], "source_policy": policy["policy_version"],
            "salary_baseline_manwon": baseline, "active_weight_sum": weights["active_weight_sum"],
            "weights": definitions, "survey_summary": {key: weights[key] for key in
                ("response_count", "important_total", "unimportant_total", "counts_match_document_reference")},
            "source_manifest": policy["raw_sources"], "document_limitation": limitation,
            "summary": counts, "companies": rows}


def source_text(sources):
    return "\n".join((f"{s.get('field', '')}: " + s["quote"] if isinstance(s["quote"], str)
                      else json.dumps(s["quote"], ensure_ascii=False)) for s in sources)


def source_locations(sources):
    return "; ".join(f"제공문서 {s['page']}쪽 {s['field']}" if "page" in s else
        f"선별파일 '{s['sheet']}'!{s['address']}" if "sheet" in s else f"제공기업목록 {s['row']}행" for s in sources)


def csv_rows(table):
    wide, detail = [], []
    for company in table["companies"]:
        c = company["criteria"]
        wide.append([company["company_id"], company["selection_order"], company["original_id"], company["catalog_link_status"],
            company["region"], company["company_name"],
            company["evaluation_status"], company["document_location"], company["document_target_job"],
            c["salary"]["numeric_value"], company["pdf_reference_salary_manwon"], company["salary_difference_manwon"],
            c["salary"]["index"], *[c[key]["raw_score"] for key in ACTIVE_IDS[1:]], c["ai_resilience"]["reference_text"],
            company["pdf_reference_ai_label"], "예" if company["included_in_salary_statistics"] else "아니오",
            ", ".join(c[key]["name"] for key in company["missing_criteria"]), company["salary_data_type"],
            company["document_page"], "\n".join(company["deferral_reasons"])])
        for key in ACTIVE_IDS:
            row = c[key]
            detail.append([company["company_id"], company["selection_order"], company["original_id"], company["region"], company["company_name"],
                row["name"], row["weight"], row["normalized_weight"], row["numeric_value"], row["unit"],
                row["reference_text"], row["raw_score"], row["index"], row["status"], source_locations(row["sources"]),
                source_text(row["sources"]), " / ".join(row["missing_information"])])
    return wide, detail


def write_outputs(table, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("기존 평가표를 보존합니다. 새로운 출력 폴더를 지정하세요.")
    output.mkdir(parents=True)
    files = {"company_evaluation_table.json": table, "evaluation_coverage.json": table["summary"],
             "document_comparison_inputs.json": {"dataset_type": "real", "model_id": table["model_id"],
                 "active_criteria": list(ACTIVE_IDS), "weights": {r["criterion_id"]: r["weight"] for r in table["weights"]},
                 "common_comparison_company_ids": [], "companies": [], "ready_for_ranking": False,
                 "deferred_companies": [{"company_id": c["company_id"], "missing_criteria": c["missing_criteria"],
                     "duplicate_of": c["duplicate_of"]} for c in table["companies"]],
                 "reason": "활성 5개 기준이 완결된 공통 비교집합이 없음"}}
    for name, value in files.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    wide, detail = csv_rows(table)
    for name, header, rows in (("기업별_평가표.csv", WIDE_HEADERS, wide), ("평가근거_상세.csv", DETAIL_HEADERS, detail)):
        with (output / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)


def run(root, output):
    root = Path(root)
    policy, records = load_user_records(root)
    config = yaml.safe_load((root / "configs/document_model.yaml").read_text(encoding="utf-8"))
    catalog, _ = load_catalog(root / "data/raw/company_catalog.csv")
    if __package__:
        from .selected50_table import selected_table_companies
    else:
        from selected50_table import selected_table_companies
    companies, selected = selected_table_companies(root, policy, catalog, records)
    weights = load_survey_weights(root / "data/raw/survey_responses.csv", config)
    if not weights["counts_match_document_reference"]:
        raise ValueError("제공 설문과 문서 모형의 가중치 집계가 다릅니다.")
    reader = PdfReader(root / "data/raw" / policy["raw_sources"]["pdf"]["stored_name"])
    exclusions, limitation = extract_pdf_notes([page.extract_text() or "" for page in reader.pages], policy)
    table = build_table(companies, weights, config, policy, {}, limitation)
    if table["summary"]["company_count"] != 50:
        raise ValueError("평가 대상이 선별 50개와 다릅니다.")
    table["selection_policy"] = selected["selection_policy"]
    table["job_guidelines"] = selected["job_guidelines"]
    table["summary"].update(original_population_count=len(catalog), selected_population_count=50,
                            mapping_counts=selected["summary"]["mapping_counts"],
                            salary_difference_count=selected["summary"]["salary_difference_count"],
                            final_selection_target=10)
    table["reference_catalog150"] = [{k: c[k] for k in ("company_id", "original_id", "company_name", "region")} for c in catalog]
    table["notes"] = ["새 엑셀 연봉 우선, PDF 수치 병기", "기준연봉 7,150만원은 기존 문서 모형의 확정값 유지",
        "150개는 원본 모집단, 선별된 50개가 평가 대상, 최종 10개는 50개 안에서 선정",
        "직군별 기준의 로봇·물리 자동화와 생성형 AI 설명을 구분하며 회사 전체 확률이나 확정 점수로 변환하지 않음"]
    write_outputs(table, output)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation/company_table")
    args = parser.parse_args()
    try:
        table = run(ROOT, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    summary = table["summary"]
    print(f"기업 {summary['company_count']}개, 평가 항목 {summary['evaluation_row_count']}행을 연결했습니다.")
    print(f"연봉 연결 {summary['salary_available_count']}개, 선별 명단 연봉 계산 단위 {summary['salary_calculation_unit_count']}개")
    print("비급여 점수와 최종 순위는 자료 부족으로 보류했습니다.")


if __name__ == "__main__":
    main()
