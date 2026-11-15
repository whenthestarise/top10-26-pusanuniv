"""새 선별 명단을 평가표의 주대상으로 연결하고 PDF는 별도 참고로 보존합니다."""

import hashlib
import json
from pathlib import Path


def load_selected(root, policy):
    path = Path(root) / "data/processed/selected50_companies.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != policy["processed_selected50_sha256"]:
        raise ValueError("선별 50개 처리 자료의 해시가 정책과 다릅니다.")
    selected = json.loads(path.read_text(encoding="utf-8"))
    if selected["source_sha256"] != policy["raw_sources"]["selected50"]["sha256"]:
        raise ValueError("선별 자료가 제공된 엑셀에 연결되지 않습니다.")
    if len(selected["companies"]) != 50 or selected["selection_policy"]["final_selection_target"] != 10:
        raise ValueError("선별 50개와 최종 10개 범위를 확인하세요.")
    return selected


def selected_table_companies(root, policy, catalog, pdf_records):
    if __package__:
        from .build_company_evaluation import record_refs
    else:
        from build_company_evaluation import record_refs
    selected = load_selected(root, policy)
    companies = []
    source = policy["raw_sources"]["selected50"]
    for row in selected["companies"]:
        raw = row["raw_fields"]
        reference = row["pdf_reference"]
        parent = row["catalog_link"]
        record = {"source_kind": "selected_workbook", "source_sheet": row["source_sheet"],
            "source_row": row["source_row"], "source_page": reference["source_page"] if reference else None,
            "raw_fields": raw, "annual_salary_manwon": row["annual_salary_manwon"],
            "location": row["location"], "evaluated_job": row["main_hiring_roles"],
            "ai_exposure_label": row["ai_exposure_label"], "salary_data_type": row["salary_source_text"],
            "salary_reference_period": None, "included_in_salary_statistics": row["annual_salary_manwon"] is not None,
            "duplicate_of": None, "exclusion_reason": None}
        selection_source = {"kind": "user_provided_xlsx", "file": source["stored_name"], "sha256": source["sha256"],
            "sheet": row["source_sheet"], "row": row["source_row"], "address": f"A{row['source_row']}:L{row['source_row']}",
            "quote": raw}
        companies.append({"company_id": row["company_id"], "selection_order": row["selection_order"],
            "original_id": parent["original_id"], "catalog_link_status": parent["status"],
            "company_name": row["company_name"], "region": row["region"], "industry": row["industry"],
            "products_business": row["products_business"], "catalog_notes": row["review_result"],
            "selection_source": selection_source, "selection_review_result": row["review_result"], "research": record,
            "pdf_reference_salary_manwon": reference["annual_salary_manwon"] if reference else None,
            "pdf_reference_ai_label": reference["ai_exposure_label"] if reference else None,
            "pdf_reference_job": reference["evaluated_job"] if reference else None,
            "pdf_reference_location": reference["location"] if reference else None,
            "pdf_reference_notes": reference["comparison_notes"] if reference else None,
            "salary_difference_manwon": row["salary_difference_manwon"],
            "salary_comparison_status": row["salary_comparison_status"],
            "pdf_reference_sources": record_refs(reference, policy, ("D", "G", "H", "I", "J", "K", "N", "T", "W"))})
    return companies, selected
