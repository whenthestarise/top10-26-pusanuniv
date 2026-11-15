"""실제 기업 평가의 입력을 사용자가 제공한 문서의 원문 필드로 제한합니다."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_user_records(root=None):
    root = Path(root or ROOT)
    policy = json.loads((root / "configs/data_policy.json").read_text(encoding="utf-8"))
    for source in policy["raw_sources"].values():
        path = root / "data/raw" / source["stored_name"]
        if digest(path) != source["sha256"]:
            raise ValueError("제공받은 원본 자료의 해시가 정책과 다릅니다.")
    path = root / "data/processed/document_companies.json"
    if digest(path) != policy["processed_companies_sha256"]:
        raise ValueError("문서에서 추출한 기업 자료가 검증된 자료와 다릅니다.")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["source_sha256"] != policy["raw_sources"]["pdf"]["sha256"]:
        raise ValueError("추출 자료가 제공받은 문서에 연결되지 않습니다.")
    return policy, {item["company_id"]: item for item in document["companies"]}


def source_locator(pdf_hash, page, field):
    return f"user_pdf:sha256:{pdf_hash}#page={page}&field={field}"


def selected_locator(workbook_hash, sheet, row, field):
    return f"user_xlsx:sha256:{workbook_hash}#sheet={sheet}&cell={field}{row}"


def enforce_evaluation_population(bundle, root=None):
    if bundle.get("dataset_type") != "real":
        return
    policy = json.loads((Path(root or ROOT) / "configs/data_policy.json").read_text(encoding="utf-8"))
    if "selected50" in policy["raw_sources"] and not bundle.get("context", {}).get("company_id", "").startswith("selected50:"):
        raise ValueError("실제 평가는 연구팀 선별 50개를 사용합니다. 원본 150개와 PDF 기업 자료는 대조용입니다.")


def validate_user_sources(bundle, root=None):
    if not isinstance(bundle, dict):
        raise ValueError("입력은 자료 객체여야 합니다.")
    if bundle.get("dataset_type") == "synthetic":
        return
    if not isinstance(bundle.get("sources"), list) or any(not isinstance(s, dict) for s in bundle["sources"]):
        raise ValueError("출처는 자료 객체의 목록이어야 합니다.")
    if not bundle["sources"]:
        return
    policy, records = load_user_records(root)
    context = bundle.get("context", {})
    if context.get("company_id", "").startswith("selected50:"):
        if __package__:
            from .selected50_table import load_selected
        else:
            from selected50_table import load_selected
        selected = load_selected(Path(root or ROOT), policy)
        row = next((c for c in selected["companies"] if c["company_id"] == context["company_id"]), None)
        if row is None:
            raise ValueError("연구팀이 선별한 50개 명단에 없는 기업입니다.")
        expected = {"company_id": row["company_id"], "company_name": row["company_name"],
                    "workplace": row["location"], "target_job": row["main_hiring_roles"]}
        if any(context.get(key) != value for key, value in expected.items()):
            raise ValueError("평가 범위가 선별 명단의 기업·기재 위치·채용 직군과 다릅니다.")
        book_hash = policy["raw_sources"]["selected50"]["sha256"]
        for source in bundle["sources"]:
            field = source.get("document_field")
            raw = row["raw_fields"].get(field)
            if (field not in row["raw_fields"] or source.get("source_kind") != "user_xlsx_cell"
                    or source.get("source_document_sha256") != book_hash
                    or source.get("source_sheet") != row["source_sheet"] or source.get("source_row") != row["source_row"]
                    or source.get("locator") != selected_locator(book_hash, row["source_sheet"], row["source_row"], field)
                    or source.get("text") != ("" if raw is None else str(raw))):
                raise ValueError("실제 평가에는 선별 엑셀의 원문 셀만 사용할 수 있습니다.")
        return
    record = records.get(context.get("company_id"))
    if record is None:
        raise ValueError("제공받은 기업 목록에 연결되지 않은 실제 기업입니다.")
    expected_context = {"company_id": record["company_id"], "company_name": record["company_name"],
                        "workplace": record["location"], "target_job": record["evaluated_job"]}
    if any(context.get(key) != value for key, value in expected_context.items()):
        raise ValueError("평가 범위가 제공 문서의 기업·기재 위치·직무와 다릅니다.")
    pdf_hash = policy["raw_sources"]["pdf"]["sha256"]
    for source in bundle["sources"]:
        field = source.get("document_field")
        if (source.get("source_kind") != "user_pdf_field" or field not in record["raw_fields"]
                or source.get("source_document_sha256") != pdf_hash
                or source.get("document_page") != record["source_page"]
                or source.get("locator") != source_locator(pdf_hash, record["source_page"], field)
                or source.get("text") != record["raw_fields"][field]):
            raise ValueError("실제 평가에는 제공받은 문서의 원문 필드만 사용할 수 있습니다.")


def build_user_bundle(record, policy):
    context = {"company_id": record["company_id"], "company_name": record["company_name"],
               "workplace": record["location"], "target_job": record["evaluated_job"]}
    pdf_hash = policy["raw_sources"]["pdf"]["sha256"]
    return {"dataset_type": "real", "context": context,
        "sources": [{"source_id": f"{record['company_id']}:K", "source_kind": "user_pdf_field",
            "source_document_sha256": pdf_hash, "document_page": record["source_page"], "document_field": "K",
            "title": policy["raw_sources"]["pdf"]["original_name"],
            "locator": source_locator(pdf_hash, record["source_page"], "K"),
            "retrieved_at": record["reference_date"], "scope": dict(context), "text": record["raw_fields"]["K"],
            "scope_note": "문서에 기재된 위치이며 실제 사업장 확인을 뜻하지 않음. AI 설명은 제공 문서의 작성자 정성 판단임."}]}


def build_selected_bundle(record, policy):
    context = {"company_id": record["company_id"], "company_name": record["company_name"],
               "workplace": record["location"], "target_job": record["main_hiring_roles"]}
    source = policy["raw_sources"]["selected50"]
    return {"dataset_type": "real", "context": context, "sources": [
        {"source_id": f"{record['company_id']}:{field}", "source_kind": "user_xlsx_cell",
         "source_document_sha256": source["sha256"], "source_sheet": record["source_sheet"],
         "source_row": record["source_row"], "document_field": field, "title": source["original_name"],
         "locator": selected_locator(source["sha256"], record["source_sheet"], record["source_row"], field),
         "retrieved_at": "제공 파일 기준일 미기재", "scope": dict(context), "text": str(record["raw_fields"][field]),
         "scope_note": "선별 명단의 직군별 정성 설명이며 실제 지원 사업장과 회사 전체 업무 비중은 미확정"}
        for field in ("I", "J", "K") if record["raw_fields"].get(field) is not None]}
