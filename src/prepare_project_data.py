"""기업 목록 CSV·설문 CSV·연구 PDF를 재현 가능한 프로젝트 입력으로 연결합니다."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import unicodedata

import yaml

if __package__:
    from .calculate_document_weights import calculate_weights
    from .calculate_survey_weights import LABEL_TO_ID, QUESTION, normalize_text
    from .extract_document_companies import extract_records, validate_and_enrich
else:
    from calculate_document_weights import calculate_weights
    from calculate_survey_weights import LABEL_TO_ID, QUESTION, normalize_text
    from extract_document_companies import extract_records, validate_and_enrich


ROOT = Path(__file__).resolve().parents[1]
UNIMPORTANT_QUESTION = "귀하가 취업할 당시 가장 중요하지 않게 여기는 항목은 무엇입니까?"
COMPANY_COLUMNS = ("지역", "기업명", "주요 산업 분야", "주요 취급 제품/서비스", "특징 및 구분")
RAW_NAMES = {
    "companies": "company_catalog.csv",
    "survey": "survey_responses.csv",
    "pdf": "직업만족도_AI분석_자료와프롬프트.pdf",
}
OUTPUT_NAMES = (
    "company_catalog.json", "document_companies.json", "document_survey_weights.json",
    "document_analysis_inputs.json", "data_readiness.json", "source_manifest.json",
)


def clean_text(value):
    return normalize_text(unicodedata.normalize("NFC", value or ""))


def read_csv(path):
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"UTF-8/CP949로 읽을 수 없는 CSV: {path}")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    headers = [clean_text(value) for value in reader.fieldnames or []]
    if not headers or any(not value for value in headers) or len(headers) != len(set(headers)):
        raise ValueError(f"CSV 열 이름 누락 또는 중복: {path}")
    reader.fieldnames = headers
    rows = []
    for number, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV {number}행의 열 개수가 다릅니다: {path}")
        rows.append({key: clean_text(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"빈 CSV: {path}")
    return rows, encoding


def load_catalog(path):
    rows, encoding = read_csv(path)
    missing = set(COMPANY_COLUMNS) - set(rows[0])
    if missing:
        raise ValueError(f"기업 목록 열 누락: {sorted(missing)}")
    companies = []
    for index, row in enumerate(rows, 1):
        if not row["기업명"] or row["지역"] not in {"부산", "울산", "경남"}:
            raise ValueError(f"기업 목록 {index}번의 기업명·지역을 확인하세요.")
        companies.append({
            "company_id": f"document:{index:03d}",
            "original_id": index,
            "company_name": row["기업명"],
            "region": row["지역"],
            "industry": row["주요 산업 분야"],
            "products_business": row["주요 취급 제품/서비스"],
            "catalog_notes": row["특징 및 구분"],
        })
    return companies, encoding


def survey_counts(rows, column):
    counts = Counter({key: 0 for key in set(LABEL_TO_ID.values())})
    distribution = Counter()
    repeated_rows = 0
    for number, row in enumerate(rows, 2):
        ids = []
        for label in (clean_text(value) for value in row[column].split(",")):
            if not label:
                continue
            if label not in LABEL_TO_ID:
                raise ValueError(f"설문 {number}행의 알 수 없는 선택지: {label}")
            ids.append(LABEL_TO_ID[label])
        selected = set(ids)
        repeated_rows += len(ids) != len(selected)
        distribution[len(selected)] += 1
        counts.update(selected)
    return dict(counts), {
        "selection_count_distribution": {str(k): v for k, v in sorted(distribution.items())},
        "rows_with_repeated_items": repeated_rows,
        "blank_answer_rows": distribution[0],
    }


def load_survey_weights(path, config):
    rows, encoding = read_csv(path)
    for column in (QUESTION, UNIMPORTANT_QUESTION):
        if column not in rows[0]:
            raise ValueError(f"설문 필수 질문 열 누락: {column}")
    important, important_audit = survey_counts(rows, QUESTION)
    unimportant, unimportant_audit = survey_counts(rows, UNIMPORTANT_QUESTION)
    resolved = deepcopy(config)
    resolved["survey"].update(
        response_count=len(rows),
        important_total=sum(important.values()),
        unimportant_total=sum(unimportant.values()),
    )
    for item in resolved["criteria"]:
        item["important_count"] = important[item["id"]]
        item["unimportant_count"] = unimportant[item["id"]]
    result = calculate_weights(resolved)
    reference_matches = (
        len(rows) == config["survey"]["response_count"]
        and sum(important.values()) == config["survey"]["important_total"]
        and sum(unimportant.values()) == config["survey"]["unimportant_total"]
        and all(
            important[item["id"]] == item["important_count"]
            and unimportant[item["id"]] == item["unimportant_count"]
            for item in config["criteria"]
        )
    )
    result.update(
        dataset_type="real", response_count=len(rows), source_encoding=encoding,
        formula=config["survey"]["formula"],
        counts_source="survey_responses.csv",
        counts_match_document_reference=reference_matches,
        source_audit={"important": important_audit, "unimportant": unimportant_audit},
    )
    return result


def join_companies(catalog, records):
    by_id = {item["original_id"]: item for item in records}
    if len(by_id) != len(records):
        raise ValueError("PDF의 원본 순번이 중복됐습니다.")
    expected_ids = {item["original_id"] for item in catalog}
    if set(by_id) - expected_ids:
        raise ValueError("기업 목록에 없는 PDF 원본 순번이 있습니다.")
    joined = []
    for item in catalog:
        record = by_id.get(item["original_id"])
        if record and any(
            clean_text(record[key]) != clean_text(item[key])
            for key in ("company_name", "region")
        ):
            raise ValueError(f"CSV와 PDF의 기업명·지역 불일치: 원본 {item['original_id']}번")
        joined.append({**item, "research": record})
    return joined


def analysis_inputs(companies, config):
    active_ids = {item["id"] for item in config["criteria"] if item["active"]}
    results = []
    for company in companies:
        research = company["research"] or {}
        criteria = {}
        for item in config["criteria"]:
            key = item["id"]
            if key not in active_ids:
                continue
            evidence = {}
            status = "자료 부족"
            if key == "salary" and research:
                status = "검증 필요"
                evidence = {
                    "average_salary_manwon": research["annual_salary_manwon"],
                    "salary_index": research["salary_index"],
                    "source_url": research["salary_source_url"],
                    "reference_period": research["salary_reference_period"],
                    "data_type": research["salary_data_type"],
                    "source_page": research["source_page"],
                }
            elif key == "ai_resilience" and research:
                status = "검증 필요"
                evidence = {
                    "evaluated_job": research["evaluated_job"],
                    "exposure_label": research["ai_exposure_label"],
                    "explanation": research["ai_exposure_explanation"],
                    "scope_limitations": research["ai_scope_limitations"],
                    "job_source_url": research["job_source_url"],
                    "source_page": research["source_page"],
                }
            criteria[key] = {
                "name": item["name"], "status": status,
                "evidence": evidence,
                # 연봉지수는 evidence에만 보존하고 비급여 원점수와 구분합니다.
                "score": None,
                "missing_information": [
                    "동일 직무·사업장에 적용할 수 있는 비교 근거와 평가 척도 확인"
                ] if key in active_ids else [],
            }
        results.append({
            "company_id": company["company_id"], "company_name": company["company_name"],
            "source_id": f"company_dataset:{company['company_id']}",
            "source_files": [RAW_NAMES["companies"], RAW_NAMES["pdf"]],
            "profile": {
                "region": company["region"], "location": research.get("location"),
                "industry": company["industry"],
                "products_business": company["products_business"],
                "hiring_roles": research.get("evaluated_job"),
            },
            "criteria": criteria, "ranking_status": "평가 보류",
            "ranking_reason": "동일 범위의 비급여 4개 점수와 선정 자격 검토가 필요함",
            "included_in_salary_statistics": research.get("included_in_salary_statistics", False),
            "duplicate_of": research.get("duplicate_of"),
        })
    return {"schema_version": "1.0", "dataset_type": "real", "model_id": config["model_id"],
            "inactive_criteria": [item["id"] for item in config["criteria"] if not item["active"]],
            "company_count": len(results), "companies": results}


def import_project_data(companies_path, survey_path, pdf_path, *, output_dir, raw_dir):
    config = yaml.safe_load((ROOT / "configs/document_model.yaml").read_text(encoding="utf-8-sig"))
    output_dir, raw_dir = Path(output_dir), Path(raw_dir)
    if output_dir.resolve() == raw_dir.resolve():
        raise ValueError("원본과 처리 결과 폴더는 달라야 합니다.")
    for name in OUTPUT_NAMES:
        if (output_dir / name).exists():
            raise FileExistsError(f"기존 결과를 보존합니다. 다른 --output-dir를 지정하세요: {output_dir / name}")
    catalog, encoding = load_catalog(companies_path)
    if len(catalog) != config["population"]["original_rows"]:
        raise ValueError("기업 CSV 행 수가 문서 모형의 모집단과 다릅니다.")
    records = extract_records(Path(pdf_path))
    validation = validate_and_enrich(records, config)
    companies = join_companies(catalog, records)
    weights = load_survey_weights(survey_path, config)
    sources = {"companies": Path(companies_path), "survey": Path(survey_path), "pdf": Path(pdf_path)}
    manifest = {}
    for key, path in sources.items():
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        destination = raw_dir / RAW_NAMES[key]
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"기존 원본과 내용이 다릅니다: {destination}")
        manifest[key] = {"original_name": unicodedata.normalize("NFC", path.name),
                         "stored_name": RAW_NAMES[key], "sha256": digest, "bytes": len(raw)}
    inputs = analysis_inputs(companies, config)
    readiness = {
        "schema_version": "1.0", "dataset_type": "real", "model_id": config["model_id"],
        "company_count": len(catalog), "region_counts": dict(Counter(item["region"] for item in catalog)),
        **validation,
        "survey_response_count": weights["response_count"],
        "survey_counts_match_document_reference": weights["counts_match_document_reference"],
        "active_criteria": [item["criterion_id"] for item in weights["criteria"] if item["active"]],
        "inactive_criteria": [item["criterion_id"] for item in weights["criteria"] if not item["active"]],
        "common_complete_comparison_count": 0, "ready_for_ranking": False,
        "missing_inputs": [
            "동일 직무·사업장 범위의 워라밸·인프라·직무 유망성·AI 안전성 원점수와 출처",
            "실제 지원 사업장 및 비교 대상 직무의 확정", "중견기업 선정 자격 검토",
        ],
        "final_top10": None,
        "regional_selection_requirement": {"부산": 1, "울산": 1, "경남": 1},
        "note": "연봉지수·설문 가중치만 계산함. PDF 내부 검산은 외부 사실 검증을 뜻하지 않음.",
    }
    payloads = {
        "company_catalog.json": {"schema_version": "1.0", "dataset_type": "real",
                                 "source_encoding": encoding, "companies": companies},
        "document_companies.json": {"schema_version": "1.0", "dataset_type": "real",
                                    "model_id": config["model_id"], "validation": validation,
                                    "source_sha256": manifest["pdf"]["sha256"], "companies": records},
        "document_survey_weights.json": weights,
        "document_analysis_inputs.json": inputs,
        "data_readiness.json": readiness,
        "source_manifest.json": {"schema_version": "1.0", "sources": manifest},
    }
    # 모든 입력 검사를 마친 뒤 원본 바이트를 보존하고 결과를 저장합니다.
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, path in sources.items():
        destination = raw_dir / RAW_NAMES[key]
        if not destination.exists():
            with destination.open("xb") as file:
                file.write(path.read_bytes())
    for name, payload in payloads.items():
        with (output_dir / name).open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    return readiness, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", type=Path, default=ROOT / "data/raw/company_catalog.csv")
    parser.add_argument("--survey", type=Path, default=ROOT / "data/raw/survey_responses.csv")
    parser.add_argument("--pdf", type=Path, default=ROOT / "data/raw" / RAW_NAMES["pdf"])
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    try:
        readiness, weights = import_project_data(
            args.companies, args.survey, args.pdf, output_dir=args.output_dir, raw_dir=args.raw_dir,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"기업 목록: {readiness['company_count']}개 / 계산 단위: {readiness['calculation_units']}개")
    print(f"설문: {weights['response_count']}명 / 중요 {weights['important_total']}건 / 비중요 {weights['unimportant_total']}건")
    print(f"활성 5개 가중치 합: {weights['active_weight_sum']:.12f}")
    print("종합순위: 비급여 점수·평가 범위·선정 자격 보완 전 보류")
    print(f"저장: {args.output_dir}")


if __name__ == "__main__":
    main()
