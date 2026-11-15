"""연구팀 선별 50개 명단과 직군 기준을 원문 그대로 읽어 대조 자료를 만듭니다."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from openpyxl import load_workbook

if __package__:
    from .data_policy import ROOT, load_user_records
    from .prepare_project_data import load_catalog
else:
    from data_policy import ROOT, load_user_records
    from prepare_project_data import load_catalog


COMPANY_SHEET = "부울경 중견기업 50선"
GUIDELINE_SHEET = "직군별 AI 대체 가능성 기준"
HEADERS = ("순번", "기업명", "지역", "본사/주요 소재지", "주요 산업", "주요 제품·사업", "평균연봉(만원)",
           "연봉 출처", "주요 채용 직군", "AI 대체 가능성", "직군별 AI 대체 가능성 상세", "채용사이트 검수 결과")
GUIDE_HEADERS = ("직군", "주요 업무", "AI 대체 가능성", "근거 및 설명")


def name_key(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text or ""))


def regional_name(text):
    return re.sub(r"\((?:부산|울산|경남)\)$", "", name_key(text))


def match_catalog(name, region, catalog):
    exact = [c for c in catalog if name_key(c["company_name"]) == name_key(name) and c["region"] == region]
    if len(exact) == 1:
        return "이름·지역 일치", exact[0]
    geographic = [c for c in catalog if regional_name(c["company_name"]) == regional_name(name) and c["region"] == region]
    if len(geographic) == 1:
        return "지역 괄호 표기 차이", geographic[0]
    return "원본 대응 미확정", None


def salary_value(raw):
    if raw is None or isinstance(raw, str) and (not raw.strip() or "미확인" in raw):
        return None
    if isinstance(raw, bool):
        raise ValueError("연봉에 논리값을 사용할 수 없습니다.")
    try:
        value = float(raw.replace(",", "").strip() if isinstance(raw, str) else raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("새 엑셀의 연봉을 숫자 또는 미확인 표기로 읽을 수 없습니다.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("새 엑셀 연봉은 유한한 양수여야 합니다.")
    return value


def parse_workbook(path, catalog, pdf_records):
    formulas = load_workbook(path, data_only=False, read_only=True)
    cached = load_workbook(path, data_only=True, read_only=True)
    try:
        if COMPANY_SHEET not in formulas or GUIDELINE_SHEET not in formulas:
            raise ValueError("선별 명단 또는 직군별 기준 시트가 없습니다.")
        sheets = {}
        for title, expected in ((COMPANY_SHEET, HEADERS), (GUIDELINE_SHEET, GUIDE_HEADERS)):
            source, values = formulas[title], cached[title]
            if tuple(c.value for c in next(source.iter_rows(min_row=1, max_row=1))) != expected:
                raise ValueError(f"제공 엑셀의 열 구성이 다릅니다: {title}")
            result = []
            for row in source.iter_rows(min_row=2):
                if not any(cell.value is not None for cell in row):
                    continue
                if any(cell.data_type == "f" and values[cell.coordinate].value is None for cell in row):
                    raise ValueError("계산 결과가 저장되지 않은 수식이 있습니다.")
                result.append((row[0].row, {cell.column_letter: values[cell.coordinate].value if cell.data_type == "f" else cell.value for cell in row}))
            sheets[title] = result
        if len(sheets[COMPANY_SHEET]) != 50:
            raise ValueError("사용자가 확정한 선별 50개와 행 수가 다릅니다.")
        companies, seen_names, seen_orders = [], set(), set()
        for source_row, raw in sheets[COMPANY_SHEET]:
            order, name, region = raw["A"], raw["B"], raw["C"]
            if not isinstance(order, int) or isinstance(order, bool) or order not in range(1, 51) or order in seen_orders:
                raise ValueError("선별 순번이 1~50의 고유 정수가 아닙니다.")
            if not isinstance(name, str) or not name.strip() or region not in {"부산", "울산", "경남"}:
                raise ValueError("선별 기업의 기업명·지역을 확인하세요.")
            key = (name_key(name), region)
            if key in seen_names:
                raise ValueError("선별 명단에 같은 기업명·지역이 중복됐습니다.")
            seen_orders.add(order)
            seen_names.add(key)
            status, parent = match_catalog(name, region, catalog)
            reference = pdf_records.get(parent["company_id"]) if parent else None
            salary = salary_value(raw["G"])
            pdf_salary = reference["annual_salary_manwon"] if reference else None
            companies.append({"company_id": f"selected50:{order:03d}", "selection_order": order,
                "company_name": name, "region": region, "location": raw["D"], "industry": raw["E"],
                "products_business": raw["F"], "annual_salary_manwon": salary, "salary_raw_value": raw["G"],
                "salary_source_text": raw["H"], "main_hiring_roles": raw["I"], "ai_exposure_label": raw["J"],
                "ai_exposure_explanation": raw["K"], "review_result": raw["L"],
                "source_sheet": COMPANY_SHEET, "source_row": source_row, "raw_fields": raw,
                "catalog_link": {"status": status, "company_id": parent["company_id"] if parent else None,
                    "original_id": parent["original_id"] if parent else None,
                    "company_name": parent["company_name"] if parent else None},
                "pdf_reference": reference,
                "salary_comparison_status": "새 엑셀 미확인" if salary is None else "대응 PDF 없음" if pdf_salary is None
                    else "자료 간 수치 차이" if salary != pdf_salary else "수치 일치",
                "salary_difference_manwon": salary - pdf_salary if salary is not None and pdf_salary is not None else None})
        guidelines = [{"guideline_id": f"job:{i:02d}", "job_group": raw["A"], "tasks": raw["B"],
                       "exposure_label": raw["C"], "explanation": raw["D"], "source_sheet": GUIDELINE_SHEET,
                       "source_row": row, "raw_fields": raw} for i, (row, raw) in enumerate(sheets[GUIDELINE_SHEET], 1)]
        return {"schema_version": "1.0", "dataset_type": "real", "selection_policy": {
            "original_population_count": 150, "selected_count": 50, "final_selection_target": 10,
            "selected_by": "연구팀 기준", "relationship": "사용자 확정: 원본 150개에서 선별한 50개",
            "salary_precedence": "새 엑셀 우선; PDF 별도 병기; 새 엑셀 미확인은 미상 유지",
            "unresolved_mapping": "대응자료 없음; 선별 명단 유지; 불확실한 기존 기업 자료를 합치지 않음"},
            "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "workbook_created": str(formulas.properties.created), "workbook_modified": str(formulas.properties.modified),
            "summary": {"company_count": len(companies), "region_counts": dict(Counter(c["region"] for c in companies)),
                "mapping_counts": dict(Counter(c["catalog_link"]["status"] for c in companies)),
                "salary_available_count": sum(c["annual_salary_manwon"] is not None for c in companies),
                "salary_difference_count": sum(c["salary_comparison_status"] == "자료 간 수치 차이" for c in companies),
                "job_guideline_count": len(guidelines)}, "companies": companies, "job_guidelines": guidelines}
    finally:
        formulas.close()
        cached.close()


def import_selected(root, output):
    root = Path(root)
    policy, records = load_user_records(root)
    source = policy["raw_sources"]["selected50"]
    catalog, _ = load_catalog(root / "data/raw/company_catalog.csv")
    data = parse_workbook(root / "data/raw" / source["stored_name"], catalog, records)
    if data["source_sha256"] != source["sha256"]:
        raise ValueError("새 엑셀의 원본 해시가 다릅니다.")
    output = Path(output)
    if output.exists():
        raise FileExistsError("기존 처리 결과를 보존합니다. 새로운 출력 파일을 지정하세요.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/selected50_companies.json")
    args = parser.parse_args()
    try:
        data = import_selected(ROOT, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"선별 기업 {len(data['companies'])}개와 직군 기준 {len(data['job_guidelines'])}개를 연결했습니다.")


if __name__ == "__main__":
    main()
