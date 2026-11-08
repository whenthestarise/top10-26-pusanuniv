from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import yaml
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "document_model.yaml"
DEFAULT_OUTPUT = (
    ROOT / "data" / "processed" / "document_companies.json"
)

HEADER_PATTERN = re.compile(
    r"^기업 원본순번\s+(\d+)\s*\|",
    re.MULTILINE,
)

FIELD_PATTERN = re.compile(
    r"^([A-Z]{1,2})\.\s*[^:\n]+:\s*",
    re.MULTILINE,
)

EXPECTED_FIELDS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {"AA"}


def parse_number(text, label):
    try:
        value = float(text.replace(",", "").strip())
    except ValueError as exc:
        raise ValueError(
            f"{label}: 숫자로 변환할 수 없습니다: {text!r}"
        ) from exc

    if not math.isfinite(value):
        raise ValueError(f"{label}: 유한한 숫자가 아닙니다.")

    return value


def extract_records(pdf_path):
    records = []
    seen_ids = set()

    reader = PdfReader(str(pdf_path))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        headers = list(HEADER_PATTERN.finditer(text))

        # 기업 비교표의 개별 기업 페이지에만 적용합니다.
        if not headers:
            continue

        if len(headers) != 1:
            raise ValueError(
                f"{page_number}쪽: 기업 페이지 구조가 예상과 다릅니다."
            )

        original_id = int(headers[0].group(1))

        # 마지막 AA 항목에 페이지 꼬리말이 섞이지 않도록 제거합니다.
        text = re.split(
            r"\n직업만족도 AI 분석 자료집 \|",
            text,
            maxsplit=1,
        )[0]

        matches = list(FIELD_PATTERN.finditer(text))
        field_ids = [match.group(1) for match in matches]

        if (
            len(field_ids) != len(EXPECTED_FIELDS)
            or set(field_ids) != EXPECTED_FIELDS
        ):
            raise ValueError(
                f"{page_number}쪽: A~AA의 27개 항목을 "
                "정확하게 읽지 못했습니다."
            )

        fields = {}

        for index, match in enumerate(matches):
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )

            fields[match.group(1)] = text[
                match.end():end
            ].strip()

        if int(fields["A"]) != original_id:
            raise ValueError(
                f"{page_number}쪽: 제목과 A열 순번이 다릅니다."
            )

        if original_id in seen_ids:
            raise ValueError(f"원본 순번 중복: {original_id}")

        seen_ids.add(original_id)

        salary = parse_number(
            fields["G"], f"{original_id}번 연봉"
        )
        original_amount = parse_number(
            fields["O"], f"{original_id}번 급여 원자료"
        )
        conversion_factor = parse_number(
            fields["P"], f"{original_id}번 환산계수"
        )

        if salary <= 0 or conversion_factor <= 0:
            raise ValueError(
                f"{original_id}번: 연봉과 환산계수는 양수여야 합니다."
            )

        # G열을 다시 환산하지 않습니다.
        # O×P는 G열의 단위 변환이 맞는지 확인하는 데만 사용합니다.
        if not math.isclose(
            salary,
            original_amount * conversion_factor,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{original_id}번: G열 연봉과 O열×P열이 다릅니다."
            )

        records.append({
            "company_id": f"document:{original_id:03d}",
            "original_id": original_id,
            "company_name": fields["B"],
            "region": fields["C"],
            "location": fields["D"],
            "industry": fields["E"],
            "products_business": fields["F"],
            "annual_salary_manwon": salary,
            "salary_data_type": fields["H"],
            "evaluated_job": fields["I"],
            "ai_exposure_label": fields["J"],
            "ai_exposure_explanation": fields["K"],
            "legal_entity_name": fields["M"],
            "salary_reference_period": fields["N"],
            "salary_source_url": fields["Q"],
            "company_source_url": fields["R"],
            "eligibility_label_from_document": fields["S"],
            "comparison_notes": fields["T"],
            "research_status_from_document": fields["U"],
            "reference_date": fields["W"],
            "job_source_url": fields["Y"],
            "ai_scope_limitations": fields["AA"],
            "source_page": page_number,
            "raw_fields": fields,
        })

    return sorted(records, key=lambda row: row["original_id"])


def validate_and_enrich(records, config):
    population = config["population"]
    baseline = float(config["salary"]["baseline"])

    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("기준연봉은 유한한 양수여야 합니다.")

    if len(records) != population["salary_available_rows"]:
        raise ValueError(
            f"연봉 보유 행 수 불일치: {len(records)}"
        )

    expected_ids = set(
        range(1, population["original_rows"] + 1)
    )
    actual_ids = {row["original_id"] for row in records}

    if not actual_ids <= expected_ids:
        raise ValueError("원본 순번 범위를 벗어난 기업이 있습니다.")

    missing_ids = sorted(expected_ids - actual_ids)

    if len(missing_ids) != population["salary_missing_rows"]:
        raise ValueError("연봉 비교표에 없는 원본 순번 수가 다릅니다.")

    by_id = {row["original_id"]: row for row in records}

    # 문서에서 명시한 중복 관계를 확인합니다.
    # 다른 문서에서 같은 순번을 잘못 제외하지 않도록 이름도 검사합니다.
    if (
        by_id.get(103, {}).get("company_name") != "CTR"
        or by_id.get(108, {}).get("company_name") != "센트랄"
    ):
        raise ValueError("문서의 CTR·센트랄 중복 관계를 확인할 수 없습니다.")

    if not math.isclose(
        by_id[103]["annual_salary_manwon"],
        by_id[108]["annual_salary_manwon"],
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise ValueError("CTR·센트랄의 연봉 값이 서로 다릅니다.")

    for row in records:
        excluded = row["original_id"] == 108

        row["included_in_salary_statistics"] = not excluded
        row["duplicate_of"] = "document:103" if excluded else None
        row["exclusion_reason"] = (
            "문서 지정 중복: 원본 103번 CTR과 동일 기업"
            if excluded
            else None
        )

        # 연봉지수는 참고용으로 보존하지만 중복 행은 통계에서 제외합니다.
        row["salary_index"] = (
            row["annual_salary_manwon"] / baseline
        )

        # 중견기업 자격은 문서 표시만 보존하며 자동 확정하지 않습니다.
        row["selection_eligibility_review"] = "추가 검토 필요"

    included = [
        row for row in records
        if row["included_in_salary_statistics"]
    ]

    excluded_count = len(records) - len(included)

    if excluded_count != population["duplicate_rows_excluded"]:
        raise ValueError("중복 제외 행 수가 설정과 다릅니다.")

    if len(included) != population["expected_calculation_units"]:
        raise ValueError("계산 단위 수가 설정과 다릅니다.")

    salaries = [
        row["annual_salary_manwon"] for row in included
    ]
    calculated_median = statistics.median(salaries)

    if not math.isclose(
        calculated_median,
        baseline,
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"중앙값 불일치: 계산값 {calculated_median}, "
            f"문서 기준값 {baseline}. 원자료 확인이 필요합니다."
        )

    return {
        "salary_available_rows": len(records),
        "duplicate_rows_excluded": excluded_count,
        "calculation_units": len(included),
        "ids_absent_from_salary_table": missing_ids,
        "baseline_manwon": baseline,
        "calculated_median_manwon": calculated_median,
        "calculated_mean_manwon": statistics.mean(salaries),
        "salary_index_median": statistics.median(
            row["salary_index"] for row in included
        ),
        "unit_conversion_check": "통과",
        "baseline_check": "통과",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )

    args = parser.parse_args()

    pdf_path = (
        args.pdf if args.pdf.is_absolute()
        else ROOT / args.pdf
    )
    output_path = (
        args.output if args.output.is_absolute()
        else ROOT / args.output
    )

    if not pdf_path.is_file():
        parser.error(f"PDF 파일이 없습니다: {pdf_path}")

    if output_path.exists():
        parser.error(
            f"기존 결과 파일이 있습니다: {output_path}\n"
            "--output으로 다른 파일명을 지정하세요."
        )

    try:
        with CONFIG_PATH.open(encoding="utf-8-sig") as file:
            config = yaml.safe_load(file)

        records = extract_records(pdf_path)
        validation = validate_and_enrich(records, config)

    except ValueError as exc:
        parser.error(str(exc))

    payload = {
        "schema_version": "1.0",
        "dataset_type": "real",
        "model_id": config["model_id"],
        "source_file": str(pdf_path),
        "source_sha256": hashlib.sha256(
            pdf_path.read_bytes()
        ).hexdigest(),
        "validation": validation,
        "note": (
            "문서 내부의 수치·구조 검산 결과입니다. "
            "외부 출처 검증, 선정 자격 확정, 종합순위 산출은 "
            "수행하지 않았습니다."
        ),
        "companies": records,
    }

    # 모든 검산을 통과한 뒤에만 결과를 저장합니다.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"연봉 보유 행: {validation['salary_available_rows']}")
    print(f"중복 제외 행: {validation['duplicate_rows_excluded']}")
    print(f"계산 단위: {validation['calculation_units']}")
    print(
        f"계산된 중앙값: "
        f"{validation['calculated_median_manwon']:,.4f}만원"
    )
    print(
        f"계산된 산술평균: "
        f"{validation['calculated_mean_manwon']:,.8f}만원"
    )
    print(
        f"연봉지수 중앙값: "
        f"{validation['salary_index_median']:.8f}"
    )
    print(f"단위 환산 검사: {validation['unit_conversion_check']}")
    print(f"문서 기준연봉 검사: {validation['baseline_check']}")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()