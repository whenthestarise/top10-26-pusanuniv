from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "부울경_중견기업50.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

COMPANY_SHEET = "부울경 중견기업 50선"
AI_REFERENCE_SHEET = "직군별 AI 대체 가능성 기준"


COLUMN_MAP = {
    "순번": "company_id",
    "기업명": "company_name",
    "지역": "region",
    "본사/주요 소재지": "location",
    "주요 산업": "industry",
    "주요 제품·사업": "products_business",
    "평균연봉(만원)": "average_salary_10k_krw",
    "연봉 출처": "salary_source",
    "주요 채용 직군": "hiring_roles",
    "AI 대체 가능성": "ai_replacement_risk_label",
    "직군별 AI 대체 가능성 상세": "ai_risk_details",
    "채용사이트 검수 결과": "job_site_validation",
}


AI_REPLACEMENT_RISK_SCORE = {
    "낮음": 0.0,
    "낮음~중간": 25.0,
    "중간": 50.0,
    "중간~높음": 75.0,
    "높음": 100.0,
}


CRITERIA_SCORE_COLUMNS = [
    "salary_score",
    "infrastructure_score",
    "work_life_balance_score",
    "ai_resilience_score",
    "job_outlook_score",
    "horizontal_culture_score",
    "career_capital_score",
]


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def clean_text(value):
    if pd.isna(value):
        return pd.NA

    return " ".join(str(value).strip().split())


def classify_salary_quality(row: pd.Series) -> str:
    salary = row["average_salary_10k_krw"]
    source = str(row["salary_source"])

    if pd.isna(salary):
        return "missing"

    if "편차 큼" in source:
        return "high_variance"

    if "추정" in source:
        return "estimated"

    return "reported"


def validate_required_columns(raw_df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in COLUMN_MAP
        if column not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "엑셀에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )


def load_and_clean_companies() -> pd.DataFrame:
    raw_df = pd.read_excel(
        INPUT_FILE,
        sheet_name=COMPANY_SHEET,
        engine="openpyxl",
    )

    validate_required_columns(raw_df)

    df = raw_df.rename(columns=COLUMN_MAP).copy()
    df = df[list(COLUMN_MAP.values())]

    text_columns = [
        column
        for column in df.columns
        if column not in {"company_id", "average_salary_10k_krw"}
    ]

    for column in text_columns:
        df[column] = df[column].map(clean_text)

    df["company_id"] = pd.to_numeric(
        df["company_id"],
        errors="coerce",
    ).astype("Int64")

    # 연봉 미확인 등의 텍스트는 결측값으로 변환합니다.
    df["average_salary_10k_krw"] = pd.to_numeric(
        df["average_salary_10k_krw"],
        errors="coerce",
    ).astype("Float64")

    df["salary_quality_flag"] = df.apply(
        classify_salary_quality,
        axis=1,
    )

    # 유효한 연봉끼리 백분위 점수로 변환합니다.
    df["salary_score"] = (
        df["average_salary_10k_krw"]
        .rank(method="average", pct=True)
        .mul(100)
        .round(2)
        .astype("Float64")
    )

    df["ai_replacement_risk_score"] = (
        df["ai_replacement_risk_label"]
        .map(AI_REPLACEMENT_RISK_SCORE)
        .astype("Float64")
    )

    # AI 대체 가능성이 낮을수록 좋은 점수가 되도록 방향을 반전합니다.
    df["ai_resilience_score"] = (
        100 - df["ai_replacement_risk_score"]
    ).astype("Float64")

    # 아직 수집하지 않은 평가 항목입니다.
    missing_score_columns = [
        "infrastructure_score",
        "work_life_balance_score",
        "job_outlook_score",
        "horizontal_culture_score",
        "career_capital_score",
    ]

    for column in missing_score_columns:
        df[column] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="Float64",
        )

    df["criteria_available_count"] = (
        df[CRITERIA_SCORE_COLUMNS]
        .notna()
        .sum(axis=1)
    )

    df["model_ready"] = (
        df["criteria_available_count"]
        == len(CRITERIA_SCORE_COLUMNS)
    )

    return df


def load_ai_reference() -> pd.DataFrame:
    reference_df = pd.read_excel(
        INPUT_FILE,
        sheet_name=AI_REFERENCE_SHEET,
        engine="openpyxl",
    )

    for column in reference_df.columns:
        reference_df[column] = reference_df[column].map(clean_text)

    return reference_df


def create_quality_report(df: pd.DataFrame) -> dict:
    known_ai_labels = set(AI_REPLACEMENT_RISK_SCORE)
    actual_ai_labels = set(
        df["ai_replacement_risk_label"]
        .dropna()
        .astype(str)
    )

    unknown_ai_labels = sorted(
        actual_ai_labels - known_ai_labels
    )

    duplicate_companies = (
        df.loc[
            df["company_name"].duplicated(keep=False),
            "company_name",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    allowed_regions = {"부산", "울산", "경남"}
    invalid_regions = sorted(
        set(df["region"].dropna().astype(str))
        - allowed_regions
    )

    salary = df["average_salary_10k_krw"]

    warnings = []

    if salary.isna().any():
        warnings.append(
            f"평균연봉이 없는 기업이 {int(salary.isna().sum())}개 있습니다."
        )

    high_variance_count = int(
        (df["salary_quality_flag"] == "high_variance").sum()
    )

    if high_variance_count:
        warnings.append(
            f"출처별 연봉 편차가 큰 기업이 {high_variance_count}개 있습니다."
        )

    if unknown_ai_labels:
        warnings.append(
            "정의되지 않은 AI 대체 가능성 값이 있습니다: "
            + ", ".join(unknown_ai_labels)
        )

    if duplicate_companies:
        warnings.append(
            "중복 기업명이 있습니다: "
            + ", ".join(duplicate_companies)
        )

    return {
        "schema_version": "1.0",
        "source_file": INPUT_FILE.name,
        "source_sha256": calculate_sha256(INPUT_FILE),
        "record_count": int(len(df)),
        "region_counts": {
            str(key): int(value)
            for key, value in df["region"].value_counts().items()
        },
        "salary": {
            "valid_count": int(salary.notna().sum()),
            "missing_count": int(salary.isna().sum()),
            "minimum": (
                float(salary.min())
                if salary.notna().any()
                else None
            ),
            "median": (
                float(salary.median())
                if salary.notna().any()
                else None
            ),
            "maximum": (
                float(salary.max())
                if salary.notna().any()
                else None
            ),
            "high_variance_count": high_variance_count,
        },
        "duplicate_company_names": duplicate_companies,
        "unknown_ai_labels": unknown_ai_labels,
        "invalid_regions": invalid_regions,
        "criteria": {
            "total_count": len(CRITERIA_SCORE_COLUMNS),
            "currently_available": [
                "salary_score",
                "ai_resilience_score",
            ],
            "not_yet_collected": [
                "infrastructure_score",
                "work_life_balance_score",
                "job_outlook_score",
                "horizontal_culture_score",
                "career_capital_score",
            ],
            "model_ready_company_count": int(df["model_ready"].sum()),
        },
        "warnings": warnings,
    }


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"원본 파일을 찾을 수 없습니다: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    companies_df = load_and_clean_companies()
    ai_reference_df = load_ai_reference()
    quality_report = create_quality_report(companies_df)

    companies_output = OUTPUT_DIR / "companies_clean.csv"
    ai_reference_output = OUTPUT_DIR / "ai_job_risk_reference.csv"
    report_output = REPORT_DIR / "data_quality_report.json"

    companies_df.to_csv(
        companies_output,
        index=False,
        encoding="utf-8-sig",
    )

    ai_reference_df.to_csv(
        ai_reference_output,
        index=False,
        encoding="utf-8-sig",
    )

    report_output.write_text(
        json.dumps(
            quality_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("1단계 전처리가 완료되었습니다.")
    print(f"기업 데이터: {companies_output}")
    print(f"AI 기준표: {ai_reference_output}")
    print(f"품질 보고서: {report_output}")
    print()
    print(json.dumps(
        quality_report,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()