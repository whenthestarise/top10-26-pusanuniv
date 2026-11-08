from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "companies_clean.csv"
OUTPUT = ROOT / "data" / "processed" / "company_analysis_inputs.json"


CRITERIA = {
    "salary": "높은 급여",
    "infrastructure": "근무 지역 인프라",
    "work_life_balance": "워라밸",
    "ai_resilience": "AI 대체 안전성",
    "job_outlook": "직무 유망성",
    "horizontal_culture": "수평적 조직문화",
    "career_capital": "경력 확보",
}

REQUIRED_COLUMNS = [
    "company_id",
    "company_name",
    "region",
    "location",
    "industry",
    "products_business",
    "average_salary_10k_krw",
    "salary_source",
    "salary_quality_flag",
    "hiring_roles",
    "ai_replacement_risk_label",
    "ai_risk_details",
    "salary_score",
    "ai_resilience_score",
]


def text_or_none(value):
    if pd.isna(value):
        return None

    result = str(value).strip()
    return result if result else None


def number_or_none(value):
    if pd.isna(value):
        return None

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"유한한 숫자가 아닙니다: {value}")

    return result


def prepare_company(row: dict) -> dict:
    company_id = text_or_none(row["company_id"])
    company_name = text_or_none(row["company_name"])

    if not company_id or not company_name:
        raise ValueError("기업 ID 또는 기업명이 비어 있습니다.")

    # 기본 상태: 근거가 확보되지 않은 항목은 점수를 계산하지 않습니다.
    criteria = {
        key: {
            "name": name,
            "status": "자료 부족",
            "evidence": {},
            "provisional_score": None,
            "verified_score": None,
            "missing_information": [],
        }
        for key, name in CRITERIA.items()
    }

    salary = number_or_none(row["average_salary_10k_krw"])
    salary_score = number_or_none(row["salary_score"])

    if salary is not None:
        if salary <= 0:
            raise ValueError(f"{company_name}: 연봉 값이 0 이하입니다.")

        criteria["salary"].update({
            "status": "검증 필요",
            "evidence": {
                "average_salary_manwon": salary,
                "source_description": text_or_none(row["salary_source"]),
                "quality_flag": text_or_none(row["salary_quality_flag"]),
            },
            "provisional_score": salary_score,
        })

    criteria["salary"]["missing_information"] = [
        "원문 출처와 기준연도 확인",
        "평균연봉의 집계 대상과 기업 간 비교 가능성 확인",
    ]

    if salary is None:
        criteria["salary"]["missing_information"].append(
            "평균연봉 금액 확보"
        )

    ai_label = text_or_none(row["ai_replacement_risk_label"])
    ai_details = text_or_none(row["ai_risk_details"])

    if ai_label is not None or ai_details is not None:
        criteria["ai_resilience"].update({
            "status": "검증 필요",
            "evidence": {
                "existing_risk_label": ai_label,
                "existing_explanation": ai_details,
            },
            "provisional_score": number_or_none(
                row["ai_resilience_score"]
            ),
        })

    criteria["ai_resilience"]["missing_information"] = [
        "평가 대상 직무의 구체적인 업무 내용",
        "기존 AI 위험도 분류의 원문 근거와 평가 방법",
    ]

    criteria["infrastructure"]["missing_information"] = [
        "실제 근무 사업장 주소",
        "교통·주거·생활시설 접근성 자료",
    ]
    criteria["work_life_balance"]["missing_information"] = [
        "실제 근로시간",
        "휴가 사용·근무 유연성 관련 근거",
    ]
    criteria["job_outlook"]["missing_information"] = [
        "대상 직무의 채용 수요와 산업 전망",
        "관련 사업의 성장·투자 근거",
    ]
    criteria["horizontal_culture"]["missing_information"] = [
        "의사소통·업무 자율성·공정성 관련 근거",
    ]
    criteria["career_capital"]["missing_information"] = [
        "직무교육·자격 지원 자료",
        "담당 업무와 프로젝트 경험 기회",
    ]

    for criterion in criteria.values():
        score = criterion["provisional_score"]
        if score is not None and not 0 <= score <= 100:
            raise ValueError(f"{company_name}: 임시 점수 범위 오류")

    return {
        "company_id": company_id,
        "company_name": company_name,
        "source_id": f"company_dataset:{company_id}",
        "source_file": INPUT.name,
        "profile": {
            "region": text_or_none(row["region"]),
            "location": text_or_none(row["location"]),
            "industry": text_or_none(row["industry"]),
            "products_business": text_or_none(row["products_business"]),
            "hiring_roles": text_or_none(row["hiring_roles"]),
        },
        "criteria": criteria,
        "ranking_status": "평가 보류",
        "ranking_reason": "7개 기준의 검증 완료 점수가 아직 확보되지 않음",
    }


def main() -> None:
    df = pd.read_csv(
        INPUT,
        encoding="utf-8-sig",
        dtype={"company_id": "string"},
    )

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing)}")

    if df.empty:
        raise ValueError("기업 데이터가 비어 있습니다.")

    if df["company_id"].duplicated().any():
        raise ValueError("기업 ID가 중복되었습니다.")

    companies = [
        prepare_company(row)
        for row in df.to_dict(orient="records")
    ]

    summary = {}
    for key, name in CRITERIA.items():
        summary[key] = {
            "name": name,
            "needs_verification": sum(
                company["criteria"][key]["status"] == "검증 필요"
                for company in companies
            ),
            "insufficient_data": sum(
                company["criteria"][key]["status"] == "자료 부족"
                for company in companies
            ),
        }

    result = {
        "schema_version": "1.0",
        "description": (
            "기존 기업 데이터의 확보 상태를 구분한 입력 파일. "
            "독립적인 사실 검증이나 LLM 평가 결과는 아님."
        ),
        "company_count": len(companies),
        "summary": summary,
        "companies": companies,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"기업 수: {len(companies)}")
    for item in summary.values():
        print(
            f"{item['name']}: "
            f"검증 필요 {item['needs_verification']}개 / "
            f"자료 부족 {item['insufficient_data']}개"
        )
    print(f"\n저장 완료: {OUTPUT}")


if __name__ == "__main__":
    main()