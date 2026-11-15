"""실제 기업과 분리한 5개 항목 계산 기능 검사 입력입니다."""

import argparse
import json
from pathlib import Path


def regional_case():
    companies = []
    for i in range(1, 16):
        region = "부산" if i <= 8 else "울산" if i <= 11 else "경남"
        salary = (14000 if i <= 8 else 9000 if i <= 11 else 7000) - 100 * i
        companies.append({"company_id": f"synthetic:{i:03d}", "company_name": f"가상 기능검사 기업 {i}",
            "region": region, "target_job": "가상 공통 직무", "comparison_scope_id": "가상동일범위",
            "values": {"salary": salary if i < 15 else None,
                       "work_life_balance": 3 if i < 15 else None,
                       "infrastructure": 3 if i < 15 else None,
                       "job_outlook": 3 if i < 15 else None,
                       "ai_resilience": 3 if i < 15 else None},
            "evidence": {}, "deferral_reasons": []})
    return {"dataset_type": "synthetic", "purpose": "가상 자료의 기능 검사 전용; 실제 기업 근거로 사용 금지",
        "comparison_cohort": {"scope_id": "가상동일범위", "target_job": "가상 공통 직무",
                              "scope_definition": "같은 가상 직무·고용 조건의 기능 검사 범위",
                              "company_ids": [c["company_id"] for c in companies[:14]]},
        "companies": companies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("기존 가상 입력을 보존합니다. 새로운 출력 파일명을 지정하세요.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(regional_case(), stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("가상 기능 검사 입력을 생성했습니다. 실제 기업 평가에 사용하지 않습니다.")


if __name__ == "__main__":
    main()
