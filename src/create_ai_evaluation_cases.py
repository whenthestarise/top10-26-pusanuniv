import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"

context = {
    "company_id": "TEST-COMPARE-001",
    "company_name": "가상기업 B",
    "workplace": "가상 정비사업장",
    "target_job": "현장 설비 정비",
}

source = {
    "source_id": "SYNTHETIC-JOB-001",
    "title": "가상기업 B 현장 설비 정비 업무 조사",
    "locator": "시험용 가상 자료 1쪽",
    "retrieved_at": "2026-11-09",
    "text": (
        "이 자료는 프로그램 시험을 위해 작성한 가상 자료다. "
        "가상기업 B의 가상 정비사업장에서 현장 설비 정비 직무를 "
        "수행하는 직원의 주요 업무를 조사했다. "
        "업무시간의 90%는 현장에서 설비를 분해하고, "
        "고장 부품을 교체하고, 다시 조립하고, "
        "시험 운전과 안전 상태 확인을 수행하는 작업이다. "
        "이 작업은 직원이 현장의 실물 설비를 직접 다뤄야 한다. "
        "나머지 10%는 점검표와 정비 결과 문서를 작성하는 업무다. "
        "문서 초안과 기록 정리는 생성형 AI의 도움을 받을 수 있지만, "
        "현장 작업 결과 확인과 설비 재가동 판단은 담당자가 수행한다. "
        "위 비율은 업무 구성에 관한 가상 관찰값이며 "
        "AI 대체 확률을 뜻하지 않는다. "
        "이 자료는 주당 근로시간, 야간근무, 휴가 사용, "
        "주변 교통과 주거, 향후 채용 수요에 관한 정보를 포함하지 않는다."
    ),
}

cases = {
    "ai_case_no_evidence.json": {
        "dataset_type": "synthetic",
        "context": context,
        "sources": [],
    },
    "ai_case_job_evidence.json": {
        "dataset_type": "synthetic",
        "context": context,
        "sources": [source],
    },
}

# 기존 시험 자료를 덮어쓰지 않습니다.
for filename in cases:
    path = INPUT_DIR / filename
    if path.exists():
        raise FileExistsError(f"이미 파일이 있습니다: {path}")

INPUT_DIR.mkdir(parents=True, exist_ok=True)

for filename, payload in cases.items():
    path = INPUT_DIR / filename

    with path.open("x", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"저장: {path}")