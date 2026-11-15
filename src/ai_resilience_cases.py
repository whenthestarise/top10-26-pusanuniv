"""단일 항목 평가의 고정 가상 사례입니다. 기대 답안은 모델에 보내지 않습니다."""

from copy import deepcopy

if __package__:
    from .ai_evidence_cases import CONTEXT, source
else:
    from ai_evidence_cases import CONTEXT, source


def build_cases():
    definitions = [
        ("physical", "현장 수행 중심", [source("SYNTHETIC-PHYSICAL",
            "이 직무의 핵심 업무 대부분은 현장에서 파손된 배관을 직접 분해하고 수리하는 작업이다.")], 5, "score_proposed"),
        ("digital", "반복 디지털 처리 중심", [source("SYNTHETIC-DIGITAL",
            "이 직무의 핵심 업무 대부분은 정해진 양식의 전산 자료를 반복 입력하고 정형 문서의 문장을 복사하는 디지털 처리다.")], 1, "score_proposed"),
        ("mixed", "디지털 처리와 현장 수행 혼합", [source("SYNTHETIC-MIXED",
            "직원의 업무는 생성형 인공지능이 자동 작성하는 정형 보고서 처리와 현장 설비의 실물 수리가 함께 구성한다.")], 3, "score_proposed"),
        ("irrelevant", "급여 자료만 있음", [source("SYNTHETIC-SALARY",
            "직원의 평균 연봉은 오천만원이다.")], None, "insufficient_evidence"),
        ("conflicting", "동일 기간의 핵심 업무 충돌", [
            source("SYNTHETIC-CONFLICT-A", "같은 조사 기간 이 직무의 핵심 업무 전부는 현장에서 설비를 직접 분해하고 수리하는 작업이다."),
            source("SYNTHETIC-CONFLICT-B", "같은 조사 기간 이 직무에는 현장 작업이 전혀 없고 핵심 업무 전부는 전산 자료를 반복 입력하는 사무 처리다."),
        ], None, "conflicting_evidence"),
        ("other_job", "다른 직무 자료", [source("SYNTHETIC-OTHER",
            "이 직무의 핵심 업무 대부분은 정해진 전산 자료의 반복 입력이다.",
            {**CONTEXT, "target_job": "다른 직무"})], None, "insufficient_evidence"),
        ("empty", "자료 없음", [], None, "insufficient_evidence"),
    ]
    return [{"case_id": key, "name": name, "expected_score": score, "expected_status": status,
             "bundle": {"dataset_type": "synthetic", "context": deepcopy(CONTEXT), "sources": sources}}
            for key, name, sources, score, status in definitions]
