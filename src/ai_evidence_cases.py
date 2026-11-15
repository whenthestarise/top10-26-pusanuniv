"""근거 선택기의 고정 가상 시험 자료입니다. 모델 입력에 기대 답안을 넣지 않습니다."""

from copy import deepcopy


CONTEXT = {"company_id": "SYNTHETIC-EVIDENCE-001", "company_name": "가상기업",
           "workplace": "가상 정비사업장", "target_job": "현장 설비 정비"}


def source(source_id, text, scope=None):
    return {"source_id": source_id, "title": "가상 직무 조사", "locator": "가상 시험 문서",
            "retrieved_at": "2026-11-14", "scope": deepcopy(scope or CONTEXT), "text": text}


def build_cases():
    other_scope = {**CONTEXT, "target_job": "회계 사무"}
    definitions = [
        ("relevant", "정비 업무 근거", [source("SYNTHETIC-MAINTENANCE", (
            "현장 설비 정비 직원은 설비를 분해하고 고장 부품을 교체한다. "
            "점검 기록의 초안 작성에는 생성형 인공지능을 사용한다. "
            "설비 재가동과 안전 상태의 최종 판단은 담당 정비사가 내린다. "
            "평균연봉은 오천이백만원이다."
        ))], ["문장0001", "문장0002", "문장0003"], False),
        ("empty", "제공 자료 없음", [], [], True),
        ("other_job", "다른 직무의 자료", [source("SYNTHETIC-ACCOUNTING", (
            "회계 사무 직원은 전표를 입력하고 정형 보고서를 작성한다. "
            "생성형 인공지능이 정형 보고서의 초안을 작성한다."
        ), other_scope)], [], True),
        ("conflicting", "핵심 업무 설명 충돌", [
            source("SYNTHETIC-SURVEY-A", "같은 조사 기간 정비 직원의 핵심 업무는 현장에서 실물 설비를 직접 분해하고 수리하는 작업이다."),
            source("SYNTHETIC-SURVEY-B", "같은 조사 기간 정비 직원의 핵심 업무는 현장 작업 없이 전산 화면에서 점검표를 대조하는 사무 처리다."),
        ], ["문장0001", "문장0002"], False),
        ("irrelevant", "같은 직무의 무관한 자료", [source("SYNTHETIC-BENEFITS", (
            "직원의 평균연봉은 오천이백만원이다. "
            "사업장 앞에서 시내버스를 이용할 수 있다. "
            "직원에게 연차 휴가를 제공한다."
        ))], [], False),
        ("held_out", "다른 표현의 확인 자료", [source("SYNTHETIC-SECONDARY", (
            "정비사는 고객에게 설비 고장 원인을 대면으로 설명하고 현장 안전 확인을 수행한다. "
            "생성형 인공지능은 주간 점검 보고서의 초안 작성을 보조한다. "
            "식당은 사업장 정문 옆에 있다."
        ))], ["문장0001", "문장0002"], False),
    ]
    return [{"case_id": key, "name": name,
             "bundle": {"dataset_type": "synthetic", "context": deepcopy(CONTEXT), "sources": sources},
             "expected_sentence_ids": ids, "expected_model_skipped": skipped}
            for key, name, sources, ids, skipped in definitions]
