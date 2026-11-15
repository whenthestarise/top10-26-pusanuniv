"""기존 실패 사례는 유지하고 새 확인 사례는 개발 사례와 나눠 고정합니다."""

from copy import deepcopy

if __package__:
    from .ai_evidence_cases import CONTEXT, source
    from .ai_resilience_cases import build_cases
else:
    from ai_evidence_cases import CONTEXT, source
    from ai_resilience_cases import build_cases


def case(key, name, texts, score, status=None):
    return {"case_id": key, "name": name, "expected_score": score,
            "expected_status": status or ("score_proposed" if score is not None else "insufficient_evidence"),
            "bundle": {"dataset_type": "synthetic", "context": deepcopy(CONTEXT),
                       "sources": [source(f"SYNTHETIC-{key}-{i}", text) for i, text in enumerate(texts)]}}


def development_cases():
    return build_cases() + [
        case("substantial", "상당 부분 반복 사무", ["이 직무의 핵심 업무의 상당 부분은 정형 문서를 분류하고 정해진 답안 문구를 반복 작성하는 작업이다."], 2),
        case("mainly", "주로 현장 수행", ["직원은 주로 현장에서 고장 난 부품을 직접 교체하고 장비를 수리한다."], 4),
        case("unknown_share", "업무는 있지만 비중 없음", ["직원은 현장에서 고장 난 설비를 직접 수리한다."], None),
        case("employee_share", "직원 수의 비율은 업무 비중이 아님", ["대부분의 직원이 현장에서 장비를 직접 수리한다."], None),
    ]


def confirmation_cases():
    return [
        case("confirm_digital", "전부 정형 전산 처리", ["담당자의 업무 전부는 전산 화면에서 정해진 분류 코드를 복사하고 신청서 항목을 반복 입력하는 일이다."], 1),
        case("confirm_human", "대부분 환자 직접 처치", ["간호 인력의 핵심 업무 대부분은 환자 곁에서 상태를 관찰하고 직접 처치하며 보호자에게 대면으로 설명하는 일이다."], 5),
        case("confirm_mainly", "주로 화물 운반", ["직원은 주로 창고에서 포장된 화물을 직접 들어 차량에 싣고 내린다."], 4),
        case("confirm_substantial", "상당 부분 정형 정보 처리", ["담당 업무의 상당 부분은 접수된 문의를 정해진 항목으로 분류하고 표준 문구를 복사해 문서를 만드는 작업이다."], 2),
        case("confirm_mixed", "정형 처리와 대면 응대", ["직원은 정해진 서식을 반복 입력하는 업무와 방문 고객의 사정을 대면으로 듣고 해결 방안을 판단하는 업무를 함께 수행한다."], 3),
        case("confirm_unknown", "운반 업무 비중 미상", ["물류 직원은 상자를 직접 들어 옮긴다."], None),
        case("confirm_no_duties", "단순 직무명과 근거 부족", ["설계와 관리 업무를 담당한다."], None),
        case("confirm_compatible", "서로 양립하는 출처", [
            "직원은 정해진 양식에 접수 번호와 항목을 반복 입력한다.",
            "같은 직원은 고객을 대면으로 응대하고 개별 사정에 따른 해결 방안을 판단한다.",
        ], 3),
    ]
