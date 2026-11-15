"""구절 분류를 기존 5개 항목 평가표에 연결하고 선정 가능 여부를 계산합니다."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import re

if __package__:
    from .build_company_evaluation import ACTIVE_IDS, run as build_table, write_outputs
    from .classify_selected_evidence import PHYSICAL, DIGITAL, check_claim, claim_units, classify, digest, function_cases
    from .classify_pdf_job_references import verify_reference_claim
    from .data_policy import ROOT, load_user_records
    from .selected50_table import load_selected
else:
    from build_company_evaluation import ACTIVE_IDS, run as build_table, write_outputs
    from classify_selected_evidence import PHYSICAL, DIGITAL, check_claim, claim_units, classify, digest, function_cases
    from classify_pdf_job_references import verify_reference_claim
    from data_policy import ROOT, load_user_records
    from selected50_table import load_selected


class RecordedGenerator:
    def __init__(self, generation):
        self.generation = generation

    def choose(self, messages, allowed):
        return self.generation


def replay_classification(item):
    if classify(item["claim"], RecordedGenerator(item["generation"])) != item:
        raise ValueError("분류 응답과 검사 결과가 원본 기록과 다릅니다.")


def verify_claim_results(items, record, book_hash, field="K", scope="company_qualitative"):
    expected = claim_units(record, book_hash, field, scope)
    if not isinstance(items, list) or [item.get("claim") for item in items] != expected:
        raise ValueError("저장된 구절 분류가 원문 범위와 다릅니다.")
    for item in items:
        check_claim(item["claim"], record, book_hash)
        replay_classification(item)


def audit_content(item):
    audited = deepcopy(item)
    text, label = item["claim"]["quote"], item["effective_label"]
    flags = []
    if label == "업무 내용 서술" and re.search(r"낮음|중간|높음", text):
        flags.append("위험 등급이 섞인 직군 설명을 실제 담당 행동으로 확인할 수 없음")
    if label in {"정성 등급·직군 설명", "업무 내용 서술"} and (PHYSICAL.search(text) or DIGITAL.search(text)):
        flags.append("원문 자동화 표지와 모델 분류의 관계를 추가 점검해야 함")
    audited["content_audit_flags"] = flags
    audited["assessment_effective_label"] = "판단 보류" if flags else label
    return audited


def selection_readiness(table, scoring_gate):
    definitions = table["weights"]
    if tuple(d["criterion_id"] for d in definitions) != ACTIVE_IDS:
        raise ValueError("활성 5개 기준과 가중치를 유지해야 합니다.")
    if any(not isinstance(d["weight"], (int, float)) or isinstance(d["weight"], bool)
           or not math.isfinite(d["weight"]) or d["weight"] <= 0 for d in definitions):
        raise ValueError("가중치는 유한한 양수여야 합니다.")
    available = dict.fromkeys(ACTIVE_IDS, 0)
    complete = []
    normalized_complete = []
    for company in table["companies"]:
        criteria = company["criteria"]
        if set(criteria) != set(ACTIVE_IDS):
            raise ValueError("기업별 평가 항목이 활성 5개와 다릅니다.")
        present = []
        for key in ACTIVE_IDS:
            value = criteria[key]["index"] if key == "salary" else criteria[key]["raw_score"]
            valid = (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                     and (value > 0 if key == "salary" else 1 <= value <= 5))
            available[key] += valid
            present.append(valid)
        if all(present):
            complete.append(company)
            if all(isinstance(criteria[k]["index"], (int, float)) and not isinstance(criteria[k]["index"], bool)
                   and math.isfinite(criteria[k]["index"]) and criteria[k]["index"] > 0 for k in ACTIVE_IDS):
                normalized_complete.append(company)
    reasons = []
    if len(complete) < 10:
        reasons.append(f"활성 5개 항목의 수치가 모두 있는 기업 {len(complete)}개로 목표 10개에 미달")
    regions = Counter(c["region"] for c in complete)
    if any(not regions[region] for region in ("부산", "울산", "경남")):
        reasons.append("완결된 공통 비교집합에서 지역별 최소 1개 조건을 확인할 수 없음")
    if scoring_gate is not True:
        reasons.append("기존 기업 점수 평가 모델의 자동 기능 검증이 미완료")
    if len(normalized_complete) < 10:
        reasons.append("활성 항목의 공통 비교집합 정규화가 완료된 기업이 10개 미만")
    if not table["summary"]["ready_for_ranking"]:
        reasons.append("비급여 공통 비교집합과 최종 순위 연결이 준비되지 않음")
    return {"active_criteria": list(ACTIVE_IDS), "weights": {d["criterion_id"]: d["weight"] for d in definitions},
            "active_weight_sum": sum(d["weight"] for d in definitions), "usable_value_counts": available,
            "all_five_complete_count": len(complete), "complete_region_counts": dict(regions),
            "all_five_normalized_count": len(normalized_complete),
            "company_count": len(table["companies"]), "target_count": 10,
            "can_confirm_top10": not reasons, "deferral_reasons": reasons, "final_top10": None,
            "reduced_criteria_scenario_used": False, "missing_values_imputed": False}


def connect_table(table, company_results, scoring_gate, pdf_references=None):
    table = deepcopy(table)
    annotations = {item["company_id"]: item for item in company_results}
    pdf_annotations = {item["company_id"]: item for item in (pdf_references or [])}
    if len(annotations) != len(company_results) or set(annotations) != {c["company_id"] for c in table["companies"]}:
        raise ValueError("기업별 분류 결과가 평가 대상과 다릅니다.")
    for company in table["companies"]:
        annotation = annotations[company["company_id"]]
        if annotation["company_name"] != company["company_name"] or annotation["ai_resilience_score"] is not None:
            raise ValueError("분류 결과를 기업 점수로 바꾸거나 다른 기업에 연결할 수 없습니다.")
        ai = company["criteria"]["ai_resilience"]
        if ai["raw_score"] is not None or ai["index"] is not None:
            raise ValueError("기존 점수가 있으면 별도로 대조해야 하며 분류 결과로 덮어쓰지 않습니다.")
        ai["content_classifications"] = [audit_content(c) for c in annotation["claims"]]
        reference = pdf_annotations.get(company["company_id"])
        ai["partial_job_reference_classifications"] = [audit_content(reference["reference_classification"])] if reference else []
        ai["reference_scope_note"] = "PDF의 일부 직무 설명과 명단의 여러 채용 직군은 범위가 달라 회사 전체 점수로 합치지 않음"
        ai["classification_counts"] = dict(Counter(c["assessment_effective_label"] for c in ai["content_classifications"]))
        ai["status"] = "정성 설명 분류 완료·점수 보류"
        ai["raw_score"] = None
        ai["index"] = None
        ai["score_gate_passed"] = scoring_gate is True
        ai["missing_information"] = list(dict.fromkeys(ai["missing_information"] + [
            "여러 채용 직군을 회사 전체로 합칠 업무·인원 비중 미기재",
            "정성 설명의 분류 결과만으로 확정 점수를 만들지 않음",
            *([] if scoring_gate is True else ["기업 점수 평가 모델의 기존 자동 기능 검증 미통과"])]))
        company["missing_criteria"] = [key for key in ACTIVE_IDS if
            company["criteria"][key]["index" if key == "salary" else "raw_score"] is None]
        company["deferral_reasons"] = [company["criteria"][key]["name"] + ": " +
            " / ".join(company["criteria"][key]["missing_information"]) for key in company["missing_criteria"]]
        company["composite_index"] = None
        company["ready_for_ranking"] = False
    table["summary"]["ready_for_ranking"] = False
    table["summary"]["final_top10"] = None
    table["summary"]["classified_company_count"] = len(annotations)
    table["summary"]["ai_content_claim_count"] = sum(len(a["claims"]) for a in annotations.values())
    table["summary"]["pdf_job_reference_classification_count"] = len(pdf_annotations)
    table["selection_feasibility"] = selection_readiness(table, scoring_gate)
    table["summary"]["usable_value_counts"] = table["selection_feasibility"]["usable_value_counts"]
    table["summary"]["common_complete_comparison_count"] = table["selection_feasibility"]["all_five_complete_count"]
    table["user_decision"] = "기존 5개 항목 평가만 유지; 2개 항목 참고 시나리오 제외"
    annotated = [item for c in table["companies"] for item in c["criteria"]["ai_resilience"]["content_classifications"]]
    table["content_audit_summary"] = {"audited_claim_category_counts": dict(Counter(a["assessment_effective_label"] for a in annotated)),
        "additional_audit_deferral_count": sum(bool(a["content_audit_flags"]) for a in annotated),
        "claim_count": len(annotated), "semantic_accuracy_independently_validated": False}
    return table


def run(root, classification_dir, pdf_reference_dir, output):
    root, folder, reference_folder, output = Path(root), Path(classification_dir), Path(pdf_reference_dir), Path(output)
    if output.exists():
        raise FileExistsError("기존 결과를 보존합니다. 새로운 출력 폴더를 지정하세요.")
    policy, records = load_user_records(root)
    selected = load_selected(root, policy)
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("status") != "completed" or not summary.get("all_source_hashes_unchanged"):
        raise ValueError("완료되고 원본 해시가 확인된 분류 결과가 필요합니다.")
    for name, expected in manifest["source_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("실제 제공 자료·설정이 모델 분류 때와 다릅니다.")
    for name, expected in manifest["implementation_sha256"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError("구절 분류에 사용한 구현과 검증 구현이 다릅니다.")
    checks = json.loads((folder / "functional_checks.json").read_text())
    cases = function_cases()
    if len(checks) != len(cases) or any(not c["passed"] for c in checks):
        raise ValueError("분류 기능 검사가 완결되지 않았습니다.")
    for stored, case in zip(checks, cases, strict=True):
        if any(stored.get(k) != v for k, v in case.items()) or stored["result"]["effective_label"] != case["expected"]:
            raise ValueError("분류 기능 검사 사례나 기대값이 다릅니다.")
        if stored["result"]["claim"] != {"quote": case["text"]}:
            raise ValueError("기능 검사 입력 구절이 다릅니다.")
        replay_classification(stored["result"])
    companies = json.loads((folder / "company_results.json").read_text())
    guides = json.loads((folder / "guideline_results.json").read_text())
    if len(companies) != 50 or len(guides) != len(selected["job_guidelines"]):
        raise ValueError("선별 50개 또는 직군 기준의 분류 결과가 부족합니다.")
    book_hash = policy["raw_sources"]["selected50"]["sha256"]
    for result, record in zip(companies, selected["companies"], strict=True):
        if result["company_id"] != record["company_id"] or result["company_name"] != record["company_name"]:
            raise ValueError("기업별 분류가 원본 명단의 기업과 다릅니다.")
        verify_claim_results(result["claims"], record, book_hash)
    for result, record in zip(guides, selected["job_guidelines"], strict=True):
        if (result["guideline_id"] != record["guideline_id"] or result["usable_as_company_empirical_evidence"]
                or result["job_group"] != record["job_group"] or result["tasks_reference"] != record["tasks"]):
            raise ValueError("직군 일반 기준을 개별 기업 실증 근거로 사용하지 않습니다.")
        verify_claim_results(result["claims"], record, book_hash, "D", "general_job_reference")
    reference_manifest = json.loads((reference_folder / "manifest.json").read_text())
    reference_summary = json.loads((reference_folder / "summary.json").read_text())
    if reference_summary.get("status") != "completed" or not reference_summary.get("all_source_hashes_unchanged"):
        raise ValueError("PDF 직무 참고 분류가 완료되지 않았습니다.")
    if (reference_manifest["candidate"] != manifest["candidate"]
            or reference_manifest["source_hashes"] != manifest["source_hashes"]
            or reference_manifest["classifier_implementation_sha256"] != manifest["implementation_sha256"]
            or reference_manifest["reference_implementation_sha256"] != digest(Path(__file__).parent / "classify_pdf_job_references.py")):
        raise ValueError("PDF 참고 분류와 원문·모델·구현이 다릅니다.")
    reference_results = json.loads((reference_folder / "reference_results.json").read_text())
    reference_targets = [c for c in selected["companies"] if c["pdf_reference"] is not None]
    if len(reference_results) != len(reference_targets):
        raise ValueError("안전하게 대응된 PDF 참고 자료 수와 다릅니다.")
    for item, record in zip(reference_results, reference_targets, strict=True):
        if (item["company_id"] != record["company_id"] or item["company_name"] != record["company_name"]
                or item["usable_as_company_wide_score"] or item["reference_job_score"] is not None):
            raise ValueError("PDF 일부 직무를 다른 기업이나 회사 전체 점수로 합칠 수 없습니다.")
        verify_reference_claim(item["reference_classification"]["claim"], record,
                               records[record["catalog_link"]["company_id"]], policy)
        replay_classification(item["reference_classification"])
    output.mkdir(parents=True)
    table = build_table(root, output / "source_table")
    gate = json.loads((root / "configs/selected_model.json").read_text()).get("functional_scoring_checks_passed")
    connected = connect_table(table, companies, gate, reference_results)
    connected["general_job_reference_classifications"] = guides
    connected["classification_run_summary"] = summary
    connected["pdf_job_reference_run_summary"] = reference_summary
    write_outputs(connected, output / "evaluation")
    (output / "selection_feasibility.json").write_text(json.dumps(connected["selection_feasibility"], ensure_ascii=False, indent=2))
    with (output / "업무근거_구절분류.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["기업ID", "기업명", "원문구절", "모델분류", "추가점검후분류", "검사보류사유", "근거셀", "원문시작", "원문끝", "AI점수"])
        for result in connected["companies"]:
            for item in result["criteria"]["ai_resilience"]["content_classifications"]:
                claim = item["claim"]
                writer.writerow([result["company_id"], result["company_name"], claim["quote"], item["model_label"],
                    item["assessment_effective_label"], " / ".join(item["validation_errors"] + item["content_audit_flags"]),
                    f"'{claim['source_sheet']}'!{claim['source_field']}{claim['source_row']}",
                    claim["start"], claim["end"], None])
    with (output / "PDF직무참고_분류.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["기업ID", "기업명", "PDF기업ID", "PDF일부직무", "PDF쪽수", "원문", "모델분류", "추가점검후분류", "보류사유", "회사전체AI점수"])
        for company in connected["companies"]:
            for item in company["criteria"]["ai_resilience"]["partial_job_reference_classifications"]:
                claim = item["claim"]
                writer.writerow([company["company_id"], company["company_name"], claim["referenced_document_company_id"],
                    claim["reference_target_job"], claim["source_page"], claim["quote"], item["model_label"],
                    item["assessment_effective_label"], " / ".join(item["validation_errors"] + item["content_audit_flags"]), None])
    return connected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--classification-dir", type=Path, required=True)
    parser.add_argument("--pdf-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        table = run(args.project_root, args.classification_dir, args.pdf_reference_dir, args.output_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"기업 {len(table['companies'])}개, 활성 5개 항목의 평가 근거와 보류 사유를 연결했습니다.")
    print("최종 10개 확정 가능:", "예" if table["selection_feasibility"]["can_confirm_top10"] else "보류")


if __name__ == "__main__":
    main()
