"""기능 검사와 제공 사실 읽기를 구분하여 다음 시험 후보를 정합니다."""


def select_for_pilot(reports, functional_count=19, provided_count=6):
    usable = [r for r in reports if r.get("functional_count") == functional_count
              and r.get("provided_data_count") == provided_count
              and r.get("provided_data_passed") == provided_count
              and r.get("status") != "runtime_error"]
    if not usable:
        return {"status": "no_candidate_passed_provided_reading", "candidate": None,
                "functional_scoring_checks_passed": False, "company_scoring_enabled": False, "ready_for_ranking": False}
    best = min(usable, key=lambda r: (-r["functional_passed"], r.get("model_peak_gib", float("inf")),
                                     r["wall_seconds"], r["candidate"]["candidate_id"]))
    return {"status": "candidate_for_provided_fact_reading", "candidate": best["candidate"],
            "functional_passed": best["functional_passed"], "functional_count": functional_count,
            "provided_data_passed": provided_count, "provided_data_count": provided_count,
            "functional_scoring_checks_passed": best["functional_passed"] == functional_count,
            "real_company_evaluation_accuracy_validated": False, "company_scoring_enabled": False, "ready_for_ranking": False,
            "note": "사실 읽기와 가상 기능 검사 결과이며 실제 기업 점수의 정확도 검증이나 선정 완료가 아님"}
