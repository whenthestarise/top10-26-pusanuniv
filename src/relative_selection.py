"""활성 5개 항목의 상대지수와 지역 조건을 함께 계산합니다."""

from copy import deepcopy
from fractions import Fraction
import math


CRITERIA = ("salary", "work_life_balance", "infrastructure", "job_outlook", "ai_resilience")
REGIONS = ("부산", "울산", "경남")
GATES = ("source_linkage", "score_validation", "comparison_scope")
CRITERION_NAMES = {"salary": "높은 급여", "work_life_balance": "워라밸", "infrastructure": "근무 지역 인프라",
                   "job_outlook": "직무 유망성", "ai_resilience": "AI 대체 안전성"}
GATE_NAMES = {"source_linkage": "원문 근거 연결", "score_validation": "점수 평가 모델", "comparison_scope": "공통 직무·범위"}


def number(value, label, low=None, high=None):
    try:
        finite = isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if isinstance(value, bool) or not finite:
        raise ValueError(f"{label}: 유한한 숫자가 필요합니다.")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{label}: 허용 범위 밖의 값입니다.")
    return Fraction(str(value))


def weights_checked(weights):
    if not isinstance(weights, dict) or set(weights) != set(CRITERIA):
        raise ValueError("기존 활성 5개 항목의 가중치를 정확히 제공해야 합니다.")
    return {k: number(weights[k], f"{k} 가중치", .9, 1.1) for k in CRITERIA}


def finite_float(value):
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("계산 결과가 저장 가능한 수치 범위를 벗어났습니다.") from exc
    if not math.isfinite(result):
        raise ValueError("계산 결과가 유한하지 않습니다.")
    return result


def _merge(left, right):
    if left is None or right["total"] > left["total"]:
        return right
    if right["total"] < left["total"]:
        return left
    return {"total": left["total"], "count": left["count"] + right["count"],
            "possible": left["possible"] | right["possible"],
            "mandatory": left["mandatory"] & right["mandatory"]}


def optimal_sets(items, top_n, require_all_regions=True):
    """동적 계획법으로 최적 합계와 모든 최적 조합의 포함 관계를 계산합니다."""
    states = {(0, 0): {"total": Fraction(0), "count": 1, "possible": frozenset(), "mandatory": frozenset()}}
    for company_id, region, score in sorted(items):
        next_states = dict(states)
        bit = 1 << REGIONS.index(region)
        for (count, mask), previous in states.items():
            if count >= top_n:
                continue
            key = (count + 1, mask | bit)
            included = {"total": previous["total"] + score, "count": previous["count"],
                        "possible": previous["possible"] | {company_id},
                        "mandatory": previous["mandatory"] | {company_id}}
            next_states[key] = _merge(next_states.get(key), included)
        states = next_states
    if require_all_regions:
        return states.get((top_n, (1 << len(REGIONS)) - 1))
    best = None
    for (count, _), state in states.items():
        if count == top_n:
            best = _merge(best, state)
    return best


def calculate_selection(payload, weights, salary_baseline=7150, top_n=10):
    """원점수의 사실성은 입력 검증 단계에서 확인하며 이 함수는 계산 규격을 검사합니다."""
    w = weights_checked(weights)
    baseline = number(salary_baseline, "기준연봉")
    if baseline <= 0:
        raise ValueError("기준연봉은 양수여야 합니다.")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < len(REGIONS):
        raise ValueError("선정 개수는 지역 수 이상인 정수여야 합니다.")
    if not isinstance(payload, dict) or payload.get("dataset_type") not in {"real", "synthetic", "demo"}:
        raise ValueError("실제 자료·가상 기능 검사·시연 중 자료 종류를 명시해야 합니다.")
    dataset = payload["dataset_type"]
    if dataset == "demo":
        metadata = payload.get("demo_metadata", {})
        if (not isinstance(metadata, dict) or metadata.get("purpose") != "mvp_presentation"
                or not isinstance(metadata.get("label"), str) or "시연" not in metadata["label"]
                or metadata.get("assumptions_recorded") is not True):
            raise ValueError("시연 모드는 시연 표시와 내부 가정 기록이 필요합니다.")
    companies = payload.get("companies")
    if not isinstance(companies, list) or not companies:
        raise ValueError("기업 목록이 비어 있거나 형식이 잘못됐습니다.")
    rows, values = {}, {}
    for company in companies:
        if not isinstance(company, dict):
            raise ValueError("기업 항목은 객체여야 합니다.")
        cid, name, region = (company.get(k) for k in ("company_id", "company_name", "region"))
        if not isinstance(cid, str) or not cid.strip() or cid in rows:
            raise ValueError("비어 있거나 중복된 기업 식별자입니다.")
        if not isinstance(name, str) or not name.strip() or region not in REGIONS:
            raise ValueError("기업명 또는 지역을 확인하세요.")
        if dataset == "synthetic" and not cid.startswith("synthetic:"):
            raise ValueError("가상 기능 검사에는 별도의 가상 기업 식별자를 사용해야 합니다.")
        raw = company.get("values")
        if not isinstance(raw, dict) or set(raw) != set(CRITERIA):
            raise ValueError("기업별 원값은 활성 5개 항목을 정확히 포함해야 합니다.")
        evidence, deferrals = company.get("evidence", {}), company.get("deferral_reasons", [])
        if not isinstance(evidence, dict) or set(evidence) - set(CRITERIA):
            raise ValueError("근거 목록은 활성 항목별 객체여야 합니다.")
        if not isinstance(deferrals, list) or any(not isinstance(v, str) for v in deferrals):
            raise ValueError("보류 사유는 문자열 목록이어야 합니다.")
        parsed = {}
        for key in CRITERIA:
            value = raw[key]
            if value is None:
                parsed[key] = None
            elif key == "salary":
                parsed[key] = number(value, f"{name} 연봉")
                if parsed[key] <= 0:
                    raise ValueError("연봉은 양수 또는 미상이어야 합니다.")
            else:
                parsed[key] = number(value, f"{name}/{key}", 1, 5)
        values[cid] = parsed
        salary_index = parsed["salary"] / baseline if parsed["salary"] is not None else None
        rows[cid] = {"company_id": cid, "company_name": name, "region": region,
            "values": deepcopy(raw), "comparison_scope_id": company.get("comparison_scope_id"),
            "target_job": company.get("target_job"), "evidence": deepcopy(company.get("evidence", {})),
            "upstream_deferral_reasons": deepcopy(company.get("deferral_reasons", [])),
            "missing_criteria": [k for k in CRITERIA if parsed[k] is None],
            "salary_index": finite_float(salary_index) if salary_index is not None else None,
            "criterion_indices": None, "criterion_explanations": None,
            "weighted_sum_T": None, "relative_index_I": None, "display_index_P": None,
            "salary_equivalent_M": None, "relative_index_exact": None, "rank": None,
            "status": "평가 보류"}
    if dataset in {"real", "demo"} and set(rows) != {f"selected50:{i:03d}" for i in range(1, 51)}:
        raise ValueError("실제·시연 계산의 후보 목록은 연구팀 선별 50개여야 합니다.")
    W = sum(w.values(), Fraction(0))
    result = {"calculation_version": "five_criteria_selection_v2", "dataset_type": dataset,
        "active_criteria": list(CRITERIA), "weights": {k: finite_float(w[k]) for k in CRITERIA},
        "active_weight_sum": finite_float(W), "salary_baseline_manwon": finite_float(baseline),
        "requested_top_n": top_n, "input_company_count": len(rows),
        "usable_value_counts": {k: sum(values[cid][k] is not None for cid in rows) for k in CRITERIA},
        "all_five_values_present_count": sum(all(values[cid][k] is not None for k in CRITERIA) for cid in rows),
        "normalization": {"status": "보류", "company_ids": [], "non_salary_means": None},
        "normalization_block_reasons": [], "selection_block_reasons": [],
        "eligible_count": 0, "shortfall": top_n, "ranked_companies": [],
        "selected_companies": [], "candidate_companies": [], "mandatory_companies": [],
        "optimal_combination_count": 0, "selection_status": "계산 보류", "boundary_tie": False,
        "objective_sum_I": None, "regional_constraint_cost_I": None,
        "final_top10": None, "ready_for_ranking": False, "real_selection_confirmed": False,
        "selection_is_synthetic": dataset == "synthetic", "missing_values_imputed": False,
        "selection_is_demo": dataset == "demo", "assumed_values_used": dataset == "demo",
        "reduced_criteria_scenario_used": False, "tie_policy": "최적 조합이 여러 개이면 임의 확정하지 않음",
        "region_minimums": {r: 1 for r in REGIONS}}
    if dataset == "real":
        gates = payload.get("validation_gates", {})
        if not isinstance(gates, dict):
            raise ValueError("실제 자료의 자동 검증 상태는 객체여야 합니다.")
        for key in GATES:
            if gates.get(key) is not True:
                result["normalization_block_reasons"].append(f"실제 자료 검증 미완료: {GATE_NAMES[key]}")
    cohort = payload.get("comparison_cohort")
    ids = []
    if cohort is None:
        result["normalization_block_reasons"].append("동일 직무·범위의 공통 비교집합이 선언되지 않음")
    else:
        if not isinstance(cohort, dict):
            raise ValueError("공통 비교집합은 객체여야 합니다.")
        ids = cohort.get("company_ids")
        if not isinstance(ids, list) or any(not isinstance(cid, str) for cid in ids) or len(ids) != len(set(ids)):
            raise ValueError("공통 비교집합에는 고유 기업 식별자 목록이 필요합니다.")
        if set(ids) - set(rows):
            raise ValueError("후보 목록에 없는 기업을 비교집합에 넣을 수 없습니다.")
        if any(not isinstance(cohort.get(k), str) or not cohort[k].strip() for k in ("scope_id", "target_job", "scope_definition")):
            raise ValueError("비교집합의 직무와 평가 범위를 명시해야 합니다.")
        result["normalization"]["scope"] = {k: cohort[k] for k in ("scope_id", "target_job", "scope_definition")}
        result["normalization"]["company_ids"] = sorted(ids)
        if not ids:
            result["normalization_block_reasons"].append("공통 비교집합이 비어 있음")
        if any(rows[cid]["comparison_scope_id"] != cohort["scope_id"] or rows[cid]["target_job"] != cohort["target_job"] for cid in ids):
            result["normalization_block_reasons"].append("공통 비교집합의 직무 또는 평가 범위가 서로 다름")
        if any(rows[cid]["missing_criteria"] for cid in ids):
            result["normalization_block_reasons"].append("선언한 공통 비교집합에 활성 항목이 미상인 기업이 있음")
    if result["normalization_block_reasons"]:
        result["selection_block_reasons"] = list(result["normalization_block_reasons"])
    else:
        means = {k: sum((values[cid][k] for cid in ids), Fraction(0)) / len(ids) for k in CRITERIA[1:]}
        result["normalization"].update(status="계산 완료", cohort_size=len(ids),
            non_salary_means={k: finite_float(v) for k, v in means.items()},
            non_salary_means_exact={k: str(v) for k, v in means.items()})
        exact_scores = {}
        for cid in ids:
            row, raw = rows[cid], values[cid]
            indices = {k: raw[k] / (baseline if k == "salary" else means[k]) for k in CRITERIA}
            terms = {k: w[k] * indices[k] for k in CRITERIA}
            T = sum(terms.values(), Fraction(0))
            I = T / W
            exact_scores[cid] = I
            row.update(status="계산 완료", criterion_indices={k: finite_float(v) for k, v in indices.items()},
                criterion_explanations={k: {"raw_value": finite_float(raw[k]), "reference_value": finite_float(baseline if k == "salary" else means[k]),
                    "index": finite_float(indices[k]), "weight": finite_float(w[k]), "weighted_term": finite_float(terms[k]),
                    "relative_index_contribution": finite_float(terms[k] / W)} for k in CRITERIA},
                weighted_sum_T=finite_float(T), relative_index_I=finite_float(I), display_index_P=finite_float(100 * I),
                salary_equivalent_M=finite_float(baseline * I), relative_index_exact=str(I))
        ordered = sorted(ids, key=lambda cid: (-exact_scores[cid], cid))
        previous, rank = None, 0
        for position, cid in enumerate(ordered, 1):
            if previous is None or exact_scores[cid] != previous:
                rank = position
            rows[cid]["rank"] = rank
            previous = exact_scores[cid]
        def brief(cid):
            return {key: deepcopy(rows[cid][key]) for key in ("company_id", "company_name", "region", "relative_index_I", "display_index_P", "rank")}
        result["ranked_companies"] = [brief(cid) for cid in ordered]
        result["eligible_count"] = len(ids)
        result["shortfall"] = max(0, top_n - len(ids))
        result["eligible_region_counts"] = {r: sum(rows[cid]["region"] == r for cid in ids) for r in REGIONS}
        if len(ids) < top_n:
            result["selection_block_reasons"].append("활성 5개 항목이 완비된 공통 비교집합의 기업 수가 선정 목표보다 적음")
        if any(not result["eligible_region_counts"][r] for r in REGIONS):
            result["selection_block_reasons"].append("공통 비교집합에서 지역별 최소 1개 조건을 충족할 수 없음")
        if not result["selection_block_reasons"]:
            items = [(cid, rows[cid]["region"], exact_scores[cid]) for cid in ids]
            best, unrestricted = optimal_sets(items, top_n), optimal_sets(items, top_n, False)
            if best is None:
                raise RuntimeError("지역 조건을 만족하는 최적 조합 계산에 실패했습니다.")
            result.update(optimal_combination_count=best["count"], objective_sum_I=finite_float(best["total"]),
                objective_sum_I_exact=str(best["total"]), regional_constraint_cost_I=finite_float(unrestricted["total"] - best["total"]),
                candidate_companies=[brief(cid) for cid in ordered if cid in best["possible"]],
                mandatory_companies=[brief(cid) for cid in ordered if cid in best["mandatory"]],
                boundary_tie=best["count"] > 1)
            if best["count"] == 1:
                selected = [cid for cid in ordered if cid in best["mandatory"]]
                if len(selected) != top_n:
                    raise RuntimeError("유일한 최적 조합의 선정 개수가 목표와 다릅니다.")
                result["selected_companies"] = [brief(cid) for cid in selected]
                result["selected_region_counts"] = {r: sum(rows[cid]["region"] == r for cid in selected) for r in REGIONS}
                result["selection_status"] = ("가상 기능 검사 선정 완료" if dataset == "synthetic" else
                                              "시연용 선정 완료" if dataset == "demo" else "선정 완료")
                for cid in selected:
                    rows[cid]["selection_reason"] = {"rule": "지역별 최소 1개를 만족하는 조합 중 상대지수 합 최대",
                        "relative_index_I": rows[cid]["relative_index_I"], "region": rows[cid]["region"],
                        "regional_constraint_affected_selection": cid not in unrestricted["possible"]}
                if dataset == "real":
                    result.update(final_top10=deepcopy(result["selected_companies"]), ready_for_ranking=True, real_selection_confirmed=True)
            else:
                result["selection_status"] = "동점 조합으로 확정 보류"
                result["selection_block_reasons"].append("최적의 선정 조합이 여러 개라 임의의 기업을 탈락시키지 않음")
        else:
            result["selection_status"] = "선정 조건 미충족"
    for cid, row in rows.items():
        if row["relative_index_I"] is None:
            row["status"] = "평가 보류" if row["missing_criteria"] else "비교집합·검증 보류"
            row["deferral_reasons"] = [*row["upstream_deferral_reasons"],
                *(f"미상 평가항목: {CRITERION_NAMES[k]}" for k in row["missing_criteria"]),
                *(result["normalization_block_reasons"] if cid in ids or not ids else ["선언된 공통 비교집합 밖의 기업"])]
    result["company_results"] = [rows[cid] for cid in sorted(rows)]
    result["deferred_company_ids"] = [cid for cid in sorted(rows) if rows[cid]["relative_index_I"] is None]
    result["calculation_completed"] = result["normalization"]["status"] == "계산 완료"
    result["note"] = "가상 기능 검사 결과는 실제 기업 선정으로 사용하지 않으며 원점수의 의미상 타당성은 별도 검증 대상"
    if dataset == "demo":
        result["missing_values_imputed"] = True
        result["demo_label"] = payload["demo_metadata"]["label"]
        result["note"] = "시연용 결과이며 실측 기업 평가 또는 검증된 최종 선정으로 취급하지 않음"
    return result
