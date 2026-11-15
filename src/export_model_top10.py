"""기업별 모델 평가와 기존 가중치로 TOP10 및 선정 근거 마크다운을 만듭니다."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import yaml

if __package__:
    from .data_policy import ROOT, load_user_records
    from .model_company_ratings import KEYS, company_fields, parse_rating, sanity_cases
    from .model_role_balance import aggregate_roles, parse_score, roles_of
    from .prepare_project_data import load_survey_weights
    from .relative_selection import CRITERIA, CRITERION_NAMES, calculate_selection
    from .selected50_table import load_selected
else:
    from data_policy import ROOT, load_user_records
    from model_company_ratings import KEYS, company_fields, parse_rating, sanity_cases
    from model_role_balance import aggregate_roles, parse_score, roles_of
    from prepare_project_data import load_survey_weights
    from relative_selection import CRITERIA, CRITERION_NAMES, calculate_selection
    from selected50_table import load_selected


RATED = tuple(CRITERIA[1:])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_result(record, result):
    if result.get("company_id") != record["company_id"] or result.get("company_name") != record["company_name"]:
        raise ValueError("모델 평가 결과의 기업이 제공 명단과 다릅니다.")
    fields = company_fields(record)
    if result.get("provided_fields") != fields:
        raise ValueError("모델에 전달한 원문이 제공 자료와 다릅니다.")
    if result.get("generation", {}).get("finish_reason") == "length":
        raise ValueError("완료되지 않은 모델 응답은 사용하지 않습니다.")
    checked = parse_rating(result["generation"]["answer"], fields)
    if checked != result.get("ratings"):
        raise ValueError("모델 원시 응답과 저장된 점수·근거가 다릅니다.")
    return checked


def build_payload(records, results, run_hash):
    if len(records) != 50 or len(results) != 50:
        raise ValueError("선별 50개의 기업과 모델 평가가 모두 필요합니다.")
    by_id = {r["company_id"]: r for r in results}
    if len(by_id) != 50 or set(by_id) != {r["company_id"] for r in records}:
        raise ValueError("평가 결과에 중복 또는 명단 밖의 기업이 있습니다.")
    scope_id = f"model-evaluation:{run_hash[:16]}"
    companies, cohort, ratings = [], [], {}
    for record in records:
        checked = validate_result(record, by_id[record["company_id"]])
        ratings[record["company_id"]] = checked
        values = {"salary": record["annual_salary_manwon"], **{k: checked[k]["score"] for k in RATED}}
        companies.append({"company_id": record["company_id"], "company_name": record["company_name"], "region": record["region"],
            "comparison_scope_id": scope_id, "target_job": "명단 주요 직군 구성의 모델 평가",
            "values": values, "evidence": {k: [{"field": f, "quote": quote} for f, quote in checked[k]["basis_quotes"].items()] for k in RATED},
            "deferral_reasons": [] if values["salary"] is not None else ["제공 평균연봉 미확인으로 종합 계산 보류"]})
        if values["salary"] is not None:
            cohort.append(record["company_id"])
    return {"dataset_type": "demo", "companies": companies,
        "demo_metadata": {"purpose": "mvp_presentation", "label": "모델 평가 결과 · 시연",
                          "assumptions_recorded": True, "score_origin": "model_interpretation", "rating_run_sha256": run_hash},
        "comparison_cohort": {"scope_id": scope_id, "target_job": "명단 주요 직군 구성의 모델 평가",
            "scope_definition": "각 기업의 제공 위치·산업·제품·주요 직군에 같은 모델 평가 규칙을 적용한 비교 범위",
            "company_ids": cohort}}, ratings


def verify_calculation(calculated):
    if len(calculated["selected_companies"]) != 10 or calculated["optimal_combination_count"] != 1:
        raise ValueError("유일한 10개 선정 조합이 아닙니다. 동점 및 선정 조건을 확인하세요.")
    if any(v < 1 for v in calculated["selected_region_counts"].values()):
        raise ValueError("지역별 최소 1개 조건을 충족하지 않습니다.")
    if calculated["real_selection_confirmed"] or calculated["final_top10"] is not None:
        raise ValueError("모델 평가 결과를 실측 검증된 선정으로 표시하지 않습니다.")
    cohort = set(calculated["normalization"]["company_ids"])
    rows = [c for c in calculated["company_results"] if c["company_id"] in cohort]
    for key in RATED:
        mean = math.fsum(c["criterion_indices"][key] for c in rows) / len(rows)
        if not math.isclose(mean, 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError("비급여 항목의 정규화 평균이 1이 아닙니다.")
    for row in rows:
        I = row["relative_index_I"]
        terms = row["criterion_explanations"]
        if (not math.isclose(math.fsum(v["relative_index_contribution"] for v in terms.values()), I, abs_tol=1e-12)
                or not math.isclose(row["weighted_sum_T"] / calculated["active_weight_sum"], I, abs_tol=1e-12)
                or not math.isclose(row["salary_equivalent_M"], calculated["salary_baseline_manwon"] * I, abs_tol=1e-8)):
            raise ValueError("가중치 기여도 또는 상대지수 검산이 일치하지 않습니다.")
    return {"normalization_means_equal_one": True, "contributions_checked": True,
            "regional_minimums_checked": True, "selected_count_checked": 10,
            "empirical_accuracy_validated": False}


def apply_refinement(records, payload, ratings, role_ratings, location_ratings, saved_balance):
    expected_roles = {role for record in records for role in roles_of(record)}
    expected_locations = {record["location"] for record in records}
    if set(role_ratings) != expected_roles or set(location_ratings) != expected_locations:
        raise ValueError("직군·위치 모델 평가가 제공 명단의 전체 구성과 다릅니다.")
    for stored in (*role_ratings.values(), *location_ratings.values()):
        generation = stored["generation"]
        if generation.get("finish_reason") == "length" or parse_score(generation["answer"]) != stored["score"]:
            raise ValueError("직군·위치 점수가 완료된 모델 원시 응답과 다릅니다.")
    recomputed = [aggregate_roles(record, role_ratings) for record in records]
    if recomputed != saved_balance:
        raise ValueError("직군별 점수와 기업별 워라밸 집계가 다릅니다.")
    by_id = {row["company_id"]: row for row in payload["companies"]}
    for record, balance in zip(records, recomputed, strict=True):
        cid = record["company_id"]
        ratings[cid]["work_life_balance"] = balance
        ratings[cid]["infrastructure"] = {"score": location_ratings[record["location"]]["score"],
            "basis_fields": ["D"], "basis_quotes": {"D": record["location"]},
            "interpretation_rule": "제공된 위치 표기에 생활권·산업 입지 연계 기준을 적용했고 같은 위치에는 같은 평가값을 부여했다",
            "score_origin": "model_location_rating", "empirically_verified": False}
        for key in ("work_life_balance", "infrastructure"):
            rated = ratings[cid][key]
            by_id[cid]["values"][key] = rated["score"]
            by_id[cid]["evidence"][key] = [{"field": field, "quote": quote} for field, quote in rated["basis_quotes"].items()]
    return recomputed


def load_refinement(root, folder, records, payload, ratings, candidate):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if (summary.get("status") != "completed" or summary.get("company_count") != 50
            or not summary.get("all_source_hashes_unchanged") or manifest["candidate"] != candidate):
        raise ValueError("같은 모델로 완료한 직군·입지 평가 기록이 필요합니다.")
    for name, expected in manifest["source_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("직군·입지 평가 이후 제공 원본이 변경됐습니다.")
    if digest(Path(__file__).parent / "model_role_balance.py") != manifest["implementation_sha256"]:
        raise ValueError("직군·입지 평가 구현의 버전이 다릅니다.")
    sanity = json.loads((folder / "sanity_checks.json").read_text())
    if len(sanity) != 2 or {row["name"] for row in sanity} != {"직군", "입지"}:
        raise ValueError("직군·입지 기능 검사 기록이 없습니다.")
    for case in sanity:
        scores = [parse_score(g["answer"]) for g in case["generations"]]
        if any(g.get("finish_reason") == "length" for g in case["generations"]) or scores != case["scores"] or not scores[0] > scores[1]:
            raise ValueError("직군·입지 기능 검사의 모델 응답이 유효하지 않습니다.")
    roles = json.loads((folder / "roles.json").read_text())
    locations = json.loads((folder / "locations.json").read_text())
    balances = json.loads((folder / "company_balance.json").read_text())
    apply_refinement(records, payload, ratings, roles, locations, balances)
    return summary


def text(value):
    return html.escape(str(value)).replace("|", "\\|").replace("\n", " ")


def render_markdown(calculated, records, ratings, summary):
    source = {r["company_id"]: r for r in records}
    company_rows = {r["company_id"]: r for r in calculated["company_results"]}
    selected_ids = [c["company_id"] for c in calculated["selected_companies"]]
    weights = calculated["weights"]
    lines = ["# 부울경 중견기업 TOP10 선정 결과", "", "기준일: 2026-11-15", "",
        "제공된 50개 기업 자료에 모델 평가와 설문 가중치를 적용해 10개 기업을 선정했다.",
        "비급여 점수는 위치·산업·제품·주요 직군 정보를 같은 기준으로 해석해 모델이 부여한 평가값이다.",
        "기업의 실제 근로조건·교통 접근성·미래 고용성과를 실측해 확인한 값과 구분한다.", "",
        "## 1. 최종 선정 10개", "",
        "| 순위 | 기업 | 지역 | 제공 연봉(만원/년) | 급여지수 | 워라밸 | 인프라 | 직무 유망성 | AI 안전성 | 종합지수 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, cid in enumerate(selected_ids, 1):
        r = company_rows[cid]
        values = r["values"]
        lines.append(f"| {rank} | {text(r['company_name'])} | {r['region']} | {values['salary']:,.0f} | {r['salary_index']:.4f} | " +
            " | ".join(str(values[k]) for k in RATED) + f" | {r['display_index_P']:.2f} |")
    lines += ["", "지역 구성: " + " · ".join(f"{k} {v}개" for k, v in calculated["selected_region_counts"].items()) + ".",
              "종합지수는 상대지수 I에 100을 곱한 값이다. 100점 상한이나 실제 만족도 비율을 뜻하지 않는다.", "",
              "## 2. 평가 항목과 가중치", "", "| 항목 | 가중치 | 정규화 비중 | 모델 점수 연결 기준 |",
              "|---|---:|---:|---|"]
    rules = {"salary": "제공 연봉 / 7,150", "work_life_balance": "산업·직군의 일정 예측 가능성과 현장 대응 부담",
             "infrastructure": "기재 위치의 생활권과 산업 입지 연계", "job_outlook": "제품·사업·직군의 기술 수요와 지속성",
             "ai_resilience": "직군별 자동화 노출과 사람의 수행·판단 역할"}
    for k in CRITERIA:
        lines.append(f"| {CRITERION_NAMES[k]} | {weights[k]:.6f} | {100*weights[k]/calculated['active_weight_sum']:.2f}% | {rules[k]} |")
    lines += ["", "비급여 4개 항목은 1~5점의 모델 평가값으로 계산했다. 공통 3점 대체 방식은 사용하지 않았다.",
              "워라밸은 주요 직군별 0.5점 단위 평가를 동일 가중 평균한 뒤 소수 첫째 자리로 반올림했다. 실제 재직 인원 비중을 가정한 값은 아니다.",
              "지역 인프라는 위치 표기만을 별도로 평가해 같은 위치에 같은 점수를 부여했다. 직무 유망성과 AI 안전성은 기업별 산업·제품·직군 설명을 함께 평가했다.",
              "기업명과 연봉은 비급여 점수 입력에서 제외했다. 원문에 없는 야근 횟수·통근 시간·성장률은 생성하지 않았다.", "",
              "### 점수 해석 기준", "",
              "- 워라밸: 1점은 매우 큰 일정 제약, 2점은 생산·운영 제약 중심, 3점은 계획·대응 업무 혼재, 4점은 계획형 전문 업무 중심, 5점은 자율성과 일정 예측 가능성이 매우 높게 평가된 업무다.",
              "- 지역 인프라: 1점은 매우 큰 입지 제약, 2점은 외곽·산업 거점 중심, 3점은 혼재·범위 불명확, 4점은 도시 생활권 연계, 5점은 구체적으로 기재된 높은 도심 접근성을 기준으로 평가한다.",
              "- 직무 유망성: 1점은 수요 축소, 2점은 기술·수요 연계가 약한 구성, 3점은 성숙 사업의 운영 수요, 4점은 기술 고도화·연구개발 연계, 5점은 첨단 제품과 신규 응용 직군의 강한 연결을 기준으로 평가한다.",
              "- AI 안전성: 1점은 반복 디지털 중심, 2점은 디지털 노출이 큰 구성, 3점은 사람 업무와 자동화 노출 혼재, 4점은 기술·연구·현장 판단 중심, 5점은 전문 판단·대인 대응·현장 수행 의존도가 매우 큰 구성을 기준으로 평가한다.", "",
              "## 3. 도출 과정", "",
              "1. 연구팀 선별 50개 기업을 대상으로 제공 위치·산업·제품·직군·AI 설명과 대응되는 PDF 일부 직무 자료를 연결했다.",
              "2. 같은 로컬 모델과 같은 평가 규칙으로 50개 기업의 비급여 4개 항목을 평가했다.",
              f"3. 제공 연봉을 확인할 수 있는 {calculated['eligible_count']}개 기업을 공통 계산 집합으로 정했다. 연봉 미확인 2개는 원값을 유지하고 종합 계산에서 보류했다.",
              "4. 비급여 각 항목을 공통 집합 평균으로 나눠 정규화하고 급여는 7,150만원을 기준으로 지수화했다.",
              "5. 기존 5개 가중치를 적용한 상대지수를 계산하고, 부산·울산·경남 각각 최소 1개를 포함하는 조합 중 지수 합이 가장 큰 10개를 선정했다.", "",
              "계산식: `급여지수 = 연봉 / 7150`, `비급여지수 = 모델평가값 / 항목평균`, `T = Σ(가중치 × 지수)`, `I = T / Σ가중치`, `종합지수 = 100 × I`.", "",
              "공통 계산 집합의 비급여 평균: " + " · ".join(f"{CRITERION_NAMES[k]} {calculated['normalization']['non_salary_means'][k]:.4f}" for k in RATED) + ".", "",
              "## 4. 기업별 선정 근거", ""]
    for rank, cid in enumerate(selected_ids, 1):
        record, row, rated = source[cid], company_rows[cid], ratings[cid]
        strongest = sorted(RATED, key=lambda k: (-row["criterion_explanations"][k]["relative_index_contribution"], k))[:2]
        lines += [f"### {rank}. {record['company_name']}", "",
            f"{record['company_name']}의 경우 제공 평균연봉 {record['annual_salary_manwon']:,.0f}만원과 " +
            f"‘{text(record['products_business'])}’ 사업 정보, ‘{text(record['main_hiring_roles'])}’ 직군 구성을 반영해 종합지수 {row['display_index_P']:.2f}로 선정했다.",
            "비급여 항목에서는 " + " · ".join(f"{CRITERION_NAMES[k]} {rated[k]['score']}점" for k in strongest) + "이 주요 가중 기여를 했다.", ""]
        for k in RATED:
            info = rated[k]
            quotes = " / ".join(f"{field}: {text(quote)}" for field, quote in info["basis_quotes"].items())
            lines.append(f"- **{CRITERION_NAMES[k]} {info['score']}점:** {info['interpretation_rule']}. 근거: {quotes}")
            if k == "work_life_balance":
                lines.append("  직군별 평가: " + " · ".join(f"{text(item['role'])} {item['score']}점" for item in info["role_scores"]) + f" → 동일 가중 평균 {info['score']}점.")
        lines += ["", "| 항목 | 계산에 사용한 값 | 기준값 | 정규화 지수 | 종합지수 기여 |",
                  "|---|---:|---:|---:|---:|"]
        for key in CRITERIA:
            term = row["criterion_explanations"][key]
            reference = calculated["salary_baseline_manwon"] if key == "salary" else calculated["normalization"]["non_salary_means"][key]
            lines.append(f"| {CRITERION_NAMES[key]} | {row['values'][key]} | {reference:.4f} | {row['criterion_indices'][key]:.4f} | {100*term['relative_index_contribution']:.2f} |")
        lines += ["", f"위 5개 기여도를 합산한 종합지수는 **{row['display_index_P']:.2f}**다. 표시는 소수 둘째 자리로 반올림했다.",
                  "**선정 판단:** 5개 항목의 가중 상대지수와 지역 조건을 함께 적용한 최적 조합에 포함되어 선정했다.", ""]
    lines += ["## 5. 검산과 확인 범위", "",
        f"- 모델 응답: 50개 기업의 네 항목 점수와 원시 응답을 모두 보존하고 형식·범위·원문 연결을 재검사했다.",
        f"- 기본 방향성 기능 검사: {summary['sanity_passed']}/{summary['sanity_count']} 통과. 이 검사는 실제 기업 평가 정확도 시험과 구분한다.",
        f"- 직군·입지 평가: {summary['role_count']}개 고유 직군과 {summary['location_count']}개 위치 표기에 같은 규칙을 적용했다.",
        "- 비급여 정규화 평균 1, 항목별 가중 기여도의 합, T/W=I, 선정 10개 및 지역별 최소 조건을 검산했다.",
        "- 외부 기업 자료는 추가하지 않았고, 원본 파일과 제공 연봉은 바꾸지 않았다.",
        "- 점수의 현실 적합성은 독립적인 기업 조사로 검증하지 않았다. 본 결과의 근거는 제공 자료의 해석과 기록된 모델 평가값이다.", "",
        "## 6. 전체 50개 기업 평가표", "",
        "| 순번 | 기업 | 지역 | 제공 연봉(만원/년) | 워라밸 | 인프라 | 직무 유망성 | AI 안전성 | 종합지수 | 선정 여부 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for record in records:
        cid, r = record["company_id"], company_rows[record["company_id"]]
        salary = f"{record['annual_salary_manwon']:,.0f}" if record["annual_salary_manwon"] is not None else "미확인"
        index = f"{r['display_index_P']:.2f}" if r["display_index_P"] is not None else "보류"
        lines.append(f"| {record['selection_order']} | {text(record['company_name'])} | {record['region']} | {salary} | " +
            " | ".join(str(ratings[cid][k]["score"]) for k in RATED) + f" | {index} | {'선정' if cid in selected_ids else '보류' if index == '보류' else '미선정'} |")
    lines += ["", "원자료 위치: `data/raw/부울경_중견기업50.xlsx`, 설문 응답 원본, 기존 PDF의 안전하게 대응된 일부 직무 자료.",
              "평가 원문·모델 응답·실행 버전·입력 해시는 별도 실행 기록에 보존했다.", ""]
    return "\n".join(lines)


def run(root, ratings_dir, refinement_dir, output, audit_dir):
    root, folder, output, audit_dir = Path(root), Path(ratings_dir), Path(output), Path(audit_dir)
    if output.exists() or audit_dir.exists():
        raise FileExistsError("기존 모델 평가 결과를 보존합니다. 새로운 출력 경로를 지정하세요.")
    if output.resolve().is_relative_to(audit_dir.resolve()) or audit_dir.resolve().is_relative_to(output.resolve()):
        raise ValueError("결과 문서와 내부 실행 기록은 별도 경로에 저장해야 합니다.")
    policy, _ = load_user_records(root)
    selected = load_selected(root, policy)
    records = selected["companies"]
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("status") != "completed" or summary.get("company_count") != 50 or not summary.get("all_source_hashes_unchanged"):
        raise ValueError("선별 50개 기업의 완료된 모델 평가 기록이 필요합니다.")
    for name, expected in manifest["source_hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("모델 평가 이후 원본 또는 설정이 변경됐습니다.")
    for name, expected in manifest["implementation_sha256"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError("모델 평가에 사용한 구현과 검증 구현이 다릅니다.")
    sanity = json.loads((folder / "sanity_checks.json").read_text())
    if len(sanity) != len(sanity_cases()):
        raise ValueError("기본 방향성 검사 기록이 완전하지 않습니다.")
    for stored, case in zip(sanity, sanity_cases(), strict=True):
        if any(stored.get(k) != v for k, v in case.items()):
            raise ValueError("기능 검사 사례가 바뀌었습니다.")
        high = parse_rating(stored["high_generation"]["answer"], case["high"])
        low = parse_rating(stored["low_generation"]["answer"], case["low"])
        if high != stored["high_ratings"] or low != stored["low_ratings"] or not high[case["criterion_id"]]["score"] > low[case["criterion_id"]]["score"]:
            raise ValueError("방향성 검사 응답 또는 통과 표시가 유효하지 않습니다.")
    results = json.loads((folder / "ratings.json").read_text())
    payload, ratings = build_payload(records, results, digest(folder / "manifest.json"))
    refinement = load_refinement(root, refinement_dir, records, payload, ratings, manifest["candidate"])
    summary = {**summary, "sanity_passed": summary["sanity_passed"] + refinement["sanity_passed"],
               "sanity_count": summary["sanity_count"] + refinement["sanity_count"],
               "role_count": refinement["role_count"], "location_count": refinement["location_count"]}
    config = yaml.safe_load((root / "configs/document_model.yaml").read_text())
    surveyed = load_survey_weights(root / "data/raw/survey_responses.csv", config)
    if not surveyed["counts_match_document_reference"]:
        raise ValueError("설문 원본과 가중치 기준이 다릅니다.")
    weights = {r["criterion_id"]: r["weight"] for r in surveyed["criteria"] if r["active"]}
    calculated = calculate_selection(payload, weights, config["salary"]["baseline"])
    checks = verify_calculation(calculated)
    rows = {r["company_id"]: r for r in calculated["company_results"]}
    source = {r["company_id"]: r for r in records}
    public_rows = [{"rank": order, "company_id": c["company_id"], "company_name": c["company_name"], "region": c["region"],
        "annual_salary_manwon": source[c["company_id"]]["annual_salary_manwon"],
        "model_scores": {k: ratings[c["company_id"]][k]["score"] for k in RATED},
        "salary_index": rows[c["company_id"]]["salary_index"], "composite_index": c["display_index_P"]}
        for order, c in enumerate(calculated["selected_companies"], 1)]
    published = {"result_type": "model_based_evaluation", "label": "모델 평가 결과", "score_type": "model_interpretation",
        "empirical_accuracy_validated": False, "calculation_checked": True, "candidate_count": 50,
        "comparison_count": calculated["eligible_count"], "region_counts": calculated["selected_region_counts"], "top10": public_rows}
    output.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    write_json(output / "top10.json", published)
    (output / "TOP10_선정결과.md").write_text(render_markdown(calculated, records, ratings, summary), encoding="utf-8")
    with (output / "기업별_모델평가.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["기업ID", "기업명", "지역", "제공연봉(만원/년)", "워라밸", "지역인프라", "직무유망성", "AI안전성", "급여지수", "종합지수", "선정여부"])
        selected_ids = {r["company_id"] for r in public_rows}
        for r in records:
            c = rows[r["company_id"]]
            writer.writerow([r["company_id"], r["company_name"], r["region"], r["annual_salary_manwon"],
                *[ratings[r["company_id"]][k]["score"] for k in RATED], c["salary_index"], c["display_index_P"],
                "선정" if r["company_id"] in selected_ids else "보류" if c["display_index_P"] is None else "미선정"])
    write_json(audit_dir / "calculation_input.json", payload)
    write_json(audit_dir / "calculation_result.json", calculated)
    write_json(audit_dir / "checks.json", checks)
    write_json(audit_dir / "applied_ratings.json", ratings)
    write_json(audit_dir / "source_manifest.json", {"rating_manifest_sha256": digest(folder / "manifest.json"),
        "ratings_sha256": digest(folder / "ratings.json"), "document_model_sha256": digest(root / "configs/document_model.yaml"),
        "refinement_manifest_sha256": digest(Path(refinement_dir) / "manifest.json"),
        "refinement_result_hashes": {name: digest(Path(refinement_dir) / name) for name in ("roles.json", "locations.json", "company_balance.json")},
        "source_policy": policy, "implementation_sha256": {name: digest(Path(__file__).parent / name) for name in (
            "export_model_top10.py", "model_company_ratings.py", "model_role_balance.py", "relative_selection.py")}})
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--ratings-dir", type=Path, required=True)
    parser.add_argument("--refinement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.project_root, args.ratings_dir, args.refinement_dir, args.output_dir, args.audit_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    for c in result["top10"]:
        print(f"{c['rank']}위 | {c['company_name']} | {c['region']} | 종합지수 {c['composite_index']:.2f}")
    print("TOP10 선정 결과와 기업별 근거를 마크다운으로 저장했습니다.")


if __name__ == "__main__":
    main()
