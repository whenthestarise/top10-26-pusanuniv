from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = ROOT / "configs" / "criteria.yaml"
SURVEY_FILE = ROOT / "data" / "raw" / "survey_responses.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

QUESTION = (
    "귀하가 취업할 당시 중요하게 여기는 항목이 무엇입니까?(3개 체크)"
)

# 설문 문구를 프로젝트의 평가 기준 ID로 연결합니다.
LABEL_TO_ID = {
    "높은 급여": "salary",
    "확보된 근무 지역 인프라": "infrastructure",
    "확보된 근무지역 인프라": "infrastructure",
    "워라벨": "work_life_balance",
    "워라밸": "work_life_balance",
    "AI대체 가능성": "ai_resilience",
    "AI 대체 가능성": "ai_resilience",
    "직무 유망성": "job_outlook",
    "수평적 조직문화": "horizontal_culture",
    "수평적 조직 문화": "horizontal_culture",
    "경력 확보": "career_capital",
}


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def load_config() -> tuple[dict, dict, float, float, float]:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("criteria.yaml 내용을 확인하세요.")

    calculation = config.get("criterion_weight_calculation", {})

    if calculation.get("method") != "min_max_selection_count":
        raise ValueError(
            "criterion_weight_calculation.method를 "
            "min_max_selection_count로 설정하세요."
        )

    if calculation.get("basis") != "selection_count":
        raise ValueError("basis는 selection_count여야 합니다.")

    criteria = config["criteria"]
    criterion_ids = [item["id"] for item in criteria]

    if len(criterion_ids) != 7 or len(set(criterion_ids)) != 7:
        raise ValueError("중복 없는 7개 평가 기준이 필요합니다.")

    if set(criterion_ids) != set(LABEL_TO_ID.values()):
        raise ValueError("YAML의 평가 기준 ID와 설문 매핑이 다릅니다.")

    names = {
        item["id"]: item["name"]
        for item in criteria
    }

    bounds = config["criterion_weight_bounds"]
    lower = float(bounds["minimum"])
    upper = float(bounds["maximum"])
    equal_weight = float(calculation["equal_counts_weight"])

    if not all(math.isfinite(v) for v in [lower, upper, equal_weight]):
        raise ValueError("가중치 설정은 유한한 숫자여야 합니다.")

    if not (0 < lower < upper and lower <= equal_weight <= upper):
        raise ValueError("가중치 범위와 동률 가중치를 확인하세요.")

    return config, names, lower, upper, equal_weight


def load_responses() -> tuple[list[set[str]], dict]:
    survey = pd.read_csv(
        SURVEY_FILE,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    survey.columns = [
        normalize_text(column)
        for column in survey.columns
    ]

    if QUESTION not in survey.columns:
        raise ValueError(
            "설문 질문 열을 찾을 수 없습니다.\n"
            f"필요한 열: {QUESTION}\n"
            f"현재 열: {survey.columns.tolist()}"
        )

    responses = []
    selection_counts = {}
    unknown_labels = set()
    duplicate_item_rows = 0

    for answer in survey[QUESTION]:
        labels = [
            normalize_text(part)
            for part in answer.split(",")
            if normalize_text(part)
        ]

        ids = []

        for label in labels:
            if label not in LABEL_TO_ID:
                unknown_labels.add(label)
            else:
                ids.append(LABEL_TO_ID[label])

        # 같은 응답에서 동일 항목을 반복해도 한 번만 집계합니다.
        selected = set(ids)

        if len(ids) != len(selected):
            duplicate_item_rows += 1

        responses.append(selected)

        size_key = str(len(selected))
        selection_counts[size_key] = (
            selection_counts.get(size_key, 0) + 1
        )

    if unknown_labels:
        raise ValueError(
            "매핑되지 않은 설문 선택지가 있습니다. "
            "LABEL_TO_ID를 확인하세요: "
            + ", ".join(sorted(unknown_labels))
        )

    audit = {
        "response_row_count": len(survey),
        "selection_count_distribution": selection_counts,
        "rows_with_repeated_items": duplicate_item_rows,
        "blank_answer_rows": sum(
            not selected for selected in responses
        ),
    }

    return responses, audit


def calculate_weights(
    responses: list[set[str]],
    names: dict,
    lower: float,
    upper: float,
    equal_weight: float,
) -> pd.DataFrame:
    if not responses or not any(responses):
        raise ValueError("집계할 유효한 선택이 없습니다.")

    counts = {
        criterion_id: sum(
            criterion_id in selected
            for selected in responses
        )
        for criterion_id in names
    }

    min_count = min(counts.values())
    max_count = max(counts.values())
    total_selections = sum(counts.values())

    rows = []

    for criterion_id, count in counts.items():
        if max_count == min_count:
            weight = equal_weight
        else:
            weight = lower + (upper - lower) * (
                (count - min_count) / (max_count - min_count)
            )

        rows.append({
            "criterion_id": criterion_id,
            "criterion_name": names[criterion_id],
            "selection_count": count,
            "selection_share": count / total_selections,
            "criterion_weight": weight,
        })

    result = pd.DataFrame(rows)

    if not result["criterion_weight"].between(
        lower - 1e-12,
        upper + 1e-12,
    ).all():
        raise ValueError("계산된 가중치가 설정 범위를 벗어났습니다.")

    # 표시 순서만 정렬합니다. 순위로 가중치를 계산하지 않습니다.
    return result.sort_values(
        ["selection_count", "criterion_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def save_result(
    mode: str,
    responses: list[set[str]],
    table: pd.DataFrame,
    audit: dict,
    config: dict,
    source_hash: str,
) -> None:
    csv_file = OUTPUT_DIR / f"survey_weights_{mode}.csv"
    yaml_file = OUTPUT_DIR / f"survey_weights_{mode}.yaml"

    # 반올림하지 않은 계산값을 저장합니다.
    table.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "schema_version": "1.0",
        "source_file": SURVEY_FILE.name,
        "source_sha256": source_hash,
        "question": QUESTION,
        "response_filter": mode,
        "included_response_count": len(responses),
        "total_selections": int(table["selection_count"].sum()),
        "source_audit": audit,
        "calculation": {
            "method": config["criterion_weight_calculation"]["method"],
            "minimum": float(config["criterion_weight_bounds"]["minimum"]),
            "maximum": float(config["criterion_weight_bounds"]["maximum"]),
            "equal_counts_weight": float(
                config["criterion_weight_calculation"]["equal_counts_weight"]
            ),
        },
        "selection_counts": {
            row["criterion_id"]: int(row["selection_count"])
            for row in table.to_dict(orient="records")
        },
        "weights": {
            row["criterion_id"]: float(row["criterion_weight"])
            for row in table.to_dict(orient="records")
        },
    }

    with yaml_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            payload,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"\n분석 기준: {mode}")
    print(f"포함 응답 수: {len(responses)}")
    print(f"총 선택 수: {payload['total_selections']}")

    display = table[
        ["criterion_name", "selection_count", "criterion_weight"]
    ].copy()

    display["criterion_weight"] = display["criterion_weight"].round(4)
    print(display.to_string(index=False))
    print(f"가중치 합계: {table['criterion_weight'].sum():.6f}")
    print(f"저장: {csv_file.name}, {yaml_file.name}")


def main() -> None:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {CONFIG_FILE}")

    if not SURVEY_FILE.exists():
        raise FileNotFoundError(f"설문 파일이 없습니다: {SURVEY_FILE}")

    config, names, lower, upper, equal_weight = load_config()
    responses, audit = load_responses()

    groups = {
        # 빈 응답은 제외하고, 나머지는 선택 개수와 무관하게 포함합니다.
        "all": [
            selected for selected in responses
            if selected
        ],
        "exactly_three": [
            selected for selected in responses
            if len(selected) == 3
        ],
    }

    # 두 분석 모두 계산 가능한지 확인한 뒤 저장합니다.
    results = {
        mode: calculate_weights(
            selected_responses,
            names,
            lower,
            upper,
            equal_weight,
        )
        for mode, selected_responses in groups.items()
    }

    source_hash = hashlib.sha256(SURVEY_FILE.read_bytes()).hexdigest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode, table in results.items():
        save_result(
            mode,
            groups[mode],
            table,
            audit,
            config,
            source_hash,
        )

    print("\n설문 가중치 계산이 완료되었습니다.")
    print("두 결과를 비교한 뒤 최종 적용할 분석 기준을 선택하세요.")


if __name__ == "__main__":
    main()