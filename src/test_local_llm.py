from __future__ import annotations

import json
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)


MODEL_ID = "Qwen/Qwen3-1.7B"

# 실제 기업 정보가 아닌 실행 시험용 가상 자료입니다.
DOCUMENT = """
[문서 ID: TEST-001]
회사명: 테스트기업 A
근무지역: 부산
2025년 평균연봉: 5,200만원
주요 채용직무: 설계, 품질관리
교육제도: 신입사원 직무교육 제공
"""

SYSTEM_PROMPT = """
너는 기업 자료 분석 보조자다.
제공된 자료만 근거로 답한다.
자료에 없는 정보는 추정하지 않고 반드시 "자료 부족"으로 표시한다.
자료에 없는 사실이나 출처를 만들어내지 않는다.
답변은 설명이나 코드 블록 없이 JSON 객체 하나만 출력한다.
"""

USER_PROMPT = f"""
다음 자료를 읽고 요청한 정보를 추출하라.

<자료>
{DOCUMENT}
</자료>

다음 4개 키를 가진 JSON 객체를 출력하라.
- company_name: 회사명 문자열
- average_salary_manwon: 평균연봉 숫자. 단위는 만원
- weekly_working_hours: 실제 주당 근로시간. 없으면 "자료 부족"
- source_id: 자료에 표시된 문서 ID
"""


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch가 GPU를 인식하지 못합니다.")

    set_seed(42)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"전체 VRAM: {total_bytes / 1024**3:.2f} GiB")
    print(f"사용 가능한 VRAM: {free_bytes / 1024**3:.2f} GiB")
    print(f"모델: {MODEL_ID}")
    print("토크나이저와 모델을 불러옵니다. 첫 실행은 다운로드가 필요합니다.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()

    print("모델 로딩 완료")
    print(f"4비트 로딩: {getattr(model, 'is_loaded_in_4bit', False)}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    if inputs["input_ids"].shape[1] > 1024:
        raise ValueError("첫 시험 입력은 1,024토큰 이하로 제한합니다.")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated = output[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    print("\n모델 응답:")
    print(answer)
    print(f"\n생성 시간: {elapsed:.2f}초")
    print(f"생성 토큰 수: {generated.numel()}")
    print(
        "PyTorch 최대 할당 메모리: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
    )

        # 원래 응답은 유지하고, 전체를 감싼 코드 블록만 제거합니다.
    json_text = answer.strip()
    lines = json_text.splitlines()

    wrapped = (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```json", "```"}
        and lines[-1].strip() == "```"
    )

    if wrapped:
        json_text = "\n".join(lines[1:-1]).strip()
        print("\n출력 형식: Markdown 코드 블록 제거 후 검사")
    else:
        print("\n출력 형식: 원본 응답 그대로 검사")

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as error:
        print(f"\n검사 결과: JSON 형식 불일치 — {error.msg}")
        print("추론은 실행됐지만 출력 형식을 개선해야 합니다.")
        return

    if not isinstance(parsed, dict):
        print("\n검사 결과: JSON 객체가 아닙니다.")
        return

    expected = {
        "company_name": "테스트기업 A",
        "average_salary_manwon": 5200,
        "weekly_working_hours": "자료 부족",
        "source_id": "TEST-001",
    }

    checks = {
        key: parsed.get(key) == value
        for key, value in expected.items()
    }

    print("\n항목별 검사:")
    for key, passed in checks.items():
        print(f"- {key}: {'통과' if passed else '확인 필요'}")

    if all(checks.values()):
        print("\n이번 기본 추출 시험을 통과했습니다.")
    else:
        print("\n추론은 실행됐지만 일부 추출 결과가 정답과 다릅니다.")


if __name__ == "__main__":
    main()