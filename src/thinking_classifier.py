"""내부 검토를 마친 뒤 허용된 분류명만 반환하는 비교용 실행기입니다."""

import time

if __package__:
    from .extract_ai_evidence import LocalGenerator, choice_trie, next_choice_tokens
else:
    from extract_ai_evidence import LocalGenerator, choice_trie, next_choice_tokens


class ThinkingClassifier(LocalGenerator):
    def choose(self, messages, allowed_responses):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        # 내부 검토 내용은 출력하거나 기록하지 않고 최종 분류명만 복호화합니다.
        prompt += "<think>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        token_count = inputs["input_ids"].shape[1]
        if token_count > 4096:
            raise ValueError("입력이 제한을 넘었습니다. 자료를 자르지 않고 중단합니다.")
        close = self.tokenizer.convert_tokens_to_ids("</think>")
        if close is None or close == self.tokenizer.unk_token_id:
            raise ValueError("모델의 내부 검토 종료 토큰을 확인할 수 없습니다.")
        inputs = inputs.to(self.device)
        self.runtime.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            reviewed = self.model.generate(
                **inputs, max_new_tokens=384, do_sample=False,
                eos_token_id=[close, self.tokenizer.eos_token_id],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_review_tokens = int(reviewed.shape[1] - token_count)
        closing_positions = (reviewed[0, token_count:] == close).nonzero()
        completed = bool(closing_positions.numel())
        reasoning_tokens = int(closing_positions[0, 0]) + 1 if completed else generated_review_tokens
        # 종료 판정이 늦어 추가 생성된 토큰이 있어도 검토 종료 뒤의 내용은 사용하지 않습니다.
        reviewed = reviewed[:, :token_count + reasoning_tokens]
        answer, final_tokens = "", 0
        if completed:
            final_start = reviewed.shape[1]
            trie = choice_trie([self.tokenizer.encode(prefix + label, add_special_tokens=False)
                                for prefix in ("", " ", "\n", "\n\n") for label in sorted(allowed_responses)],
                               self.tokenizer.eos_token_id)
            with self.torch.inference_mode():
                output = self.model.generate(
                    input_ids=reviewed, attention_mask=self.torch.ones_like(reviewed),
                    max_new_tokens=32, do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=lambda _, sequence: next_choice_tokens(trie, sequence[final_start:].tolist()),
                )
            final = output[0, final_start:]
            final_tokens = int(final.numel())
            answer = self.tokenizer.decode(final, skip_special_tokens=True).strip()
        self.runtime.synchronize(self.device)
        return {"answer": answer, "model_id": "Qwen/Qwen3-1.7B", "device": str(self.device),
                "input_tokens": token_count, "generated_tokens": generated_review_tokens + final_tokens,
                "reasoning_tokens": reasoning_tokens, "reasoning_completed": completed,
                "discarded_after_review_tokens": generated_review_tokens - reasoning_tokens,
                "generation_seconds": round(time.perf_counter() - started, 3),
                "current_allocated_gib": self.runtime.current_allocated_gib(self.device),
                "classification_mode": "review_then_constrained_label"}
