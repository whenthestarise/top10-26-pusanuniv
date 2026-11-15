"""고정 버전으로 내려받은 맥 전용 모델을 같은 평가 인터페이스로 실행합니다."""

import os
from pathlib import Path
import time

if __package__:
    from .extract_ai_evidence import choice_trie, next_choice_tokens
else:
    from extract_ai_evidence import choice_trie, next_choice_tokens


class MlxGenerator:
    def __init__(self, model_id, revision, hf_home=None):
        if hf_home:
            os.environ["HF_HOME"] = str(Path(hf_home).expanduser().resolve())
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm import load, stream_generate
        self.mx, self.stream_generate = mx, stream_generate
        self.model_id, self.revision = model_id, revision
        if not revision or len(revision) != 40:
            raise ValueError("모델의 고정된 버전 식별자가 필요합니다.")
        path = snapshot_download(model_id, revision=revision, local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "README.md"])
        print("내려받은 후보 모델을 읽습니다.", flush=True)
        self.model, self.tokenizer = load(path, tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
        mx.eval(self.model.parameters())
        self.max_peak_gib = round(mx.get_active_memory() / 1024**3, 3)

    def choose(self, messages, allowed_responses):
        return self(messages, allowed_responses=allowed_responses)

    def __call__(self, messages, allowed_responses=None, max_tokens=32):
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        token_count = len(tokens)
        if token_count > 4096:
            raise ValueError("입력이 제한을 넘었습니다. 자료를 자르지 않고 중단합니다.")
        options = {}
        if allowed_responses is not None:
            eos = self.tokenizer.eos_token_id
            trie = choice_trie([self.tokenizer.encode(prefix + label, add_special_tokens=False)
                               for prefix in ("", " ", "\n", "\n\n") for label in sorted(allowed_responses)], eos)
            # 허용되지 않은 위치는 음의 무한대로 유지하고 허용 위치에 원래 점수를 놓습니다.
            def processor(history, logits):
                allowed = next_choice_tokens(trie, history[token_count:].tolist())
                indices = self.mx.array(allowed)
                mask = self.mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
                mask[:, indices] = logits[:, indices]
                return mask
            options["logits_processors"] = [processor]
        self.mx.synchronize()
        self.mx.reset_peak_memory()
        started = time.perf_counter()
        pieces, last = [], None
        for response in self.stream_generate(self.model, self.tokenizer, prompt=tokens,
                                               max_tokens=max_tokens, **options):
            pieces.append(response.text)
            last = response
        self.mx.synchronize()
        self.max_peak_gib = max(self.max_peak_gib, round(self.mx.get_peak_memory() / 1024**3, 3))
        return {"answer": "".join(pieces).strip(), "model_id": self.model_id,
                "model_revision": self.revision, "backend": "mlx", "device": "metal",
                "input_tokens": token_count, "generated_tokens": last.generation_tokens if last else 0,
                "generation_seconds": round(time.perf_counter() - started, 3),
                "current_allocated_gib": round(self.mx.get_active_memory() / 1024**3, 3),
                "peak_allocated_gib": round(self.mx.get_peak_memory() / 1024**3, 3),
                "finish_reason": last.finish_reason if last else None}
