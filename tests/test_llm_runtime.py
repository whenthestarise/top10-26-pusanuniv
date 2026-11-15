import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src import llm_runtime


class TestDeviceSelection(unittest.TestCase):
    def test_auto_uses_mps_on_apple_gpu(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            self.assertEqual(llm_runtime.select_device().type, "mps")

    def test_auto_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            self.assertEqual(llm_runtime.select_device().type, "cpu")

    def test_explicit_unavailable_device_fails_clearly(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "--device cpu"):
                llm_runtime.select_device("mps")

    def test_cpu_does_not_call_gpu_memory_or_sync_functions(self):
        device = torch.device("cpu")
        with patch("torch.cuda.synchronize") as cuda_sync, patch(
            "torch.mps.synchronize"
        ) as mps_sync:
            llm_runtime.synchronize(device)
            llm_runtime.reset_memory_stats(device)
            self.assertIsNone(llm_runtime.peak_allocated_gib(device))
            self.assertIsNone(llm_runtime.current_allocated_gib(device))
            cuda_sync.assert_not_called()
            mps_sync.assert_not_called()

    def test_mps_current_memory_is_not_reported_as_peak(self):
        device = torch.device("mps")
        with patch("torch.mps.current_allocated_memory", return_value=1024**3):
            self.assertEqual(llm_runtime.current_allocated_gib(device), 1.0)
            self.assertIsNone(llm_runtime.peak_allocated_gib(device))


class TestLocalQwenLoading(unittest.TestCase):
    def test_save_load_and_generate_without_network(self):
        """실제 Qwen3 모델을 작게 만들어 새 라이브러리의 로딩·추론 경로를 검사합니다."""
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            Qwen3ForCausalLM(config).save_pretrained(directory)
            device = torch.device("cpu")
            model = llm_runtime.load_model(directory, device)
            self.assertEqual(model.dtype, torch.float32)
            self.assertFalse(model.training)
            with torch.inference_mode():
                result = model.generate(
                    torch.tensor([[1, 3, 4]], device=device),
                    attention_mask=torch.ones((1, 3), dtype=torch.long, device=device),
                    max_new_tokens=2,
                    do_sample=False,
                )
            self.assertGreater(result.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
