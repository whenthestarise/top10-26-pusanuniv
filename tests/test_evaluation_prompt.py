import json
from pathlib import Path
import unittest

import yaml

from src.evaluate_company_local import PROMPT_VERSION, build_output_contract, build_user_prompt


class TestEvaluationPrompt(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = yaml.safe_load((root / "configs/ai_evaluation.yaml").read_text())
        self.context = {"company_id": "TEST-001", "company_name": "가상기업",
                        "workplace": "가상사업장", "target_job": "설비 정비"}

    def test_contract_has_constraints_without_prefilled_evaluation(self):
        prompt = build_user_prompt(self.config, self.context, [])
        self.assertIn("output_contract", prompt)
        self.assertNotIn('"status": "insufficient_evidence"', prompt)
        self.assertNotIn('"score": null', prompt)
        self.assertNotIn('"reason": "자료가 부족합니다."', prompt)
        contract = build_output_contract(self.config, self.context)
        self.assertEqual(contract["company_id"], "TEST-001")
        self.assertEqual(set(contract["criteria"]), set(self.config["criteria"]))
        self.assertNotIn("required_fields", contract)
        self.assertEqual(PROMPT_VERSION, "evaluation_v2_no_prefilled_answer")

    def test_source_text_remains_reference_data(self):
        sources = [{"source_id": "SYNTHETIC-001", "text": "업무시간의 90%는 현장 정비다."}]
        prompt = build_user_prompt(self.config, self.context, sources)
        payload = prompt.split("\n입력 자료:\n", 1)[1].split("\n출력 형식 명세", 1)[0]
        self.assertEqual(json.loads(payload), {"context": self.context, "sources": sources})


if __name__ == "__main__":
    unittest.main()
