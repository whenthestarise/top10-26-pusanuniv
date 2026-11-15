from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.relative_selection import CRITERIA
from src.run_relative_selection import adapt_assessment, run
from src.selection_function_cases import regional_case


ROOT = Path(__file__).resolve().parents[1]


class TestSelectionCommand(unittest.TestCase):
    def test_synthetic_cli_writes_ten_with_regions_and_never_real_final_list(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "synthetic.json"
            source.write_text(json.dumps(regional_case(), ensure_ascii=False))
            result = run(ROOT, folder / "output", source)
            self.assertEqual(len(result["selected_companies"]), 10)
            self.assertEqual(result["selected_region_counts"], {"부산": 8, "울산": 1, "경남": 1})
            self.assertFalse(result["real_selection_confirmed"])
            self.assertIsNone(result["final_top10"])
            stored = json.loads((folder / "output/selection_result.json").read_text())
            self.assertEqual(stored, result)
            self.assertEqual((folder / "output/기업별_선정계산.csv").read_bytes()[:3], b"\xef\xbb\xbf")

    def test_external_real_payload_cannot_bypass_source_rebuilding(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.json", Path(temp) / "out"
            payload = regional_case()
            payload["dataset_type"] = "real"
            source.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                run(ROOT, output, source)
            self.assertFalse(output.exists())

    def test_duplicate_json_fields_are_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.json", Path(temp) / "out"
            source.write_text('{"dataset_type":"real","dataset_type":"synthetic"}')
            with self.assertRaises(ValueError):
                run(ROOT, output, source)
            self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            marker = folder / "유지.txt"
            marker.write_text("기존 파일")
            with self.assertRaises(FileExistsError):
                run(ROOT, folder)
            self.assertEqual(marker.read_text(), "기존 파일")

    def test_real_adapter_keeps_missing_scores_and_does_not_infer_common_job(self):
        table = {"dataset_type": "real", "weights": [{"criterion_id": k, "weight": 1} for k in CRITERIA], "companies": []}
        for i in range(1, 51):
            table["companies"].append({"company_id": f"selected50:{i:03d}", "company_name": f"가상 검증 {i}",
                "region": "부산", "deferral_reasons": ["자료 부족"],
                "criteria": {k: {"numeric_value": 7000 if k == "salary" else None, "raw_score": None,
                                  "sources": [{"quote": "가상 원문"}]} for k in CRITERIA}})
        original = deepcopy(table)
        result = adapt_assessment(table, {"functional_scoring_checks_passed": False, "company_scoring_enabled": False})
        self.assertEqual(table, original)
        self.assertIsNone(result["comparison_cohort"])
        self.assertIsNone(result["companies"][0]["values"]["ai_resilience"])
        self.assertIsNone(result["companies"][0]["target_job"])
        self.assertFalse(result["validation_gates"]["score_validation"])
        self.assertFalse(result["validation_gates"]["comparison_scope"])
        result["companies"][0]["evidence"]["salary"][0]["quote"] = "복사 후 변경"
        self.assertEqual(table, original)
