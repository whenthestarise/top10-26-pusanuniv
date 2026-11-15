from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.data_policy import build_user_bundle, load_user_records, validate_user_sources


class TestProvidedSourcesOnly(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ("configs", "data/raw", "data/processed"):
            (self.root / folder).mkdir(parents=True)
        raw = {}
        for key in ("companies", "survey", "pdf"):
            filename = key + ".txt"
            path = self.root / "data/raw" / filename
            path.write_text("기능 검사 전용 가상 자료 " + key)
            raw[key] = {"original_name": filename, "stored_name": filename,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.record = {"company_id": "SYNTHETIC-POLICY", "company_name": "가상 기업",
                       "location": "가상 위치", "evaluated_job": "가상 직무", "source_page": 1,
                       "reference_date": "2026-11-15", "raw_fields": {"K": "기능 검사 전용 가상 원문"}}
        data = self.root / "data/processed/document_companies.json"
        data.write_text(json.dumps({"source_sha256": raw["pdf"]["sha256"], "companies": [self.record]}, ensure_ascii=False))
        self.policy = {"raw_sources": raw, "processed_companies_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
        (self.root / "configs/data_policy.json").write_text(json.dumps(self.policy))
        self.bundle = build_user_bundle(self.record, self.policy)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_document_field_is_accepted(self):
        validate_user_sources(self.bundle, self.root)

    def test_added_text_external_url_or_false_provenance_is_rejected(self):
        for updates in ({"text": "외부에서 가져온 내용"}, {"locator": "https://external.invalid/job"},
                        {"source_kind": "official_job_description"}, {"document_field": "Z"},
                        {"source_document_sha256": "다른 해시"}, {"document_page": 2}):
            bundle = deepcopy(self.bundle)
            bundle["sources"][0].update(updates)
            with self.assertRaises(ValueError):
                validate_user_sources(bundle, self.root)

    def test_unknown_company_or_job_cannot_enter_real_evaluation(self):
        for key in ("company_id", "company_name", "target_job", "workplace"):
            bundle = deepcopy(self.bundle)
            bundle["context"][key] = "다른 값"
            with self.assertRaises(ValueError):
                validate_user_sources(bundle, self.root)

    def test_modified_original_or_processed_records_are_rejected(self):
        path = self.root / "data/raw/pdf.txt"
        original = path.read_bytes()
        path.write_text("바뀐 원본")
        with self.assertRaises(ValueError):
            load_user_records(self.root)
        path.write_bytes(original)
        path = self.root / "data/processed/document_companies.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaises(ValueError):
            load_user_records(self.root)

    def test_synthetic_function_checks_are_allowed_without_real_source_files(self):
        validate_user_sources({"dataset_type": "synthetic"}, Path("/없는/경로"))

    def test_malformed_sources_fail_without_loading_a_model(self):
        with self.assertRaises(ValueError):
            validate_user_sources({"dataset_type": "real", "sources": ["잘못된 출처"]}, self.root)
