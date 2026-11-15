from copy import deepcopy
import unittest

from src.classify_pdf_job_references import reference_claim, verify_reference_claim


class TestPdfJobReference(unittest.TestCase):
    def setUp(self):
        self.pdf = {"company_id": "document:090", "source_page": 2, "evaluated_job": "가상 일부 직무",
                    "location": "가상 위치", "raw_fields": {"K": "담당자는 원문\n업무를 수행한다."}}
        self.selected = {"company_id": "selected50:001", "catalog_link": {"company_id": "document:090"},
                         "pdf_reference": self.pdf}
        self.policy = {"raw_sources": {"pdf": {"sha256": "가상해시"}}}

    def test_pdf_job_reference_keeps_entire_original_and_separate_scope(self):
        claim = reference_claim(self.selected, self.pdf, self.policy)
        verify_reference_claim(claim, self.selected, self.pdf, self.policy)
        self.assertEqual(claim["quote"], self.pdf["raw_fields"]["K"])
        self.assertEqual(claim["reference_target_job"], "가상 일부 직무")
        self.assertEqual(claim["source_scope"], "provided_pdf_partial_job_reference")

    def test_wrong_parent_or_changed_job_scope_cannot_be_merged(self):
        wrong = deepcopy(self.selected)
        wrong["catalog_link"]["company_id"] = "document:001"
        with self.assertRaises(ValueError):
            reference_claim(wrong, self.pdf, self.policy)
        claim = reference_claim(self.selected, self.pdf, self.policy)
        claim["reference_target_job"] = "회사 전체 직무"
        with self.assertRaises(ValueError):
            verify_reference_claim(claim, self.selected, self.pdf, self.policy)
