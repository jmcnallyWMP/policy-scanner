from policy_scanner.schema import (
    PolicyHeader, CoverageElement, ExtractionCitation, ProcessingLog
)

def test_policy_header_required_fields():
    h = PolicyHeader(
        policy_id="POL-001", file_name="test.pdf", carrier="Chubb",
        named_insured="Acme Corp", policy_number="D95637139",
        effective_date="2025-01-01", expiration_date="2026-01-01",
        policy_type="primary", claims_made_or_occurrence="occurrence",
        page_count=87, model_used="gpt-4o", processed_at="2026-03-28T00:00:00Z",
        declarations_source_text="Declarations page text here"
    )
    assert h.policy_id == "POL-001"
    assert h.declarations_source_text  # must be preserved for Phase 2 classifier

def test_coverage_element_included_state():
    c = CoverageElement(
        element_id="COV-001", policy_id="POL-001",
        coverage_name="General Aggregate",
        coverage_section="Commercial General Liability",
        limit_amount=2000000, aggregate_limit=2000000,
        deductible_or_retention=None, is_sublimit=False,
        included_state=None, confidence=0.97,
        section_type="coverage"
    )
    assert c.included_state is None  # None = explicit dollar value, not "Included"

def test_extraction_citation_required():
    cit = ExtractionCitation(
        citation_id="CIT-001", element_id="COV-001",
        page_number=14, section="coverage_form",
        passage="General Aggregate Limit $2,000,000"
    )
    assert cit.passage  # passage must never be empty

def test_processing_log_cost_estimate():
    log = ProcessingLog(
        log_id="LOG-001", policy_id="POL-001",
        prompt_stage="declarations",
        input_tokens=10000, output_tokens=500,
        model_tier="gpt-4o", retry_count=0,
        cost_estimate_usd=0.030, duration_ms=1200, status="SUCCESS"
    )
    assert log.cost_estimate_usd > 0
