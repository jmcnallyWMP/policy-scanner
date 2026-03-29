from pydantic import BaseModel
from typing import Optional, Literal

IncludedState = Literal["INHERIT_AGGREGATE", "SUBLIMIT_WITHIN_COVERAGE", "COVERED_WITH_DEDUCTIBLE"]
SectionType = Literal["declarations", "coverage", "endorsement"]
PolicyType = Literal["primary", "excess", "umbrella", "wc", "auto", "cyber", "unknown"]
ClaimsBasis = Literal["claims-made", "occurrence", "unknown"]


class PolicyHeader(BaseModel):
    policy_id: str
    file_name: str
    carrier: str
    named_insured: str
    policy_number: Optional[str] = None
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    policy_type: PolicyType = "unknown"
    claims_made_or_occurrence: ClaimsBasis = "unknown"
    page_count: int
    model_used: str
    processed_at: str
    declarations_source_text: str  # preserved for Phase 2 classifier input


class CoverageElement(BaseModel):
    element_id: str
    policy_id: str
    coverage_name: str
    coverage_section: str
    limit_amount: Optional[float] = None
    aggregate_limit: Optional[float] = None
    deductible_or_retention: Optional[float] = None
    is_sublimit: bool = False
    included_state: Optional[IncludedState] = None  # None = explicit dollar value present
    confidence: float  # deterministic completeness score, not LLM self-reported
    section_type: SectionType


class ExtractionCitation(BaseModel):
    citation_id: str
    element_id: str
    page_number: int
    section: str
    passage: str  # 50-100 char excerpt from source PDF; must never be empty


class ProcessingLog(BaseModel):
    log_id: str
    policy_id: str
    prompt_stage: str
    input_tokens: int
    output_tokens: int
    model_tier: str
    retry_count: int
    cost_estimate_usd: float
    duration_ms: int
    status: Literal["SUCCESS", "FAILED", "TRUNCATED", "RETRIED"]
