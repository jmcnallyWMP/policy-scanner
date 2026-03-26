# Databricks notebook source
# MAGIC %md
# MAGIC # Insurance Policy PDF Extraction PoC (Databricks Claude / Foundation Model API)
# MAGIC
# MAGIC This notebook runs end to end:
# MAGIC 1. Reads PDF files from a Unity Catalog Volume
# MAGIC 2. Parses each PDF with Databricks `ai_parse_document`
# MAGIC 3. Builds page-aware text for LLM extraction
# MAGIC 4. Saves raw parsed text to a Delta table
# MAGIC 5. Calls a Databricks Foundation Model endpoint (Claude) or a stub for testing
# MAGIC 6. Saves structured outputs and curated policy / coverage tables for Streamlit

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Required Libraries

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType

VOLUME_PATH = "/Volumes/datascience/default/policy_pdfs"
RAW_TEXT_TABLE = "datascience.default.policy_pdf_text"
LLM_OUTPUT_TABLE = "datascience.default.policy_llm_output"
POLICY_TABLE_NAME = "datascience.default.policy_summary"
COVERAGE_TABLE_NAME = "datascience.default.policy_coverages"
EXPOSURE_TABLE_NAME = "datascience.default.policy_exposures"
VIEW_NAME = "datascience.default.v_policy_dashboard"

# Set to False when ready to call Claude in Databricks.
USE_STUB = False

# Databricks Foundation Model endpoint.
DATABRICKS_FM_ENDPOINT = "databricks-claude-opus-4-6"

# Cap how much text is sent to the model for a single file.
MAX_CHARS_TO_SEND = 120000
MAX_PDFS = None
REQUEST_TIMEOUT_SECONDS = 180
AI_PARSE_DOCUMENT_VERSION = "2.0"

print("Configuration loaded")
print(f"Volume: {VOLUME_PATH}")
print(f"Raw text table: {RAW_TEXT_TABLE}")
print(f"LLM output table: {LLM_OUTPUT_TABLE}")
print(f"Policy table: {POLICY_TABLE_NAME}")
print(f"Coverage table: {COVERAGE_TABLE_NAME}")
print(f"Exposure table: {EXPOSURE_TABLE_NAME}")
print(f"Dashboard view: {VIEW_NAME}")
print(f"USE_STUB: {USE_STUB}")
print(f"Databricks FM endpoint: {DATABRICKS_FM_ENDPOINT}")
print(f"ai_parse_document version: {AI_PARSE_DOCUMENT_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify PDFs Exist

# COMMAND ----------

files = dbutils.fs.ls(VOLUME_PATH)
display(files)

# COMMAND ----------

pdf_files = [f for f in files if f.path.lower().endswith(".pdf")]

if MAX_PDFS is not None:
    pdf_files = pdf_files[:MAX_PDFS]

print(f"PDF files found: {len(pdf_files)}")
for f in pdf_files:
    print(f.path)

if not pdf_files:
    raise ValueError(f"No PDF files found in {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse PDFs with ai_parse_document

# COMMAND ----------

import json
from pyspark.sql import functions as F

source_pattern = f"{VOLUME_PATH}/*.pdf"

binary_df = spark.read.format("binaryFile").load(source_pattern)

if MAX_PDFS is not None:
    binary_df = binary_df.limit(MAX_PDFS)

parsed_variant_df = (
    binary_df
    .select(
        F.col("path").alias("file_path"),
        F.element_at(F.split(F.col("path"), "/"), -1).alias("file_name"),
        F.expr(
            f"""ai_parse_document(
                    content,
                    map(
                        'version', '{AI_PARSE_DOCUMENT_VERSION}',
                        'descriptionElementTypes', ''
                    )
                )"""
        ).alias("parsed_doc")
    )
)

# Convert VARIANT to JSON string for inspection/display
parsed_preview_df = parsed_variant_df.select(
    "file_path",
    "file_name",
    F.to_json(F.col("parsed_doc")).alias("parsed_doc_json")
)

display(parsed_preview_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert Parsed Variant Output to Page-Aware Text

# COMMAND ----------

from pyspark.sql import Row

parsed_json_df = parsed_variant_df.select(
    "file_name",
    "file_path",
    F.to_json(F.col("parsed_doc")).alias("parsed_doc_json")
)

parsed_rows = parsed_json_df.collect()


def _safe_page_id(element: dict) -> int | None:
    bbox = element.get("bbox") or []
    if not bbox:
        return None

    first_bbox = bbox[0] or {}
    page_id = first_bbox.get("page_id")
    if page_id is None:
        return None

    try:
        return int(page_id)
    except Exception:
        return None


SKIP_ELEMENT_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
}


def build_policy_text_from_parsed_json(parsed_doc_json: str) -> tuple[str | None, int | None, str | None]:
    try:
        parsed = json.loads(parsed_doc_json or "{}")
    except Exception as e:
        return None, None, f"Failed to parse ai_parse_document JSON: {str(e)}"

    error_status = parsed.get("error_status")
    if error_status:
        return None, None, f"ai_parse_document error_status: {error_status}"

    document = parsed.get("document") or {}
    elements = document.get("elements") or []

    page_parts: dict[int, list[str]] = {}
    max_page_id = 0

    for element in elements:
        if not isinstance(element, dict):
            continue

        element_type = (element.get("type") or "").strip().lower()
        if element_type in SKIP_ELEMENT_TYPES:
            continue

        content = element.get("content")
        if content is None:
            continue

        content = str(content).strip()
        if not content:
            continue

        page_id = _safe_page_id(element)
        if page_id is None or page_id < 1:
            page_id = 1

        max_page_id = max(max_page_id, page_id)
        page_parts.setdefault(page_id, []).append(content)

    if not page_parts:
        return None, None, "No text-like content returned by ai_parse_document"

    ordered_pages = []
    for page_id in sorted(page_parts.keys()):
        page_text = "\n\n".join(page_parts[page_id]).strip()
        ordered_pages.append(
            f"=== PAGE {page_id} START ===\n{page_text}\n=== PAGE {page_id} END ==="
        )

    return "\n\n".join(ordered_pages).strip(), max_page_id or None, None


records = []

for row in parsed_rows:
    policy_text, page_count, error_message = build_policy_text_from_parsed_json(row.parsed_doc_json)
    extraction_status = "SUCCESS" if error_message is None else "FAILED"

    records.append(
        Row(
            file_name=row.file_name,
            file_path=row.file_path,
            page_count=page_count,
            policy_text=policy_text,
            extraction_status=extraction_status,
            error_message=error_message,
        )
    )

raw_text_schema = StructType([
    StructField("file_name", StringType(), True),
    StructField("file_path", StringType(), True),
    StructField("page_count", IntegerType(), True),
    StructField("policy_text", StringType(), True),
    StructField("extraction_status", StringType(), True),
    StructField("error_message", StringType(), True),
])

raw_text_df = spark.createDataFrame(records, schema=raw_text_schema)
display(raw_text_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Extracted Policy Text

# COMMAND ----------

(
    raw_text_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(RAW_TEXT_TABLE)
)

policy_text_df = raw_text_df.filter("extraction_status = 'SUCCESS'").select(
    "file_name",
    "file_path",
    "page_count",
    "policy_text"
)

policy_text_df.createOrReplaceTempView("policy_text_view")

print(f"Saved raw text to {RAW_TEXT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt Template and Model Helpers

# COMMAND ----------

PROMPT_TEMPLATE = """
You are an expert insurance policy analyst. Extract all coverage-related information from the provided insurance document into structured JSON.

🎯 Objective

Identify and extract every distinct insuring element in the policy. A “coverage” is not limited to top-level coverage names—it includes all limits, sublimits, aggregates, deductibles, endorsements, and scheduled variations.

🧠 Definition of Coverage

A coverage includes ANY of the following:

Base coverages

Primary insuring agreements (e.g., Property, General Liability, Business Income)

Sublimits and extensions

Additional Coverages

Coverage Extensions

Special limits within a broader coverage

Limit components (MUST be separate entries)

Each Occurrence

Aggregate limits (all types)

Products-Completed Operations Aggregate

Personal & Advertising Injury

Medical Expense

Damage to Premises Rented to You

Any other named limit type

Deductibles / Retentions (MUST be included)

Flat deductibles

Percentage deductibles

Waiting periods (e.g., 24-hour, 72-hour)

Coverage-specific deductibles (e.g., flood, earthquake)

Scheduled or segmented limits

By building

By location

By premises

By schedule group

By coverage territory

Endorsement-provided or modified coverages

Any coverage, limit, or deductible introduced or modified by endorsement

Scheduled endorsement limits must be extracted individually

“Included” coverages

Items marked as:

“Included”

“Shown as Included”

No explicit limit
→ These MUST still be extracted with null limits

⚠️ Critical Extraction Rules
1. Do NOT collapse multiple limits into one

If a section lists multiple limits for the same coverage, extract each as a separate coverage entry.

Example:

General Liability → extract:

Each Occurrence

General Aggregate

Products-Completed Ops Aggregate

Personal & Advertising Injury

Medical Expense

Damage to Premises

2. Deductibles are REQUIRED

Extract ALL deductibles as separate coverage entries when they are:

Coverage-specific

Listed in a deductible schedule

Time-based (waiting periods)

Percentage-based

3. Split scheduled variations

If limits differ by:

building

location

premises group

schedule

→ Create separate entries for each variation.

4. Endorsements must be fully expanded

For endorsements:

Extract each scheduled item

Extract each limit within the endorsement

Extract any modified limits (even if duplicative of base coverage)

5. Include null-limit coverages

If a coverage is present but:

limit is not shown

marked as “included”

→ include it with:

"limit_amount": null

"aggregate_limit": null

6. Prefer over-extraction to under-extraction

If uncertain whether something qualifies as a coverage:
→ INCLUDE IT

🏷️ Naming Rules

Preserve full specificity of the coverage name

Include qualifiers such as:

“Each Occurrence”

“Aggregate”

“Per Building”

“In Transit”

“At Described Premises”

Do NOT merge distinct entries into a generalized name

📦 Output Schema

Return JSON in the following structure:

{
  "policy": {
    "policy_number": string,
    "carrier": string,
    "named_insured": string,
    "effective_date": string,
    "expiration_date": string,
    "policy_type": string,
    "claims_made_or_occurrence": string | null,
    "confidence": number,
    "page_reference": number[]
  },
  "coverages": [
    {
      "coverage_name": string,
      "coverage_section": string,
      "coverage_type": "base | sublimit | aggregate | deductible | endorsement | extension",
      "limit_amount": number | null,
      "aggregate_limit": number | null,
      "deductible_or_retention": number | null,
      "is_sublimit": boolean,
      "confidence": number,
      "page_reference": number[]
    }
  ],
  "exposures": [
    {
      "exposure_type": string,
      "exposure_value": number,
      "basis": string,
      "confidence": number,
      "page_reference": number[]
    }
  ]
}
🔍 Additional Extraction Requirements

Capture both limit_amount and aggregate_limit when present

If only one exists, populate the other as null

Maintain numeric values (no text like “Included” in numeric fields)

Deductibles should be numeric where possible (convert time-based deductibles to numeric hours)

Include page references for traceability

Assign a confidence score (0–1) based on clarity of extraction

🚫 Do NOT

Do not summarize

Do not group multiple coverages into one

Do not omit deductibles or aggregates

Do not ignore endorsements

Do not skip entries with missing limits

✅ Success Criteria

A correct output will:

Include all base coverages + all subcomponents

Break out every limit, aggregate, and deductible

Fully expand endorsements and schedules

Contain no collapsed entries

Use this guidance to extract a complete and exhaustive representation of all coverages in the document.

POLICY TEXT:
{policy_text}
"""

# COMMAND ----------

import re
from datetime import datetime
import mlflow.deployments


def _strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_string(text: str) -> str:
    text = _strip_code_fences(text)
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        json.loads(candidate)
        return candidate

    raise ValueError(f"Could not isolate valid JSON from model output: {text[:2000]}")


def _repair_truncated_json(text: str) -> str:
    text = _strip_code_fences(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in text")
    text = text[start:]
    text = text.rstrip()
    text = re.sub(r'[,:\s]+$', '', text)
    if text.endswith('"') and text.count('"') % 2 != 0:
        text = text[:text.rfind('"')]
        text = re.sub(r'[,:\s]+$', '', text)

    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            stack.append('}')
        elif ch == '[':
            stack.append(']')
        elif ch in ('}', ']') and stack and stack[-1] == ch:
            stack.pop()

    while stack:
        text += stack.pop()

    json.loads(text)
    return text


def get_deploy_client():
    return mlflow.deployments.get_deploy_client("databricks")


def call_databricks_foundation_model(prompt: str) -> dict:
    client = get_deploy_client()
    return client.predict(
        endpoint=DATABRICKS_FM_ENDPOINT,
        inputs={
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
        },
    )


def extract_content_string(model_response: dict) -> str:
    choices = model_response.get("choices", [])
    if not choices:
        raise ValueError(f"No choices returned by model: {model_response}")

    message = choices[0].get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        combined = "\n".join([p for p in text_parts if p])
        if combined:
            return combined

    raise ValueError(f"Unable to extract text content from model response: {model_response}")


def is_truncated(model_response: dict) -> bool:
    choices = model_response.get("choices", [])
    if not choices:
        return False
    finish_reason = choices[0].get("finish_reason")
    return finish_reason in {"length", "max_tokens"}


def parse_model_json(model_response: dict) -> str:
    content = extract_content_string(model_response)
    truncated = is_truncated(model_response)

    try:
        clean_json = _extract_json_string(content)
        return clean_json
    except Exception:
        if not truncated:
            raise
        print("  -> Response was truncated, attempting JSON repair...")

    repaired = _repair_truncated_json(content)
    print("  -> JSON repair succeeded")
    return repaired


print("Helper functions loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stub Output for Testing

# COMMAND ----------


def stub_extraction(file_name: str, file_path: str, page_count: int | None, policy_text: str) -> dict:
    stub_json = {
        "policy": {
            "policy_number": "STUB-001",
            "carrier": "Stub Carrier",
            "named_insured": "Stub Insured",
            "effective_date": "2025-01-01",
            "expiration_date": "2026-01-01",
            "policy_type": "Primary",
            "claims_made_or_occurrence": "Occurrence",
            "confidence": 0.9,
            "page_reference": [1]
        },
        "coverages": [
            {
                "coverage_name": "Business Personal Property",
                "coverage_section": "Declarations",
                "coverage_type": "base",
                "limit_amount": 1000000,
                "aggregate_limit": None,
                "deductible_or_retention": 5000,
                "is_sublimit": False,
                "confidence": 0.9,
                "page_reference": [1]
            }
        ],
        "exposures": []
    }

    return {
        "file_name": file_name,
        "file_path": file_path,
        "page_count": page_count,
        "request_chars": len(policy_text or ""),
        "model_name": "STUB",
        "extraction_status": "SUCCESS",
        "model_output_json": json.dumps(stub_json),
        "raw_model_response": json.dumps(stub_json),
        "error_message": None,
        "processed_at": datetime.utcnow().isoformat(),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Model Extraction

# COMMAND ----------

rows = policy_text_df.collect()

outputs = []

for r in rows:
    policy_text = (r.policy_text or "")[:MAX_CHARS_TO_SEND]
    prompt = PROMPT_TEMPLATE.replace("{policy_text}", policy_text)
    raw_content = None

    try:
        if USE_STUB:
            result = stub_extraction(
                file_name=r.file_name,
                file_path=r.file_path,
                page_count=r.page_count,
                policy_text=policy_text,
            )
        else:
            print(f"Processing: {r.file_name} ({len(policy_text):,} chars)...")
            model_response = call_databricks_foundation_model(prompt)
            raw_content = extract_content_string(model_response)
            parsed_json = parse_model_json(model_response)
            result = {
                "file_name": r.file_name,
                "file_path": r.file_path,
                "page_count": r.page_count,
                "request_chars": len(policy_text),
                "model_name": DATABRICKS_FM_ENDPOINT,
                "extraction_status": "SUCCESS",
                "model_output_json": parsed_json,
                "raw_model_response": raw_content,
                "error_message": None,
                "processed_at": datetime.utcnow().isoformat(),
            }
            print("  -> SUCCESS")
    except Exception as e:
        print(f"  -> FAILED: {e}")
        result = {
            "file_name": r.file_name,
            "file_path": r.file_path,
            "page_count": r.page_count,
            "request_chars": len(policy_text),
            "model_name": DATABRICKS_FM_ENDPOINT if not USE_STUB else "STUB",
            "extraction_status": "FAILED",
            "model_output_json": None,
            "raw_model_response": raw_content,
            "error_message": str(e),
            "processed_at": datetime.utcnow().isoformat(),
        }

    outputs.append(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Structured Model Output

# COMMAND ----------

from pyspark.sql.types import LongType, DoubleType, BooleanType

llm_output_schema = StructType([
    StructField("file_name", StringType(), True),
    StructField("file_path", StringType(), True),
    StructField("page_count", IntegerType(), True),
    StructField("request_chars", LongType(), True),
    StructField("model_name", StringType(), True),
    StructField("extraction_status", StringType(), True),
    StructField("model_output_json", StringType(), True),
    StructField("raw_model_response", StringType(), True),
    StructField("error_message", StringType(), True),
    StructField("processed_at", StringType(), True),
])

llm_df = spark.createDataFrame(outputs, schema=llm_output_schema)

(
    llm_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(LLM_OUTPUT_TABLE)
)

print(f"Created llm_df with {llm_df.count()} rows")
print(f"Saved model output to {LLM_OUTPUT_TABLE}")
display(llm_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse JSON into Structured Columns for Downstream Analysis

# COMMAND ----------

from pyspark.sql.functions import col, from_json, explode_outer

policy_schema = """
STRUCT<
  policy: STRUCT<
    policy_number: STRING,
    carrier: STRING,
    named_insured: STRING,
    effective_date: STRING,
    expiration_date: STRING,
    policy_type: STRING,
    claims_made_or_occurrence: STRING,
    confidence: DOUBLE,
    page_reference: ARRAY<INT>
  >,
  coverages: ARRAY<STRUCT<
    coverage_name: STRING,
    coverage_section: STRING,
    coverage_type: STRING,
    limit_amount: DOUBLE,
    aggregate_limit: DOUBLE,
    deductible_or_retention: DOUBLE,
    is_sublimit: BOOLEAN,
    confidence: DOUBLE,
    page_reference: ARRAY<INT>
  >>,
  exposures: ARRAY<STRUCT<
    exposure_type: STRING,
    exposure_value: DOUBLE,
    basis: STRING,
    confidence: DOUBLE,
    page_reference: ARRAY<INT>
  >>
>
"""

parsed_df = (
    llm_df
    .filter("extraction_status = 'SUCCESS' AND model_output_json IS NOT NULL")
    .withColumn("parsed_json", from_json(col("model_output_json"), policy_schema))
)

policy_df = parsed_df.select(
    "file_name",
    "file_path",
    "page_count",
    col("parsed_json.policy.policy_number").alias("policy_number"),
    col("parsed_json.policy.carrier").alias("carrier"),
    col("parsed_json.policy.named_insured").alias("named_insured"),
    col("parsed_json.policy.effective_date").alias("effective_date"),
    col("parsed_json.policy.expiration_date").alias("expiration_date"),
    col("parsed_json.policy.policy_type").alias("policy_type"),
    col("parsed_json.policy.claims_made_or_occurrence").alias("claims_made_or_occurrence"),
    col("parsed_json.policy.confidence").alias("policy_confidence"),
    col("parsed_json.policy.page_reference").alias("policy_page_reference"),
    col("processed_at")
)

coverage_df = (
    parsed_df
    .select(
        "file_name",
        "file_path",
        explode_outer(col("parsed_json.coverages")).alias("coverage")
    )
    .select(
        "file_name",
        "file_path",
        col("coverage.coverage_name").alias("coverage_name"),
        col("coverage.coverage_section").alias("coverage_section"),
        col("coverage.coverage_type").alias("coverage_type"),
        col("coverage.limit_amount").alias("limit_amount"),
        col("coverage.aggregate_limit").alias("aggregate_limit"),
        col("coverage.deductible_or_retention").alias("deductible_or_retention"),
        col("coverage.is_sublimit").alias("is_sublimit"),
        col("coverage.confidence").alias("coverage_confidence"),
        col("coverage.page_reference").alias("coverage_page_reference")
    )
)

exposure_df = (
    parsed_df
    .select(
        "file_name",
        "file_path",
        explode_outer(col("parsed_json.exposures")).alias("exposure")
    )
    .select(
        "file_name",
        "file_path",
        col("exposure.exposure_type").alias("exposure_type"),
        col("exposure.exposure_value").alias("exposure_value"),
        col("exposure.basis").alias("basis"),
        col("exposure.confidence").alias("exposure_confidence"),
        col("exposure.page_reference").alias("exposure_page_reference")
    )
)

display(policy_df)
display(coverage_df)
display(exposure_df)

# COMMAND ----------

CATALOG = "datascience"
SCHEMA = "default"

POLICY_TABLE_NAME = f"{CATALOG}.{SCHEMA}.policy_summary"
COVERAGE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.policy_coverages"
LLM_TABLE_NAME = f"{CATALOG}.{SCHEMA}.policy_llm_output_v2"

policy_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(POLICY_TABLE_NAME)
coverage_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(COVERAGE_TABLE_NAME)
llm_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(LLM_TABLE_NAME)

print(f"Saved: {POLICY_TABLE_NAME}, {COVERAGE_TABLE_NAME}, {LLM_TABLE_NAME}")

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE VIEW datascience.default.v_policy_dashboard AS
SELECT
    p.file_name,
    p.file_path,
    p.policy_number,
    p.carrier,
    p.named_insured,
    p.effective_date,
    p.expiration_date,
    p.policy_type,
    p.claims_made_or_occurrence,
    p.policy_confidence,
    COUNT(c.coverage_name) AS coverage_count
FROM datascience.default.policy_summary p
LEFT JOIN datascience.default.policy_coverages c
    ON p.file_name = c.file_name
   AND p.file_path = c.file_path
GROUP BY
    p.file_name,
    p.file_path,
    p.policy_number,
    p.carrier,
    p.named_insured,
    p.effective_date,
    p.expiration_date,
    p.policy_type,
    p.claims_made_or_occurrence,
    p.policy_confidence
""")
