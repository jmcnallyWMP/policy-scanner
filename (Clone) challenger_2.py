# Databricks notebook source
# MAGIC %md
# MAGIC # Insurance Policy PDF Extraction PoC (Databricks Claude / Foundation Model API)
# MAGIC
# MAGIC This notebook runs end to end:
# MAGIC 1. Reads PDF files from a Unity Catalog Volume
# MAGIC 2. Extracts page-aware text from each PDF
# MAGIC 3. Saves raw text to a Delta table
# MAGIC 4. Calls a Databricks Foundation Model endpoint (Claude) or a stub for testing
# MAGIC 5. Saves structured outputs and run metadata to a Delta table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Required Libraries

# COMMAND ----------

#%pip install streamlit

# COMMAND ----------

# MAGIC %pip install pypdf mlflow

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

# Set to False when ready to call Claude in Databricks.
USE_STUB = False

# Databricks Foundation Model endpoint.
# Replace with the exact endpoint you see under Serving.
# Common examples:
# - anthropic-claude-3-7-sonnet
# - anthropic-claude-3-sonnet
# - anthropic-claude-3-haiku
DATABRICKS_FM_ENDPOINT = "databricks-claude-opus-4-6"

# Cap how much text is sent to the model for a single file.
# Increase only if your endpoint/model context window supports it.
MAX_CHARS_TO_SEND = 120000
MAX_PDFS = None
REQUEST_TIMEOUT_SECONDS = 180


print("Configuration loaded")
print(f"Volume: {VOLUME_PATH}")
print(f"Raw text table: {RAW_TEXT_TABLE}")
print(f"LLM output table: {LLM_OUTPUT_TABLE}")
print(f"USE_STUB: {USE_STUB}")
print(f"Databricks FM endpoint: {DATABRICKS_FM_ENDPOINT}")

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
# MAGIC ## Define PDF Text Extraction Function

# COMMAND ----------

from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> tuple[str, int]:
    """Extract text from a PDF and preserve page boundaries for downstream page reference logic."""
    reader = PdfReader(pdf_path.replace("dbfs:/", "/", 1))
    text_parts = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            page_text = f"[PAGE ERROR: {str(e)}]"

        text_parts.append(f"\n\n=== PAGE {i} START ===\n{page_text}\n=== PAGE {i} END ===")

    return "\n".join(text_parts).strip(), len(reader.pages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Text from PDFs

# COMMAND ----------

from pyspark.sql import Row

records = []

for f in pdf_files:
    try:
        text, page_count = extract_text_from_pdf(f.path)
        records.append(
            Row(
                file_name=f.name,
                file_path=f.path,
                page_count=page_count,
                policy_text=text,
                extraction_status="SUCCESS",
                error_message=None,
            )
        )
        print(f"Processed: {f.name}")
    except Exception as e:
        records.append(
            Row(
                file_name=f.name,
                file_path=f.path,
                page_count=None,
                policy_text=None,
                extraction_status="FAILED",
                error_message=str(e),
            )
        )
        print(f"Failed: {f.name} -> {str(e)}")

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

policy_text_df = raw_text_df.select("file_name", "file_path", "policy_text")

policy_text_df.createOrReplaceTempView("policy_text_view")

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

# DBTITLE 1,Cell 19
import json
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
    """Best-effort JSON extraction for models that sometimes wrap output with extra text."""
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
    """Attempt to repair truncated JSON by closing unclosed braces and brackets."""
    text = _strip_code_fences(text)
    # Find the outermost JSON object
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in text")
    text = text[start:]

    # Remove any trailing partial key/value (incomplete string or number)
    # Trim back to the last complete value
    text = text.rstrip()
    # Remove trailing comma or colon that would make JSON invalid
    text = re.sub(r'[,:\s]+$', '', text)
    # If the last char is a quote that starts an incomplete string, remove it
    if text.endswith('"') and text.count('"') % 2 != 0:
        text = text[:text.rfind('"')]
        text = re.sub(r'[,:\s]+$', '', text)

    # Count unclosed braces and brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')

    # Close them in reverse order by scanning what was opened
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

    # Append closing characters in reverse order
    closing = ''.join(reversed(stack))
    repaired = text + closing

    # Validate the repair worked
    parsed = json.loads(repaired)
    return json.dumps(parsed)


def extract_content_string(model_response: dict) -> str:
    """Extract just the content text from a model response (for clean storage)."""
    if isinstance(model_response, dict):
        choices = model_response.get("choices")
        if choices and isinstance(choices, list):
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message") or {}
                if isinstance(message, dict):
                    content = message.get("content")
                    if content:
                        return content
        for key in ["output_text", "text", "prediction", "predictions", "candidates"]:
            val = model_response.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, list) and len(val) > 0:
                first = val[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    content = first.get("text") or first.get("content")
                    if content:
                        return content
    return json.dumps(model_response)


def is_truncated(model_response: dict) -> bool:
    """Check if the model response was truncated due to max_tokens."""
    if isinstance(model_response, dict):
        # Check finish_reason in choices
        choices = model_response.get("choices", [])
        if choices and isinstance(choices, list):
            finish_reason = choices[0].get("finish_reason", "")
            if finish_reason == "length":
                return True
        # Check usage
        usage = model_response.get("usage", {})
        if isinstance(usage, dict):
            completion = usage.get("completion_tokens", 0)
            # If completion tokens equals or is very close to max_tokens, likely truncated
            if completion >= 15900:  # Close to our 16000 limit
                return True
    return False


def stub_extraction(file_name: str, file_path: str, page_count: int, policy_text: str) -> dict:
    """Stub output so the notebook can be tested end to end without calling the model."""
    stub_json = {
        "policy": {
            "policy_number": None,
            "carrier": None,
            "named_insured": None,
            "effective_date": None,
            "expiration_date": None,
            "policy_type": None,
            "claims_made_or_occurrence": None,
            "confidence": 0.35,
            "page_reference": [1] if page_count and page_count > 0 else [],
        },
        "coverages": [
            {
                "coverage_name": "Sample Coverage",
                "coverage_section": "Sample Section",
                "limit_amount": None,
                "aggregate_limit": None,
                "deductible_or_retention": None,
                "is_sublimit": False,
                "confidence": 0.25,
                "page_reference": [1] if page_count and page_count > 0 else [],
            }
        ],
        "exposures": [],
    }
    return {
        "file_name": file_name,
        "file_path": file_path,
        "page_count": page_count,
        "request_chars": len(policy_text or ""),
        "model_name": "STUB",
        "extraction_status": "SUCCESS",
        "model_output_json": json.dumps(stub_json),
        "raw_model_response": json.dumps({"stub": True}),
        "error_message": None,
        "processed_at": datetime.utcnow().isoformat(),
    }


def call_databricks_foundation_model(prompt: str) -> dict:
    """Call a Databricks FM endpoint such as anthropic-claude-3-sonnet."""
    client = mlflow.deployments.get_deploy_client("databricks")

    inputs = {
        "messages": [
            {
                "role": "system",
                "content": "You extract structured insurance-policy JSON. Return valid JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 16000,
    }

    return client.predict(endpoint=DATABRICKS_FM_ENDPOINT, inputs=inputs)


def parse_model_json(model_response: dict) -> str:
    """Return a clean JSON string from a Databricks FM response."""
    content = extract_content_string(model_response)
    truncated = is_truncated(model_response)

    # Try standard parsing first
    try:
        clean_json = _extract_json_string(content)
        return clean_json
    except Exception as e:
        if not truncated:
            raise
        # If truncated, attempt repair
        print(f"  -> Response was truncated, attempting JSON repair...")

    # Attempt to repair truncated JSON
    repaired = _repair_truncated_json(content)
    print(f"  -> JSON repair succeeded")
    return repaired


print("Helper functions loaded (with truncation repair)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Model Extraction

# COMMAND ----------

rows = policy_text_df.collect()

outputs = []

for r in rows:
    policy_text = (r.policy_text or "")[:MAX_CHARS_TO_SEND]
    prompt = PROMPT_TEMPLATE.replace("{policy_text}", policy_text)
    raw_content = None  # Store the model's content string (not the full API response)

    try:
        if USE_STUB:
            result = stub_extraction(
                file_name=r.file_name,
                file_path=r.file_path,
                page_count=None,
                policy_text=policy_text,
            )
        else:
            print(f"Processing: {r.file_name} ({len(policy_text):,} chars)...")
            model_response = call_databricks_foundation_model(prompt)
            raw_content = extract_content_string(model_response)  # Clean content text
            parsed_json = parse_model_json(model_response)
            result = {
                "file_name": r.file_name,
                "file_path": r.file_path,
                "page_count": None,
                "request_chars": len(policy_text),
                "model_name": DATABRICKS_FM_ENDPOINT,
                "extraction_status": "SUCCESS",
                "model_output_json": parsed_json,
                "raw_model_response": raw_content,
                "error_message": None,
                "processed_at": datetime.utcnow().isoformat(),
            }
            print(f"  -> SUCCESS")
    except Exception as e:
        print(f"  -> FAILED: {e}")
        result = {
            "file_name": r.file_name,
            "file_path": r.file_path,
            "page_count": None,
            "request_chars": len(policy_text),
            "model_name": DATABRICKS_FM_ENDPOINT if not USE_STUB else "STUB",
            "extraction_status": "FAILED",
            "model_output_json": None,
            "raw_model_response": raw_content,  # Preserved for debugging
            "error_message": str(e),
            "processed_at": datetime.utcnow().isoformat(),
        }

    outputs.append(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Structured Model Output

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

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

print(f"Created llm_df with {llm_df.count()} rows")
display(llm_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Parse JSON into Structured Columns for Downstream Analysis

# COMMAND ----------

# Optional: Parse JSON into Structured Columns for Downstream Analysis

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
    col("parsed_json.policy.policy_number").alias("policy_number"),
    col("parsed_json.policy.carrier").alias("carrier"),
    col("parsed_json.policy.named_insured").alias("named_insured"),
    col("parsed_json.policy.effective_date").alias("effective_date"),
    col("parsed_json.policy.expiration_date").alias("expiration_date"),
    col("parsed_json.policy.policy_type").alias("policy_type"),
    col("parsed_json.policy.claims_made_or_occurrence").alias("claims_made_or_occurrence"),
    col("parsed_json.policy.confidence").alias("policy_confidence"),
    col("parsed_json.policy.page_reference").alias("policy_page_reference")
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

POLICY_TABLE_NAME = "policy_df"
COVERAGE_TABLE_NAME = "coverage_df"
LLM_TABLE_NAME = "llm_output_v2"

policy_df.write.format("delta").mode("overwrite").saveAsTable(POLICY_TABLE_NAME)
coverage_df.write.format("delta").mode("overwrite").saveAsTable(COVERAGE_TABLE_NAME)
llm_df.write.format("delta").mode("overwrite").saveAsTable(LLM_TABLE_NAME)

print(f"Saved: {POLICY_TABLE_NAME}, {COVERAGE_TABLE_NAME}, {LLM_TABLE_NAME}")
