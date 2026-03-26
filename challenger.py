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

# MAGIC %pip install streamlit

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
You are an expert insurance coverage analyst extracting structured data from commercial insurance policy PDFs.

Your job is to read the ENTIRE document and return a single valid JSON object only.

You must analyze all pages, including:
- declarations
- coverage parts
- forms schedules
- endorsements
- insuring agreements
- additional coverages
- extensions of coverage
- sublimits
- conditions pages
- location schedules
- underlying insurance references
- excess / umbrella / follow-form language

Do not stop after the declarations page.
Do not summarize.
Do not explain your reasoning.
Return JSON only.

====================
PRIMARY EXTRACTION GOAL
====================

Extract:
1. policy-level information
2. ALL coverages
3. ALL exposures explicitly stated

A coverage must be captured if it appears anywhere in the document, including:
- declarations
- coverage forms
- endorsement titles
- schedule entries
- additional coverages
- extensions
- sublimits
- separate insuring agreements
- manuscript or amendatory endorsements
- layered / excess / umbrella provisions
- contingent business interruption or dependent property language
- coverage enhancements that create a distinct insuring grant or distinct limit

Do NOT omit a coverage just because:
- it appears only in an endorsement
- it has no explicit numeric limit
- it modifies another form
- it is part of “additional coverages”
- it is part of a schedule
- it is a sublimit rather than a primary limit
- it is expressed as time, waiting period, attachment point, or retention rather than a standard deductible

If multiple distinct coverages are listed in one section, create separate coverage entries.
If one coverage has multiple distinct sublimits, create separate coverage entries.
Do not merge separate coverage grants into one row.

====================
COVERAGE DISCOVERY PROCESS
====================

Before producing final JSON, you must mentally perform this checklist across the full document:

1. Identify every form, endorsement, and coverage part by title or form number.
2. Identify every insuring agreement and every “additional coverage” / “coverage extension”.
3. Identify every listed sublimit, extension limit, and scheduled special coverage.
4. Identify any deductible, retention, SIR, waiting period, or attachment point tied to each coverage.
5. Then produce the final JSON with one row per distinct coverage item.

Important:
A form title alone can be a coverage if it grants or modifies coverage.
An endorsement that adds a new covered cause of loss, extension, or special limit should be captured as its own coverage row.

====================
DEDUCTIBLE / RETENTION RULES
====================

For each coverage entry, extract deductible_or_retention whenever explicitly stated for that coverage.

Capture deductible_or_retention when the document states any of the following:
- deductible
- retention
- self-insured retention
- SIR
- waiting period
- franchise deductible
- minimum deductible
- percentage deductible only if an explicit numeric amount is also stated
- attachment point in excess / umbrella structures
- “x/s” or “excess of” attachment amount

Rules:
- Convert stated numeric dollar values to numbers only, removing $ and commas.
- If the deductible/retention is expressed as hours or days, convert to numeric duration only.
  Examples:
  - “24 hours” -> 24
  - “72 hours” -> 72
- If layered wording appears, such as “$5,000,000 excess of $5,000,000”:
  - limit_amount = 5000000
  - deductible_or_retention = 5000000
- If only a percentage is given and no explicit numeric amount is stated, set deductible_or_retention = null.
- If multiple deductibles apply to the same coverage and they are clearly separate coverage situations, create separate coverage entries.
- If the deductible/retention is not explicitly tied to a specific coverage, do not guess; return null for that coverage.
- Do not confuse aggregate limit with deductible/retention.

Examples of deductible/retention phrases to recognize:
- “Deductible: $25,000 each occurrence”
- “Retention: $100,000”
- “Self-Insured Retention: $250,000”
- “Waiting Period: 72 hours”
- “$10,000,000 excess of $5,000,000”
- “Applies subject to a $50,000 wind/hail deductible”

====================
NUMERIC EXTRACTION RULES
====================

- Extract only values explicitly stated in the document.
- Do not infer missing values.
- If a value is not explicitly stated, return null.
- Convert dollar values to numeric values.
  Example: "$5,000,000" -> 5000000
- If both per-occurrence and aggregate are stated, capture both.
- If a coverage exists but the limit is not explicitly stated, still create the coverage entry and set the numeric fields to null where needed.
- Do not drop a coverage row because deductible_or_retention is missing.
- Do not drop a coverage row because limit_amount is missing.

====================
COVERAGE GRANULARITY RULES
====================

Create separate coverage entries when:
- different insuring agreements are listed separately
- different sublimits are listed separately
- the same coverage has separate limits for described vs undescribed premises
- the same coverage has separate in-transit vs at-premises limits
- the same coverage has separate aggregate and each-occurrence subcomponents that behave like separate listed coverages
- a form or endorsement introduces a distinct covered item or extension

Set is_sublimit = true when the coverage is a sublimit, extension, special limit, or otherwise not the main policy/coverage-part limit.

Set is_sublimit = false when the row represents the main coverage part, main insuring agreement, or top-level endorsement/form coverage without a subordinate sublimit structure.

====================
EXPOSURE RULES
====================

Extract exposures only if explicitly stated, such as:
- payroll
- sales
- revenue
- TIV
- locations
- vehicles
- employees
- units
- square footage
- premiums when clearly stated as rated exposure or premium value

If no exposures are explicitly stated, return an empty array.

====================
CONFIDENCE RULES
====================

Confidence must reflect certainty:
- 0.95 to 1.00 = explicitly stated and unambiguous
- 0.75 to 0.94 = clearly supported but requires minor interpretation
- 0.50 to 0.74 = partially clear
- below 0.50 = uncertain

Do not assign uniformly high confidence.

====================
PAGE REFERENCES
====================

Use PDF page numbers based on the page markers in the supplied text.
If evidence comes from multiple pages, include all relevant pages.
Example:
"page_reference": [12, 13]

====================
JSON SCHEMA (MUST MATCH EXACTLY)
====================

{
  "policy": {
    "policy_number": "",
    "carrier": "",
    "named_insured": "",
    "effective_date": "",
    "expiration_date": "",
    "policy_type": "",
    "claims_made_or_occurrence": "",
    "confidence": 0.0,
    "page_reference": []
  },
  "coverages": [
    {
      "coverage_name": "",
      "coverage_section": "",
      "limit_amount": null,
      "aggregate_limit": null,
      "deductible_or_retention": null,
      "is_sublimit": false,
      "confidence": 0.0,
      "page_reference": []
    }
  ],
  "exposures": [
    {
      "exposure_type": "",
      "exposure_value": null,
      "basis": "",
      "confidence": 0.0,
      "page_reference": []
    }
  ]
}

FINAL RULES:
- Return one JSON object only
- Return valid JSON only
- Do not use markdown
- Do not add commentary
- Do not omit coverages found anywhere in the document
- Do not invent values
- If a coverage exists but its numeric values are not explicit, still include the coverage row

Begin analysis now.

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
