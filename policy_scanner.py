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
You are an expert insurance contract analyst. Your task is to analyze the entire document and extract structured, analytics-ready data according to the schema below. You must read the FULL document before producing output. Do not stop after the declarations page. Do not extract only summary sections. You must scan endorsements, schedules, and all insuring agreements. 🔷 OBJECTIVE Convert the unstructured insurance policy into structured JSON. You must: Extract ALL policy-level information. Extract ALL coverage sections, including: Primary coverages Insuring agreements Endorsement-based coverages Sublimits Excess or umbrella layers Follow-form provisions Business Interruption and Contingent Business Interruption separately Extract numeric values exactly as stated. If a value is not explicitly stated, return null. Do NOT infer or assume missing information. Do NOT hallucinate. Include a confidence score between 0 and 1 for each object. Include page numbers where each item was found. Return valid JSON only. Do not include commentary or explanation outside of the JSON. 🔷 SPECIAL HANDLING RULES Policy-Level Rules Identify whether the policy is: Primary Excess Umbrella Follow-form Identify whether it is: Claims-made Occurrence-based If layered (e.g., \"$5M xs $5M\"): Extract limit_amount = 5000000 Extract deductible_or_retention = attachment point (5000000) If the policy references underlying insurance: Extract only the limits applicable to THIS policy. Coverage Extraction Rules Extract every coverage section even if nested in endorsements. Extract each insuring agreement separately. If sublimits exist within a coverage: Create separate entries. Set \"is_sublimit\": true If multiple limits apply within one coverage: Create separate entries. Do not merge distinct coverage sections. Do not omit Contingent Business Interruption if present. Do not omit Payment Card Loss if present. Do not omit Regulatory Proceedings if present. Numeric Handling Rules Convert dollar values to numeric (remove $, commas). \"$5,000,000\" → 5000000 If aggregate is separate from per occurrence: Extract both. If deductible or retention exists: Extract as numeric. If waiting period exists: Include numeric hours inside deductible_or_retention field. If unclear → return null. Exposure Extraction Rules If exposures are listed (e.g., payroll, revenue, locations, vehicles): Extract: exposure_type exposure_value basis If no exposures explicitly stated → return empty array. 🔷 JSON SCHEMA (MUST FOLLOW EXACTLY) { \"policy\": { \"policy_number\": \"\", \"carrier\": \"\", \"named_insured\": \"\", \"effective_date\": \"\", \"expiration_date\": \"\", \"policy_type\": \"\", \"claims_made_or_occurrence\": \"\", \"confidence\": 0.0, \"page_reference\": [] }, \"coverages\": [ { \"coverage_name\": \"\", \"coverage_section\": \"\", \"limit_amount\": null, \"aggregate_limit\": null, \"deductible_or_retention\": null, \"is_sublimit\": false, \"confidence\": 0.0, \"page_reference\": [] } ], \"exposures\": [ { \"exposure_type\": \"\", \"exposure_value\": null, \"basis\": \"\", \"confidence\": 0.0, \"page_reference\": [] } ] } 🔷 CONFIDENCE SCORING Confidence must reflect certainty: 0.95–1.0 → explicitly stated 0.75–0.94 → clear but minor interpretation required 0.50–0.74 → partially clear Below 0.50 → uncertain Confidence must not be uniformly high. 🔷 PAGE REFERENCES Use page numbers from the PDF. If multiple pages used → include all. If page numbers not visible → use sequential page count. Example: \"page_reference\": [12, 13] 🔷 FINAL INSTRUCTIONS Analyze the entire attached document. Return valid JSON only. Do not include markdown. Do not summarize. Do not truncate. Do not add explanation. 

RULES:
- Use ONLY values explicitly stated in the document
- Do NOT return null if a value exists
- If a value truly does not exist, return null
- Extract ALL coverages found anywhere in the document
- Return ONLY valid JSON (no explanations)

Begin analysis now.

POLICY TEXT:
{policy_text}

OUTPUT JSON:
{
  "policy": {
    "policy_number": "",
    "carrier": "",
    "named_insured": "",
    "effective_date": "",
    "expiration_date": "",
    "policy_type": "",
    "claims_made_or_occurrence": "",
    "coverages": [
      {
        "coverage_name": "",
        "coverage_section": "",
        "limit_amount": null,
        "aggregate_limit": null,
        "deductible_or_retention": null,
        "is_sublimit": false
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
}
"""
print(f"Prompt template loaded ({len(PROMPT_TEMPLATE)} chars)")

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
