import pandas as pd
import streamlit as st
from databricks import sql

st.set_page_config(page_title="Insurance Policy Extraction PoC", layout="wide")

@st.cache_resource
def get_connection():
    return sql.connect(
        server_hostname=st.secrets["DATABRICKS_HOST"],
        http_path=st.secrets["DATABRICKS_HTTP_PATH"],
        access_token=st.secrets["DATABRICKS_TOKEN"],
    )

@st.cache_data(ttl=300)
def run_query(query: str) -> pd.DataFrame:
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)

dashboard_df = run_query("""
SELECT *
FROM datascience.default.v_policy_dashboard
ORDER BY file_name
""")

st.title("Insurance Policy Extraction PoC")

if dashboard_df.empty:
    st.error("No policy data found in datascience.default.v_policy_dashboard.")
    st.stop()

policy_options = dashboard_df["file_name"].dropna().tolist()
selected_file = st.selectbox("Select a policy file", policy_options)

selected_policy_df = dashboard_df[dashboard_df["file_name"] == selected_file]

coverages_df = run_query(f"""
SELECT
    file_name,
    coverage_name,
    coverage_section,
    limit_amount,
    aggregate_limit,
    deductible_or_retention,
    is_sublimit,
    coverage_confidence,
    coverage_page_reference
FROM datascience.default.policy_coverages
WHERE file_name = '{selected_file.replace("'", "''")}'
ORDER BY coverage_name
""")

st.subheader("Policy Summary")
st.dataframe(selected_policy_df, use_container_width=True, hide_index=True)

st.subheader("Coverage Details")
st.dataframe(coverages_df, use_container_width=True, hide_index=True)