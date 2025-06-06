import streamlit as st
import pandas as pd
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import io
import base64
from tqdm import tqdm

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("Spam Classification Interface (Batch + Progress)")

# --- Constants ---
API_URL = "http://0.0.0.0:8102/classify"
REQUIRED_COLUMNS = ['Id', 'Topic', 'Title', 'Content', 'Description', 'Sentiment', 'SiteName', 'SiteId', 'Type']
CATEGORY_OPTIONS = ["finance", "real_estate", "ewallet", "healthcare_insurance", "ecommerce", "education", "logistic_delivery", "energy_fuels", "fnb", "investment"]
BATCH_SIZE = 10
MAX_WORKERS = 10

# --- Utility Functions ---
def get_excel_download_link(df: pd.DataFrame, filename: str) -> str:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel file</a>'

def convert_to_api_format(row: pd.Series) -> Dict:
    return {
        "id": str(row['Id']),
        "topic": str(row['Topic']),
        "title": str(row['Title']),
        "content": str(row['Content']),
        "description": str(row['Description']),
        "sentiment": str(row['Sentiment']),
        "site_name": str(row['SiteName']),
        "site_id": str(row['SiteId']),
        "label": str(row.get('Label', '')),
        "type": str(row['Type'])
    }

def chunk_data(data: List[Dict], batch_size: int = 10) -> List[List[Dict]]:
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def call_api_batch(data_batch: List[Dict], category: str) -> List[Dict]:
    try:
        payload = {
            "category": category,
            "data": data_batch
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        if result.get("result") == 1 and "data" in result and "results" in result["data"]:
            return result["data"]["results"]
        else:
            return []
    except Exception:
        return []

def process_data_parallel(json_data: List[Dict], category: str, progress_placeholder) -> List[Dict]:
    results = []
    data_batches = chunk_data(json_data, batch_size=BATCH_SIZE)
    total_batches = len(data_batches)

    progress_bar = progress_placeholder.progress(0, text="Starting batch processing...")
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(call_api_batch, batch, category): batch for batch in data_batches}
        for future in as_completed(futures):
            batch_result = future.result()
            if batch_result:
                results.extend(batch_result)
            completed_batches += 1
            progress_bar.progress(completed_batches / total_batches, text=f"Processed {completed_batches}/{total_batches} batches")

    return results

# --- Main Streamlit UI ---
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file).head(1000)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.success("File uploaded and validated successfully!")
            st.markdown("### Preview Uploaded Data")
            st.dataframe(df.head(20), use_container_width=True)

            json_data = [convert_to_api_format(row) for _, row in df.iterrows()]
            category = st.selectbox("Select Category", CATEGORY_OPTIONS)

            if st.button("Classify Spam (Batch + Parallel)"):
                with st.spinner("Preparing classification..."):
                    progress_placeholder = st.empty()
                    results = process_data_parallel(json_data, category, progress_placeholder)

                if results:
                    result_df = df.copy()
                    result_df['spam'] = None
                    result_df['language'] = None

                    for result in results:
                        idx = result_df[result_df['Id'] == result['id']].index
                        if not idx.empty:
                            result_df.loc[idx, 'spam'] = result['spam']
                            result_df.loc[idx, 'language'] = result['lang']

                    st.success("✅ Classification completed!")
                    st.markdown("### Results")
                    st.dataframe(result_df, use_container_width=True)

                    st.markdown(get_excel_download_link(result_df, "classified_results.xlsx"), unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No results returned from the API.")
    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
else:
    st.info("📤 Please upload an Excel file to begin.")
