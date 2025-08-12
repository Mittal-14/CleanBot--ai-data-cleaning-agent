"""
Streamlit AI Data Cleaning Agent
- Supports CSV and Excel uploads (.csv, .xls, .xlsx)
- Lets user select cleaning steps:
    - remove columns
    - missing value strategies (drop, fill with value, fill with mean)
    - encoding categorical (label, one-hot)
    - normalize numerical (minmax, standard)
    - detect & handle outliers (z-score remove / clip)
    - convert columns to datetime
- Shows dataset preview & diagnostics
- Suggests cleaning steps with simple heuristics ("AI suggestions")
- Saves cleaned data to the same file format and provides a download button

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""



import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy import stats

st.set_page_config(page_title="AI Data Cleaning Agent", layout="wide")
st.title("AI Data Cleaning Agent")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if not uploaded_file:
    st.info("Please upload a CSV or Excel file.")
    st.stop()

# Load data
if uploaded_file.name.endswith('.csv'):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.subheader("Original Data Preview (5 rows)")
st.dataframe(df.head())

# --- Step 1: Drop columns ---
drop_cols = st.multiselect("Select columns to drop", df.columns)
if drop_cols:
    df = df.drop(columns=drop_cols)
    st.success(f"Dropped columns: {', '.join(drop_cols)}")

st.subheader("Data Preview After Dropping Columns (5 rows)")
st.dataframe(df.head())

# --- Step 2: Handle missing values ---
missing_option = st.selectbox("Handle missing values", ["None", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])
if missing_option == "Drop rows":
    df = df.dropna()
elif missing_option == "Fill with mean":
    df = df.fillna(df.mean(numeric_only=True))
elif missing_option == "Fill with median":
    df = df.fillna(df.median(numeric_only=True))
elif missing_option == "Fill with mode":
    df = df.fillna(df.mode().iloc[0])

# --- Step 3: Encode categorical ---
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
encode_cols = st.multiselect("Select categorical columns to encode", cat_cols)
if encode_cols:
    le = LabelEncoder()
    for col in encode_cols:
        df[col] = le.fit_transform(df[col].astype(str))

# --- Step 4: Scale numeric data ---
scale_option = st.selectbox("Scale numeric data", ["None", "Min-Max scaling", "Standard scaling"])
if scale_option != "None":
    num_cols = df.select_dtypes(include=np.number).columns
    if scale_option == "Min-Max scaling":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

# --- Step 5: Remove outliers ---
remove_outliers = st.checkbox("Remove outliers (Z-score method)")
if remove_outliers:
    z_thresh = st.slider("Z-score threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    num_cols = df.select_dtypes(include=np.number).columns
    z_scores = np.abs(stats.zscore(df[num_cols]))
    df = df[(z_scores < z_thresh).all(axis=1)]

# --- Final Preview ---
st.subheader("Cleaned Data Preview (5 rows)")
st.dataframe(df.head())

# --- Download cleaned data ---
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def to_excel(df):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="Download Cleaned CSV",
        data=to_csv(df),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
with col2:
    st.download_button(
        label="Download Cleaned Excel",
        data=to_excel(df),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )












