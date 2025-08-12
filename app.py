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










# """
# Streamlit AI Data Cleaning Agent
# - Supports CSV and Excel uploads (.csv, .xls, .xlsx)
# - Lets user select cleaning steps:
#     - remove columns
#     - missing value strategies (drop, fill with value, fill with mean)
#     - encoding categorical (label, one-hot)
#     - normalize numerical (minmax, standard)
#     - detect & handle outliers (z-score remove / clip)
#     - convert columns to datetime
# - Shows dataset preview & diagnostics
# - Suggests cleaning steps with simple heuristics ("AI suggestions")
# - Saves cleaned data to the same file format and provides a download button

# Run:
#     pip install -r requirements.txt
#     streamlit run streamlit_app.py
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# from io import BytesIO
# import base64
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
# from scipy import stats

# st.set_page_config(page_title="AI Data Cleaning Agent", layout="wide")

# # ----------------------
# # Helper functions
# # ----------------------

# @st.cache_data(show_spinner=False)
# def read_file(uploaded_file):
#     """Read CSV or Excel into a pandas DataFrame. Uses caching for repeated reads."""
#     name = uploaded_file.name.lower()
#     try:
#         if name.endswith(".csv"):
#             # Let pandas infer encoding; if huge file, user will be warned in UI
#             df = pd.read_csv(uploaded_file)
#         elif name.endswith((".xls", ".xlsx")):
#             df = pd.read_excel(uploaded_file)
#         else:
#             raise ValueError("Unsupported file format. Upload CSV or Excel.")
#     except Exception as e:
#         raise
#     return df

# def suggest_cleaning_steps(df, n_top=5):
#     """
#     Simple heuristic-based suggestions (lightweight 'AI'): 
#     - Columns with many missing values
#     - Columns with many unique values (maybe IDs)
#     - Categorical columns
#     - Numerical columns with outliers
#     """
#     suggestions = []
#     n_rows = len(df)
#     missing = df.isna().mean().sort_values(ascending=False)
#     high_missing = missing[missing > 0.3]  # more than 30%
#     if not high_missing.empty:
#         suggestions.append(f"Drop or impute columns with >30% missing: {', '.join(high_missing.index.tolist())}")

#     cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#     if cat_cols:
#         suggestions.append(f"Consider encoding categorical columns: {', '.join(cat_cols[:n_top])}")

#     num_cols = df.select_dtypes(include=[np.number]).columns
#     if len(num_cols) > 0:
#         # detect columns with high skew / outliers via zscore
#         outlier_cols = []
#         for c in num_cols:
#             col = df[c].dropna()
#             if col.shape[0] < 2:
#                 continue
#             z = np.abs(stats.zscore(col))
#             if (z > 3).any():
#                 outlier_cols.append(c)
#         if outlier_cols:
#             suggestions.append(f"Outliers detected in numeric columns: {', '.join(outlier_cols)} — consider removal or transformation")

#     # Columns that look like datetimes
#     maybe_dates = []
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             sample = df[col].dropna().astype(str).head(10).tolist()
#             if any((':' in s or '-' in s or '/' in s) for s in sample):
#                 maybe_dates.append(col)
#     if maybe_dates:
#         suggestions.append(f"Columns that might be datetimes: {', '.join(maybe_dates[:n_top])}")

#     # If nothing found, give a general suggestion
#     if not suggestions:
#         suggestions.append("No strong issues detected — preview the data and choose operations as needed.")
#     return suggestions

# def remove_columns(df, cols):
#     return df.drop(columns=cols, errors='ignore')

# def handle_missing_values(df, strategy, fill_value=None):
#     df = df.copy()
#     if strategy == "drop":
#         return df.dropna()
#     elif strategy == "fill":
#         return df.fillna(fill_value)
#     elif strategy == "mean":
#         for col in df.select_dtypes(include=[np.number]).columns:
#             df[col] = df[col].fillna(df[col].mean())
#         return df
#     return df

# def encode_categorical(df, method):
#     df = df.copy()
#     cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#     if method == "label":
#         enc = LabelEncoder()
#         for col in cat_cols:
#             try:
#                 df[col] = enc.fit_transform(df[col].astype(str))
#             except Exception:
#                 # fallback: leave column as is if encoding fails
#                 pass
#     elif method == "onehot":
#         if cat_cols:
#             df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
#     return df

# def normalize_data(df, method):
#     df = df.copy()
#     num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     if not num_cols:
#         return df
#     if method == "minmax":
#         scaler = MinMaxScaler()
#         df[num_cols] = scaler.fit_transform(df[num_cols])
#     elif method == "standard":
#         scaler = StandardScaler()
#         df[num_cols] = scaler.fit_transform(df[num_cols])
#     return df

# def detect_and_handle_outliers(df, strategy, z_thresh=3):
#     """
#     strategy: 'none', 'remove', 'clip'
#     remove -> drop rows where any numeric column has |z| > z_thresh
#     clip -> clip numeric values to z_thresh * std
#     """
#     df = df.copy()
#     num_cols = df.select_dtypes(include=[np.number]).columns
#     if strategy == "remove" and len(num_cols) > 0:
#         z = np.abs(stats.zscore(df[num_cols].dropna()))
#         # if any column has NaNs removed differently, handle shape carefully
#         # if z.ndim == 1:
#         #     mask = z < z_thresh
#         # else:
#         #     mask = (z < z_thresh).all(axis=1)
#         # df = df.loc[mask]
#         mask = pd.Series(True, index=df.index)
#         for c in num_cols:
#             col = df[c]
#             z = (col - col.mean()) / col.std()
#             mask &= z.abs().fillna(0) < z_thresh
#         df = df.loc[mask]

#         return df
#     elif strategy == "clip" and len(num_cols) > 0:
#         for c in num_cols:
#             col = df[c]
#             mean = col.mean()
#             std = col.std()
#             low, high = mean - z_thresh*std, mean + z_thresh*std
#             df[c] = col.clip(lower=low, upper=high)
#         return df
#     return df

# def convert_datetime_columns(df, columns, dayfirst=False):
#     df = df.copy()
#     for col in columns:
#         try:
#             df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=dayfirst)
#         except Exception:
#             df[col] = pd.to_datetime(df[col], errors='coerce')
#     return df

# def df_to_bytes(df, original_filename):
#     """
#     Return bytes buffer of cleaned DataFrame in same format as original file.
#     """
#     buf = BytesIO()
#     lower = original_filename.lower()
#     if lower.endswith(".csv"):
#         df.to_csv(buf, index=False)
#         mime = "text/csv"
#         out_name = f"cleaned_{original_filename}"
#     else:
#         # default to Excel
#         try:
#             df.to_excel(buf, index=False, engine='openpyxl')
#         except Exception:
#             # fallback to csv if openpyxl not available
#             df.to_csv(buf, index=False)
#             mime = "text/csv"
#             out_name = f"cleaned_{original_filename}"
#             return buf.getvalue(), out_name, mime
#         mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#         out_name = f"cleaned_{original_filename}"
#     return buf.getvalue(), out_name, mime

# # ----------------------
# # UI
# # ----------------------

# st.title("🧹 AI Data Cleaning Agent (Streamlit)")

# uploaded_file = st.file_uploader("Upload dataset (CSV or Excel)", type=["csv", "xls", "xlsx"])

# if not uploaded_file:
#     st.info("Upload a CSV or Excel file to start. Tip: For large files (>200k rows) consider saving as Parquet or sampling.")
#     st.stop()

# # Read file and show basic info
# with st.spinner("Reading file..."):
#     try:
#         df = read_file(uploaded_file)
#     except Exception as e:
#         st.error(f"Could not read file: {e}")
#         st.stop()

# st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# # Show data preview and basic diagnostics
# st.subheader("Preview & Diagnostics")
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.write("First 5 rows:")
#     st.dataframe(df.head(), use_container_width=True)

# with col2:
#     st.write("Quick diagnostics")
#     st.write(f"- Rows: {df.shape[0]:,}")
#     st.write(f"- Columns: {df.shape[1]:,}")
#     missing_pct = (df.isna().mean()*100).round(2).sort_values(ascending=False)
#     if not missing_pct.empty:
#         st.write("Top missing percentages:")
#         st.table(missing_pct[missing_pct > 0].head(5))

# # Suggestions (AI-ish)
# st.subheader("Smart Suggestions")
# suggestions = suggest_cleaning_steps(df)
# for s in suggestions:
#     st.info(s)

# st.markdown("---")

# # ----------------------
# # User-configurable cleaning options
# # ----------------------

# st.subheader("Configure Cleaning Steps")

# # Column removal
# cols = list(df.columns)
# cols_to_remove = st.multiselect("Remove columns (select to drop)", cols)

# # Missing values
# st.markdown("**Missing values**")
# missing_choice = st.selectbox("Strategy", ["none", "drop rows with any NaN", "fill with custom value", "fill numeric with mean"])
# fill_value = None
# if missing_choice == "fill with custom value":
#     fill_value = st.text_input("Value to fill missing entries with (applies to all columns)", value="N/A")

# # Encoding categorical
# st.markdown("**Categorical encoding**")
# encoding_choice = st.selectbox("Encoding method", ["none", "label encoding", "one-hot encoding"])

# # Normalization
# st.markdown("**Normalize numerical features**")
# normalize_choice = st.selectbox("Normalize", ["none", "min-max (0-1)", "standard (z-score)"])

# # Outliers
# st.markdown("**Outlier handling**")
# outlier_choice = st.selectbox("Outlier strategy", ["none", "remove rows with z > 3", "clip to z=3"])
# z_thresh = st.slider("Z-threshold for detection/clipping", min_value=2.0, max_value=5.0, value=3.0, step=0.5)

# # Datetime conversion
# st.markdown("**Date/time conversion**")
# maybe_datetime = [c for c in df.columns if df[c].dtype == 'object']
# datetime_cols = st.multiselect("Columns to convert to datetime (if any)", maybe_datetime)

# # Advanced / custom options box
# with st.expander("Advanced / preview-only options"):
#     sample_rows = st.number_input("Preview sample size (0 = full)", min_value=0, max_value=10000, value=5)
#     keep_index = st.checkbox("Keep original index in output", value=False)

# # Run cleaning
# run = st.button("Run cleaning")

# if not run:
#     st.info("Configure options and click **Run cleaning** when ready.")
#     st.stop()

# # ----------------------
# # Execute cleaning pipeline
# # ----------------------

# with st.spinner("Cleaning data..."):
#     working = df.copy(deep=False)

#     # Remove columns
#     if cols_to_remove:
#         working = remove_columns(working, cols_to_remove)

#     # Missing values
#     if missing_choice != "none":
#         if missing_choice == "drop rows with any NaN":
#             working = handle_missing_values(working, "drop")
#         elif missing_choice == "fill with custom value":
#             working = handle_missing_values(working, "fill", fill_value)
#         elif missing_choice == "fill numeric with mean":
#             working = handle_missing_values(working, "mean")

#     # Datetime conversion (do before encoding/normalization)
#     if datetime_cols:
#         working = convert_datetime_columns(working, datetime_cols)

#     # Encoding categorical
#     if encoding_choice != "none":
#         if encoding_choice == "label encoding":
#             working = encode_categorical(working, "label")
#         elif encoding_choice == "one-hot encoding":
#             working = encode_categorical(working, "onehot")

#     # Normalize
#     if normalize_choice != "none":
#         if normalize_choice == "min-max (0-1)":
#             working = normalize_data(working, "minmax")
#         elif normalize_choice == "standard (z-score)":
#             working = normalize_data(working, "standard")

#     # Outliers
#     if outlier_choice != "none":
#         if outlier_choice == "remove rows with z > 3":
#             working = detect_and_handle_outliers(working, "remove", z_thresh=z_thresh)
#         elif outlier_choice == "clip to z=3":
#             working = detect_and_handle_outliers(working, "clip", z_thresh=z_thresh)

#     # Optionally restore index
#     # if keep_index:
#     #     working.reset_index(inplace=False)

#     working = working.reset_index(drop=False)   # keep index as a column


# # Final checks
# if working.empty:
#     st.error("After cleaning, the dataset is empty. Adjust choices and try again.")
#     st.stop()

# st.success(f"Cleaning complete — result: {working.shape[0]:,} rows × {working.shape[1]:,} columns")

# # Show preview and allow download
# st.subheader("Cleaned Preview")
# st.dataframe(working.head(sample_rows if sample_rows > 0 else 5), use_container_width=True)

# # Prepare download bytes
# try:
#     btn_bytes, out_name, mime = df_to_bytes(working, uploaded_file.name)
# except Exception as e:
#     st.error(f"Failed to prepare download: {e}")
#     st.stop()

# st.download_button(
#     label="Download cleaned file",
#     data=btn_bytes,
#     file_name=out_name,
#     mime=mime
# )

# # Also let user inspect changes (diff-ish): show columns removed and new columns
# orig_cols = set(df.columns)
# new_cols = set(working.columns)
# removed = orig_cols - new_cols
# added = new_cols - orig_cols

# st.markdown("---")
# st.write("Summary of structural changes:")
# if removed:
#     st.write(f"- Removed columns: {', '.join(sorted(list(removed)))[:1000]}")
# else:
#     st.write("- Removed columns: None")

# if added:
#     st.write(f"- Added columns (e.g., from one-hot): {', '.join(sorted(list(added)))[:1000]}")
# else:
#     st.write("- Added columns: None")

# st.info("Tip: For very large datasets, run cleaning in batches or convert source to Parquet to speed loading and reduce memory usage.")
