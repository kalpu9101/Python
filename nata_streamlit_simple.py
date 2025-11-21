
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

st.set_page_config(layout="wide", page_title="NATA Supermarket - Simple UI")

# ------------------ CONFIG (exact user paths) ------------------
DATA_CSV = Path("/content/drive/My Drive/Code PPA/Nata Supermarket/df_cleaned.csv")
MODELS_DIR = Path("/content/drive/My Drive/Nata_Supermarket_Trained_Models")
OUTPUT_PRED_CSV = Path("/content/drive/My Drive/Code PPA/Nata Supermarket/df_with_preds.csv")
# ---------------------------------------------------------------

st.title("NATA Supermarket — Simple Prediction UI")
st.markdown(f"Data: `{DATA_CSV}`  |  Models dir: `{MODELS_DIR}`")

# Load data
if not DATA_CSV.exists():
    st.error(f"df_cleaned.csv not found at {DATA_CSV}")
    st.stop()

df = pd.read_csv(DATA_CSV)
df_cleaned = df.copy()

# ensure TotalSpending convenience column (if Mnt* exist)
mnt_cols = [c for c in df_cleaned.columns if c.startswith("Mnt")]
if "TotalSpending" not in df_cleaned.columns:
    if mnt_cols:
        df_cleaned["TotalSpending"] = df_cleaned[mnt_cols].sum(axis=1)
    else:
        df_cleaned["TotalSpending"] = 0.0

# Detect columns
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_cleaned.select_dtypes(include=["object","category"]).columns.tolist()

# List model files
if MODELS_DIR.exists():
    model_files = sorted([p for p in MODELS_DIR.iterdir() if p.suffix.lower() in (".pkl", ".joblib")])
else:
    model_files = []

st.sidebar.header("Controls")
st.sidebar.write("Found models:", [p.name for p in model_files] if model_files else "None")

# Choose model target (derived from filenames if possible) or select Mnt columns
available_targets = []
for p in model_files:
    stem = p.stem
    if stem.lower().startswith("model_"):
        available_targets.append(stem[len("model_"):])
    else:
        available_targets.append(stem)
if not available_targets:
    available_targets = [c for c in df_cleaned.columns if c.startswith("Mnt")]

chosen_target = st.sidebar.selectbox("Choose model/target", options=available_targets)

# Map chosen target to a file path if present
chosen_model_path = None
for p in model_files:
    if chosen_target.lower() in p.stem.lower():
        chosen_model_path = p
        break

# New-customer input form
st.sidebar.subheader("New customer input")
user_input = {}
with st.sidebar.form("input_form"):
    for c in categorical_cols:
        opts = df_cleaned[c].dropna().unique().tolist()
        if len(opts) > 0 and len(opts) <= 40:
            user_input[c] = st.selectbox(c, options=opts, index=0)
        else:
            user_input[c] = st.text_input(c, value="")
    for c in numeric_cols:
        mn = float(np.nanmin(df_cleaned[c].astype(float)))
        mx = float(np.nanmax(df_cleaned[c].astype(float)))
        med = float(np.nanmedian(df_cleaned[c].astype(float)))
        if mn == mx:
            user_input[c] = st.number_input(c, value=med)
        else:
            try:
                user_input[c] = st.slider(c, min_value=mn, max_value=mx, value=med)
            except Exception:
                user_input[c] = st.number_input(c, value=med)
    submit = st.form_submit_button("Apply inputs")

# Main layout: left preview, right actions
left, right = st.columns([2,1])
with left:
    st.subheader("Dataset preview")
    st.dataframe(df_cleaned.head(200), use_container_width=True)

with right:
    st.subheader("Quick info")
    st.write(f"Rows: {df_cleaned.shape[0]}")
    st.write(f"Columns: {df_cleaned.shape[1]}")
    st.write("Mnt columns:", mnt_cols)

# Similar customers (numeric)
if submit:
    st.subheader("Similar customers (numeric features)")
    num_df = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
    if len(numeric_cols) > 0:
        nbrs = NearestNeighbors(n_neighbors=6).fit(num_df.values)
        inp = {k: v for k, v in user_input.items() if k in numeric_cols}
        inp_df = pd.DataFrame([inp])
        for c in numeric_cols:
            if c not in inp_df.columns:
                inp_df[c] = num_df[c].median()
        inp_df = inp_df[numeric_cols].astype(float).fillna(num_df.median())
        distances, indices = nbrs.kneighbors(inp_df.values)
        sim = df_cleaned.iloc[indices[0][1:6]].copy()
        sim["distance"] = distances[0][1:6]
        st.dataframe(sim.reset_index(drop=True))
    else:
        st.info("No numeric columns available for similarity.")

# Single prediction using chosen pretrained model
st.subheader("Predict using pretrained model")
if st.button("Predict for new customer"):
    if chosen_model_path is None:
        st.error("No model file found for the selected target.")
    else:
        try:
            model = joblib.load(chosen_model_path)
        except Exception as e:
            st.error(f"Failed to load model {chosen_model_path}: {e}")
            model = None

        if model is not None:
            new_row = pd.DataFrame([user_input])
            for c in df_cleaned.columns:
                if c not in new_row.columns:
                    new_row[c] = np.nan
            new_row = new_row[df_cleaned.columns]

            pred = None
            try:
                pred = model.predict(new_row)[0]
            except Exception:
                try:
                    num_cols = new_row.select_dtypes(include=[np.number]).columns
                    pred = model.predict(new_row[num_cols].fillna(df_cleaned[num_cols].median()))[0]
                except Exception as e2:
                    st.error(f"Prediction failed: {e2}")

            if pred is not None:
                st.success(f"Predicted {chosen_target}: {pred:.4f}")
            else:
                st.error("Prediction did not return a value.")

# Batch predict all models
st.subheader("Batch predict (all models) and save")
if st.button("Run batch prediction and save to Drive"):
    if not model_files:
        st.error("No model files found in models directory.")
    else:
        out_df = df_cleaned.copy()
        failed = []
        for p in model_files:
            try:
                m = joblib.load(p)
                try:
                    preds = m.predict(df_cleaned)
                except Exception:
                    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                    preds = m.predict(df_cleaned[num_cols].fillna(df_cleaned[num_cols].median()))
                out_col = f"pred_{p.stem}"
                out_df[out_col] = preds
            except Exception as e:
                failed.append((p.name, str(e)))
        try:
            out_df.to_csv(OUTPUT_PRED_CSV, index=False)
            st.success(f"Saved predictions to {OUTPUT_PRED_CSV}")
            if failed:
                st.warning(f"Some models failed: {failed}")
        except Exception as e:
            st.error(f"Failed to save predictions CSV: {e}")

st.markdown("Done.")
