# nata_streamlit_no_models.py
# Minimal Streamlit UI that DOES NOT load pretrained .pkl files (avoids large binaries).
# Instead: if predictions are needed, it trains small, in-memory models on demand
# (no model files written). This avoids uploading large .pkl to GitHub.
#
# Expectation: Put df_cleaned.csv in the same GitHub repo/folder as the app:
#   df_cleaned.csv
#
# Usage:
#   streamlit run nata_streamlit_no_models.py

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

# Scikit-learn imports used only when training on-demand
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors

st.set_page_config(layout="wide", page_title="NATA Supermarket — No External Models")

# ------------------ CONFIG: local repo-relative CSV ------------------
DATA_CSV = Path("df_cleaned.csv")   # file must be in the same GitHub repo/folder
st.title("NATA Supermarket — Simple UI (no external .pkl models)")

# Load data
if not DATA_CSV.exists():
    st.error(f"df_cleaned.csv not found at repo path: {DATA_CSV}\nPlease upload the CSV into the repository.")
    st.stop()

df = pd.read_csv(DATA_CSV)
df_cleaned = df.copy()

# convenience: derive TotalSpending if Mnt* present
mnt_cols = [c for c in df_cleaned.columns if c.lower().startswith("mnt")]
if "TotalSpending" not in df_cleaned.columns:
    if mnt_cols:
        df_cleaned["TotalSpending"] = df_cleaned[mnt_cols].sum(axis=1)
    else:
        df_cleaned["TotalSpending"] = 0.0

# detect columns
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_cleaned.select_dtypes(include=['object','category']).columns.tolist()

st.sidebar.header("Controls")
st.sidebar.write(f"Rows: {df_cleaned.shape[0]}  Columns: {df_cleaned.shape[1]}")

# choose a numeric target (MNT variable or Income/TotalSpending)
default_targets = [c for c in df_cleaned.columns if c.startswith("Mnt")]
if "Income" in df_cleaned.columns:
    default_targets.insert(0, "Income")
if "TotalSpending" in df_cleaned.columns:
    default_targets.insert(0, "TotalSpending")
default_targets = list(dict.fromkeys(default_targets))  # keep order unique

target = st.sidebar.selectbox("Choose numeric target to predict (will train on-demand)", options=default_targets or numeric_cols, index=0)

# user input form produced from df_cleaned columns
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
            # slider sometimes fails on very large ranges; use try/except
            try:
                user_input[c] = st.slider(c, min_value=mn, max_value=mx, value=med)
            except Exception:
                user_input[c] = st.number_input(c, value=med)
    submit_inputs = st.form_submit_button("Apply inputs")

# main layout
left, right = st.columns([2,1])
with left:
    st.subheader("Dataset preview")
    st.dataframe(df_cleaned.head(200), use_container_width=True)

with right:
    st.subheader("Quick info")
    st.write("Mnt columns detected:", mnt_cols)
    st.write("Numeric columns count:", len(numeric_cols))
    st.write("Categorical columns count:", len(categorical_cols))

# similarity search if user provided input
if submit_inputs:
    st.subheader("Similar customers (by numeric features)")
    if len(numeric_cols) == 0:
        st.info("No numeric columns available for similarity.")
    else:
        num_df = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
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

# On-demand lightweight training + predict (no saving of models).
st.subheader("Train lightweight model on-demand & predict (no external .pkl required)")
st.markdown("This will train a small model in-memory (no artifact files). Use this if you cannot upload large .pkl files to GitHub/Streamlit Cloud.")

train_button = st.button("Train & Predict (lightweight)")

if train_button:
    # Prepare X, y
    X = df_cleaned.drop(columns=[target]) if target in df_cleaned.columns else df_cleaned.drop(columns=[target])
    y = df_cleaned[target].fillna(df_cleaned[target].median())

    # identify numeric & categorical features for pipeline
    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X.select_dtypes(include=['object','category']).columns.tolist()

    # Keep the pipeline light: fewer trees and simple imputers
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_feats),
        ("cat", categorical_transformer, cat_feats)
    ], sparse_threshold=0)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1))  # lightweight
    ])

    # Split and train (small train for speed)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        st.write(f"Lightweight model trained — MAE: {mae:.3f}, R2: {r2:.3f}")
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # predict for the provided new customer if inputs exist
    if submit_inputs:
        new_row = pd.DataFrame([user_input])
        # ensure columns exist for pipeline
        for c in X.columns:
            if c not in new_row.columns:
                new_row[c] = np.nan
        new_row = new_row[X.columns]
        try:
            pred_new = model.predict(new_row)[0]
            st.success(f"Predicted {target} for provided customer: {pred_new:.3f}")
        except Exception as e:
            st.error(f"Prediction on new input failed: {e}")

    # optional: batch-predict the whole df and offer download (no model files written)
    if st.checkbox("Batch predict entire dataset (adds columns to in-memory copy)"):
        try:
            all_preds = model.predict(X)
            out = df_cleaned.copy()
            out[f"pred_{target}"] = all_preds
            # provide download button for user to download CSV
            csv = out.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", data=csv, file_name=f"df_with_pred_{target}.csv", mime="text/csv")
            st.success("Batch predictions ready for download (no files written to server).")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.markdown("---")
st.markdown("Notes: this app trains small models in RAM on-demand. If you need production-grade performance or persistent models, consider storing model artifacts in an external object store (S3, Azure Blob) and load them at runtime.")
