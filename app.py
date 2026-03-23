"""
AKI Predictor - Streamlit Web Application
Based on LightGBM model trained on thoracic and abdominal aortic surgery data.
Model: best_with_intraoperative (LightGBM), Cutoff: 0.0125
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import os

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CUTOFF = 0.007434122745536748

# Feature order as used in the notebook (best_with_intraoperative)
FEATURES = [
    'BMI', 'PLR', 'NLR', 'UCR', 'Age (years)',
    'Serum albumin (g/L)', 'Urea (mmol/L)', 'SII',
    'Operation time (mins)', 'Surgery type'
]

# Numeric features that receive RobustScaling (all except Surgery type).
# Order MUST match notebook `final_numeric = [c for c in numeric_cols if c in best_with_intraoperative]`
# (iteration order of `numeric_cols`, not the same as FEATURES / model column order).
SCALER_NUMERIC_ORDER = [
    'Operation time (mins)',
    'Age (years)',
    'BMI',
    'Serum albumin (g/L)',
    'Urea (mmol/L)',
    'NLR',
    'PLR',
    'SII',
    'UCR',
]

# Skewed numeric features that receive log1p transformation before scaling
SKEWED_FEATURES = [
    'Operation time (mins)', 'Urea (mmol/L)',
    'NLR', 'PLR', 'SII'
]

# Surgery type encoding (as in notebook)
SURGERY_MAP = {'EVAR': 0, 'TEVAR': 1, 'OSR': 2}

# Input ranges (min, max) — values outside these are flagged as out-of-range
RANGES = {
    'Age (years)':                  (19.00,  94.00),
    'BMI':                          (15.22,  40.90),
    'NLR':                          (0.37,   44.50),
    'PLR':                          (37.50,  733.33),
    'SII':                          (57.06,  9093.33),
    'Urea (mmol/L)':                (1.50,   43.60),
    'UCR':                          (5.63,   42.08),
    'Serum albumin (g/L)':          (23.00,  55.00),
    'Operation time (mins)':        (11.00,  900.00),
}

# Model artifact paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH  = os.path.join(MODEL_DIR, 'lgbm_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'robust_scaler.pkl')


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    """Load the trained LightGBM model and RobustScaler from disk."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def preprocess(inputs: dict, scaler) -> pd.DataFrame:
    """
    Replicate the preprocessing pipeline from the notebook:
      1. log1p transform on skewed numeric features
      2. RobustScaler on all numeric features
      3. Surgery type encoding (already integer)
    Returns a DataFrame with sanitised column names (as LightGBM expects).
    """
    df = pd.DataFrame([inputs], columns=FEATURES)

    # Step 1 – log1p on skewed features
    for col in SKEWED_FEATURES:
        df[col] = np.log1p(df[col])

    # Step 2 – RobustScaler on numeric features (column order = fit order in notebook)
    df[SCALER_NUMERIC_ORDER] = scaler.transform(df[SCALER_NUMERIC_ORDER])

    # Step 3 – sanitise column names (matches notebook behaviour)
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    return df


def predict(model, df_processed: pd.DataFrame):
    """Return predicted probability for AKI = 1."""
    prob = model.predict_proba(df_processed)[0, 1]
    return float(prob)


def make_shap_waterfall(model, df_processed: pd.DataFrame, raw_inputs: dict) -> bytes:
    """
    Generate a SHAP waterfall plot and return it as PNG bytes.
    The display values shown in the plot use the original (unscaled) inputs.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_processed)

    # For binary LightGBM, shap_values may be a list [neg_class, pos_class]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv = shap_values[1]
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        sv = shap_values if not isinstance(shap_values, list) else shap_values[0]
        base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    # Build display data with original feature names (no sanitization in SHAP labels).
    surgery_inv = {v: k for k, v in SURGERY_MAP.items()}
    display_data = {}
    for feat in FEATURES:
        if feat == 'Surgery type':
            display_data[feat] = surgery_inv.get(int(raw_inputs[feat]), str(raw_inputs[feat]))
        else:
            display_data[feat] = round(raw_inputs[feat], 4)

    display_df = pd.DataFrame([display_data], columns=FEATURES)

    explanation = shap.Explanation(
        values=sv[0],
        base_values=float(base_val),
        data=display_df.iloc[0].values,
        feature_names=list(display_df.columns)
    )

    # Use a tall figure to accommodate all features clearly
    plt.figure(figsize=(12, 7))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 7)
    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def validate_ranges(inputs: dict):
    """Return list of (field_name, value, min_val, max_val) for out-of-range fields."""
    errors = []
    for field, (lo, hi) in RANGES.items():
        val = inputs.get(field)
        if val is not None and not (lo <= val <= hi):
            errors.append((field, val, lo, hi))
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AKI Predictor for Thoracic and Abdominal Aortic Surgery",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main container */
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Title */
    h1 { color: #1a3a5c; font-size: 2rem !important; }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 5px solid #f9a825;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        color: #555;
        line-height: 1.6;
    }
    .disclaimer-box strong { color: #e65100; }

    /* Section headers */
    .section-header {
        background-color: #1a3a5c;
        color: white;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        margin-top: 0.4rem;
    }

    /* Result boxes */
    .result-positive {
        background-color: #ffebee;
        border: 2px solid #c62828;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .result-negative {
        background-color: #e8f5e9;
        border: 2px solid #2e7d32;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .result-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; }
    .result-prob  { font-size: 1.1rem; margin-top: 6px; color: #444; }
    .result-cutoff { font-size: 0.82rem; color: #888; margin-top: 4px; }

    /* Divider */
    hr { margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────────────────────────────────────

artifacts_ok = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)

if artifacts_ok:
    try:
        model, scaler = load_artifacts()
        artifacts_loaded = True
    except Exception as e:
        artifacts_loaded = False
        load_error = str(e)
else:
    artifacts_loaded = False
    load_error = "Model files not found in `models/` directory."


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏥 Postoperative AKI Predictor for Thoracic and Abdominal Aortic Surgery")

st.markdown("""
<div class="disclaimer-box">
<strong>⚠️ Disclaimer:</strong>
This LightGBM-based prediction model is intended <strong>only for thoracic and abdominal aortic surgery</strong>
and has <strong>not undergone external validation</strong>.
On the internal test set, the model achieved:
<strong>Recall&nbsp;0.588</strong> (95% CI 0.25–0.818),
<strong>Precision&nbsp;0.238</strong> (95% CI 0.119–0.819),
<strong>FPR&nbsp;0.198</strong> (95% CI 0.006–0.366).
Results are provided <strong>for model validation purposes only</strong> and do
<strong>not constitute clinical advice</strong>.
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(
        f"**Model artifacts could not be loaded.** {load_error}\n\n"
        "Please place `lgbm_model.pkl` and `robust_scaler.pkl` in the `models/` folder "
        "next to `app.py`, then restart the app."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Layout: left form | right results
# ─────────────────────────────────────────────────────────────────────────────

col_form, col_result = st.columns([1, 1], gap="large")

# ── LEFT COLUMN: Input Form ──────────────────────────────────────────────────
with col_form:
    st.markdown("### Patient Information")

    with st.form("prediction_form"):

        # ── ① Patient Characteristics ────────────────────────────────────────
        st.markdown('<div class="section-header">① Patient Characteristics</div>',
                    unsafe_allow_html=True)

        age = st.number_input(
            "Age (years)",
            min_value=0.0, max_value=150.0, value=65.0, step=1.0,
            help="Valid range: 19.00 – 94.00 years"
        )
        bmi = st.number_input(
            "BMI",
            min_value=0.0, max_value=100.0, value=24.0, step=0.1,
            format="%.2f",
            help="Valid range: 15.22 – 40.90"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ② Laboratory Biomarkers ───────────────────────────────────────────
        st.markdown('<div class="section-header">② Laboratory Biomarkers</div>',
                    unsafe_allow_html=True)

        nlr = st.number_input(
            "NLR (Neutrophil-to-Lymphocyte Ratio)",
            min_value=0.0, max_value=200.0, value=2.5, step=0.01,
            format="%.2f",
            help="Valid range: 0.37 – 44.50"
        )
        plr = st.number_input(
            "PLR (Platelet-to-Lymphocyte Ratio)",
            min_value=0.0, max_value=2000.0, value=120.0, step=0.1,
            format="%.2f",
            help="Valid range: 37.50 – 733.33"
        )
        sii = st.number_input(
            "SII (Systemic Immune-Inflammation Index)",
            min_value=0.0, max_value=50000.0, value=500.0, step=1.0,
            format="%.2f",
            help="Valid range: 57.06 – 9093.33"
        )
        urea = st.number_input(
            "Urea (mmol/L)",
            min_value=0.0, max_value=200.0, value=6.0, step=0.1,
            format="%.2f",
            help="Valid range: 1.50 – 43.60"
        )
        ucr = st.number_input(
            "UCR (Urea-to-Creatinine Ratio)",
            min_value=0.0, max_value=200.0, value=15.0, step=0.01,
            format="%.2f",
            help="Valid range: 5.63 – 42.08"
        )
        albumin = st.number_input(
            "Serum Albumin (g/L)",
            min_value=0.0, max_value=100.0, value=41.0, step=0.1,
            format="%.2f",
            help="Valid range: 23.00 – 55.00"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ③ Intraoperative Variables ────────────────────────────────────────
        st.markdown('<div class="section-header">③ Intraoperative Variables</div>',
                    unsafe_allow_html=True)

        op_time = st.number_input(
            "Operation Time (mins)",
            min_value=0.0, max_value=2000.0, value=60.0, step=1.0,
            format="%.1f",
            help="Valid range: 11.00 – 900.00 minutes"
        )
        surgery_type_label = st.selectbox(
            "Surgery Type",
            options=["EVAR", "TEVAR", "OSR"],
            help=(
                "EVAR = Endovascular Aortic Repair | "
                "TEVAR = Thoracic Endovascular Aortic Repair | "
                "OSR = Open Surgical Repair"
            )
        )

        submitted = st.form_submit_button("🔍 Predict AKI Risk", use_container_width=True)


# ── RIGHT COLUMN: Results ────────────────────────────────────────────────────
with col_result:
    st.markdown("### Prediction Results")

    if not submitted:
        st.info("Fill in the patient information on the left and click **Predict AKI Risk** to see results.")
    else:
        # Collect raw inputs
        surgery_encoded = SURGERY_MAP[surgery_type_label]
        raw_inputs = {
            'BMI':                          bmi,
            'PLR':                          plr,
            'NLR':                          nlr,
            'UCR':                          ucr,
            'Age (years)':                  age,
            'Serum albumin (g/L)':          albumin,
            'Urea (mmol/L)':                urea,
            'SII':                          sii,
            'Operation time (mins)':        op_time,
            'Surgery type':                 surgery_encoded,
        }

        # Validate ranges
        range_errors = validate_ranges(raw_inputs)

        if range_errors:
            st.error("**Out-of-range values detected. Prediction cannot be made.**")
            for field, val, lo, hi in range_errors:
                st.warning(f"• **{field}**: entered `{val}`, valid range is `{lo} – {hi}`")
        else:
            with st.spinner("Running prediction…"):
                try:
                    # Preprocess
                    df_processed = preprocess(raw_inputs, scaler)

                    # Predict
                    prob = predict(model, df_processed)
                    aki_pred = int(prob >= CUTOFF)

                    # Display prediction result
                    if aki_pred == 1:
                        st.markdown("""
                        <div class="result-positive">
                            <div class="result-title" style="color:#c62828;">
                                ⚠️ High AKI Risk
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-negative">
                            <div class="result-title" style="color:#2e7d32;">
                                ✅ Low AKI Risk
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # SHAP waterfall plot
                    st.markdown("#### SHAP Feature Contribution (Waterfall Plot)")
                    with st.spinner("Generating SHAP explanation…"):
                        try:
                            shap_png = make_shap_waterfall(model, df_processed, raw_inputs)
                            st.image(shap_png, use_container_width=True)
                            st.caption(
                                "The waterfall plot shows each feature's contribution to the "
                                "predicted probability. Red bars push the prediction higher "
                                "(towards AKI); blue bars push it lower."
                            )
                        except Exception as shap_err:
                            st.warning(f"SHAP plot could not be generated: {shap_err}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.8rem;'>"
    "For research use only. Not for clinical decision-making."
    "</div>",
    unsafe_allow_html=True
)
