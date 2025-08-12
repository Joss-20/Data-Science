"""
CostMind — SMT Scrap/OEE Predictor (Streamlit, CM Theme + Bosch Data)
----------------------------------------------------------------------
- Uses your dataset by default: /mnt/data/Bosch_SMT_Final_Training_Data.csv
- Predicts high scrap (classification on `High_Scrap`) or low OEE (`Low_OEE`).
- English UI, CostMind theme, feature engineering for SMT context.
- Avoids target leakage by excluding scrap/OEE metrics when predicting their flags.

Run locally:
  pip install -U streamlit pandas numpy scikit-learn lightgbm joblib matplotlib
  streamlit run app.py

Deploy (Streamlit Cloud) — requirements.txt:
  streamlit
  pandas
  numpy
  scikit-learn
  lightgbm
  joblib
  matplotlib
"""
from __future__ import annotations
import os, math
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# -------------------------------
# Paths
# -------------------------------
DEFAULT_CSV = "/mnt/data/Bosch_SMT_Final_Training_Data.csv"
PREFERRED_TARGETS = ["High_Scrap", "Low_OEE", "Defect_Rate"]

# -------------------------------
# Page & Theme (CostMind)
# -------------------------------
st.set_page_config(page_title="CostMind — SMT Scrap/OEE Predictor", layout="wide")

CSS = """
<style>
  :root{--bg:#070a12;--panel:#0b1220;--panelA:rgba(11,18,32,.72);--text:#e6f1ff;--muted:#a0aec0;--brand:#14ACBA;--accent:#7C3AED;--radius:20px;--lime:#a6ff00;--primary-color:#a6ff00}
  html,body,[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 800px at 80% -10%, rgba(20,172,186,.30) 0%, transparent 60%),var(--bg)!important;color:var(--text)}
  [data-testid="stHeader"], footer{background:transparent!important}
  [data-testid="stSidebar"]>div{background:var(--panel)!important;border-right:1px solid #1f2a3a}
  .cm-card{background:linear-gradient(var(--panelA),var(--panelA)) padding-box, linear-gradient(135deg, color-mix(in oklab,var(--brand)35%,transparent), color-mix(in oklab,var(--accent)25%,transparent)) border-box; border:1px solid transparent; border-radius:var(--radius); box-shadow:0 20px 40px rgba(0,0,0,.45)}
  .cm-hero{margin:-1rem 0 1rem; padding:1rem 1.2rem; border-radius:16px; border:1px solid #223047; background:linear-gradient(180deg, rgba(20,172,186,.06), rgba(124,58,237,.04));}
  .cm-pill{display:inline-block;padding:.35rem .6rem;border-radius:9999px;border:1px solid #1f3a46;background:rgba(20,172,186,.08);color:var(--muted);margin-right:.4rem}
  .stButton>button{border-radius:999px;border:1px solid #0f2d36;background:linear-gradient(180deg, color-mix(in oklab,var(--brand)55%,#0a2b33), color-mix(in oklab,var(--brand)25%,#071a1f));color:var(--text)}
  .stButton>button:hover{filter:brightness(1.05)}
  .stTextInput>div>div>input,.stNumberInput input,.stSelectbox>div>div{background:#0b1220;border:1px solid #334155;color:var(--text);border-radius:10px}
  .stDataFrame{filter:saturate(1.05)}
  /* Lime accents for any default reds */
  div[data-testid="stMetricDelta"]{color:var(--lime)!important}
  div[data-testid="stMetricDelta"] svg{fill:var(--lime)!important}
  div[role="progressbar"]>div{background:var(--lime)!important}
  .stMarkdown a{color:var(--lime)}
  /* Slider text & wrapper */
  [data-testid="stSlider"]{ color: var(--lime)!important }
  [data-testid="stSlider"] div[data-baseweb="slider"]{ padding:0!important; box-shadow:none!important }
  /* Slider overrides */
  [data-testid="stSlider"] [role="slider"]{background:var(--lime)!important;border:0!important;box-shadow:none!important}
  [data-testid="stSlider"] div[data-baseweb="slider"] > div { background:#334155!important; height:6px!important; box-shadow:none!important }
  [data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: var(--lime)!important; }
  [data-testid="stSlider"] [role="slider"]:focus{box-shadow:0 0 0 2px rgba(166,255,0,.35)!important;outline:none}
  [data-testid="stSlider"] [role="slider"]:hover{filter:none}
  /* --- Slider refinements: remove padding/margins and force value color --- */
  /* Remove residual padding/margins around the track */
  [data-testid="stSlider"] div[data-baseweb="slider"]{ padding:0!important; margin:0!important; box-shadow:none!important }
  [data-testid="stSlider"] div[data-baseweb="slider"]>div{ margin:0!important }
  /* Unfilled/filled track */
  [data-testid="stSlider"] div[data-baseweb="slider"]>div{ background:#334155!important; height:6px!important }
  [data-testid="stSlider"] div[data-baseweb="slider"]>div>div{ background:var(--lime)!important }
  /* Thumb (no glow) */
  [data-testid="stSlider"] [role="slider"]{ background:var(--lime)!important; border:0!important; box-shadow:none!important }
  [data-testid="stSlider"] [role="slider"]:focus{ box-shadow:0 0 0 2px rgba(166,255,0,.35)!important }
  /* Label/value row: keep label white, force numeric value lime */
  [data-testid="stSlider"] label{ color:#e6f1ff!important }
  [data-testid="stSlider"] [data-testid="stWidgetValue"]{ color:var(--lime)!important }
  /* Fallback selectors for older Streamlit DOMs */
  [data-testid="stSlider"] label+div span,
  [data-testid="stSlider"] label+div div{ color:var(--lime)!important }
  /* Remove any residual glow/shadow on slider bars */
  [data-testid="stSlider"] div[data-baseweb="slider"] > div,
  [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    box-shadow: none !important;
    filter: none !important;
    background-image: none !important;
    border: 0 !important;
  }
  [data-testid="stSlider"] *::before,
  [data-testid="stSlider"] *::after {
    box-shadow: none !important;
    filter: none !important;
  }
</style>
"""

if st.query_params.get("embed", ["false"]) [0].lower() == "true":
    st.markdown("""
    <style>header[data-testid=\"stHeader\"], footer{visibility:hidden;height:0}</style>
    """, unsafe_allow_html=True)

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="cm-hero cm-card">
      <span class="cm-pill">Automotive Electronics — SMT</span>
      <h1 style="margin:.4rem 0 0;font-size:1.6rem;">Predict High Scrap & Low OEE</h1>
      <div style="color:var(--muted)">Train or load a model to flag at‑risk shifts/lines based on SMT process & operations signals.</div>
    </div>
    """, unsafe_allow_html=True,
)

# -------------------------------
# Built-in small demo generator
# -------------------------------
DEMO_ROWS = 600

def make_small_demo(n: int = DEMO_ROWS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fechas = pd.date_range('2025-03-01', periods=n, freq='H')
    turno = rng.choice(list('ABC'), size=n, p=[0.35, 0.40, 0.25])
    modelo = rng.choice(['ECU-A','ECU-B','ECU-C','ECU-D'], size=n)

    # Process baseline
    temp = rng.normal(205, 7, size=n)                 # °C
    paste = rng.normal(150, 9, size=n)                # μm
    humid = rng.normal(45, 8, size=n)                 # %
    misal = rng.normal(0.05, 0.018, size=n)           # mm equivalent

    # Production time and events
    t_eff = rng.integers(300, 540, size=n)            # effective minutes (5h–9h)
    change = rng.integers(4, 28, size=n)              # changeover minutes
    paros = np.clip(rng.normal(18, 9, size=n), 0, None)  # downtime minutes

    uph_base = rng.normal(120, 12, size=n)            # units per hour
    uph_penalty = 1 - (np.clip(misal, 0, 0.12) * 2 + np.clip((temp-205)/25, -0.4, 0.4)) * 0.2
    uph = np.clip(uph_base * uph_penalty, 60, 180)
    unidades = (uph * (t_eff/60)).astype(int)

    # Risk score -> High_Scrap / Low_OEE
    logit = (
        -2.2
        + 0.035*(temp-205)
        + 0.012*(paste-150)
        + 0.025*(humid-45)
        + 3.2*np.clip(misal-0.05, -0.03, 0.06)
        + 0.015*(paros-18)
        + 0.018*(change-10)
        + (turno=='C')*0.25
        + (modelo=='ECU-D')*0.2
    )
    p_def = 1/(1+np.exp(-logit))
    high_scrap = rng.binomial(1, np.clip(p_def, 0.02, 0.85))

    oee_loss = (paros/t_eff) + np.clip(change/t_eff, 0, 0.25) + np.clip((0.12 - (uph/120))/0.12, 0, 0.4)
    low_oee = (oee_loss > np.quantile(oee_loss, 0.7)).astype(int)

    demo = pd.DataFrame({
        'Fecha': fechas,
        'Turno': turno,
        'Modelo': modelo,
        'Tiempo_Cambio_Modelo_min': change,
        'Tiempo_Produccion_efectivo_min': t_eff,
        'Paros_Linea_min': np.round(paros,1),
        'Unidades_Producidas': unidades,
        'Avg_Solder_Temp_C': np.round(temp,2),
        'Avg_Paste_Thickness_um': np.round(paste,1),
        'Avg_Humidity': np.round(humid,1),
        'Avg_Misalignment': np.round(misal,3),
        'High_Scrap': high_scrap.astype(int),
        'Low_OEE': low_oee.astype(int)
    })
    return demo

# -------------------------------
# Load Data
# -------------------------------
st.sidebar.header("Data")
source = st.sidebar.selectbox("Data source", ["Small demo (built-in)", "Bosch CSV (server)", "Upload CSV"], index=0)

upload = st.sidebar.file_uploader("Upload CSV (Bosch SMT schema)", type=["csv"]) if source=="Upload CSV" else None

if source == "Small demo (built-in)":
    df = make_small_demo()
elif source == "Bosch CSV (server)":
    if os.path.exists(DEFAULT_CSV):
        df_full = pd.read_csv(DEFAULT_CSV)
        # Keep it fast by sampling up to 10k rows while preserving target balance if available
        sample_n = min(len(df_full), 10000)
        if sample_n < len(df_full) and any(t in df_full.columns for t in PREFERRED_TARGETS):
            tcol = next(t for t in PREFERRED_TARGETS if t in df_full.columns)
            # stratified sample
            df = (df_full.groupby(df_full[tcol]).apply(lambda g: g.sample(frac=sample_n/len(df_full), random_state=42)).reset_index(drop=True))
        else:
            df = df_full.sample(n=sample_n, random_state=42) if len(df_full)>sample_n else df_full
    else:
        st.warning("Server CSV not found; falling back to small demo.")
        df = make_small_demo()
elif source == "Upload CSV":
    if upload is not None:
        df = pd.read_csv(upload)
    else:
        st.info("Upload a CSV to continue, or pick a different data source.")
        st.stop()
else:
    df = make_small_demo()

# Show preview
st.write("### Data preview")
st.container().markdown('<div class="cm-card">', unsafe_allow_html=True)
st.write(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
st.dataframe(df.head(20), use_container_width=True)
st.container().markdown('</div>', unsafe_allow_html=True)

# Target selection
candidates = [c for c in PREFERRED_TARGETS if c in df.columns]
# Fallbacks if not found
if not candidates:
    candidates = [c for c in df.columns if c.lower() in {"target","label","y"}]
if not candidates:
    candidates = [df.columns[-1]]

target_col = st.sidebar.selectbox("Target column", candidates, index=0)

# -------------------------------
# Feature Engineering (SMT‑aware)
# -------------------------------
def engineer_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.copy()
    y_raw = X.pop(target)

    # Parse date features
    if "Fecha" in X.columns:
        try:
            dt = pd.to_datetime(X["Fecha"], errors="coerce")
            X["weekday"] = dt.dt.weekday
            X["month"] = dt.dt.month
        except Exception:
            X["weekday"] = -1; X["month"] = -1

    # Derived ratios
    def safe_div(a,b):
        a = pd.to_numeric(a, errors='coerce'); b = pd.to_numeric(b, errors='coerce');
        out = a / b.replace(0,np.nan)
        return out.replace([np.inf,-np.inf], np.nan).fillna(0)

    if {"Tiempo_Cambio_Modelo_min","Tiempo_Produccion_efectivo_min"}.issubset(X.columns):
        X["changeover_ratio"] = safe_div(X["Tiempo_Cambio_Modelo_min"], X["Tiempo_Produccion_efectivo_min"]) 
    if {"Paros_Linea_min","Tiempo_Produccion_efectivo_min"}.issubset(X.columns):
        X["downtime_ratio"] = safe_div(X["Paros_Linea_min"], X["Tiempo_Produccion_efectivo_min"]) 
    if {"Unidades_Producidas","Tiempo_Produccion_efectivo_min"}.issubset(X.columns):
        X["uph"] = safe_div(X["Unidades_Producidas"], X["Tiempo_Produccion_efectivo_min"]/60.0)

    # Deviation features for process signals
    for col in ["Avg_Solder_Temp_C","Avg_Paste_Thickness_um","Avg_Humidity","Avg_Misalignment"]:
        if col in X.columns:
            med = pd.to_numeric(X[col], errors='coerce').median()
            X[f"{col}_dev"] = pd.to_numeric(X[col], errors='coerce') - med

    # Avoid leakage: drop metrics tightly coupled to target
    leakage_cols = []
    if target.lower() == "high_scrap":
        leakage_cols += [c for c in ["Scrap_Unidades","Defect_Rate","Defectos_Detectados"] if c in X.columns]
    if target.lower() == "low_oee":
        leakage_cols += [c for c in ["OEE_%"] if c in X.columns]
    X = X.drop(columns=leakage_cols, errors='ignore')

    # Build binary target from raw
    if y_raw.dtype == object:
        y = y_raw.astype(str).str.lower().str.strip().isin({"1","true","yes","y","high","low","si"}).astype(int)
    else:
        # If it's numeric/float like Defect_Rate, convert to high/low via top quantile
        if target == "Defect_Rate":
            thr = np.nanpercentile(pd.to_numeric(y_raw, errors='coerce'), 80)
            y = (pd.to_numeric(y_raw, errors='coerce') >= thr).astype(int)
        else:
            y = (pd.to_numeric(y_raw, errors='coerce').fillna(0) > 0.5).astype(int)

    return X, y

X, y = engineer_features(df, target_col)

# Identify feature types
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

with st.expander("Feature types & leakage guard"):
    st.write({"numeric": num_cols, "categorical": cat_cols})

# -------------------------------
# Model options
# -------------------------------
st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Algorithm", [opt for opt in (["LightGBM"] if HAS_LGBM else []) + ["Random Forest","Logistic Regression"]], index=0 if HAS_LGBM else 1)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 9999, 42, 1)

# Preprocessor
num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])

# Class imbalance handling
pos_rate = float(y.mean())
scale_pos_weight = None
if HAS_LGBM and model_name == "LightGBM":
    try:
        scale_pos_weight = (1-pos_rate)/max(pos_rate,1e-6)
    except Exception:
        scale_pos_weight = None

# Estimator
if HAS_LGBM and model_name == "LightGBM":
    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        class_weight=None,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )
elif model_name == "Random Forest":
    clf = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state, class_weight="balanced")
else:
    clf = LogisticRegression(max_iter=400, class_weight="balanced")

pipe = Pipeline([("pre", pre), ("clf", clf)])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:,1]

st.sidebar.header("Threshold")
th = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01)
y_pred = (proba >= th).astype(int)

# -------------------------------
# Metrics & charts
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
pre_ = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, proba)
except Exception:
    auc = float("nan")

st.markdown('<div class="cm-card" style="padding:.6rem 1rem">', unsafe_allow_html=True)
col1,col2,col3,col4,col5 = st.columns(5)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("Precision", f"{pre_:.3f}")
col3.metric("Recall", f"{rec:.3f}")
col4.metric("F1", f"{f1:.3f}")
col5.metric("ROC AUC", f"{auc:.3f}")
st.markdown('</div>', unsafe_allow_html=True)

# Theme for matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
mpl.rcParams.update({'figure.facecolor':'#0b1220','axes.facecolor':'#0b1220','axes.edgecolor':'#334155','axes.labelcolor':'#e6f1ff','xtick.color':'#a0aec0','ytick.color':'#a0aec0','grid.color':'#1f2a3a','text.color':'#e6f1ff'})

mpl.rcParams['axes.prop_cycle'] = cycler(color=['#a6ff00', '#14ACBA', '#7C3AED', '#eab308'])

st.markdown('<div class="cm-card" style="padding:1rem">', unsafe_allow_html=True)
st.write("#### Confusion Matrix")
st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

cc1, cc2 = st.columns(2)
with cc1:
    st.write("#### ROC Curve")
    fig1, ax1 = plt.subplots(); RocCurveDisplay.from_predictions(y_test, proba, ax=ax1, name="Model"); ax1.grid(True, linestyle='--', alpha=.25); st.pyplot(fig1)
with cc2:
    st.write("#### Precision–Recall Curve")
    fig2, ax2 = plt.subplots(); PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax2, name="Model"); ax2.grid(True, linestyle='--', alpha=.25); st.pyplot(fig2)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Batch predictions + downloads
# -------------------------------
st.markdown('<div class="cm-card" style="padding:1rem">', unsafe_allow_html=True)
with st.expander("Batch predictions & downloads", expanded=True):
    all_proba = pipe.predict_proba(X)[:,1]
    all_pred = (all_proba >= th).astype(int)
    out = df.copy(); out["pred_proba"] = all_proba; out["pred_label"] = all_pred
    st.dataframe(out.head(20), use_container_width=True)

    csv = out.to_csv(index=False).encode()
    st.download_button("⬇️ Download predictions CSV", csv, file_name="smt_predictions.csv", mime="text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Ops ROI — scrap reduction / OEE uplift (illustrative)
# -------------------------------
st.markdown('<div class="cm-card" style="padding:1rem">', unsafe_allow_html=True)
st.write("### Estimate Monthly Impact (illustrative)")
col1, col2, col3 = st.columns(3)
scrap_per_unit = col1.number_input("Scrap cost per defective unit ($)", min_value=0.0, value=8.0, step=0.5)
units_per_month = col2.number_input("Units per month", min_value=0, value=200000, step=1000)
oee_value_per_point = col3.number_input("Value per OEE point ($/mo)", min_value=0.0, value=5000.0, step=500.0)

# Scrap/OEE impact logic
# Use overall class balance for baseline (more stable than just test split)
baseline_scrap_rate = float(y.mean())

if target_col.lower() == "high_scrap":
    prevent_frac = st.slider("Assumed prevention rate when flagged", 0.0, 1.0, 0.6, 0.05)
    detection = rec  # model recall on positives (High_Scrap) at current threshold

    # Expected (not rounded) savings to avoid truncating to zero for small rates
    expected_saved_units = units_per_month * baseline_scrap_rate * max(detection, 0.0) * prevent_frac
    scrap_savings = expected_saved_units * scrap_per_unit

    # Friendly hints if savings look like zero
    if baseline_scrap_rate == 0:
        st.info("No positive High_Scrap examples found in the selected data — baseline scrap rate is 0%.")
    elif detection == 0:
        st.info("Model recall is 0 at this threshold. Try lowering the threshold to catch more positives.")
    elif prevent_frac == 0:
        st.info("Prevention rate is set to 0%. Increase it to simulate avoided scrap when flagged.")

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Baseline scrap rate", f"{baseline_scrap_rate*100:.2f}%")
    rc2.metric("Saved units/month (expected)", f"{expected_saved_units:,.1f}")
    rc3.metric("High_Scrap monthly estimated savings", f"${scrap_savings:,.0f}")
    st.caption(f"Recall @ threshold: {detection:.2f}  •  Prevention: {prevent_frac:.2f}")

else:
    st.info("Switch Target column to 'High_Scrap' to compute scrap savings. Current target is not High_Scrap.")
    # Simple OEE value proxy using recall (illustrative only)
    oee_points_recovered = rec  # assume 1 OEE point recoverable at full recall
    est_value = oee_points_recovered * oee_value_per_point
    rc1, rc2 = st.columns(2)
    rc1.metric("Estimated OEE points recovered", f"{oee_points_recovered:.2f}")
    rc2.metric("Estimated OEE value/month", f"${est_value:,.0f}")

st.caption("These are illustrative estimates. For accurate ROI, align with accounting, test coverage, and prevention rate assumptions.")
st.markdown('</div>', unsafe_allow_html=True)

