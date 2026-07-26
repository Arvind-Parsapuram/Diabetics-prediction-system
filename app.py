import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #8B0000;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #8B0000;
        margin-top: 1rem;
        border-left: 4px solid #8B0000;
        padding-left: 0.6rem;
    }
    .metric-card {
        background: #f9f9f9;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border: 1px solid #ddd;
    }
    .result-positive {
        background-color: #ffe0e0;
        border-left: 6px solid #cc0000;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 600;
        color: #8B0000;
    }
    .result-negative {
        background-color: #e0f7e9;
        border-left: 6px solid #2e7d32;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 600;
        color: #1b5e20;
    }
    .info-box {
        background: #fff8f8;
        border: 1px solid #f5c6c6;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #444;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load & preprocess data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("diabetes.csv")

    # ── Step 1: Mark biological zeros as NaN ──────────────────────────────────
    # Columns where 0 is physiologically impossible
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_clean = df.copy()
    df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)

    # ── Step 2: Semi-supervised XGBoost imputation for Insulin ───────────────
    # Use rows where Insulin is known to train the imputer
    insulin_features = ["Pregnancies", "Glucose", "BloodPressure",
                        "SkinThickness", "BMI", "DiabetesPedigreeFunction", "Age"]

    known_mask   = df_clean["Insulin"].notna()
    unknown_mask = df_clean["Insulin"].isna()

    X_known   = df_clean.loc[known_mask,   insulin_features].fillna(df_clean[insulin_features].median())
    y_known   = df_clean.loc[known_mask,   "Insulin"]
    X_unknown = df_clean.loc[unknown_mask, insulin_features].fillna(df_clean[insulin_features].median())

    xgb_imputer = XGBRegressor(n_estimators=200, max_depth=4,
                               learning_rate=0.05, random_state=42,
                               verbosity=0)
    xgb_imputer.fit(X_known, y_known)
    df_clean.loc[unknown_mask, "Insulin"] = xgb_imputer.predict(X_unknown).clip(min=0)

    # ── Step 3: Impute remaining NaNs with median ─────────────────────────────
    for col in zero_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

    return df, df_clean, xgb_imputer


@st.cache_resource
def train_model(df_clean):
    X = df_clean.drop("Outcome", axis=1)
    y = df_clean["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Random Forest Classifier (ensemble of decision trees) ────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc  = accuracy_score(y_test,  rf.predict(X_test))
    report    = classification_report(y_test, rf.predict(X_test), output_dict=True)

    return rf, X_train, X_test, y_train, y_test, train_acc, test_acc, report, list(X.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Load everything
# ─────────────────────────────────────────────────────────────────────────────
df_raw, df_clean, xgb_imputer = load_and_preprocess()
rf, X_train, X_test, y_train, y_test, train_acc, test_acc, report, feature_cols = train_model(df_clean)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – patient input
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/diabetes.png", width=80)
st.sidebar.markdown("## 🩺 Patient Parameters")
st.sidebar.markdown("Adjust the sliders to match the patient's clinical data.")

def user_report():
    pregnancies = st.sidebar.slider("Pregnancies",        0,   17,  3)
    glucose     = st.sidebar.slider("Glucose (mg/dL)",   44,  200, 120)
    bp          = st.sidebar.slider("Blood Pressure (mmHg)", 24, 122, 70)
    skin        = st.sidebar.slider("Skin Thickness (mm)",    7, 100,  20)
    insulin     = st.sidebar.slider("Insulin (μU/mL)",    14,  846,  79)
    bmi         = st.sidebar.slider("BMI",                18,   67,  25)
    dpf         = st.sidebar.slider("Diabetes Pedigree Function", 0.08, 2.42, 0.47)
    age         = st.sidebar.slider("Age (years)",        21,   88,  33)

    return {
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": bp, "SkinThickness": skin,
        "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age
    }

patient_data = user_report()
user_df = pd.DataFrame([patient_data])


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🩺 Diabetes Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pima Indian Diabetes Dataset · Random Forest + XGBoost Imputation · Streamlit</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Model Performance", "🔍 Patient Prediction"])


# ── Tab 1: Data Overview ──────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Dataset Statistics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records",    len(df_raw))
    col2.metric("Diabetic Cases",   int(df_raw["Outcome"].sum()))
    col3.metric("Non-Diabetic",     int((df_raw["Outcome"] == 0).sum()))

    st.markdown('<div class="section-title">Raw Data Summary (Pima Indian Dataset)</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown('<div class="section-title">Missing Value Analysis (zeros replaced with NaN)</div>', unsafe_allow_html=True)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    missing = (df_raw[zero_cols] == 0).sum().rename("Zero / Missing Count")
    st.bar_chart(missing)

    st.markdown('<div class="section-title">Feature Distributions (after XGBoost Imputation)</div>', unsafe_allow_html=True)
    st.bar_chart(df_clean.drop("Outcome", axis=1).mean().rename("Mean Value"))


# ── Tab 2: Model Performance ──────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Model Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Step 1 – XGBoost Imputer:</b> A gradient-boosted regressor trained on rows with observed insulin
    values predicts missing Insulin entries (semi-supervised approach). This improves data quality
    before classification.
    </div>
    <div class="info-box">
    <b>Step 2 – Random Forest Classifier:</b> An ensemble of 200 decision trees (max_depth = 8,
    class-balanced, stratified 80/20 split) performs the final diabetic / non-diabetic classification.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Accuracy</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("🏋️ Train Accuracy", f"{train_acc*100:.2f}%")
    c2.metric("🧪 Test Accuracy",  f"{test_acc*100:.2f}%")

    st.markdown('<div class="section-title">Classification Report (Test Set)</div>', unsafe_allow_html=True)
    rdf = pd.DataFrame(report).T.drop("accuracy", errors="ignore")
    st.dataframe(rdf.style.format("{:.2f}"), use_container_width=True)

    st.markdown('<div class="section-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    importance_df = pd.DataFrame({
        "Feature":   feature_cols,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False).set_index("Feature")
    st.bar_chart(importance_df)


# ── Tab 3: Patient Prediction ─────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Patient Input Summary</div>', unsafe_allow_html=True)
    st.dataframe(user_df, use_container_width=True)

    # Predict
    prediction    = rf.predict(user_df)[0]
    proba         = rf.predict_proba(user_df)[0]
    risk_pct      = proba[1] * 100

    st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)
    st.progress(int(risk_pct), text=f"Diabetes Risk Score: {risk_pct:.1f}%")

    st.markdown('<div class="section-title">Diagnosis</div>', unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(
            f'<div class="result-positive">⚠️ High Risk: The model predicts <b>Diabetic</b> '
            f'(confidence {risk_pct:.1f}%). Please consult a healthcare professional.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-negative">✅ Low Risk: The model predicts <b>Non-Diabetic</b> '
            f'(confidence {100-risk_pct:.1f}%). Maintain a healthy lifestyle!</div>',
            unsafe_allow_html=True
        )

    st.markdown("""
    <br><small style="color:#888;">
    ⚠️ <b>Disclaimer:</b> This tool is for educational and research purposes only.
    It is not a substitute for professional medical diagnosis or advice.
    </small>
    """, unsafe_allow_html=True)
