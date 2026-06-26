import streamlit as st
import numpy as np
import joblib

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Custom Styling
# -------------------------------------------------

st.markdown("""
<style>
    /* Overall app background */
    .stApp {
        background-color: #f7f8fa;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #374151;
        border-left: 4px solid #ef4444;
        padding-left: 0.6rem;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Card container — simple bordered box instead of a background-color
       swap, so we never need to fight the widget's own internal colors */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    /* ============================================================
       GLOBAL WIDGET STYLING
       Streamlit/BaseWeb widgets carry their own internal background
       and text colors that vary by theme. Instead of scoping fixes to
       a card wrapper (fragile, theme-dependent), we set colors
       directly and globally on every relevant widget so they are
       readable no matter what theme the visitor's browser/system uses.
       ============================================================ */

    /* All widget labels (Age, Sex, Resting Blood Pressure, etc.) */
    .stSlider label,
    .stSelectbox label,
    .stNumberInput label,
    .stTextInput label,
    .stRadio label,
    label[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] p {
        color: #1f2937 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    /* Number input boxes */
    .stNumberInput input {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
    }

    .stNumberInput button {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
    }

    /* Selectbox closed (collapsed) box */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
    }

    .stSelectbox div[data-baseweb="select"] span {
        color: #111827 !important;
    }

    .stSelectbox svg {
        fill: #111827 !important;
    }

    /* Selectbox dropdown popup menu (renders in a portal, separate
       from the card, so it always needs its own global rule) */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #ffffff !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] li,
    ul[data-testid="stSelectboxVirtualDropdown"] li span,
    ul[data-testid="stSelectboxVirtualDropdown"] li div {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #f3f4f6 !important;
    }

    /* Slider: track, numeric value above the thumb, and min/max labels */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider [data-testid="stSliderThumbValue"],
    .stSlider div[role="slider"] {
        color: #1f2937 !important;
    }

    .stSlider [data-baseweb="slider"] div[role="slider"]::after {
        color: #1f2937 !important;
    }

    /* Help-icon tooltip text */
    div[data-testid="stTooltipIcon"] svg {
        fill: #6b7280 !important;
    }

    /* Section text / captions / help text in general */
    .stMarkdown p, .stCaption {
        color: #374151;
    }

    /* Disclaimer banner */
    .disclaimer-box {
        background-color: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: #92400e;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
    }

    /* Risk result cards */
    .risk-card {
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        border: 2px solid;
    }

    .risk-card-low {
        background-color: #ecfdf5;
        border-color: #10b981;
        color: #065f46;
    }

    .risk-card-moderate {
        background-color: #fffbeb;
        border-color: #f59e0b;
        color: #92400e;
    }

    .risk-card-high {
        background-color: #fef2f2;
        border-color: #ef4444;
        color: #991b1b;
    }

    .risk-label {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .risk-sub {
        font-size: 0.95rem;
        opacity: 0.85;
    }

    .risk-percentage {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }

    /* Custom field labels — rendered by us via st.markdown instead of
       relying on Streamlit's native widget label, which can inherit
       low-contrast colors from the active theme that are hard to
       override with CSS alone */
    .field-label {
        color: #1f2937 !important;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
        margin-top: 0.4rem;
        display: block;
    }

    /* Footer */
    .footer-caption {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 1rem;
    }

    /* Sidebar metric badges */
    .sidebar-metric {
        background-color: #f3f4f6;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    metrics = joblib.load("metrics.pkl")
    return model, scaler, metrics

try:
    model, scaler, metrics = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError as e:
    artifacts_loaded = False
    load_error = str(e)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("❤️ Heart Disease Predictor")

st.sidebar.markdown("### About")
st.sidebar.write(
    "This app estimates the likelihood of heart disease using a "
    "**Logistic Regression** model trained on the Cleveland Heart "
    "Disease Dataset."
)

st.sidebar.markdown("### Model Snapshot")

if artifacts_loaded:
    st.sidebar.markdown(
        f"<div class='sidebar-metric'>🎯 Accuracy: <b>{metrics['accuracy']*100:.2f}%</b></div>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f"<div class='sidebar-metric'>📈 ROC-AUC: <b>{metrics['roc_auc']:.3f}</b></div>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.warning("Model files not found in this directory yet.")

st.sidebar.markdown("### Risk Bands")
st.sidebar.markdown("🟢 **Low** — below 30%")
st.sidebar.markdown("🟡 **Moderate** — 30% to 60%")
st.sidebar.markdown("🔴 **High** — above 60%")

st.sidebar.markdown("---")
st.sidebar.markdown("**Developer:** Ishita Chandra")

# -------------------------------------------------
# Custom Label Helper
# -------------------------------------------------

def field_label(text):
    """Renders a label we fully control instead of relying on Streamlit's
    native widget label, whose color can be hard to override reliably
    across light/dark themes."""
    st.markdown(f"<span class='field-label'>{text}</span>", unsafe_allow_html=True)

# -------------------------------------------------
# Main Page Header
# -------------------------------------------------

st.markdown("<div class='main-title'>❤️ Heart Disease Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-subtitle'>Fill in the patient's clinical details to estimate heart disease risk</div>",
    unsafe_allow_html=True
)

st.markdown("""
<div class='disclaimer-box'>
⚕️ <b>Disclaimer:</b> This tool is for educational purposes only. It is not a substitute
for professional medical advice, diagnosis, or treatment. Always consult a qualified
healthcare provider with any questions about a medical condition.
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(
        "⚠️ Could not find `heart_model.pkl`, `scaler.pkl`, or `metrics.pkl` in the app "
        "directory. Predictions are disabled until these files are present."
    )

# -------------------------------------------------
# Input Fields — Demographics & Basic Vitals
# -------------------------------------------------

st.markdown("<div class='section-header'>🧍 Demographics & Vitals</div>", unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        field_label("Age")
        age = st.slider(
            "Age",
            20, 100, 50,
            help="Patient's age in years.",
            label_visibility="collapsed"
        )

        field_label("Sex")
        sex = st.selectbox(
            "Sex",
            ["Female", "Male"],
            label_visibility="collapsed"
        )

    with col2:
        field_label("Resting Blood Pressure (mm Hg)")
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80, max_value=250, value=120,
            help="Blood pressure while resting (mm Hg).",
            label_visibility="collapsed"
        )

        field_label("Cholesterol (mg/dL)")
        chol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100, max_value=600, value=200,
            help="Serum cholesterol level (mg/dL).",
            label_visibility="collapsed"
        )

# -------------------------------------------------
# Input Fields — Symptoms & Test Results
# -------------------------------------------------

st.markdown("<div class='section-header'>🩺 Symptoms & Test Results</div>", unsafe_allow_html=True)

with st.container(border=True):
    col3, col4 = st.columns(2)

    with col3:
        field_label("Chest Pain Type")
        cp = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ],
            help="""
Typical Angina: Chest pain due to reduced blood flow.

Atypical Angina: Chest pain with unusual characteristics.

Non-anginal Pain: Chest pain unrelated to the heart.

Asymptomatic: No chest pain.
""",
            label_visibility="collapsed"
        )

        field_label("Fasting Blood Sugar > 120 mg/dL")
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            ["No", "Yes"],
            help="Whether fasting blood sugar exceeds 120 mg/dL.",
            label_visibility="collapsed"
        )

        field_label("Resting ECG")
        restecg = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy"
            ],
            help="Electrocardiogram results while resting.",
            label_visibility="collapsed"
        )

    with col4:
        field_label("Exercise-Induced Angina")
        exang = st.selectbox(
            "Exercise-Induced Angina",
            ["No", "Yes"],
            help="Chest pain during physical exercise.",
            label_visibility="collapsed"
        )

        field_label("Thalassemia")
        thal = st.selectbox(
            "Thalassemia",
            [
                "Normal",
                "Fixed Defect",
                "Reversible Defect"
            ],
            help="""
Normal: Normal blood flow.

Fixed Defect: Permanent reduction in blood flow.

Reversible Defect: Reduced blood flow during exercise.
""",
            label_visibility="collapsed"
        )

        field_label("Major Vessels Visible (Fluoroscopy)")
        ca = st.selectbox(
            "Major Vessels Visible (Fluoroscopy)",
            [0, 1, 2, 3],
            help="Number of major blood vessels visible via fluoroscopy.",
            label_visibility="collapsed"
        )

# -------------------------------------------------
# Input Fields — Exercise Response
# -------------------------------------------------

st.markdown("<div class='section-header'>🏃 Exercise Response</div>", unsafe_allow_html=True)

with st.container(border=True):
    col5, col6 = st.columns(2)

    with col5:
        field_label("Maximum Heart Rate Achieved")
        thalach = st.slider(
            "Maximum Heart Rate Achieved",
            60, 220, 150,
            help="Maximum heart rate achieved during exercise.",
            label_visibility="collapsed"
        )

    with col6:
        field_label("ST Depression (Old Peak)")
        oldpeak = st.slider(
            "ST Depression (Old Peak)",
            0.0, 7.0, 1.0, step=0.1,
            help="ST depression induced by exercise compared with rest.",
            label_visibility="collapsed"
        )

    field_label("ST Segment Slope")
    slope = st.selectbox(
        "ST Segment Slope",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ],
        help="Slope of the peak exercise ST segment.",
        label_visibility="collapsed"
    )

# -------------------------------------------------
# Convert Inputs
# -------------------------------------------------

sex_val = 1 if sex == "Male" else 0

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_map = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2
}

cp_val = cp_map[cp]
restecg_val = restecg_map[restecg]
slope_val = slope_map[slope]
thal_val = thal_map[thal]

fbs_val = 1 if fbs == "Yes" else 0
exang_val = 1 if exang == "Yes" else 0

# -------------------------------------------------
# Risk Tier Helper
# -------------------------------------------------

def get_risk_tier(probability):
    """Returns (tier_name, css_class, emoji) based on probability."""
    if probability < 0.30:
        return "Low Risk", "risk-card-low", "✅"
    elif probability < 0.60:
        return "Moderate Risk", "risk-card-moderate", "⚠️"
    else:
        return "High Risk", "risk-card-high", "🚨"

# -------------------------------------------------
# Prediction
# -------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True,
    disabled=not artifacts_loaded,
    type="primary"
)

if predict_clicked and artifacts_loaded:

    sample = np.array([[
        age,
        sex_val,
        cp_val,
        trestbps,
        chol,
        fbs_val,
        restecg_val,
        thalach,
        exang_val,
        oldpeak,
        slope_val,
        ca,
        thal_val
    ]])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]

    tier_name, tier_class, tier_emoji = get_risk_tier(probability)

    st.divider()
    st.subheader("Prediction Result")

    st.markdown(f"""
    <div class='risk-card {tier_class}'>
        <div class='risk-label'>{tier_emoji} {tier_name}</div>
        <div class='risk-percentage'>{probability*100:.1f}%</div>
        <div class='risk-sub'>Estimated probability of heart disease</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(probability))

    # Quick-glance scale showing where this result falls
    scale_col1, scale_col2, scale_col3 = st.columns(3)
    with scale_col1:
        st.markdown(
            f"{'🟢' if tier_name == 'Low Risk' else '⚪'} **Low**<br><span style='color:#6b7280;font-size:0.85rem'>0–30%</span>",
            unsafe_allow_html=True
        )
    with scale_col2:
        st.markdown(
            f"{'🟡' if tier_name == 'Moderate Risk' else '⚪'} **Moderate**<br><span style='color:#6b7280;font-size:0.85rem'>30–60%</span>",
            unsafe_allow_html=True
        )
    with scale_col3:
        st.markdown(
            f"{'🔴' if tier_name == 'High Risk' else '⚪'} **High**<br><span style='color:#6b7280;font-size:0.85rem'>60–100%</span>",
            unsafe_allow_html=True
        )

    if tier_name == "High Risk":
        st.warning(
            "This result suggests a high likelihood of heart disease based on the inputs "
            "provided. Please consult a cardiologist or healthcare provider promptly."
        )
    elif tier_name == "Moderate Risk":
        st.info(
            "This result suggests some elevated risk factors. Consider discussing these "
            "results with a healthcare provider for further evaluation."
        )
    else:
        st.success(
            "This result suggests a lower likelihood of heart disease based on the inputs "
            "provided. Routine checkups are still recommended."
        )

    st.divider()
    st.subheader("Model Performance")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.2f}%")
        st.metric("🎯 Precision", f"{metrics['precision']*100:.2f}%")
        st.metric("📈 ROC-AUC", f"{metrics['roc_auc']:.3f}")

    with c2:
        st.metric("🔍 Recall", f"{metrics['recall']*100:.2f}%")
        st.metric("⚖️ F1 Score", f"{metrics['f1']*100:.2f}%")

# -------------------------------------------------
# Expandable Section
# -------------------------------------------------

with st.expander("ℹ️ About this Model"):

    st.markdown("""
### Machine Learning Model

- **Algorithm:** Logistic Regression
- **Dataset:** Cleveland Heart Disease Dataset
- **Training Samples:** 237
- **Testing Samples:** 60
- **Number of Features:** 13

### Evaluation Metrics

- Accuracy: **73.33%**
- Precision: **70.00%**
- Recall: **75.00%**
- F1 Score: **72.41%**
- ROC-AUC: **0.839**

### Risk Bands Used in This App

- 🟢 **Low Risk:** predicted probability below 30%
- 🟡 **Moderate Risk:** predicted probability between 30% and 60%
- 🔴 **High Risk:** predicted probability above 60%

This application predicts the probability of heart disease based on clinical attributes.

**Disclaimer:** This tool is for educational purposes only and should not be used for medical diagnosis.
""")

st.divider()

st.markdown(
    "<div class='footer-caption'>Built with ❤️ using Python, Scikit-learn and Streamlit • © Ishita Chandra</div>",
    unsafe_allow_html=True
)