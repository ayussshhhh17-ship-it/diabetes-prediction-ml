import streamlit as st
import numpy as np
import pickle
import os

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction | ML App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS – Premium Medical Dark UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
        min-height: 100vh;
    }

    /* Hide default Streamlit header */
    #MainMenu, footer, header { visibility: hidden; }

    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, rgba(0,212,255,0.15) 0%, rgba(9,9,121,0.2) 50%, rgba(0,212,255,0.05) 100%);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0,212,255,0.05) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b61ff, #00d4ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        margin: 0;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        border-color: rgba(0,212,255,0.4);
        background: rgba(0,212,255,0.05);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    /* Feature info cards */
    .feature-info {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        border-left: 3px solid #7b61ff;
    }

    /* Result card */
    .result-diabetic {
        background: linear-gradient(135deg, rgba(255,59,59,0.15), rgba(255,59,59,0.05));
        border: 2px solid rgba(255,59,59,0.5);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        animation: fadeIn 0.5s ease;
    }
    .result-healthy {
        background: linear-gradient(135deg, rgba(0,255,163,0.15), rgba(0,255,163,0.05));
        border: 2px solid rgba(0,255,163,0.5);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        animation: fadeIn 0.5s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-subtitle {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.65);
    }

    /* Predict button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00d4ff, #7b61ff);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.9rem 2rem;
        border: none;
        border-radius: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 20px rgba(0,212,255,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,212,255,0.5);
    }

    /* Sliders & inputs */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #7b61ff) !important;
    }
    label {
        color: rgba(255,255,255,0.85) !important;
        font-weight: 500 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 30, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255,255,255,0.8);
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: rgba(255,255,255,0.9);
        border-bottom: 2px solid rgba(0,212,255,0.3);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Progress bar custom */
    .progress-container {
        background: rgba(255,255,255,0.08);
        border-radius: 50px;
        height: 8px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .progress-fill-red {
        background: linear-gradient(90deg, #ff6b6b, #ff3b3b);
        height: 100%;
        border-radius: 50px;
        transition: width 1s ease;
    }
    .progress-fill-green {
        background: linear-gradient(90deg, #00ffa3, #00c774);
        height: 100%;
        border-radius: 50px;
        transition: width 1s ease;
    }

    /* Warning banner */
    .disclaimer {
        background: rgba(255,165,0,0.08);
        border: 1px solid rgba(255,165,0,0.25);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        font-size: 0.8rem;
        color: rgba(255,165,0,0.85);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model & Scaler
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model = pickle.load(open(os.path.join(base, 'diabetes_model.sav'), 'rb'))
    scaler = pickle.load(open(os.path.join(base, 'scaler.sav'), 'rb'))
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 About This App")
    st.markdown("""
    This app uses a **Support Vector Machine (SVM)** trained on the **PIMA Indians Diabetes Dataset** to predict diabetes risk.

    ---
    **Dataset:** PIMA Indians Diabetes  
    **Algorithm:** SVM (Linear Kernel)  
    **Features:** 8 clinical inputs  
    **Accuracy:** ~78% on test data  

    ---
    """)

    st.markdown("### 📋 Feature Reference")
    features_info = {
        "Pregnancies": "Number of times pregnant",
        "Glucose": "Plasma glucose concentration (mg/dL)",
        "Blood Pressure": "Diastolic blood pressure (mm Hg)",
        "Skin Thickness": "Triceps skin fold thickness (mm)",
        "Insulin": "2-Hour serum insulin (mu U/ml)",
        "BMI": "Body Mass Index (kg/m²)",
        "DPF": "Diabetes Pedigree Function (genetic score)",
        "Age": "Age in years"
    }
    for feat, desc in features_info.items():
        st.markdown(f'<div class="feature-info"><b>{feat}:</b> {desc}</div>', unsafe_allow_html=True)

    st.markdown('<div class="disclaimer">⚠️ <b>Disclaimer:</b> This tool is for educational purposes only and does not substitute professional medical advice.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🩺 Diabetes Risk Predictor</div>
    <div class="hero-subtitle">AI-powered clinical decision support using Support Vector Machine · PIMA Indians Dataset</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model status
# ─────────────────────────────────────────────
if not model_loaded:
    st.error(f"❌ Could not load model files. Please run `diabetes_prediction.py` first.\n\nError: {model_error}")
    st.stop()

# ─────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Enter Patient Clinical Data</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input(
        "🤰 Pregnancies",
        min_value=0, max_value=20, value=1, step=1,
        help="Number of times the patient has been pregnant"
    )
    glucose = st.slider(
        "🩸 Glucose (mg/dL)",
        min_value=0, max_value=300, value=110,
        help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test"
    )
    blood_pressure = st.slider(
        "💓 Blood Pressure (mm Hg)",
        min_value=0, max_value=150, value=72,
        help="Diastolic blood pressure"
    )

with col2:
    skin_thickness = st.slider(
        "📏 Skin Thickness (mm)",
        min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness"
    )
    insulin = st.slider(
        "💉 Insulin (mu U/ml)",
        min_value=0, max_value=900, value=80,
        help="2-Hour serum insulin level"
    )
    bmi = st.number_input(
        "⚖️ BMI (kg/m²)",
        min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f",
        help="Body Mass Index"
    )

with col3:
    dpf = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.35, step=0.001, format="%.3f",
        help="A function that scores likelihood of diabetes based on family history"
    )
    age = st.slider(
        "🎂 Age (years)",
        min_value=1, max_value=120, value=33,
        help="Age of the patient in years"
    )

    # Risk indicator
    risk_factors = 0
    if glucose > 140: risk_factors += 1
    if bmi > 30: risk_factors += 1
    if blood_pressure > 90: risk_factors += 1
    if age > 45: risk_factors += 1
    if insulin > 200: risk_factors += 1
    if pregnancies > 6: risk_factors += 1

    st.markdown(f"""
    <div class="metric-card" style="margin-top:1rem;">
        <div class="metric-value" style="color:{'#ff6b6b' if risk_factors >= 3 else '#ffd166' if risk_factors >= 1 else '#00ffa3'}">
            {risk_factors}/6
        </div>
        <div class="metric-label">Risk Indicators</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
st.markdown("---")
predict_col, _ = st.columns([1, 2])

with predict_col:
    predict_btn = st.button("🔍 Predict Diabetes Risk", key="predict")

# ─────────────────────────────────────────────
# Prediction & Result
# ─────────────────────────────────────────────
if predict_btn:
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    std_data = scaler.transform(input_data)
    prediction = model.predict(std_data)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)

    res_col, metrics_col = st.columns([1.2, 1])

    with res_col:
        if prediction[0] == 1:
            st.markdown("""
            <div class="result-diabetic">
                <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">⚠️</div>
                <div class="result-title" style="color: #ff6b6b;">Diabetic Risk Detected</div>
                <div class="result-subtitle">
                    Based on the clinical parameters provided, this patient shows<br>
                    indicators <b>consistent with diabetes</b>. Please consult a physician.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-healthy">
                <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">✅</div>
                <div class="result-title" style="color: #00ffa3;">Low Diabetes Risk</div>
                <div class="result-subtitle">
                    Based on the clinical parameters provided, this patient shows<br>
                    indicators <b>consistent with a non-diabetic profile</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with metrics_col:
        st.markdown('<div class="section-header">📈 Clinical Snapshot</div>', unsafe_allow_html=True)

        def risk_bar(label, value, threshold, unit, color_over="#ff6b6b", color_ok="#00ffa3"):
            is_high = value > threshold
            fill_color = color_over if is_high else color_ok
            pct = min(int((value / (threshold * 1.5)) * 100), 100)
            status = "⚠️ High" if is_high else "✅ Normal"
            return f"""
            <div style="margin-bottom:0.8rem;">
                <div style="display:flex; justify-content:space-between; color:rgba(255,255,255,0.8); font-size:0.85rem; margin-bottom:3px;">
                    <span>{label}</span>
                    <span style="color:{fill_color};">{value} {unit} · {status}</span>
                </div>
                <div class="progress-container">
                    <div style="background:{fill_color}; width:{pct}%; height:100%; border-radius:50px;"></div>
                </div>
            </div>"""

        bars_html = (
            risk_bar("Glucose", glucose, 140, "mg/dL") +
            risk_bar("Blood Pressure", blood_pressure, 90, "mm Hg") +
            risk_bar("BMI", bmi, 30, "kg/m²") +
            risk_bar("Insulin", insulin, 200, "mu U/ml") +
            risk_bar("Age", age, 45, "yrs")
        )
        st.markdown(bars_html, unsafe_allow_html=True)

    # Input summary table
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Input Summary</div>', unsafe_allow_html=True)

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    inputs = [
        ("🤰 Pregnancies", pregnancies, ""),
        ("🩸 Glucose", glucose, "mg/dL"),
        ("💓 Blood Pressure", blood_pressure, "mm Hg"),
        ("📏 Skin Thickness", skin_thickness, "mm"),
        ("💉 Insulin", insulin, "μU/ml"),
        ("⚖️ BMI", bmi, "kg/m²"),
        ("🧬 DPF", dpf, ""),
        ("🎂 Age", age, "yrs"),
    ]
    cols = [summary_col1, summary_col2, summary_col3, summary_col4]
    for i, (label, val, unit) in enumerate(inputs):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem; color:rgba(255,255,255,0.5); margin-bottom:4px;">{label}</div>
                <div class="metric-value" style="font-size:1.4rem;">{val}</div>
                <div class="metric-label">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.3); font-size:0.78rem; margin-top:3rem; padding-top:1.5rem; border-top:1px solid rgba(255,255,255,0.06);">
    Built with ❤️ using Python · Scikit-learn · Streamlit &nbsp;|&nbsp; PIMA Indians Diabetes Dataset
    <br><i>For educational purposes only — not a substitute for professional medical diagnosis.</i>
</div>
""", unsafe_allow_html=True)
