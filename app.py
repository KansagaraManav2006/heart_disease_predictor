import streamlit as st
from datetime import datetime
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

from utils import (
    ArtifactLoadError,
    build_diabetes_features,
    build_heart_features,
    load_diabetes_model,
    load_diabetes_scaler,
    load_heart_model,
    load_heart_scaler,
    predict_diabetes,
    predict_heart,
)


def build_pdf_report(disease_name, patient_name, inputs, prediction_label, probability_percent):
    if FPDF is None:
        return None
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    safe_name = patient_name.strip() or "Unknown"
    pdf = FPDF()
    pdf.set_margins(left=12, top=12, right=12)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    page_width = getattr(pdf, "epw", None) or (pdf.w - pdf.l_margin - pdf.r_margin)

    pdf.set_fill_color(26, 33, 62)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.rect(pdf.l_margin, pdf.t_margin, page_width, 16, style="F")
    pdf.set_xy(pdf.l_margin + 4, pdf.t_margin + 4)
    pdf.cell(page_width - 8, 8, "Disease Prediction Report")

    pdf.set_text_color(0, 0, 0)
    pdf.ln(20)
    pdf.set_font("Helvetica", size=11)
    left_col = page_width * 0.6
    right_col = page_width - left_col
    pdf.cell(left_col, 7, f"Generated: {timestamp}")
    pdf.cell(right_col, 7, f"Condition: {disease_name}", ln=1)
    pdf.set_font("Helvetica", style="B", size=13)
    pdf.cell(left_col, 8, f"Patient Name: {safe_name}")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(right_col, 8, "", ln=1)
    pdf.ln(2)

    pdf.set_fill_color(240, 245, 250)
    pdf.set_text_color(26, 33, 62)
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(page_width, 8, "Inputs", ln=1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", size=11)

    label_width = 55
    value_width = page_width - label_width

    def draw_kv_row(label, value):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", style="B", size=11)
        pdf.cell(label_width, 7, str(label))
        pdf.set_font("Helvetica", size=11)
        try:
            pdf.multi_cell(value_width, 7, str(value), new_x="LMARGIN", new_y="NEXT")
        except TypeError:
            pdf.multi_cell(value_width, 7, str(value))

    for key, value in inputs.items():
        draw_kv_row(key, value)

    pdf.ln(2)
    pdf.set_fill_color(240, 245, 250)
    pdf.set_text_color(26, 33, 62)
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(page_width, 8, "Result", ln=1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", size=11)

    risk_color = (244, 67, 54) if prediction_label == "High Risk" else (76, 175, 80)
    pdf.set_text_color(*risk_color)
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(page_width, 7, f"Prediction: {prediction_label}", ln=1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(page_width, 7, f"Risk Level: {probability_percent:.1f}%", ln=1)

    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, (bytes, bytearray)):
        return bytes(pdf_output)
    return str(pdf_output).encode("latin1")


def load_artifacts():
    try:
        diabetes_model = load_diabetes_model()
        heart_model = load_heart_model()
        diabetes_scaler = load_diabetes_scaler()
        heart_scaler = load_heart_scaler()
    except ArtifactLoadError as exc:
        st.error(f"Failed to load model artifacts: {exc}")
        st.stop()
    return diabetes_model, heart_model, diabetes_scaler, heart_scaler


def inject_theme():
    st.markdown(
        """
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@400;500;600;700&display=swap');
    
    /* Main background with warm gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 25%, #1f1f3d 50%, #2d2d5a 75%, #1a1a2e 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating particles effect using pseudo-elements */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #ff9500, transparent),
            radial-gradient(2px 2px at 40px 70px, #ffcc00, transparent),
            radial-gradient(2px 2px at 50px 160px, #ff6b35, transparent),
            radial-gradient(2px 2px at 90px 40px, #ffd700, transparent),
            radial-gradient(2px 2px at 130px 80px, #ff9500, transparent),
            radial-gradient(2px 2px at 160px 120px, #ffcc00, transparent),
            radial-gradient(2px 2px at 200px 200px, #ff6b35, transparent),
            radial-gradient(2px 2px at 250px 50px, #ffd700, transparent),
            radial-gradient(2px 2px at 300px 150px, #ff9500, transparent);
        background-repeat: repeat;
        background-size: 350px 350px;
        opacity: 0.25;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Main container styling */
    .main .block-container {
        padding: 2rem;
        position: relative;
        z-index: 1;
    }
    
    /* Header styling - Warm Orange */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.15) 0%, rgba(255, 204, 0, 0.1) 50%, rgba(255, 107, 53, 0.15) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 149, 0, 0.4);
        box-shadow: 
            0 0 30px rgba(255, 149, 0, 0.2),
            inset 0 0 30px rgba(255, 149, 0, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-family: 'Montserrat', sans-serif;
        color: #ff9500;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(255, 149, 0, 0.5);
        letter-spacing: 3px;
    }
    
    .main-header p {
        font-family: 'Poppins', sans-serif;
        color: #ffd700;
        font-size: 1.2rem;
        letter-spacing: 2px;
    }
    
    /* All text styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif !important;
        color: #ff9500 !important;
        text-shadow: 0 0 10px rgba(255, 149, 0, 0.3);
    }
    
    p, span, label, .stMarkdown {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Card styling */
    .prediction-card {
        background: rgba(30, 30, 60, 0.8);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 149, 0, 0.3);
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(255, 149, 0, 0.1);
    }
    
    /* Button styling - Orange Glow */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.3) 0%, rgba(255, 204, 0, 0.2) 100%);
        color: #ffd700;
        border: 2px solid #ff9500;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Montserrat', sans-serif;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        box-shadow: 
            0 0 15px rgba(255, 149, 0, 0.3),
            inset 0 0 15px rgba(255, 149, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.5) 0%, rgba(255, 204, 0, 0.4) 100%);
        box-shadow: 
            0 0 30px rgba(255, 149, 0, 0.5),
            0 0 60px rgba(255, 149, 0, 0.3),
            inset 0 0 20px rgba(255, 149, 0, 0.2);
        transform: translateY(-2px);
        color: #ffffff;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, rgba(45, 45, 80, 0.9) 0%, rgba(60, 60, 100, 0.9) 100%) !important;
        border: 2px solid rgba(255, 149, 0, 0.5) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-family: 'Poppins', sans-serif !important;
        padding: 0.5rem !important;
        box-shadow: 0 0 10px rgba(255, 149, 0, 0.1), inset 0 0 10px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #ffd700 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.4), inset 0 0 10px rgba(0, 0, 0, 0.2) !important;
        background: linear-gradient(135deg, rgba(55, 55, 90, 0.95) 0%, rgba(70, 70, 110, 0.95) 100%) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, rgba(45, 45, 80, 0.9) 0%, rgba(60, 60, 100, 0.9) 100%) !important;
        border: 2px solid rgba(255, 149, 0, 0.5) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 10px rgba(255, 149, 0, 0.1) !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #ffd700 !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.3) !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #ffd700 !important;
    }
    
    .stCheckbox > label > div[data-testid="stCheckbox"] {
        background: rgba(45, 45, 80, 0.9) !important;
        border: 2px solid rgba(255, 149, 0, 0.5) !important;
        border-radius: 5px !important;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: rgba(76, 175, 80, 0.15) !important;
        border: 1px solid #4caf50 !important;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.2);
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.15) !important;
        border: 1px solid #f44336 !important;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(244, 67, 54, 0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(26, 26, 46, 0.98) 0%, rgba(45, 45, 90, 0.95) 100%);
        border-right: 1px solid rgba(255, 149, 0, 0.3);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffd700;
    }
    
    /* Info box - Orange Theme */
    .info-box {
        background: rgba(255, 149, 0, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 149, 0, 0.3);
        margin: 1rem 0;
        box-shadow: 0 0 15px rgba(255, 149, 0, 0.1);
    }
    
    .info-box strong {
        color: #ffd700;
    }
    
    /* Section divider - Golden line */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #ff9500, #ffd700, #ff6b35, #ff9500, transparent);
        border-radius: 2px;
        margin: 1.5rem 0;
        box-shadow: 0 0 10px rgba(255, 149, 0, 0.5);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff6b35, #ff9500, #ffd700) !important;
        box-shadow: 0 0 10px rgba(255, 149, 0, 0.5);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'Montserrat', sans-serif !important;
        color: #ffd700 !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 60, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #ff9500, #ffd700);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #ffd700, #ff6b35);
    }
    
    /* Glowing border animation for sections */
    @keyframes borderGlow {
        0%, 100% { border-color: rgba(255, 149, 0, 0.3); }
        50% { border-color: rgba(255, 204, 0, 0.6); }
    }
    
    .stExpander {
        border: 1px solid rgba(255, 149, 0, 0.3);
        border-radius: 10px;
        animation: borderGlow 3s ease-in-out infinite;
    }
</style>
""",
        unsafe_allow_html=True,
    )

def render_header():
    st.markdown(
        """
<div class="main-header">
    <h1>🏥 DISEASE PREDICTION SYSTEM</h1>
    <p>⚡ AI-POWERED HEALTH RISK ASSESSMENT ⚡</p>
    <p style="font-size: 0.9rem; color: #ff9500; margin-top: 0.5rem;">DIABETES • HEART DISEASE • SMART ANALYSIS</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
    <div style="text-align: center; padding: 1rem;">
        <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" width="100" style="filter: drop-shadow(0 0 10px #ff9500);">
        <h2 style="font-family: 'Montserrat', sans-serif; color: #ff9500; margin: 0.5rem 0; font-size: 1.2rem; text-shadow: 0 0 10px rgba(255, 149, 0, 0.5);">HEALTH PREDICTOR</h2>
        <p style="color: #ffd700; font-size: 0.8rem; letter-spacing: 2px;"></p>
    </div>
    """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📋 SYSTEM INFO")
        st.info(
            """
    This application uses **Machine Learning** to predict the risk of:
    - 🍬 **Diabetes**
    - ❤️ **Heart Disease**

    Enter your health metrics and get instant predictions!
    """
        )

        st.markdown("---")

        st.markdown("### 🔬 HOW IT WORKS")
        st.markdown(
            """
    ```
    [1] SELECT DISEASE TYPE
    [2] INPUT HEALTH DATA  
    [3] INITIATE SCAN
    [4] RECEIVE AI ANALYSIS
    ```
    """
        )

        st.markdown("---")
        st.markdown("### ⚠️ DISCLAIMER")
        st.warning(
            "This tool is for educational purposes only. Always consult a healthcare professional for medical advice."
        )


def render_diabetes_section(diabetes_model, diabetes_scaler, patient_name):
    col_img, col_title = st.columns([1, 2])
    with col_img:
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem;">
            <img src="https://cdn-icons-png.flaticon.com/512/2751/2751460.png" width="120" style="filter: drop-shadow(0 0 15px #ff9500);">
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_title:
        st.markdown("## 🍬 DIABETES RISK SCAN")
        st.markdown(
            """
        <p style="color: #ffd700; font-family: 'Poppins', sans-serif; font-size: 1rem;">
            Advanced glucose & metabolic analysis system
        </p>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <div class="info-box">
        📡 <strong>INITIALIZING DIABETES ANALYSIS MODULE...</strong> Enter health parameters below.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 👤 BIOMETRIC DATA")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🎂 Age (years) [1-120]", 1, 120, 30, help="Enter your age in years (Range: 1-120)")
        gender_opt = st.selectbox("⚧ Gender", ["Female", "Male", "Other"], index=0)
    with col2:
        bmi = st.number_input("⚖️ BMI [10.0-60.0]", 10.0, 60.0, 25.0, help="Body Mass Index (Range: 10.0-60.0 kg/m²)")
        smoking_opt = st.selectbox(
            "🚬 Smoking History",
            ["never", "former", "ever", "current", "not current"],
            index=0,
        )

    st.markdown("#### 🏥 MEDICAL RECORDS")
    col3, col4 = st.columns(2)
    with col3:
        hypertension_opt = st.selectbox(
            "💊 Hypertension", ["No", "Yes"], index=0, help="Do you have high blood pressure?"
        )
        hba1c = st.number_input(
            "🔬 HbA1c Level [3.0-15.0]", 3.0, 15.0, 5.5, help="Hemoglobin A1c percentage (Range: 3.0-15.0%)"
        )
    with col4:
        heart_disease_opt = st.selectbox("❤️ Heart Disease History", ["No", "Yes"], index=0)
        glucose = st.number_input(
            "🩸 Blood Glucose Level [50-300]", 50, 300, 100, help="Fasting blood glucose (Range: 50-300 mg/dL)"
        )

    st.markdown("---")

    if st.button("⚡ INITIATE DIABETES SCAN", use_container_width=True):
        with st.spinner("🔄 ANALYZING BIOMETRIC DATA..."):
            diabetes_features = build_diabetes_features(
                age=age,
                hypertension_opt=hypertension_opt,
                heart_disease_opt=heart_disease_opt,
                bmi=bmi,
                hba1c=hba1c,
                glucose=glucose,
                gender_opt=gender_opt,
                smoking_opt=smoking_opt,
            )

            prediction, probability = predict_diabetes(
                diabetes_model,
                diabetes_scaler,
                diabetes_features,
            )
            probability_percent = probability * 100

            st.markdown("### 📊 SCAN RESULTS")

            st.progress(probability)

            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK DETECTED**")
                    st.markdown(
                        f"""
                    <div style="background: rgba(244, 67, 54, 0.15); padding: 1rem; border-radius: 10px; border: 1px solid #f44336; box-shadow: 0 0 20px rgba(244, 67, 54, 0.2);">
                        <h4 style="color: #f44336; margin: 0; font-family: 'Montserrat', sans-serif;">⚠️ RISK LEVEL: {probability_percent:.1f}%</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #ff8a80;">RECOMMENDATION: Consult healthcare provider immediately.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("✅ **LOW RISK DETECTED**")
                    st.markdown(
                        f"""
                    <div style="background: rgba(76, 175, 80, 0.15); padding: 1rem; border-radius: 10px; border: 1px solid #4caf50; box-shadow: 0 0 20px rgba(76, 175, 80, 0.2);">
                        <h4 style="color: #4caf50; margin: 0; font-family: 'Montserrat', sans-serif;">✅ RISK LEVEL: {probability_percent:.1f}%</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #a5d6a7;">STATUS: Maintain healthy lifestyle protocols.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            with col_res2:
                st.metric(
                    "RISK INDEX",
                    f"{probability_percent:.1f}%",
                    delta="CRITICAL" if prediction == 1 else "NORMAL",
                    delta_color="inverse",
                )

            # Generate PDF report
            prediction_label = "High Risk" if prediction == 1 else "Low Risk"
            inputs_dict = {
                "Age": age,
                "Gender": gender_opt,
                "BMI": bmi,
                "Smoking History": smoking_opt,
                "Hypertension": hypertension_opt,
                "Heart Disease": heart_disease_opt,
                "HbA1c Level": hba1c,
                "Blood Glucose Level": glucose,
            }
            pdf_bytes = build_pdf_report(
                disease_name="Diabetes",
                patient_name=patient_name,
                inputs=inputs_dict,
                prediction_label=prediction_label,
                probability_percent=probability_percent,
            )
            if pdf_bytes:
                safe_filename = (patient_name.strip() or "Unknown").replace(" ", "_")
                st.download_button(
                    label="📥 DOWNLOAD REPORT",
                    data=pdf_bytes,
                    file_name=f"Diabetes_Report_{safe_filename}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


def render_heart_section(heart_model, heart_scaler, patient_name):
    col_img, col_title = st.columns([1, 2])
    with col_img:
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem;">
            <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png" width="120" style="filter: drop-shadow(0 0 15px #e53935);">
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_title:
        st.markdown("## ❤️ CARDIAC RISK SCAN")
        st.markdown(
            """
        <p style="color: #ff8a65; font-family: 'Poppins', sans-serif; font-size: 1rem;">
            Cardiovascular health monitoring & risk assessment
        </p>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <div class="info-box">
        📡 <strong>INITIALIZING CARDIAC ANALYSIS MODULE...</strong> Enter cardiovascular parameters.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 👤 BIOMETRIC DATA")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🎂 Age (years) [1-120]", 1, 120, 45, help="Enter your age in years (Range: 1-120)")
        gender = st.selectbox("⚧ Gender", ["Male", "Female"], index=0)
    with col2:
        height_cm = st.number_input(
            "📏 Height (cm) [120-220]", 120, 220, 170, help="Height in centimeters (Range: 120-220)"
        )
        weight_kg = st.number_input(
            "⚖️ Weight (kg) [30-200]", 30.0, 200.0, 70.0, help="Weight in kilograms (Range: 30-200)"
        )

    st.markdown("#### 🩺 VITAL STATISTICS")
    col3, col4 = st.columns(2)
    with col3:
        systolic_bp = st.number_input(
            "🔴 Systolic BP (mmHg) [80-200]", 80, 200, 120, help="Upper blood pressure reading (Range: 80-200)"
        )
        cholesterol = st.number_input(
            "🧪 Cholesterol (mg/dL) [100-400]", 100, 400, 200, help="Cholesterol level (Range: 100-400)"
        )
    with col4:
        diastolic_bp = st.number_input(
            "🔵 Diastolic BP (mmHg) [50-120]", 50, 120, 80, help="Lower blood pressure reading (Range: 50-120)"
        )
        glucose = st.number_input(
            "🩸 Glucose (mg/dL) [50-300]", 50, 300, 100, help="Blood glucose level (Range: 50-300)"
        )

    st.markdown("#### 🏃 LIFESTYLE PARAMETERS")
    col5, col6, col7 = st.columns(3)
    with col5:
        smoke = st.checkbox("🚬 Smoker", value=False)
    with col6:
        alco = st.checkbox("🍷 Alcohol Use", value=False)
    with col7:
        active = st.checkbox("🏃 Physically Active", value=True)

    st.markdown("---")

    if st.button("⚡ INITIATE CARDIAC SCAN", use_container_width=True):
        with st.spinner("🔄 ANALYZING CARDIOVASCULAR DATA..."):
            heart_features, bmi_val = build_heart_features(
                age=age,
                gender=gender,
                height_cm=height_cm,
                weight_kg=weight_kg,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                cholesterol=cholesterol,
                glucose=glucose,
                smoke=smoke,
                alco=alco,
                active=active,
            )

            prediction, probability = predict_heart(
                heart_model,
                heart_scaler,
                heart_features,
            )
            probability_percent = probability * 100

            st.markdown("### 📊 SCAN RESULTS")

            st.progress(probability)

            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK DETECTED**")
                    st.markdown(
                        f"""
                    <div style="background: rgba(244, 67, 54, 0.15); padding: 1rem; border-radius: 10px; border: 1px solid #f44336; box-shadow: 0 0 20px rgba(244, 67, 54, 0.2);">
                        <h4 style="color: #f44336; margin: 0; font-family: 'Montserrat', sans-serif;">⚠️ RISK LEVEL: {probability_percent:.1f}%</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #ff8a80;">RECOMMENDATION: Consult cardiologist immediately.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("✅ **LOW RISK DETECTED**")
                    st.markdown(
                        f"""
                    <div style="background: rgba(76, 175, 80, 0.15); padding: 1rem; border-radius: 10px; border: 1px solid #4caf50; box-shadow: 0 0 20px rgba(76, 175, 80, 0.2);">
                        <h4 style="color: #4caf50; margin: 0; font-family: 'Montserrat', sans-serif;">✅ RISK LEVEL: {probability_percent:.1f}%</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #a5d6a7;">STATUS: Cardiac health parameters within normal range.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            with col_res2:
                st.metric(
                    "RISK INDEX",
                    f"{probability_percent:.1f}%",
                    delta="CRITICAL" if prediction == 1 else "NORMAL",
                    delta_color="inverse",
                )
                st.metric("BMI", f"{bmi_val:.1f}")

            # Generate PDF report
            prediction_label = "High Risk" if prediction == 1 else "Low Risk"
            inputs_dict = {
                "Age": age,
                "Gender": gender,
                "Height (cm)": height_cm,
                "Weight (kg)": weight_kg,
                "BMI": f"{bmi_val:.1f}",
                "Systolic BP (mmHg)": systolic_bp,
                "Diastolic BP (mmHg)": diastolic_bp,
                "Cholesterol (mg/dL)": cholesterol,
                "Glucose (mg/dL)": glucose,
                "Smoker": "Yes" if smoke else "No",
                "Alcohol Use": "Yes" if alco else "No",
                "Physically Active": "Yes" if active else "No",
            }
            pdf_bytes = build_pdf_report(
                disease_name="Heart Disease",
                patient_name=patient_name,
                inputs=inputs_dict,
                prediction_label=prediction_label,
                probability_percent=probability_percent,
            )
            if pdf_bytes:
                safe_filename = (patient_name.strip() or "Unknown").replace(" ", "_")
                st.download_button(
                    label="📥 DOWNLOAD REPORT",
                    data=pdf_bytes,
                    file_name=f"Heart_Disease_Report_{safe_filename}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


def render_footer():
    st.markdown("---")
    st.markdown(
        """
<div style="text-align: center; padding: 1.5rem; background: rgba(30, 30, 60, 0.6); border-radius: 15px; border: 1px solid rgba(255, 149, 0, 0.3);">
    <p style="font-family: 'Montserrat', sans-serif; color: #ff9500; margin: 0; letter-spacing: 2px;">
        ⚡ POWERED BY MACHINE LEARNING ⚡
    </p>
    <p style="color: #ffd700; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
        DISEASE PREDICTION SYSTEM 
    </p>
    <p style="color: #fff; font-size: 0.7rem; margin-top: 0.5rem;">
        🔬 Built By Smit Kansagara
    </p>
</div>
""",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Disease Prediction System",
        page_icon="🩺",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    inject_theme()

    diabetes_model, heart_model, diabetes_scaler, heart_scaler = load_artifacts()

    render_header()
    render_sidebar()

    st.markdown("### 👤 PATIENT NAME")
    patient_name = st.text_input("Enter your name")

    st.markdown("### 🎯 SELECT SCAN TYPE")
    disease = st.selectbox(
        "Choose a condition",
        ["Diabetes", "Heart Disease"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if disease == "Diabetes":
        render_diabetes_section(diabetes_model, diabetes_scaler, patient_name)
    else:
        render_heart_section(heart_model, heart_scaler, patient_name)

    render_footer()


if __name__ == "__main__":
    main()