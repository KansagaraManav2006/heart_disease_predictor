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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    .main .block-container {
        max-width: 100%;
        padding: 2rem 1.5rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 700;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.75rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.3rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    p {
        color: #cbd5e1;
        line-height: 1.6;
    }
    
    .header {
        background: #1e293b;
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .header h1 {
        color: #22c55e;
        margin: 0 0 0.5rem 0;
    }
    
    .header p {
        margin: 0.5rem 0;
        color: #cbd5e1;
    }
    
    .info-box {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1.5rem 0;
    }
    
    .info-box strong {
        color: #22c55e;
    }
    
    .stButton > button {
        background: #22c55e;
        color: white;
        border: none;
        border-radius: 8px;
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #16a34a;
        transform: translateY(-2px);
    }
    
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
        padding: 0.75rem !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #22c55e !important;
        box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2) !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #22c55e !important;
    }
    
    .stCheckbox > label {
        color: #e2e8f0 !important;
    }
    
    .stSuccess {
        background: rgba(34, 197, 94, 0.15) !important;
        border: 1px solid #22c55e !important;
        border-radius: 8px !important;
        color: #dcfce7 !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid #ef4444 !important;
        border-radius: 8px !important;
        color: #fee2e2 !important;
    }
    
    .stWarning {
        background: rgba(251, 146, 60, 0.15) !important;
        border: 1px solid #fb923c !important;
        border-radius: 8px !important;
        color: #ffedd5 !important;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%) !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #22c55e !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }
    
    .divider {
        height: 1px;
        background: rgba(34, 197, 94, 0.3);
        margin: 1.5rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2.5rem 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #94a3b8;
    }
    
    .footer p {
        margin: 0.5rem 0;
        color: #94a3b8;
    }
    
    footer {
        visibility: hidden;
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        h1 {
            font-size: 1.75rem;
        }
        
        h2 {
            font-size: 1.3rem;
        }
        
        .header {
            padding: 1.5rem 1rem;
        }
    }
</style>
""",
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
<div class="header">
    <h1>Disease Prediction System</h1>
    <p>AI-Powered Health Risk Assessment</p>
    <p style="font-size: 0.95rem; margin-top: 1rem;">
        Machine learning models to estimate health risks for Diabetes and Heart Disease.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_diabetes_section(diabetes_model, diabetes_scaler, patient_name):
    st.markdown("## Diabetes Risk Assessment")
    st.markdown(
        """
<div class="info-box">
    <strong>Analysis Module</strong><br>
    Enter your health parameters for personalized diabetes risk evaluation.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Biometric Data")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", 1, 120, 30, help="Range: 1-120", key="diab_age")
        gender_opt = st.selectbox("Gender", ["Female", "Male", "Other"], index=0, key="diab_gender")
    with col2:
        bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0, help="Range: 10.0-60.0", key="diab_bmi")
        smoking_opt = st.selectbox(
            "Smoking History",
            ["never", "former", "ever", "current", "not current"],
            index=0,
            key="diab_smoking"
        )

    st.markdown("### Medical Records")
    col3, col4 = st.columns(2)
    with col3:
        hypertension_opt = st.selectbox(
            "Hypertension", ["No", "Yes"], index=0, key="diab_hypertension"
        )
        hba1c = st.number_input(
            "HbA1c Level (%)", 3.0, 15.0, 5.5, help="Range: 3.0-15.0%", key="diab_hba1c"
        )
    with col4:
        heart_disease_opt = st.selectbox("Heart Disease History", ["No", "Yes"], index=0, key="diab_heart_disease")
        glucose = st.number_input(
            "Blood Glucose (mg/dL)", 50, 300, 100, help="Range: 50-300", key="diab_glucose"
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("Run Diabetes Analysis", use_container_width=True, key="diab_scan"):
        with st.spinner("Analyzing biometric data..."):
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

            st.markdown("### Analysis Results")
            st.progress(probability)

            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                if prediction == 1:
                    st.error("HIGH RISK DETECTED")
                    st.markdown(
                        f"**Risk Level: {probability_percent:.1f}%**\n\nRecommendation: Consult healthcare provider immediately.",
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("LOW RISK DETECTED")
                    st.markdown(
                        f"**Risk Level: {probability_percent:.1f}%**\n\nStatus: Maintain healthy lifestyle protocols.",
                        unsafe_allow_html=True,
                    )

            with col_res2:
                st.metric(
                    "RISK INDEX",
                    f"{probability_percent:.1f}%",
                    delta="HIGH" if prediction == 1 else "LOW",
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
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"Diabetes_Report_{safe_filename}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


def render_heart_section(heart_model, heart_scaler, patient_name):
    st.markdown("## Cardiac Health Assessment")
    st.markdown(
        """
<div class="info-box">
    <strong>Cardiovascular Risk Module</strong><br>
    Enter your cardiovascular parameters for personalized cardiac risk evaluation.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Biometric Data")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", 1, 120, 45, help="Range: 1-120", key="heart_age")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0, key="heart_gender")
    with col2:
        height_cm = st.number_input(
            "Height (cm)", 120, 220, 170, help="Range: 120-220", key="heart_height"
        )
        weight_kg = st.number_input(
            "Weight (kg)", 30.0, 200.0, 70.0, help="Range: 30-200", key="heart_weight"
        )

    st.markdown("### Vital Statistics")
    col3, col4 = st.columns(2)
    with col3:
        systolic_bp = st.number_input(
            "Systolic BP (mmHg)", 80, 200, 120, help="Range: 80-200", key="heart_systolic"
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", 100, 400, 200, help="Range: 100-400", key="heart_chol"
        )
    with col4:
        diastolic_bp = st.number_input(
            "Diastolic BP (mmHg)", 50, 120, 80, help="Range: 50-120", key="heart_diastolic"
        )
        glucose = st.number_input(
            "Blood Glucose (mg/dL)", 50, 300, 100, help="Range: 50-300", key="heart_glucose"
        )

    st.markdown("### Lifestyle Factors")
    col5, col6, col7 = st.columns(3)
    with col5:
        smoke = st.checkbox("Smoker", value=False, key="heart_smoke")
    with col6:
        alco = st.checkbox("Alcohol Use", value=False, key="heart_alco")
    with col7:
        active = st.checkbox("Physically Active", value=True, key="heart_active")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("Run Cardiac Analysis", use_container_width=True, key="heart_scan"):
        with st.spinner("Analyzing cardiovascular data..."):
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

            st.markdown("### Analysis Results")
            st.progress(probability)

            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                if prediction == 1:
                    st.error("HIGH RISK DETECTED")
                    st.markdown(
                        f"**Risk Level: {probability_percent:.1f}%**\n\nRecommendation: Consult cardiologist immediately.",
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("LOW RISK DETECTED")
                    st.markdown(
                        f"**Risk Level: {probability_percent:.1f}%**\n\nStatus: Cardiac health parameters within normal range.",
                        unsafe_allow_html=True,
                    )

            with col_res2:
                st.metric(
                    "RISK INDEX",
                    f"{probability_percent:.1f}%",
                    delta="HIGH" if prediction == 1 else "LOW",
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
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"Heart_Disease_Report_{safe_filename}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


def render_footer():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="footer">
    <p><strong>Disease Prediction System</strong></p>
    <p>Built with Streamlit and Scikit-learn</p>
    <p>This system is for educational purposes only and is not a substitute for professional medical advice.</p>
    <p>Always consult qualified healthcare professionals for proper diagnosis and treatment.</p>
    <p style="margin-top: 1rem; color: #64748b; font-size: 0.85rem;">© 2026 Health AI</p>
</div>
""",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Disease Prediction System",
        page_icon="🩺",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    inject_theme()
    render_header()

    # Hide sidebar
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    diabetes_model, heart_model, diabetes_scaler, heart_scaler = load_artifacts()

    st.markdown("### Patient Information")
    patient_name = st.text_input("Enter your name (optional)")

    st.markdown("### Select Assessment Type")
    disease = st.selectbox(
        "Choose a condition to assess",
        ["Diabetes", "Heart Disease"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if disease == "Diabetes":
        render_diabetes_section(diabetes_model, diabetes_scaler, patient_name)
    else:
        render_heart_section(heart_model, heart_scaler, patient_name)

    render_footer()


if __name__ == "__main__":
    main()
