# app.py
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from io import BytesIO
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "models/kidney_focal_finetuned.keras"
IMG_SIZE = 256
CLASS_NAMES = ["cyst", "normal", "stone", "tumor"]

st.set_page_config(
    page_title="AI Based CKD Tracker",
    page_icon="🧪",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# -------------------------
# PDF GENERATOR
# -------------------------
def create_pdf(patient_details, final_class, df, chart_buf):
    pdf_buf = BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.darkblue)
    c.drawString(50, height - 50, "AI Based Disease Progression Report")
    c.setFillColor(colors.black)

    # Patient details
    c.setFont("Helvetica-Bold", 14)
    y = height - 100
    c.drawString(50, y, "Patient Details:")
    c.setFont("Helvetica", 12)
    y -= 30
    for key, value in patient_details.items():
        c.drawString(60, y, f"{key}: {value}")
        y -= 20

    # Prediction
    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.red)
    c.drawString(50, y, f"Final Prediction: {final_class.upper()}")
    c.setFillColor(colors.black)

    # Probabilities
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Prediction Probabilities:")
    y -= 25
    for i, row in df.iterrows():
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"{row['Disease']}: {row['Probability (%)']}%")
        y -= 18

    # Chart
    y -= 30
    img_reader = ImageReader(chart_buf)
    c.drawImage(img_reader, 50, y - 220, width=400, height=220)

    c.showPage()
    c.save()
    pdf_buf.seek(0)
    return pdf_buf


# -------------------------
# SESSION STATE
# -------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"


# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
def sidebar_navigation():
    st.sidebar.markdown("## 🧭 Navigation")

    menu = {
        "Home": "🏠",
        "Patient Details": "🧑‍⚕",
        "Prediction": "📊"
    }

    for step, icon in menu.items():
        if st.session_state["page"] == step:
            st.sidebar.markdown(
                f"<div style='padding:8px; border-radius:8px; background:#2E86C1; color:white; font-weight:bold;'>{icon} {step}</div>",
                unsafe_allow_html=True
            )
        else:
            if st.sidebar.button(f"{icon} {step}"):
                if step == "Prediction" and "patient_details" not in st.session_state:
                    st.warning("⚠ Please fill Patient Details first!")
                else:
                    st.session_state["page"] = step

sidebar_navigation()


# -------------------------
# HOME PAGE
# -------------------------
if st.session_state["page"] == "Home":
    st.title("🧪 AI Based Kidney Disease Progression Tracker")

    st.markdown(
        """
        <div style='padding:20px; border-radius:12px; background:#EAF2F8'>
            <h3>👋 Welcome!</h3>
            <p>This AI-powered system helps in <b>detecting and tracking kidney diseases</b> 
            from medical images.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🧬 Detects *Cyst, Stone, Tumor, Normal Kidney* conditions")
    with col2:
        st.success("📊 Provides *probability analysis & charts*")
    with col3:
        st.warning("📝 Generates a *PDF medical report*")

    st.markdown("---")
    if st.button("➡ Start Patient Registration", type="primary"):
        st.session_state["page"] = "Patient Details"


# -------------------------
# PATIENT DETAILS PAGE
# -------------------------
elif st.session_state["page"] == "Patient Details":
    st.title("🧑‍⚕ Patient Registration")

    with st.form("patient_form"):
        st.markdown("### 📝 Enter Patient Information")

        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Patient Name *")
            age = st.number_input("Age *", min_value=1, max_value=120, step=1)
        with col2:
            gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
            patient_id = st.text_input("Patient ID / Record No. *")

        submit_btn = st.form_submit_button("💾 Save Details")

    if submit_btn:
        if not name or not patient_id:
            st.error("⚠ All fields marked * are mandatory.")
        else:
            st.session_state["patient_details"] = {
                "Name": name,
                "Age": age,
                "Gender": gender,
                "Patient ID": patient_id,
            }
            st.success("✅ Patient details saved successfully!")

    if "patient_details" in st.session_state:
        if st.button("➡ Proceed to Prediction", type="primary"):
            st.session_state["page"] = "Prediction"


# -------------------------
# PREDICTION PAGE
# -------------------------
elif st.session_state["page"] == "Prediction":
    st.title("📊 Kidney Disease Prediction")

    if "patient_details" not in st.session_state:
        st.warning("⚠ Please fill in *Patient Details* first!")
        st.stop()

    st.markdown("*Step 3 of 3: Upload image and get predictions*")

    uploaded_file = st.file_uploader("Upload Kidney Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("🔍 Analyzing image..."):
                # Preprocess
                img = Image.open(uploaded_file).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = image.img_to_array(img) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)

                # Predict
                preds = model.predict(img_batch)[0]
                percentages = (preds * 100).round(2)
                final_idx = np.argmax(preds)
                final_class = CLASS_NAMES[final_idx]
                final_percentage = percentages[final_idx]

                # Results dataframe
                df = pd.DataFrame({
                    "Disease": CLASS_NAMES,
                    "Probability (%)": percentages
                })

            # Show patient details
            st.subheader("🧑‍⚕ Patient Details")
            st.markdown("---")
            for key, value in st.session_state["patient_details"].items():
                st.markdown(f"- *{key}:* {value}")

            # Show prediction
            st.subheader("🔍 Prediction Result")
            st.success(f"✅ *{final_class.upper()}* detected with *{final_percentage}% probability*")

            # Show probability chart
            st.subheader("📊 Probability Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(CLASS_NAMES, percentages, color=["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            st.pyplot(fig)

            # Save chart buffer for PDF
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            # Generate report
            pdf_file = create_pdf(st.session_state["patient_details"], final_class, df, buf)
            st.download_button(
                label="📥 Download Report (PDF)",
                data=pdf_file,
                file_name=f"Kidney_Report_{st.session_state['patient_details']['Patient ID']}.pdf",
                mime="application/pdf",
                type="primary"
            )