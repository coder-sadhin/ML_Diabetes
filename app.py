# =====================================================
# Diabetes Prediction System - FINAL BUG-FREE VERSION
# =====================================================

import os
import pickle
import numpy as np
import pandas as pd
import gradio as gr
import sklearn

# -----------------------------------------------------
# Environment safety (proxy fix)
# -----------------------------------------------------
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

# -----------------------------------------------------
# Load trained model
# -----------------------------------------------------
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

if isinstance(model, dict):
    raise ValueError(
        "❌ diabetes_model.pkl contains a dictionary.\n"
        "Please save ONLY the trained model."
    )

# -----------------------------------------------------
# Load scaler
# -----------------------------------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("✅ Model Loaded:", type(model))
print("✅ Scaler Loaded:", type(scaler))

# -----------------------------------------------------
# Feature order (MUST match training)
# -----------------------------------------------------
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Insulin_Glucose_Ratio"
]

# -----------------------------------------------------
# Prediction function (WARNING-FREE)
# -----------------------------------------------------
def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
):
    insulin_glucose_ratio = insulin / (glucose + 1e-6)

    input_df = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age,
        insulin_glucose_ratio
    ]], columns=FEATURES)

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Preserve feature names (IMPORTANT)
    input_scaled_df = pd.DataFrame(
        input_scaled,
        columns=FEATURES
    )

    pred = model.predict(input_scaled_df)[0]
    prob = model.predict_proba(input_scaled_df)[0][1]

    if pred == 1:
        return (
            "🩺 **Diabetes Detected**\n\n"
            f"🔴 Risk Probability: **{prob:.2%}**"
        )
    else:
        return (
            "✅ **No Diabetes Detected**\n\n"
            f"🟢 Risk Probability: **{prob:.2%}**"
        )

# -----------------------------------------------------
# Gradio UI
# -----------------------------------------------------
with gr.Blocks() as app:

    gr.Markdown("""
    # 🧠 Diabetes Prediction System
    **Machine Learning–Based Medical Decision Support Tool**
    """)

    with gr.Row():
        with gr.Column():
            pregnancies = gr.Number(label="Pregnancies", value=2)
            glucose = gr.Number(label="Glucose Level", value=120)
            blood_pressure = gr.Number(label="Blood Pressure", value=70)
            skin_thickness = gr.Number(label="Skin Thickness", value=20)
            insulin = gr.Number(label="Insulin", value=80)

        with gr.Column():
            bmi = gr.Number(label="BMI", value=28.0)
            dpf = gr.Number(label="Diabetes Pedigree Function", value=0.5)
            age = gr.Number(label="Age", value=35)

    predict_btn = gr.Button("🔍 Predict Diabetes", variant="primary")
    output = gr.Markdown()

    predict_btn.click(
        fn=predict_diabetes,
        inputs=[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ],
        outputs=output
    )

    

# -----------------------------------------------------
# Launch app (Gradio 6.x compliant)
# -----------------------------------------------------
if __name__ == "__main__":
    app.launch(
        theme=gr.themes.Soft(),
        share=False
    )
