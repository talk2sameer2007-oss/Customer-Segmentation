import gradio as gr
import joblib
import pandas as pd
import numpy as np

# ==========================
# Load Model Package
# ==========================

package = joblib.load("customer_segmentation_kprototypes.pkl")

model = package["model"]
scaler = package["scaler"]
numerical_columns = package["numerical_columns"]
categorical_columns = package["categorical_columns"]
categorical_indices = package["categorical_indices"]


# ==========================
# Prediction Function
# ==========================

def predict_customer(
    age,
    income,
    coverage,
    premium,
    purchase_year,
    gender,
    marital_status,
    education,
    geographic,
    occupation,
    behavioral,
    interaction,
    insurance_products,
    policy_type,
    customer_preference,
    communication_channel,
    contact_time,
    language
):

    try:

        # Numerical Features
        numeric_df = pd.DataFrame({
            "Age":[age],
            "Income Level":[income],
            "Coverage Amount":[coverage],
            "Premium Amount":[premium],
            "Purchase Year":[purchase_year]
        })

        # Scale Numerical Features
        scaled_numeric = scaler.transform(numeric_df)

        # Categorical Features
        categorical_df = pd.DataFrame({
            "Gender":[gender],
            "Marital Status":[marital_status],
            "Education Level":[education],
            "Geographic Information":[geographic],
            "Occupation":[occupation],
            "Behavioral Data":[behavioral],
            "Interactions with Customer Service":[interaction],
            "Insurance Products Owned":[insurance_products],
            "Policy Type":[policy_type],
            "Customer Preferences":[customer_preference],
            "Preferred Communication Channel":[communication_channel],
            "Preferred Contact Time":[contact_time],
            "Preferred Language":[language]
        })

        # Merge Numeric + Categorical
        X = np.concatenate(
            [
                scaled_numeric,
                categorical_df.values
            ],
            axis=1
        )

        # Predict Cluster
        cluster = model.predict(
            X,
            categorical=categorical_indices
        )[0]

        return f"🎯 Predicted Customer Segment : Cluster {cluster + 1}"

    except Exception as e:
        return f"Error : {e}"

# ==========================
# Gradio Interface
# ==========================

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🛡️ Customer Segmentation System

    Predict the customer segment using the trained **K-Prototypes Model**.

    ---
    ### Developed by **Sameer Chopra**
    **Roll No.: 241020**
    """)

    gr.Markdown("## 📊 Numerical Features")

    with gr.Row():

        age = gr.Number(
            label="Age",
            value=35
        )

        income = gr.Number(
            label="Income Level",
            value=60000
        )

    with gr.Row():

        coverage = gr.Number(
            label="Coverage Amount",
            value=250000
        )

        premium = gr.Number(
            label="Premium Amount",
            value=15000
        )

    purchase_year = gr.Number(
        label="Purchase Year",
        value=2024
    )

    gr.Markdown("## 📝 Customer Information")

    with gr.Row():

        gender = gr.Dropdown(
            ["Male", "Female"],
            value="Male",
            label="Gender"
        )

        marital_status = gr.Dropdown(
            ["Single", "Married", "Divorced", "Widowed"],
            value="Single",
            label="Marital Status"
        )

    with gr.Row():

        education = gr.Dropdown(
            [
                "High School",
                "Bachelor",
                "Master",
                "PhD"
            ],
            value="Bachelor",
            label="Education Level"
        )

        geographic = gr.Dropdown(
            [
                "Urban",
                "Suburban",
                "Rural"
            ],
            value="Urban",
            label="Geographic Information"
        )

    occupation = gr.Dropdown(
        [
            "Employed",
            "Business",
            "Self-Employed",
            "Student",
            "Retired",
            "Unemployed"
        ],
        value="Employed",
        label="Occupation"
    )

    behavioral = gr.Dropdown(
        [
            "Low",
            "Medium",
            "High"
        ],
        value="Medium",
        label="Behavioral Data"
    )

    interaction = gr.Dropdown(
        [
            "Low",
            "Medium",
            "High"
        ],
        value="Medium",
        label="Interactions with Customer Service"
    )

    insurance_products = gr.Dropdown(
        [
            "1",
            "2",
            "3",
            "4",
            "5"
        ],
        value="2",
        label="Insurance Products Owned"
    )

    policy_type = gr.Dropdown(
        [
            "Health",
            "Life",
            "Vehicle",
            "Home",
            "Travel"
        ],
        value="Health",
        label="Policy Type"
    )

    customer_preference = gr.Dropdown(
        [
            "Price",
            "Coverage",
            "Service",
            "Benefits"
        ],
        value="Coverage",
        label="Customer Preferences"
    )

    communication_channel = gr.Dropdown(
        [
            "Email",
            "Phone",
            "SMS",
            "Mobile App"
        ],
        value="Email",
        label="Preferred Communication Channel"
    )

    contact_time = gr.Dropdown(
        [
            "Morning",
            "Afternoon",
            "Evening"
        ],
        value="Morning",
        label="Preferred Contact Time"
    )

    language = gr.Dropdown(
        [
            "English",
            "Hindi",
            "Spanish",
            "French"
        ],
        value="English",
        label="Preferred Language"
    )

    predict_btn = gr.Button(
        "Predict Customer Segment",
        variant="primary"
    )

    output = gr.Textbox(
        label="Prediction Result"
    )
# ==========================
# Button Action
# ==========================

    predict_btn.click(
        fn=predict_customer,
        inputs=[
            age,
            income,
            coverage,
            premium,
            purchase_year,
            gender,
            marital_status,
            education,
            geographic,
            occupation,
            behavioral,
            interaction,
            insurance_products,
            policy_type,
            customer_preference,
            communication_channel,
            contact_time,
            language
        ],
        outputs=output
    )

    # ==========================
    # Example Values
    # ==========================

    gr.Examples(
        examples=[
            [
                35,
                65000,
                300000,
                18000,
                2023,
                "Male",
                "Married",
                "Bachelor",
                "Urban",
                "Employed",
                "High",
                "Medium",
                "2",
                "Health",
                "Coverage",
                "Email",
                "Morning",
                "English"
            ],
            [
                48,
                90000,
                600000,
                32000,
                2022,
                "Female",
                "Married",
                "Master",
                "Suburban",
                "Business",
                "Medium",
                "High",
                "3",
                "Life",
                "Benefits",
                "Phone",
                "Evening",
                "Hindi"
            ],
            [
                27,
                45000,
                150000,
                9000,
                2024,
                "Male",
                "Single",
                "Bachelor",
                "Urban",
                "Self-Employed",
                "Low",
                "Low",
                "1",
                "Vehicle",
                "Price",
                "SMS",
                "Afternoon",
                "English"
            ]
        ],
        inputs=[
            age,
            income,
            coverage,
            premium,
            purchase_year,
            gender,
            marital_status,
            education,
            geographic,
            occupation,
            behavioral,
            interaction,
            insurance_products,
            policy_type,
            customer_preference,
            communication_channel,
            contact_time,
            language
        ]
    )

    gr.Markdown(
        """
        ---
        ### 📌 Instructions
        - Enter customer information.
        - Click **Predict Customer Segment**.
        - The trained K-Prototypes model will predict the most suitable cluster.

        **Developed by Sameer Chopra**  
        **Roll No. 241020**
        """
    )


# ==========================
# Launch App
# ==========================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
