import os
import gradio as gr
import joblib
import numpy as np
import pandas as pd

# ==========================
# Load Model Package
# ==========================

try:
    package = joblib.load("customer_segmentation_kprototypes.pkl")
    model = package["model"]
    scaler = package["scaler"]
    numerical_columns = package["numerical_columns"]
    categorical_columns = package["categorical_columns"]
    categorical_indices = package["categorical_indices"]
except Exception as e:
    print(f"Warning: Model file loading failed. ({e})")

# ==========================================
# Cluster Metadata & Persona Mappings
# ==========================================

CLUSTER_PERSONAS = {
    0: {
        "title": "💎 High-Value Loyalists",
        "subtitle": "Tier 1 Premium Segment",
        "description": "High-income customers with premium coverage policies. Highly engaged with minimal churn risk.",
        "badge_color": "#2D150B",
        "badge_text_color": "#FFFFFF",
        "bg_color": "#F7F3F0",
        "border_color": "#8FA353",
        "recommendation": "Cross-sell premium wealth management products and offer exclusive VIP loyalty rewards.",
        "retention_score": "96%",
        "value_tier": "High ($$$)",
    },
    1: {
        "title": "🎓 Young Digital Budget Seekers",
        "subtitle": "Growth & Mobile-First Segment",
        "description": "Younger demographic looking for cost-effective basic coverage with digital-first interaction preferences.",
        "badge_color": "#E65A72",
        "badge_text_color": "#FFFFFF",
        "bg_color": "#F7F3F0",
        "border_color": "#E65A72",
        "recommendation": "Offer flexible pay-as-you-go insurance options via mobile app and SMS notifications.",
        "retention_score": "74%",
        "value_tier": "Moderate ($)",
    },
    2: {
        "title": "🛡️ Family Protection Focused",
        "subtitle": "Core Life & Health Segment",
        "description": "Mid-age married individuals prioritizing comprehensive health, vehicle, and life coverage for dependents.",
        "badge_color": "#2D150B",
        "badge_text_color": "#FFFFFF",
        "bg_color": "#F7F3F0",
        "border_color": "#8FA353",
        "recommendation": "Promote bundled family coverage packages and long-term savings plans.",
        "retention_score": "88%",
        "value_tier": "Medium-High ($$)",
    },
    3: {
        "title": "⚠️ High-Service Demand Segment",
        "subtitle": "Attention Required / Retention Risk",
        "description": "Frequent customer service interactions with moderate policy spend and higher retention sensitivity.",
        "badge_color": "#E65A72",
        "badge_text_color": "#FFFFFF",
        "bg_color": "#F7F3F0",
        "border_color": "#E65A72",
        "recommendation": "Assign dedicated customer support reps to resolve inquiries quickly and offer targeted renewal discounts.",
        "retention_score": "52%",
        "value_tier": "Sensitive ($$)",
    },
}

# ==========================
# Real-Time KPI Function
# ==========================


def update_live_metrics(income, coverage, premium, interaction):
    loss_ratio = (
        round((premium / coverage) * 100, 2) if coverage and coverage > 0 else 0
    )

    risk_label = "🟢 Low Risk"
    risk_color = "#4D7C0F"
    if interaction == "High" or loss_ratio > 8:
        risk_label = "🔴 High Attention"
        risk_color = "#E65A72"
    elif interaction == "Medium" or loss_ratio > 4:
        risk_label = "🟡 Moderate Attention"
        risk_color = "#D97706"

    summary_html = f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 5px;">
        <div style="background: #F7F3F0; border: 1px solid #EAE3DC; padding: 14px 10px; border-radius: 20px; text-align: center;">
            <div style="color: #6B564E; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Premium Ratio</div>
            <div style="color: #2D150B; font-size: 18px; font-weight: 800; margin-top: 4px;">{loss_ratio}%</div>
        </div>
        <div style="background: #F7F3F0; border: 1px solid #EAE3DC; padding: 14px 10px; border-radius: 20px; text-align: center;">
            <div style="color: #6B564E; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Service Level</div>
            <div style="color: #2D150B; font-size: 16px; font-weight: 800; margin-top: 5px;">{interaction}</div>
        </div>
        <div style="background: #F7F3F0; border: 1px solid #EAE3DC; padding: 14px 10px; border-radius: 20px; text-align: center;">
            <div style="color: #6B564E; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Retention Radar</div>
            <div style="color: {risk_color}; font-size: 14px; font-weight: 800; margin-top: 6px;">{risk_label}</div>
        </div>
    </div>
    """
    return summary_html


# ==========================
# Main Prediction Function
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
    language,
):
    try:
        numeric_df = pd.DataFrame(
            {
                "Age": [age],
                "Income Level": [income],
                "Coverage Amount": [coverage],
                "Premium Amount": [premium],
                "Purchase Year": [purchase_year],
            }
        )
        scaled_numeric = scaler.transform(numeric_df)

        categorical_df = pd.DataFrame(
            {
                "Gender": [gender],
                "Marital Status": [marital_status],
                "Education Level": [education],
                "Geographic Information": [geographic],
                "Occupation": [occupation],
                "Behavioral Data": [behavioral],
                "Interactions with Customer Service": [interaction],
                "Insurance Products Owned": [insurance_products],
                "Policy Type": [policy_type],
                "Customer Preferences": [customer_preference],
                "Preferred Communication Channel": [communication_channel],
                "Preferred Contact Time": [contact_time],
                "Preferred Language": [language],
            }
        )

        X = np.concatenate([scaled_numeric, categorical_df.values], axis=1)
        cluster_id = int(model.predict(X, categorical=categorical_indices)[0])

        persona = CLUSTER_PERSONAS.get(
            cluster_id,
            {
                "title": f"Cluster {cluster_id + 1} Segment",
                "subtitle": "Standard Classification",
                "description": "Standard profile matching default cluster characteristics.",
                "badge_color": "#2D150B",
                "badge_text_color": "#FFFFFF",
                "bg_color": "#F7F3F0",
                "border_color": "#8FA353",
                "recommendation": "Apply standard customer engagement and marketing strategy.",
                "retention_score": "80%",
                "value_tier": "Standard",
            },
        )

        html_output = f"""
        <div style="
            background: #FFFFFF; 
            border: 2px solid {persona['border_color']}; 
            border-radius: 28px; 
            padding: 24px; 
            margin-top: 10px;
            box-shadow: 0 15px 35px rgba(45, 21, 11, 0.08);
            font-family: 'Inter', -apple-system, sans-serif;
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                <span style="
                    background-color: {persona['badge_color']}; 
                    color: {persona['badge_text_color']}; 
                    font-size: 11px; 
                    font-weight: 800; 
                    letter-spacing: 0.6px;
                    padding: 6px 16px; 
                    border-radius: 20px;
                    text-transform: uppercase;
                ">
                    Cluster {cluster_id + 1}
                </span>
                <span style="color: #6B564E; font-size: 12px; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                    <span style="height: 8px; width: 8px; background-color: #8FA353; border-radius: 50%; display: inline-block;"></span>
                    Active Profile
                </span>
            </div>

            <h2 style="color: #2D150B; font-size: 22px; font-weight: 800; margin: 0 0 4px 0;">
                {persona['title']}
            </h2>
            <div style="color: #6B564E; font-size: 13px; font-weight: 700; margin-bottom: 14px;">
                {persona['subtitle']}
            </div>
            
            <p style="color: #524038; font-size: 14px; margin: 0 0 20px 0; line-height: 1.6;">
                {persona['description']}
            </p>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px;">
                <div style="background: #F7F3F0; padding: 14px; border-radius: 20px;">
                    <span style="color: #6B564E; font-size: 11px; font-weight: 700; text-transform: uppercase; display: block; margin-bottom: 4px;">Predicted Retention</span>
                    <span style="color: #2D150B; font-size: 20px; font-weight: 800;">{persona['retention_score']}</span>
                </div>
                <div style="background: #F7F3F0; padding: 14px; border-radius: 20px;">
                    <span style="color: #6B564E; font-size: 11px; font-weight: 700; text-transform: uppercase; display: block; margin-bottom: 4px;">Value Tier</span>
                    <span style="color: #2D150B; font-size: 20px; font-weight: 800;">{persona['value_tier']}</span>
                </div>
            </div>

            <div style="display: flex; gap: 14px; align-items: flex-start; background: #F7F3F0; padding: 16px; border-radius: 20px; border-left: 5px solid {persona['border_color']};">
                <span style="font-size: 20px; line-height: 1;">💡</span>
                <div>
                    <strong style="color: #2D150B; font-size: 12px; display: block; margin-bottom: 4px; text-transform: uppercase;">Action Strategy:</strong>
                    <p style="color: #524038; font-size: 13.5px; margin: 0; line-height: 1.5;">{persona['recommendation']}</p>
                </div>
            </div>
        </div>
        """
        return html_output

    except Exception as e:
        return f"""
        <div style="background: #FDEDEE; border: 2px solid #E65A72; color: #2D150B; padding: 16px; border-radius: 20px; font-family: sans-serif;">
            ⚠️ <strong>Model Notice:</strong> {e}
        </div>
        """


# ==========================================
# Soft Pill-Style Custom CSS
# ==========================================

css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

/* Mesh Gradient Outer Canvas */
body, .gradio-container, .main, div.gradio-container {
    background: radial-gradient(circle at 10% 20%, #FF6F3C 0%, #FFAA5A 35%, #FFF1E6 75%, #FFFFFF 100%) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    min-height: 100vh;
    padding: 20px 0;
}

/* White Card Containers */
.block, .form, .gr-form, div[class*="block"], div[class*="form"] {
    background-color: #FFFFFF !important;
    border: none !important;
    border-radius: 36px !important;
    box-shadow: 0 20px 50px rgba(180, 80, 20, 0.12) !important;
    padding: 20px !important;
}

/* Soft Pill Input Controls */
input, select, textarea, .gr-input, input[type="number"], input[type="text"] {
    background-color: #F7F3F0 !important;
    border: 2px solid transparent !important;
    color: #2D150B !important;
    border-radius: 9999px !important;
    padding: 12px 20px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

/* Active Hover & Focus Olive Green Outline (Matching Image 3rd field) */
input:focus, select:focus, textarea:focus, .gr-input:focus-within {
    border-color: #8FA353 !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 0 0 4px rgba(143, 163, 83, 0.15) !important;
    outline: none !important;
}

/* Dropdown Override */
.gr-dropdown, .gr-dropdown *, fieldset {
    background-color: #F7F3F0 !important;
    border-color: transparent !important;
    border-radius: 9999px !important;
    color: #2D150B !important;
}

ul.options, .options, .dropdown-options, ul[role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EAE3DC !important;
    border-radius: 20px !important;
    box-shadow: 0 15px 30px rgba(45, 21, 11, 0.1) !important;
}

li.item, .option, li[role="option"] {
    background-color: #FFFFFF !important;
    color: #2D150B !important;
    border-radius: 12px !important;
    margin: 4px 6px !important;
    font-weight: 600 !important;
}

li.item:hover, .option:hover, li[role="option"]:hover {
    background-color: #F7F3F0 !important;
    color: #8FA353 !important;
}

/* Labels - Dark Brown Bold Typography */
label span, div[data-testid="block-label"], .gr-label {
    color: #2D150B !important;
    font-weight: 800 !important;
    font-size: 13px !important;
    margin-bottom: 8px !important;
    text-transform: none !important;
    background: transparent !important;
}

/* App Header Card */
.main-header {
    background: #FFFFFF;
    border-radius: 36px;
    padding: 30px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 20px 50px rgba(180, 80, 20, 0.12);
}

.main-header h1 {
    color: #2D150B !important;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}

.developer-card {
    background: #F7F3F0;
    border-radius: 9999px;
    padding: 8px 22px;
    display: inline-flex;
    gap: 12px;
    align-items: center;
    margin-top: 12px;
}

.dev-info {
    color: #6B564E;
    font-size: 13px;
    font-weight: 700;
}

.dev-info span {
    color: #2D150B;
    font-weight: 800;
}

.card-title {
    color: #2D150B;
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 14px;
}

/* Range Slider */
input[type="range"] {
    accent-color: #2D150B !important;
}

/* Primary Scoring Button */
button.primary-btn {
    background: #2D150B !important;
    color: #FFFFFF !important;
    border-radius: 9999px !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    padding: 16px !important;
    border: none !important;
    box-shadow: 0 8px 25px rgba(45, 21, 11, 0.25) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    width: 100% !important;
}

button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(45, 21, 11, 0.35) !important;
}
"""

# ==========================
# Gradio Interface Setup
# ==========================

with gr.Blocks(css=css, title="Customer Analytics AI Engine") as demo:

    # Main Header
    gr.HTML("""
    <div class="main-header">
        <h1>🛡️ Customer Segmentation Intelligence</h1>
        <p style="color: #6B564E; margin: 0; font-weight: 600;">Enterprise Machine Learning Dashboard for Real-Time Persona Scoring</p>
        <div class="developer-card">
            <div class="dev-info">Developer: <span>Sameer Chopra</span></div>
            <div class="dev-info">•</div>
            <div class="dev-info">Roll No: <span>241020</span></div>
        </div>
    </div>
    """)

    # Layout Row
    with gr.Row(equal_height=False):

        # Left Column: Inputs
        with gr.Column(scale=3):

            with gr.Group():
                gr.HTML(
                    '<div class="card-title">💵 Financial Profile & Coverage</div>'
                )
                with gr.Row():
                    age = gr.Slider(
                        label="Age", minimum=18, maximum=80, value=35, step=1
                    )
                    income = gr.Number(label="Income Level ($)", value=60000)
                with gr.Row():
                    coverage = gr.Number(
                        label="Coverage Amount ($)", value=250000
                    )
                    premium = gr.Number(label="Premium Amount ($)", value=15000)

            with gr.Group():
                gr.HTML(
                    '<div class="card-title">👤 Demographics & Background</div>'
                )
                with gr.Row():
                    gender = gr.Dropdown(
                        ["Male", "Female"], value="Male", label="Gender"
                    )
                    marital_status = gr.Dropdown(
                        ["Single", "Married", "Divorced", "Widowed"],
                        value="Single",
                        label="Marital Status",
                    )
                with gr.Row():
                    education = gr.Dropdown(
                        ["High School", "Bachelor", "Master", "PhD"],
                        value="Bachelor",
                        label="Education Level",
                    )
                    geographic = gr.Dropdown(
                        ["Urban", "Suburban", "Rural"],
                        value="Urban",
                        label="Geography",
                    )

            with gr.Group():
                gr.HTML(
                    '<div class="card-title">⚙️ Behavioral & Policy Details</div>'
                )
                with gr.Row():
                    occupation = gr.Dropdown(
                        [
                            "Employed",
                            "Business",
                            "Self-Employed",
                            "Student",
                            "Retired",
                            "Unemployed",
                        ],
                        value="Employed",
                        label="Occupation",
                    )
                    policy_type = gr.Dropdown(
                        ["Health", "Life", "Vehicle", "Home", "Travel"],
                        value="Health",
                        label="Policy Type",
                    )
                with gr.Row():
                    behavioral = gr.Dropdown(
                        ["Low", "Medium", "High"],
                        value="Medium",
                        label="Activity Score",
                    )
                    interaction = gr.Dropdown(
                        ["Low", "Medium", "High"],
                        value="Medium",
                        label="Service Frequency",
                    )
                    insurance_products = gr.Dropdown(
                        ["1", "2", "3", "4", "5"],
                        value="2",
                        label="Products Owned",
                    )

            with gr.Group():
                gr.HTML(
                    '<div class="card-title">📱 Channel & Interaction Preferences</div>'
                )
                with gr.Row():
                    customer_preference = gr.Dropdown(
                        ["Price", "Coverage", "Service", "Benefits"],
                        value="Coverage",
                        label="Primary Value Priority",
                    )
                    communication_channel = gr.Dropdown(
                        ["Email", "Phone", "SMS", "Mobile App"],
                        value="Email",
                        label="Preferred Channel",
                    )
                with gr.Row():
                    contact_time = gr.Dropdown(
                        ["Morning", "Afternoon", "Evening"],
                        value="Morning",
                        label="Preferred Time",
                    )
                    language = gr.Dropdown(
                        ["English", "Hindi", "Spanish", "French"],
                        value="English",
                        label="Language",
                    )

            purchase_year = gr.Number(value=2024, visible=False)

        # Right Column: Scoring & Persona Insights
        with gr.Column(scale=2):

            with gr.Group():
                gr.HTML('<div class="card-title">📊 Live Telemetry Bar</div>')
                live_kpi_display = gr.HTML()

            predict_btn = gr.Button(
                "⚡ Score Customer Segment", elem_classes=["primary-btn"]
            )

            with gr.Group():
                gr.HTML(
                    '<div class="card-title">🎯 Cluster Persona Insights</div>'
                )
                output = gr.HTML()

    # Event Handlers
    kpi_inputs = [income, coverage, premium, interaction]
    for inp in kpi_inputs:
        inp.change(
            fn=update_live_metrics, inputs=kpi_inputs, outputs=live_kpi_display
        )

    demo.load(
        fn=update_live_metrics, inputs=kpi_inputs, outputs=live_kpi_display
    )

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
            language,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
