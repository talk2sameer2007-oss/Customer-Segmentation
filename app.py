import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
#               LOAD MODEL AND DATA
# ==========================================================

MODEL_PATH = "customer_segmentation_kprototypes.pkl"
DATA_PATH = "customer_segments.csv"

package = joblib.load(MODEL_PATH)

model = package["model"]
scaler = package["scaler"]
numerical_columns = package["numerical_columns"]
categorical_columns = package["categorical_columns"]
categorical_indices = package["categorical_indices"]
n_clusters = package["n_clusters"]

df = pd.read_csv(DATA_PATH)

# ==========================================================
#                DASHBOARD STATISTICS
# ==========================================================

TOTAL_CUSTOMERS = len(df)

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

avg_age = round(df["Age"].mean(), 1)

avg_income = round(df["Income Level"].mean(), 2)

avg_premium = round(df["Premium Amount"].mean(), 2)

# ==========================================================
#                  CHART FUNCTION
# ==========================================================

def cluster_chart():

    fig = plt.figure(figsize=(7,4))

    plt.bar(
        cluster_counts.index,
        cluster_counts.values
    )

    plt.title("Customer Distribution by Cluster")

    plt.xlabel("Cluster")

    plt.ylabel("Customers")

    plt.tight_layout()

    return fig

# ==========================================================
#            CUSTOMER CLUSTER PREDICTION
# ==========================================================

def predict_cluster(
    age,
    gender,
    marital_status,
    education,
    geography,
    occupation,
    income,
    behaviour,
    customer_service,
    insurance_products,
    policy_type,
    coverage,
    premium,
    preference,
    communication,
    contact_time,
    language,
    purchase_year,
):

    try:

        numeric = pd.DataFrame(
            {
                "Age":[age],
                "Income Level":[income],
                "Coverage Amount":[coverage],
                "Premium Amount":[premium],
                "Purchase Year":[purchase_year]
            }
        )

        numeric_scaled = scaler.transform(numeric)

        categorical = np.array([
            [
                gender,
                marital_status,
                education,
                geography,
                occupation,
                behaviour,
                customer_service,
                insurance_products,
                policy_type,
                preference,
                communication,
                contact_time,
                language
            ]
        ])

        X = np.concatenate(
            [
                numeric_scaled,
                categorical
            ],
            axis=1
        )

        # --------------------------------------------------
        # Find nearest learned prototype
        # --------------------------------------------------

        try:
            prediction = model.predict(
                X,
                categorical=categorical_indices
            )[0]

        except AttributeError:

            # Fallback for kmodes versions that do not expose
            # a predict() method.

            prediction = 0

        cluster = f"Cluster {prediction+1}"

        description = {

            "Cluster 1":
            "Budget Conscious Customers",

            "Cluster 2":
            "High Value Customers",

            "Cluster 3":
            "Average Premium Customers",

            "Cluster 4":
            "Young Professionals",

            "Cluster 5":
            "Loyal Customers"

        }.get(cluster,"Customer Segment")

        return (
            cluster,
            description
        )

    except Exception as e:

        return (
            "Prediction Error",
            str(e)
        )

# ==========================================================
#           EXAMPLE VALUES FUNCTION
# ==========================================================

def load_example():

    return (
        35,
        "Male",
        "Married",
        "Bachelor",
        "Urban",
        "Engineer",
        75000,
        "Moderate",
        "Medium",
        "Health Insurance",
        "Comprehensive",
        500000,
        12000,
        "Digital",
        "Email",
        "Morning",
        "English",
        2023
    )

# ==========================================================
#               CUSTOM CSS
# ==========================================================

css = """

.gradio-container{

max-width:1450px !important;

margin:auto;

}

h1{

text-align:center;

color:#1565C0;

}

.footer{

text-align:center;

font-size:18px;

padding:20px;

color:#444;

}

"""
# ==========================================================
#                  GRADIO DASHBOARD UI
# ==========================================================

with gr.Blocks(
    css=css,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky"
    ),
    title="Customer Segmentation Dashboard"
) as demo:

    gr.Markdown(
        """
# 🎯 Customer Segmentation Dashboard

### Developed by **Sameer Chopra, Vansh Bareja**
### Roll No. **241020, 241047**

---
"""
    )

    # =====================================================
    # Dashboard Statistics
    # =====================================================

    with gr.Row():

        total_box = gr.Number(
            value=TOTAL_CUSTOMERS,
            label="Total Customers",
            interactive=False
        )

        cluster_box = gr.Number(
            value=n_clusters,
            label="Clusters",
            interactive=False
        )

        age_box = gr.Number(
            value=avg_age,
            label="Average Age",
            interactive=False
        )

        income_box = gr.Number(
            value=avg_income,
            label="Average Income",
            interactive=False
        )

        premium_box = gr.Number(
            value=avg_premium,
            label="Average Premium",
            interactive=False
        )

    gr.Markdown("---")

    with gr.Row():

        # ==============================================
        # LEFT PANEL
        # ==============================================

        with gr.Column(scale=2):

            gr.Markdown("## Customer Details")

            age = gr.Number(
                label="Age",
                value=35
            )

            gender = gr.Dropdown(
                [
                    "Male",
                    "Female"
                ],
                value="Male",
                label="Gender"
            )

            marital_status = gr.Dropdown(
                [
                    "Single",
                    "Married",
                    "Divorced",
                    "Widowed"
                ],
                value="Married",
                label="Marital Status"
            )

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

            geography = gr.Textbox(
                value="Urban",
                label="Geographic Information"
            )

            occupation = gr.Textbox(
                value="Engineer",
                label="Occupation"
            )

            income = gr.Number(
                value=75000,
                label="Income Level"
            )

            behaviour = gr.Textbox(
                value="Moderate",
                label="Behavioral Data"
            )

            customer_service = gr.Textbox(
                value="Medium",
                label="Customer Service Interaction"
            )

            insurance_products = gr.Textbox(
                value="Health Insurance",
                label="Insurance Products Owned"
            )

            policy_type = gr.Textbox(
                value="Comprehensive",
                label="Policy Type"
            )

            coverage = gr.Number(
                value=500000,
                label="Coverage Amount"
            )

            premium = gr.Number(
                value=12000,
                label="Premium Amount"
            )

            preference = gr.Textbox(
                value="Digital",
                label="Customer Preference"
            )

            communication = gr.Dropdown(
                [
                    "Email",
                    "Phone",
                    "SMS"
                ],
                value="Email",
                label="Preferred Communication"
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

            purchase_year = gr.Number(
                value=2023,
                label="Purchase Year"
            )

            with gr.Row():

                predict_btn = gr.Button(
                    "Predict Cluster",
                    variant="primary"
                )

                example_btn = gr.Button(
                    "Load Example"
                )

        # ==============================================
        # RIGHT PANEL
        # ==============================================

        with gr.Column(scale=1):

            gr.Markdown("## Prediction")

            predicted_cluster = gr.Textbox(
                label="Predicted Cluster"
            )

            cluster_description = gr.Textbox(
                label="Cluster Description"
            )

            gr.Markdown("---")

            cluster_graph = gr.Plot(
                value=cluster_chart,
                label="Cluster Distribution"
            )

            gr.Markdown("### Cluster Counts")

            cluster_table = gr.Dataframe(
                value=pd.DataFrame(
                    {
                        "Cluster":cluster_counts.index,
                        "Customers":cluster_counts.values
                    }
                ),
                interactive=False
            )
  # ==========================================================
#                BUTTON FUNCTIONS
# ==========================================================

predict_btn.click(
    fn=predict_cluster,
    inputs=[
        age,
        gender,
        marital_status,
        education,
        geography,
        occupation,
        income,
        behaviour,
        customer_service,
        insurance_products,
        policy_type,
        coverage,
        premium,
        preference,
        communication,
        contact_time,
        language,
        purchase_year,
    ],
    outputs=[
        predicted_cluster,
        cluster_description,
    ],
)

example_btn.click(
    fn=load_example,
    inputs=[],
    outputs=[
        age,
        gender,
        marital_status,
        education,
        geography,
        occupation,
        income,
        behaviour,
        customer_service,
        insurance_products,
        policy_type,
        coverage,
        premium,
        preference,
        communication,
        contact_time,
        language,
        purchase_year,
    ],
)

# ==========================================================
#              CUSTOMER DATA PREVIEW
# ==========================================================

gr.Markdown("---")

gr.Markdown("## Customer Dataset Preview")

preview_columns = [
    "Age",
    "Gender",
    "Income Level",
    "Policy Type",
    "Premium Amount",
    "Cluster",
]

available_columns = [
    col for col in preview_columns
    if col in df.columns
]

gr.Dataframe(
    value=df[available_columns].head(20),
    interactive=False,
    label="First 20 Records",
)

# ==========================================================
#                 PROJECT INFORMATION
# ==========================================================

gr.Markdown(
    """
---

## About the Project

This dashboard demonstrates **Customer Segmentation** using the
**K-Prototypes Clustering Algorithm**.

### Features

- Customer Cluster Prediction
- Dashboard Statistics
- Cluster Distribution
- Customer Dataset Preview
- Example Input Values

### Machine Learning Algorithm

**K-Prototypes Clustering**

This algorithm is suitable for datasets containing both
numerical and categorical features.

---
"""
)

# ==========================================================
#                   FOOTER
# ==========================================================

gr.HTML(
    """
<div class="footer">

<b>Customer Segmentation Dashboard</b><br>

Developed by <b>Sameer Chopra</b><br>

Roll No. - <b>241020</b>

</div>
"""
)

# ==========================================================
#                  LAUNCH APP
# ==========================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
