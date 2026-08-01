import gradio as gr
import joblib
import pandas as pd

# Load the trained K-Prototypes model
model = joblib.load("customer_segmentation_kprototypes.pkl")


def predict_segment(
    age,
    gender,
    annual_income,
    spending_score,
    profession,
    work_experience,
    family_size
):
    try:
        # Create DataFrame
        data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Annual Income ($)": [annual_income],
            "Spending Score (1-100)": [spending_score],
            "Profession": [profession],
            "Work Experience": [work_experience],
            "Family Size": [family_size]
        })

        prediction = model.predict(data)

        return f"Customer belongs to Segment: {prediction[0]}"

    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🛍️ Customer Segmentation System

        Predict the customer segment using the trained **K-Prototypes Model**.

        ---
        **Developed by Sameer Chopra**  
        **Roll No.: 241020**
        """
    )

    with gr.Row():
        age = gr.Number(label="Age", value=30)

        gender = gr.Dropdown(
            ["Male", "Female"],
            value="Male",
            label="Gender"
        )

    with gr.Row():
        annual_income = gr.Number(
            label="Annual Income ($)",
            value=60000
        )

        spending_score = gr.Slider(
            minimum=1,
            maximum=100,
            value=50,
            step=1,
            label="Spending Score"
        )

    profession = gr.Dropdown(
        [
            "Artist",
            "Doctor",
            "Engineer",
            "Entertainment",
            "Executive",
            "Healthcare",
            "Homemaker",
            "Lawyer",
            "Marketing",
            "Other"
        ],
        value="Engineer",
        label="Profession"
    )

    with gr.Row():
        work_experience = gr.Number(
            label="Work Experience (Years)",
            value=5
        )

        family_size = gr.Number(
            label="Family Size",
            value=4
        )

    predict_btn = gr.Button("Predict Segment")

    output = gr.Textbox(
        label="Prediction"
    )

    predict_btn.click(
        predict_segment,
        inputs=[
            age,
            gender,
            annual_income,
            spending_score,
            profession,
            work_experience,
            family_size
        ],
        outputs=output
    )

    gr.Examples(
        examples=[
            [25, "Male", 45000, 82, "Engineer", 2, 3],
            [42, "Female", 90000, 35, "Doctor", 15, 4],
            [30, "Female", 60000, 65, "Marketing", 6, 2]
        ],
        inputs=[
            age,
            gender,
            annual_income,
            spending_score,
            profession,
            work_experience,
            family_size
        ]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
