!pip install gradio

import requests
import gradio as gr

# Define the FastAPI endpoint URL (replace with the actual ngrok URL you get after running the FastAPI app)
api_url = "https://ec01-34-173-107-35.ngrok-free.app/"

# Function to call FastAPI and make predictions
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    # Prepare the input data as a dictionary
    params = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'diabetes_pedigree': diabetes_pedigree,
        'age': age
    }

    # Make a GET request to the FastAPI endpoint
    response = requests.get(api_url, params=params)

    # Extract the prediction result from the JSON response
    result = response.json()

    return result['prediction']

# Gradio UI setup
input_fields = [
    gr.Number(label="Pregnancies"),
    gr.Number(label="Glucose Level"),
    gr.Number(label="Blood Pressure"),
    gr.Number(label="Skin Thickness"),
    gr.Number(label="Insulin"),
    gr.Number(label="BMI"),
    gr.Number(label="Diabetes Pedigree Function"),
    gr.Number(label="Age"),
]

output_text = gr.Textbox(label="Prediction")

# Launch Gradio interface
gr.Interface(
    fn=predict_diabetes,  # Function to call FastAPI
    inputs=input_fields,  # Input fields for user data
    outputs=output_text,  # Output box for the prediction
    title="Diabetes Prediction",  # Title for your app
    description="Enter the data to predict if the person is diabetic or not"
).launch()
