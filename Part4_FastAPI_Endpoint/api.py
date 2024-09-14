!pip install fastapi nest-asyncio pyngrok uvicorn

!ngrok config add-authtoken 2m18J8IrWllezMcpaT8DybJ5Dma_744oLUTbJ8YwW52RgZUSZ

from fastapi import FastAPI
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import joblib
import numpy as np

# Load the pre-trained model and scaler using joblib
scaler = joblib.load('scaler.joblib')
loaded_model = joblib.load('filename.joblib')

# Create FastAPI instance
app = FastAPI()

# Define a function to make predictions
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    # Prepare the input data as a tuple
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)

    # Convert the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array to fit the model input
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(std_data)

    # Determine if the person is diabetic or not
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# FastAPI POST endpoint for prediction
@app.get('/')
async def predict(pregnancies: float, glucose: float, blood_pressure: float, skin_thickness: float, insulin: float, bmi: float, diabetes_pedigree: float, age: float):
    # Make the prediction
    result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)

    # Return the prediction result
    return {'prediction': result}

# Example route to test if the API is working
@app.get('/')
async def index():
    return {'message': 'Welcome to the Diabetes Prediction API'}

# Setup ngrok for public access
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)

# Required for running in notebooks or certain environments
nest_asyncio.apply()

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, port=8000)
