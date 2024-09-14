# Diabetes-Prediction-ML

This repository contains the implementation of a complete machine learning pipeline for predicting [your dataset's problem] using FastAPI and Gradio for deployment.

## Folder Structure
- **Part1_to_Part3_Data_Cleaning_Model_Building_Model_IO**: Contains the script `data_pipeline.py` that handles data cleaning, model building, and saving/loading of the model.
- **Part4_FastAPI_Endpoint**: Contains the FastAPI script `api.py` to expose the model as an API endpoint.
- **Part5_Deployment**: Contains the Gradio UI script `app.py` for interacting with the FastAPI endpoint and deployed on Hugging Face Spaces.

## How to Run Each Part

### Part 1 to Part 3: Data Cleaning, Model Building, and Model Saving/Loading
1. Navigate to the `Part1_to_Part3_Data_Cleaning_Model_Building_Model_IO` folder.
2. Run the `data_pipeline.py` script:
    ```bash
    python data_pipeline.py
    ```
3. This will:
   - Load and clean the dataset.
   - Build the machine learning model.
   - Save the trained model and scaler to files in the `saved_model/` folder.

### Part 4: FastAPI Endpoint
1. Navigate to the `Part4_FastAPI_Endpoint` folder.
2. Install the required packages using:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the FastAPI server by running:
    ```bash
    uvicorn api:app --reload
    ```
4. The API will be running at `http://localhost:8000`.

### Part 5: Deployment on Hugging Face Spaces
1. Navigate to the `Part5_Deployment` folder.
2. The script `app.py` uses Gradio to interact with the FastAPI endpoint.
3. Deploy this on Hugging Face Spaces by following the Hugging Face [documentation](https://huggingface.co/docs).

## Dependencies
- Python 3.7+
- pandas
- scikit-learn
- joblib
- FastAPI
- uvicorn
- gradio
