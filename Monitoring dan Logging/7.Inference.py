import pandas as pd
import numpy as np
import time
import requests
import os

def simulate_inference():
    print("Starting Model Inference Simulation...")
    
    # Load sample data for inference
    data_path = 'Eksperimen_SML_Stefan-Aprilio/Membangun_model/iris_preprocessing'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        sample = df.sample(1).drop('class', axis=1).to_dict(orient='records')[0]
        print(f"Sample Input: {sample}")
    else:
        print("Preprocessed data not found, using dummy input.")
        sample = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}

    # In a real scenario, we would POST to an MLflow or FastAPI endpoint
    # For now, we simulate the logic of a request
    print("Sending inference request to model service...")
    time.sleep(0.5)
    
    prediction = np.random.randint(0, 3)
    print(f"Model Prediction: {prediction}")
    
    print("Inference completed successfully.")

if __name__ == "__main__":
    while True:
        simulate_inference()
        print("-" * 30)
        time.sleep(10)
