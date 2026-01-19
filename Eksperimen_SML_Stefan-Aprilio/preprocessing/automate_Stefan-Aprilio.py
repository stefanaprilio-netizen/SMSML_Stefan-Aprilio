import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_iris_data(input_path, output_path):
    print(f"Reading data from {input_path}...")
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    
    try:
        df = pd.read_csv(input_path, header=None, names=columns)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        
        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])
        
        scaler = StandardScaler()
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        df[features] = scaler.fit_transform(df[features])
        
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, '..', 'iris_raw.csv')
    processed_path = os.path.join(script_dir, 'iris_preprocessing.csv')
    
    if os.path.exists(raw_path):
        preprocess_iris_data(raw_path, processed_path)
    else:
        # Try local path if called from project root
        raw_path = 'Eksperimen_SML_Stefan-Aprilio/iris_raw.csv'
        processed_path = 'Eksperimen_SML_Stefan-Aprilio/preprocessing/iris_preprocessing.csv'
        if os.path.exists(raw_path):
             preprocess_iris_data(raw_path, processed_path)
        else:
            print(f"Could not find {raw_path}")
