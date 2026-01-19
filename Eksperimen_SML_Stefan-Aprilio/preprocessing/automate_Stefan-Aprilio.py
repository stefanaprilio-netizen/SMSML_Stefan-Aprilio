import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def preprocess_adult_data(input_path, output_path):
    print(f"Reading data from {input_path}...")
    
    # Define columns since the raw data might not have headers
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex', 
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    try:
        # Check if file has header or not (UCI data usually doesn't, but my download scripts might have added it)
        df_check = pd.read_csv(input_path, nrows=5)
        if set(columns).issubset(df_check.columns):
            df = pd.read_csv(input_path)
        else:
            df = pd.read_csv(input_path, header=None, names=columns)
            
        print("Data loaded successfully.")
        
        # 1. Handling missing values (replacing '?' with NaN if exists)
        df.replace(' ?', np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # 2. Removing duplicates
        df.drop_duplicates(inplace=True)
        
        # 3. Encoding categorical data
        le = LabelEncoder()
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
            
        print("Preprocessing complete.")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save preprocessed data
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different potential locations for adult_raw/adult_new.csv
    potential_raw_paths = [
        os.path.join(script_dir, '..', 'adult_raw', 'adult_new.csv'),
        os.path.join(os.getcwd(), 'Eksperimen_SML_Stefan-Aprilio', 'adult_raw', 'adult_new.csv'),
        os.path.join(os.getcwd(), 'adult_raw', 'adult_new.csv')
    ]
    
    raw_path = None
    for path in potential_raw_paths:
        if os.path.exists(path):
            raw_path = path
            break
            
    if raw_path:
        # Define output path relative to script directory
        processed_path = os.path.join(script_dir, 'adult_preprocessing', 'adult_cleaned.csv')
        preprocess_adult_data(raw_path, processed_path)
    else:
        print("Could not find adult_raw/adult.csv in any expected locations.")
