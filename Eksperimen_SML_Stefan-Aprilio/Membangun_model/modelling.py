import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

# Set MLflow tracking URI to localhost
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def train_baseline():
    # Load data
    data_path = 'iris_preprocessing'
    if not os.path.exists(data_path):
        data_path = 'Eksperimen_SML_Stefan-Aprilio/Membangun_model/iris_preprocessing'
    
    df = pd.read_csv(data_path)
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Iris_Baseline_Experiment")
    
    with mlflow.start_run(run_name="Baseline_RF"):
        n_estimators = 100
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Baseline Accuracy: {acc}")
        print(f"Baseline F1 Score: {f1}")
        
        # Log params and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(clf, "model")
        
        print("Baseline model logged to MLflow.")

if __name__ == "__main__":
    train_baseline()
