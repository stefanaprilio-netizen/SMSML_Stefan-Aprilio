import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import os

# Initialize DagsHub
# Based on your repo: stefanaprilio-netizen/SMSML_Stefan-Aprilio
repo_owner = "stefanaprilio-netizen"
repo_name = "SMSML_Stefan-Aprilio"

# Initialize DagsHub for MLflow tracking
try:
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
except Exception as e:
    print(f"DagsHub init failed (likely no token configured): {e}")
    # Fallback to localhost if DagsHub fails
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

def train_tuned():
    # Load data
    data_path = 'iris_preprocessing'
    if not os.path.exists(data_path):
        data_path = 'Eksperimen_SML_Stefan-Aprilio/Membangun_model/iris_preprocessing'
    
    df = pd.read_csv(data_path)
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Iris_Tuning_Experiment")
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    with mlflow.start_run(run_name="Tuned_RF"):
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Tuned Accuracy: {acc}")
        print(f"Best Params: {best_params}")
        
        # Log best params and metrics
        for param, value in best_params.items():
            mlflow.log_param(param, value)
            
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "tuned_model")
        
        # Save a local artifact example
        with open("tuning_summary.txt", "w") as f:
            f.write(f"Best Params: {best_params}\nAccuracy: {acc}\n")
        mlflow.log_artifact("tuning_summary.txt")
        
        print("Tuned model and artifacts logged to MLflow/DagsHub.")

if __name__ == "__main__":
    train_tuned()
