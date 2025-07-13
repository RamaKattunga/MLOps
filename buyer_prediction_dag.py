from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import logging
import sys
import os

# Add model_development.py to path for import (assume it's in /src)
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from model_development import train_and_log_model, generate_synthetic_data

# Set up logging
logging.basicConfig(level=logging.INFO)

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 7, 13),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'buyer_prediction_pipeline',
    default_args=default_args,
    description='End-to-end MLOps for Buyer Prediction',
    schedule_interval=timedelta(days=1),  # Daily; change to '@weekly' for retraining
    catchup=False,
)

def train_model_task(**kwargs):
    """Task to train and log model."""
    try:
        train_and_log_model(train_size=1000, test_size=200, val_size=200)
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

def predict_on_new_data_task(**kwargs):
    """Task to load model from MLflow and predict on new data."""
    try:
        # Load latest registered model
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions("BuyerPredictionModel", stages=["None"])[0].version
        model_uri = f"models:/BuyerPredictionModel/{latest_version}"
        model = mlflow.sklearn.load_model(model_uri)
        logging.info(f"Loaded model version {latest_version}.")

        # Generate new data (simulate incoming data)
        X_new, _ = generate_synthetic_data(n_samples=50)  # No labels for inference
        scaler = StandardScaler()  # Re-fit scaler; in prod, load persisted scaler
        X_new_scaled = scaler.fit_transform(X_new)  # Note: In prod, use same scaler as training

        # Predict probabilities
        probs = model.predict_proba(X_new_scaled)[:, 1]
        predictions_df = pd.DataFrame({'buyer_id': range(len(probs)), 'buy_probability': probs})
        
        # Log predictions as artifact (e.g., save to CSV)
        output_path = "/tmp/new_predictions.csv"  # In prod, save to S3/DB
        predictions_df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved: {output_path}. Sample: {predictions_df.head().to_dict()}")

        # In a real setup, push to a dashboard or alert if probs > threshold
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise

# Define tasks
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    provide_context=True,
    dag=dag,
)

predict_task = PythonOperator(
    task_id='predict_on_new_data',
    python_callable=predict_on_new_data_task,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
train_task >> predict_task
