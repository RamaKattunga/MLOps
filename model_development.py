import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Set up logging for detailed tracing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_synthetic_data(n_samples=1000, n_features=6, n_classes=2, random_state=42):
    """
    Generate synthetic buyer data.
    Features: age, income, past_purchases, is_b2b (0/1), company_size, engagement_score.
    Target: will_buy (0/1).
    B2B samples have higher company_size; B2C have varied age/income.
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                               n_informative=4, random_state=random_state)
    # Customize features for realism
    df = pd.DataFrame(X, columns=['age', 'income', 'past_purchases', 'is_b2b', 'company_size', 'engagement_score'])
    df['age'] = np.clip(np.abs(df['age'] * 10 + 35), 18, 70)  # Age 18-70
    df['income'] = np.clip(np.abs(df['income'] * 10000 + 50000), 20000, 200000)  # Income 20k-200k
    df['past_purchases'] = np.clip(np.abs(df['past_purchases'] * 2), 0, 10)  # 0-10 purchases
    df['is_b2b'] = np.round(np.clip(df['is_b2b'], 0, 1))  # Binary
    df['company_size'] = np.where(df['is_b2b'] == 1, np.clip(np.abs(df['company_size'] * 50 + 100), 10, 1000), 0)  # B2B: 10-1000 employees
    df['engagement_score'] = np.clip(np.abs(df['engagement_score'] * 10), 0, 100)  # 0-100
    df['will_buy'] = y
    logging.info(f"Generated data: {df.shape}, class balance: {np.mean(y):.2f}")
    return df.drop('will_buy', axis=1), df['will_buy']

def train_and_log_model(train_size=1000, test_size=200, val_size=200, random_state=42):
    """
    Full model development pipeline with MLflow logging.
    - Generate data
    - Split: train/test/val
    - Scale features
    - Train LogisticRegression
    - Evaluate on test/val
    - Log params, metrics, artifacts, model
    """
    # Generate data
    X_train, y_train = generate_synthetic_data(n_samples=train_size, random_state=random_state)
    X_test, y_test = generate_synthetic_data(n_samples=test_size, random_state=random_state + 1)
    X_val, y_val = generate_synthetic_data(n_samples=val_size, random_state=random_state + 2)
    
    # Further split train for internal validation if needed, but here we use separate sets
    logging.info("Data generated and split.")

    # Preprocessing: Scale features (important for LogisticRegression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    # Start MLflow run
    with mlflow.start_run(run_name="Buyer_Prediction_Training"):
        # Parameters (hyperparams)
        params = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": random_state}
        mlflow.log_params(params)
        
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)
        logging.info("Model trained.")

        # Predictions and metrics on test
        y_pred_test = model.predict(X_test_scaled)
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]  # Probability of buying
        metrics_test = {
            "accuracy_test": accuracy_score(y_test, y_pred_test),
            "precision_test": precision_score(y_test, y_pred_test),
            "recall_test": recall_score(y_test, y_pred_test),
            "roc_auc_test": roc_auc_score(y_test, y_prob_test)
        }
        mlflow.log_metrics(metrics_test)
        
        # Metrics on validation
        y_pred_val = model.predict(X_val_scaled)
        y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
        metrics_val = {
            "accuracy_val": accuracy_score(y_val, y_pred_val),
            "precision_val": precision_score(y_val, y_pred_val),
            "recall_val": recall_score(y_val, y_pred_val),
            "roc_auc_val": roc_auc_score(y_val, y_prob_val)
        }
        mlflow.log_metrics(metrics_val)
        logging.info(f"Metrics logged: Test ROC-AUC={metrics_test['roc_auc_test']:.3f}, Val ROC-AUC={metrics_val['roc_auc_val']:.3f}")

        # Log artifact: Confusion matrix plot
        fig, ax = plt.subplots()
        sns.heatmap(pd.crosstab(y_test, y_pred_test), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Test Confusion Matrix")
        plot_path = "confusion_matrix.png"
        fig.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)  # Clean up
        
        # Log model
        mlflow.sklearn.log_model(model, "buyer_model")
        
        # Register model in MLflow registry
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/buyer_model"
        registered_model = mlflow.register_model(model_uri, "BuyerPredictionModel")
        logging.info(f"Model registered: {registered_model.name} version {registered_model.version}")

if __name__ == "__main__":
    train_and_log_model()
