import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
 
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv 
load_dotenv()


# Configuration
DATA_PATH = "workflow-ci/MLProject/bank_transactions_preprocessing.csv"
EXPERIMENT_NAME = "fraud-detection"
CONTAMINATION = 0.05
RANDOM_STATE = 42
N_ESTIMATORS = 100

# MLflow setup
def mlflow_setup():
    """
    Setup MLflow tracking URI
    """
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    dagshub_username = os.environ.get("DAGSHUB_USERNAME")
    dagshub_repo = os.environ.get("DAGSHUB_REPO")

    if dagshub_token and dagshub_username and dagshub_repo:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        tracking_uri = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI set to: {tracking_uri}")
    else:
        mlflow.set_tracking_uri("mlruns")
        print("MLflow setup locally completed")

    mlflow.set_experiment(EXPERIMENT_NAME)


# Load data
def load_data(data_path=DATA_PATH):
    """
    Load preprocessed CSV and scale features
    """
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(df)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=df.columns)

    print(f"Data loaded and scaled. Shape: {df.shape}")
    return df, scaler, X_scaled


# Artifacts
def plot_anomaly_score(anomaly_scores, labels, save_path="anomaly_score_distribution.png"):
    """
    Plot and save anomaly score distribution
    """
    os.makedirs("models/isolation_forest", exist_ok=True)
    save_path = f"models/isolation_forest/{save_path}"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(anomaly_scores[labels == 0], bins=60, alpha=0.6, color="blue", label="Normal")
    ax.hist(anomaly_scores[labels == 1], bins=60, alpha=0.6, color="red", label="Fraud")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Threshold")
    ax.set_title("Anomaly Score Distribution")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Artifact saved: {save_path}")
    return save_path


def plot_fraud_proportion(labels, save_path="fraud_proportion.png"):
    """
    Plot and save fraud vs normal transaction proportion chart
    """
    os.makedirs("models/isolation_forest", exist_ok=True)
    save_path = f"models/isolation_forest/{save_path}"
    counts = pd.Series(labels).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Normal", "Fraud"], counts.values, color=["blue", "red"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200, f"{val:,}\n({val/len(labels):.1%})", ha="center", fontsize=10)
    ax.set_title("Predicted Fraud vs Normal Transactions")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Artifact saved: {save_path}")
    return save_path


# Model training
def train_model(df, X_scaled, contamination=CONTAMINATION, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS):
    """
    Train Isolation Forest
    """
    mlflow.autolog(log_models=True)
    with mlflow.start_run(run_name="isolation-forest-autolog") as run:
        # Train model
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        model.fit(X_scaled)

        # Predict
        predictions = model.predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        labels = (predictions == -1).astype(int)
        n_fraud = int(labels.sum())
        n_normal = int((labels == 0).sum()) 

        # Log parameters
        mlflow.log_param("data_path", DATA_PATH)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", df.shape[1])
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_estimators", n_estimators)

        # Log metrics
        mlflow.log_metric("n_anomalies", n_fraud)
        mlflow.log_metric("n_normal", n_normal)
        mlflow.log_metric("anomaly_rate", round(n_fraud / len(labels), 4))
        mlflow.log_metric("mean_anomaly_score", round(float(np.mean(anomaly_scores)), 4))
        mlflow.log_metric("std_anomaly_score", round(float(np.std(anomaly_scores)), 4))

        # Log Artifacts for plots
        mlflow.log_artifact(plot_anomaly_score(anomaly_scores, labels))
        mlflow.log_artifact(plot_fraud_proportion(labels))

        # Log Artifact for predictions CSV
        os.makedirs("models/isolation_forest", exist_ok=True)

        df_result = df.copy()
        df_result["AnomalyScore"] = anomaly_scores
        df_result["IsFraud"] = labels
        result_path = "models/isolation_forest/fraud_predictions.csv"

        df_result.to_csv(result_path, index=False)
        mlflow.log_artifact(result_path)

        input_example = X_scaled.head()
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=infer_signature(X_scaled, model.predict(X_scaled)),
        )

        try:
            # Register model to MLflow Model Registry
            model_uri = f"runs:/{run.info.run_id}/model"
            registered = mlflow.register_model(
                model_uri=model_uri,
                name="fraud-detection",
            )

            # Set alias for model version
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                name="fraud-detection",
                alias="champion",
                version=registered.version,
            )
        except Exception as e:
            print(f"Model registry skipped: {e}")

        metrics = {
            "n_anomalies": n_fraud,
            "n_normal": n_normal,
            "anomaly_rate": round(n_fraud / len(labels), 4),
            "mean_anomaly_score": round(float(np.mean(anomaly_scores)), 4),
            "std_anomaly_score": round(float(np.std(anomaly_scores)), 4),
        }
 
        return run.info.run_id, metrics
    

def args_parser():
    parser = argparse.ArgumentParser(
        description="Train Isolation Forest for Fraud Detection",
    )

    parser.add_argument(
        "--contamination", 
        type=float, 
        default=CONTAMINATION, 
        help="Expected proportion of anomalies in the data",
    )
    parser.add_argument(
        "--random_state", type=int, 
        default=RANDOM_STATE, 
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n_estimators", 
        type=int, 
        default=N_ESTIMATORS, 
        help="Number of trees in the Isolation Forest",
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="workflow-ci/MLProject/bank_transactions_preprocessing.csv", 
        help="Path to preprocessed dataset CSV",
    )

    return parser.parse_args()


def main():
    args = args_parser()
    try:
        mlflow_setup()

        df, scaler, X_scaled = load_data(args.dataset)
        run_id, metrics = train_model(
            df, 
            X_scaled,
            contamination=args.contamination,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )

        print(f"Run ID: {run_id}")
        print(f"Anomaly rate: {metrics['anomaly_rate']:.2%}")
        print(f"Fraud count: {metrics['n_anomalies']:,}")
        print(f"Normal count: {metrics['n_normal']:,}")

        os.makedirs("models", exist_ok=True)
        with open("models/model_run_id.txt", "w") as f:
            f.write(run_id)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    main()