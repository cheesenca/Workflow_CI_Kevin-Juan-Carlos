import requests
import json
import time
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
 
# Logging setup
logging.basicConfig(
    filename="api_model_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
 
# Config
API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "bank_transactions_preprocessing.csv"
CONTAMINATION = 0.05
RANDOM_STATE = 42
N_SAMPLES = 10  
FEATURE_NAMES = [
    "TransactionAmount", "TransactionType", "Location", "Channel",
    "CustomerAge", "CustomerOccupation", "TransactionDuration",
    "LoginAttempts", "AccountBalance", "TransactionMonth",
    "TransactionDayOfWeek", "TransactionYear", "LoginAttempts_Bin",
]
 
 
def load_sample_data(data_path=DATA_PATH, n=N_SAMPLES):
    """
    Loads data from CSV file
    """
    df = pd.read_csv(data_path)
    sample = df.sample(n=n, random_state=None)
    scaler = StandardScaler()
    scaler.fit(df)
    X_scaled = scaler.transform(sample)

    return X_scaled.tolist(), sample
 
 
def send_request(data):
    """
    Sends a POST request containing payload data to the prediction API and logs
    """
    payload = {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": data,
        }
    }
    headers = {"Content-Type": "application/json"}
    start_time = time.time()
 
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=10,
        )
        response_time = time.time() - start_time
 
        if response.status_code == 200:
            prediction = response.json()
            logging.info(
                f"Request: {len(data)} samples | "
                f"Response: {prediction} | "
                f"Response Time: {response_time:.4f}s"
            )
            print(f"Prediction: {prediction}")
            print(f"Response Time: {response_time:.4f}s")
            return prediction
        else:
            logging.error(f"Error {response.status_code}: {response.text}")
            print(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        print(f"Exception: {str(e)}")
        return None
 
 
def main():
    # Load sample data
    data, sample = load_sample_data()
    result = send_request(data)

    if result:
        predictions = result.get("predictions", [])
        n_fraud = sum(1 for p in predictions if p == -1)
        n_normal = sum(1 for p in predictions if p == 1)
        print(f"\nSummary:")
        print(f"Total: {len(predictions)}")
        print(f"Fraud (-1): {n_fraud}")
        print(f"Normal (1): {n_normal}")
 
 
if __name__ == "__main__":
    main()