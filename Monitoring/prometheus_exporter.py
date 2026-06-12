import time
import psutil
import requests
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# MLServer endpoint
MODEL_API_URL = "http://127.0.0.1:8080/invocations"

# Prometheus metrics
# Total requests received
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests received",
)

# Latency API
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Throughput (requests per interval)
THROUGHPUT = Counter(
    "http_requests_throughput",
    "Total number of requests processed",
)

# Total error
ERROR_COUNT = Counter(
    "http_requests_errors_total",
    "Total number of failed requests",
)
 
# CPU usage
CPU_USAGE = Gauge(
    "system_cpu_usage",
    "CPU Usage Percentage",
)

# RAM usage
RAM_USAGE = Gauge(
    "system_ram_usage",
    "RAM Usage Percentage",
)

# Total fraud detected
FRAUD_DETECTED = Counter(
    "fraud_total_detected",
    "Total number of fraud transactions detected",
)

# Total normal transactions detected
NORMAL_DETECTED = Counter(
    "fraud_total_normal",
    "Total number of normal transactions",
)

# Fraud rate per batch
FRAUD_RATE = Gauge(
    "fraud_rate_per_batch",
    "Proportion of fraud",
)

# Mean anomaly score 
MEAN_ANOMALY_SCORE = Gauge(
    "fraud_mean_anomaly_score",
    "Mean anomaly score",
)
 
# Min anomaly score
MIN_ANOMALY_SCORE = Gauge(
    "fraud_min_anomaly_score",
    "Min anomaly score",
)

# Max anomaly score
MAX_ANOMALY_SCORE = Gauge(
    "fraud_max_anomaly_score",
    "Max anomaly score",
)

# Latency gauge
LAST_LATENCY = Gauge(
    "fraud_last_prediction_latency_seconds",
    "Last prediction latency in seconds",
)

# Batch size
BATCH_SIZE = Gauge(
    "fraud_batch_size",
    "Number of transactions in the latest batch",
)
 
# Model uptime
START_TIME = time.time()
MODEL_UPTIME = Gauge(
    "fraud_model_uptime_seconds",
    "Time since model exporter started in seconds",
)

@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Expose application metrics for Prometheus scraping
    """
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    MODEL_UPTIME.set(round(time.time() - START_TIME, 2))
 
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Acts as a proxy to forward incoming inference requests to the underlying model API
    """
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    data = request.get_json()

    try:
        headers = {"Content-Type": "application/json"}
        payload_to_send = data

        response = requests.post(
            MODEL_API_URL,
            json=payload_to_send,
            headers=headers,
            timeout=30,
        )
        duration = time.time() - start_time

        # Log latency
        REQUEST_LATENCY.observe(duration)
        LAST_LATENCY.set(round(duration, 6))

        # Get batch size
        if "inputs" in payload_to_send and len(payload_to_send["inputs"]) > 0:
            batch_sz = payload_to_send["inputs"][0].get("shape", [1, 1])[0]
        else:
            batch_sz = 1
        BATCH_SIZE.set(batch_sz)

        if response.status_code == 200:
            result = response.json()

            # Handle both response formats
            if "predictions" in result:
                predictions = result["predictions"]
            elif "outputs" in result and len(result["outputs"]) > 0:
                predictions = result["outputs"][0].get("data", [])
            else:
                predictions = []

            n_fraud = sum(1 for p in predictions if p == -1)
            n_normal = sum(1 for p in predictions if p == 1)
            n_total = len(predictions)

            FRAUD_DETECTED.inc(n_fraud)
            NORMAL_DETECTED.inc(n_normal)

            if n_total > 0:
                FRAUD_RATE.set(round(n_fraud / n_total, 4))

            return jsonify(result)
        else:
            ERROR_COUNT.inc()
            return jsonify({"error": response.text}), response.status_code
    except Exception as e:
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        ERROR_COUNT.inc()

        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """
    Health check monitoring
    """
    return jsonify({
        "status": "OK",
        "uptime": round(time.time() - START_TIME, 2),
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
    }), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)