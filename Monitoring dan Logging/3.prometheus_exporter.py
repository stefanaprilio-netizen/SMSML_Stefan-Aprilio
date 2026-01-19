import time
import random
from prometheus_client import start_http_server, Counter, Histogram

# Define custom metrics
INFERENCE_COUNT = Counter('model_inference_total', 'Total number of model inferences')
INFERENCE_LATENCY = Histogram('model_inference_latency_seconds', 'Inference latency in seconds')
MODEL_ACCURACY = Counter('model_accuracy_hits_total', 'Number of correct predictions')

def process_request():
    """Simulates processing a request."""
    INFERENCE_COUNT.inc()
    
    start_time = time.time()
    # Simulate processing time
    time.sleep(random.uniform(0.01, 0.2))
    INFERENCE_LATENCY.observe(time.time() - start_time)
    
    # Simulate accuracy (correct/incorrect)
    if random.random() > 0.05:
        MODEL_ACCURACY.inc()

if __name__ == "__main__":
    # Start the exporter on port 9090
    port = 9090
    start_http_server(port)
    print(f"Prometheus Exporter started on port {port}")
    
    # Simulating continuous monitoring data
    while True:
        process_request()
        time.sleep(random.uniform(1, 5))
