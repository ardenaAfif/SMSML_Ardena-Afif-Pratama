from prometheus_client import start_http_server, Summary, Counter, Gauge
import random
import time

# Metriks
REQUEST_TIME = Summary('inference_request_processing_seconds', 'Time spent processing inference request')
TOTAL_REQUESTS = Counter('inference_total_requests', 'Total number of inference requests')
FAILED_REQUESTS = Counter('inference_failed_requests', 'Total number of failed inference requests')
CURRENT_LATENCY = Gauge('inference_current_latency_ms', 'Current latency in ms')
MODEL_ACCURACY = Gauge('inference_model_accuracy', 'Model accuracy')
CPU_USAGE = Gauge('inference_cpu_usage', 'CPU usage (%)')
MEMORY_USAGE = Gauge('inference_memory_usage', 'Memory usage (MB)')
THROUGHPUT = Gauge('inference_throughput', 'Number of requests per second')
ACTIVE_SESSIONS = Gauge('inference_active_sessions', 'Active user sessions')
ERROR_RATE = Gauge('inference_error_rate', 'Inference error rate (%)')

@REQUEST_TIME.time()
def simulate_inference():
    TOTAL_REQUESTS.inc()
    latency = random.uniform(10, 500)
    CURRENT_LATENCY.set(latency)
    time.sleep(latency / 1000)

    if random.random() < 0.1:
        FAILED_REQUESTS.inc()

    MODEL_ACCURACY.set(random.uniform(0.85, 0.95))
    CPU_USAGE.set(random.uniform(10, 90))
    MEMORY_USAGE.set(random.uniform(500, 1500))
    THROUGHPUT.set(random.uniform(5, 100))
    ACTIVE_SESSIONS.set(random.randint(1, 20))
    ERROR_RATE.set((FAILED_REQUESTS._value.get() / TOTAL_REQUESTS._value.get()) * 100)

if __name__ == '__main__':
    start_http_server(8001)
    while True:
        simulate_inference()
        time.sleep(2)