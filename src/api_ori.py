from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Gauge, generate_latest
import random
import math
import time

app = FastAPI()

# Simulated metric
predicted_cpu = Gauge("upf_predicted_cpu_utilization", "Predicted CPU usage", ["namespace", "pod"])

# Simulated CPU prediction function
def simulate_predicted_cpu(t: float) -> float:
    # Simulates a wave between 30% and 90% CPU
    return 60 + 30 * math.sin(t / 30.0) + random.uniform(-5, 5)

@app.get("/metrics")
def get_metrics():
    # Simulate time using Unix time to change prediction over time
    t = time.time()
    value = simulate_predicted_cpu(t) / 100  # Convert to percentage
    predicted_cpu.labels(namespace="free5gc", pod="free5gc-premier-free5gc-upf-upf-645b59fd95-fmvkh").set(value)
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")