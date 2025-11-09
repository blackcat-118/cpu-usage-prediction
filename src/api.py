from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
# from fastapi.responses import JSONResponse
from collections import defaultdict
from prometheus_client import Gauge, generate_latest
from timesfm_predictor import TimesfmPredictor
from metrics_client import fetch_metrics
import threading
import time
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
import json
import wandb

app = FastAPI()
CONTEXT_LEN = 30
PRED_LEN = 6

# Store predictors per pod
predictors = {}
metrics_buffer = {}
predictor_heartbeat = {}
# Prometheus metrics
predicted_cpu = Gauge("upf_predicted_cpu_utilization", "Predicted CPU usage", ["namespace", "pod"])

BROKER_IP = "140.113.208.76"  # Replace with your broker IP
MQTT_TOPIC = "upf/metrics"
time_interval = 20 # seconds
previous_metrics = defaultdict()

def on_message(client, userdata, msg):
    global previous_metrics
    try:
        metrics = json.loads(msg.payload.decode())
        # metrics should be a list of dicts: [{"pod": "upf-1", "cpu_usage": 0.12, "timestamp": "2025-09-07 09:04:19"}, ...]

        for m in metrics:
            pod = m["pod"]
            ts_str = m["timestamp"]
            ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")

            # ensure predictor exists
            if pod not in predictors:
                predictors[pod] = TimesfmPredictor(pod_name=pod, context_len=CONTEXT_LEN, pred_len=PRED_LEN)
                # predictors[pod].finetune()
            m["cpu_usage"] = float(m["cpu_usage"]) * 100.0
            m["pod_num"] = len(predictors)  # add pod_num for potential feature use
            last_data = previous_metrics.get(pod, None)
            if last_data:
                last_ts = datetime.strptime(last_data["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
                gap = (ts - last_ts).total_seconds()

                # fill missing points with LOCF
                if gap > time_interval:
                    steps = int(gap // time_interval)
                    for i in range(1, steps):
                        carried = last_data.copy()
                        carried_ts = last_ts + timedelta(seconds=i * time_interval)
                        carried["timestamp"] = carried_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                        predictors[pod].add_metrics(carried)
                        print(f"[LOCF] Inserted carried-forward metric for {pod} at {carried['timestamp']}")

            # add the current metric
            predictors[pod].add_metrics(m)
            predictor_heartbeat[pod] = 3  # reset heartbeat counter
            previous_metrics[pod] = m  # update last metric for this pod

        print("Received metrics:", metrics)

    except Exception as e:
        print(f"Error processing MQTT message: {e}")

        
def start_mqtt_subscriber():
    client = mqtt.Client()
    client.connect(BROKER_IP, 1883, 60)
    client.subscribe(MQTT_TOPIC)
    client.on_message = on_message
    thread = threading.Thread(target=client.loop_forever, daemon=True)
    thread.start()

def release_resources():
    for pod in predictors.values():
        pod.terminate()
        del pod

    time.sleep(10)   # wait for threads to release resources

def background_finetune(interval: int = 60):
    def loop():
        while True:
            try:
                for pod in predictors:
                    predictors[pod].finetune()
                    # Optionally: POST preds to another service here
            except Exception as e:
                print(f"Background finetune/predict error: {e}")
            time.sleep(interval)
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

@app.on_event("startup")
def startup_event():
    start_mqtt_subscriber()
    # background_finetune(interval=300)  # every 10 minutes

@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down, releasing resources...")
    release_resources()

@app.get("/metrics")
def predict():
    try:
        result = {}
        for pod, predictor in predictors.items():
            if predictor_heartbeat.get(pod, 0) == 0:
                # skip dead predictors
                p = predictors.pop(pod)
                p.terminate()
                del p
                predictor_heartbeat.pop(pod)
                print(f"Removed inactive predictor for pod {pod}")
                continue
            elif predictor_heartbeat.get(pod, 0) < 3:
                print(f"Warning: Predictor for pod {pod} has low heartbeat {predictor_heartbeat[pod]}")
                predictor_heartbeat[pod] -= 1 # decrement heartbeat counter
                continue
            if predictor.get_metrics_length() == 0:
                continue

            predictor_heartbeat[pod] -= 1 # decrement heartbeat counter
            preds = predictor.predict() / 100.0
            # for idx, val in enumerate(preds):
            predicted_cpu.labels(namespace="free5gc", pod=pod).set(preds)
            result[pod] = preds
            
        print("Predicted CPU usage:", result)
        return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/buffer")
def get_metrics():
    return Response(content=metrics_buffer)
