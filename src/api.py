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

BROKER_IP = "broker.emqx.io"  # Replace with your broker IP
MQTT_TOPIC = "upf/metrics"
time_interval = 10 # seconds
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

# @app.post("/finetune")
# def finetune():
#     try:
#         for pod, predictor in predictors.items():
#             predictor.finetune()
#         return {"status": "finetuned"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def predict():
    try:
        result = {}
        for pod, predictor in predictors.items():
            if predictor_heartbeat.get(pod, 0) == 0:
                continue
            if predictor.get_metrics_length() < CONTEXT_LEN:
                continue

            predictor_heartbeat[pod] -= 1 # decrement heartbeat counter
            preds = predictor.predict()
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

# @app.post("/evaluate")
# def evaluate(payload: dict):
#     # payload: {"pod": "upf-1", "ground_truth": [...]}
#     pod = payload.get("pod")
#     ground_truth = payload.get("ground_truth")
#     if pod not in predictors:
#         raise HTTPException(status_code=404, detail="Pod not found")
#     preds = predictors[pod].predict()
#     score = predictors[pod].evaluate(ground_truth, preds)
#     return {"mae": score}