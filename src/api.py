from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Gauge, generate_latest
from predictor import CpuUsagePredictor
from metrics_client import fetch_metrics
import threading
import time

app = FastAPI()
REMOTE_METRICS_URL = "http://140.113.208.74:9090/cpu_metrics"
CONTEXT_LEN = 256
PRED_LEN = 10

# Store predictors per pod
predictors = {}
metrics_buffer = {}

predicted_cpu = Gauge("upf_predicted_cpu_utilization", "Predicted CPU usage", ["namespace", "pod"])

def background_metrics_collector():
    def collect():
        while True:
            metrics = fetch_metrics(REMOTE_METRICS_URL)
            if metrics:
                for m in metrics:
                    pod = m["pod"]
                    cpu = m["cpu_usage"]
                    if pod not in metrics_buffer:
                        metrics_buffer[pod] = []
                    metrics_buffer[pod].append(cpu)
                    # Keep only the latest CONTEXT_LEN + PRED_LEN points
                    metrics_buffer[pod] = metrics_buffer[pod][-(CONTEXT_LEN + PRED_LEN):]
            time.sleep(5)
    thread = threading.Thread(target=collect, daemon=True)
    thread.start()

def background_finetune_and_predict():
    def loop():
        while True:
            try:
                for pod, buffer in metrics_buffer.items():
                    if pod not in predictors:
                        predictors[pod] = CpuUsagePredictor(context_len=CONTEXT_LEN, pred_len=PRED_LEN)
                    predictors[pod].set_buffer(buffer)
                    predictors[pod].finetune()
                    preds = predictors[pod].predict()
                    for idx, val in enumerate(preds):
                        predicted_cpu.labels(namespace="free5gc", pod=pod).set(val)
                    # Optionally: POST preds to another service here
            except Exception as e:
                print(f"Background finetune/predict error: {e}")
            time.sleep(10)
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

@app.on_event("startup")
def startup_event():
    background_metrics_collector()
    background_finetune_and_predict()

@app.post("/finetune")
def finetune():
    try:
        for pod, predictor in predictors.items():
            predictor.finetune()
        return {"status": "finetuned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def predict():
    try:
        result = {}
        for pod, predictor in predictors.items():
            preds = predictor.predict()
            for idx, val in enumerate(preds):
                predicted_cpu.labels(namespace="free5gc", pod=pod).set(val)
            result[pod] = preds
        return JSONResponse(content={"cpu_usage": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/buffer")
def get_metrics():
    return JSONResponse(content=metrics_buffer)

@app.post("/evaluate")
def evaluate(payload: dict):
    # payload: {"pod": "upf-1", "ground_truth": [...]}
    pod = payload.get("pod")
    ground_truth = payload.get("ground_truth")
    if pod not in predictors:
        raise HTTPException(status_code=404, detail="Pod not found")
    preds = predictors[pod].predict()
    score = predictors[pod].evaluate(ground_truth, preds)
    return {"mae": score}