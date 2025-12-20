import os
os.environ.setdefault("WANDB_MODE", "offline")
import pandas as pd
from datetime import datetime
from tcn_predictor import TCNPredictor

# Load a small slice from pretrained data
CSV = "pretrained_data.csv"
df = pd.read_csv(CSV)

pod = "test-tcn"
pred = TCNPredictor(pod_name=pod, context_len=32, pred_len=8, pretrained_path=CSV)

# Feed ~80 points of metrics to the predictor
n = min(80, len(df))
for i in range(n):
    row = df.iloc[i].to_dict()
    # tolerate optional fields
    metrics = {
        "timestamp": row.get("timestamp"),
        "cpu_usage": float(row.get("cpu_usage", 0.0)),
        "n3": float(row.get("n3", 0.0)),
        "n4": float(row.get("n4", 0.0)),
        "n6": float(row.get("n6", 0.0)),
        "pod_num": float(row.get("pod_num", 0.0)),
        "session_count": float(row.get("session_count", 0.0)),
    }
    pred.add_metrics(metrics)

scalar = pred.predict(plot=True)
print("Predicted scalar (max horizon):", scalar)

pred.terminate()
