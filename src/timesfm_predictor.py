import os
from collections import defaultdict
import numpy as np
import pandas as pd
import timesfm
from timesfm import patched_decoder, data_loader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import time

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'

class TimesfmPredictor:
    def __init__(self, context_len=64, pred_len=8, checkpoint_dir=None):
        self.context_len = context_len
        self.pred_len = pred_len
        self.checkpoint_dir = checkpoint_dir
        self.tfm = self._load_pretrained_model()
        # collected metrics buffer
        self.cpu_buffer = []   # [(timestamp, value)]
        self.covariates_buffer = defaultdict(list)
        self.past_predicted_cpu = []
        self.covariate_keys = ["n3", "n4", "n6"]
        self._set_covariates()
        # 用來保存每次預測的結果
        self.pred_history = []   # list of dicts: {"preds": [...], "gt": [...]}

    def _set_covariates(self):
        padding = list(np.zeros(shape=(self.pred_len,), dtype=float))
        self.covariates_buffer = {k: padding for k in self.covariate_keys}
        print(f"Initialized covariates buffer with keys: {self.covariates_buffer}")
        
    def _load_pretrained_model(self):
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=128,
                num_layers=10,
                use_positional_embedding=False,
                context_len=512,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-jax"
            ),
        )
        return tfm

    def add_metrics(self, metrics: dict):
        """
        Append new metrics to the buffer.
        metrics 必須包含 {"timestamp": ..., "cpu_usage": ...}
        """
        # print(f"Adding metrics: {metrics}")
        ts = pd.to_datetime(metrics["timestamp"])
        self.cpu_buffer.append((ts, metrics["cpu_usage"]))
        
        for k, v in metrics.items():
            if k in self.covariate_keys:
                self.covariates_buffer[k].append((ts, v))
                
        # Keep only the latest context_len points
        # We use data buffer to store the data for prediction or finetuning
        # self.data_buffer = self.data_buffer[-(self.context_len + self.pred_len):]

    def finetune(self):
        """Finetune the model using buffered metrics."""
        # This is a placeholder for actual fine-tuning logic.
        # You can implement the full JAX/Praxis training loop here as in your notebook.
        
        pass

    def predict(self, plot=True):
        """Predict the next pred_len cpu usage values."""
        if len(self.cpu_buffer) < self.context_len:
            raise ValueError("Not enough data for prediction")
        
        cpu_values = [v for _, v in self.cpu_buffer[-self.context_len:]]
        cpu = np.array(cpu_values).reshape(1, -1)
        # covariates = {k: [v[-(self.context_len+self.pred_len):]] for k, v in self.covariates_buffer.items()}
        # try:
        #     preds, _ = self.tfm.forecast_with_covariates(cpu, dynamic_numerical_covariates=covariates, normalize_xreg_target_per_input=False)
        # except Exception as e:
        #     print(f"Error forecasting: {e}")
        #     return []
        preds, _ = self.tfm.forecast_with_covariates(
            cpu,
            dynamic_numerical_covariates={
                k: [[v for _, v in self.covariates_buffer[k][- (self.context_len + self.pred_len):]]]
                for k in self.covariate_keys
            },
            normalize_xreg_target_per_input=False
        )
        preds = preds[0].tolist()
        print(f"Predictions: {preds}")
        
        # ground truth: 預測 horizon 的未來段落（如果有的話）
        if len(self.pred_history) > 0:
            gt_ts = [ts for ts, _ in self.cpu_buffer[-self.pred_len:]]
            gt_values = [v for _, v in self.cpu_buffer[-self.pred_len:]]
            print(f"Ground truth timestamps: {gt_ts}, values: {gt_values}")
            last_pred = self.pred_history[-1]
            if last_pred["start_ts"] != gt_ts[0]:
                print("Warning: Ground truth timestamps do not align with last prediction start time.")
                print(f"Last prediction start_ts: {last_pred['start_ts']}, GT timestamps: {gt_ts}")
            eval_score = self.evaluate(gt_values, last_pred["preds"])
            print(f"Evaluation (MAE): {eval_score}")
        else:
            gt_values = []
            eval_score = None
            
        # 存預測，帶上 timestamp 範圍
        start_ts = self.cpu_buffer[-1][0]  # 最後一個已知點的時間
        self.pred_history.append({
            "start_ts": start_ts + timedelta(seconds=10),
            "preds": preds,
            "gt": gt_values,
            "mae": eval_score
        })
        
        if plot:
            self.plot()
            self.plot_evaluation()
        
        return max(preds)

    def evaluate(self, ground_truth: list, predictions: list, plot: bool = True):
        """Evaluate predictions (e.g., MAE)."""
        n = min(len(ground_truth), len(predictions))
        if n == 0:
            return None
        return float(np.mean(np.abs(np.array(ground_truth[:n]) - np.array(predictions[:n]))))
    
    def plot(self):
        """Plot ground truth and predictions with timestamp x-axis."""
        plt.figure(figsize=(12, 6))
        ts_axis = [ts for ts, _ in self.cpu_buffer]
        values = [v for _, v in self.cpu_buffer]
        plt.plot(ts_axis, values, label="Ground Truth")

        # 繪製每段預測
        pred_ts = []
        pred = []
        for ph in self.pred_history:
            if "start_ts" in ph and len(ph["preds"]) > 0:
                pred_ts.extend([ph["start_ts"] + timedelta(seconds=10) * (i) for i in range(len(ph["preds"]))])
                pred.extend(ph["preds"])
                print(pred_ts)
        plt.plot(pred_ts, pred, "--", label="Prediction")
        plt.legend()
        plt.grid(True)

        # 設定 x 軸時間格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))  # 最多顯示 10 個 tick
        plt.gcf().autofmt_xdate(rotation=45)

        plt.title("CPU Usage Forecast")
        plt.xlabel("Time")
        plt.ylabel("CPU Usage")
        plt.tight_layout()
        plt.savefig("../prediction/cpu_usage.png")
        plt.close()
        
    def plot_evaluation(self):
        """Plot evaluation metric (MAE) over prediction history."""
        maes = [ph["mae"] for ph in self.pred_history if ph["mae"] is not None]
        ts_list = [ph["start_ts"] for ph in self.pred_history if ph["mae"] is not None]

        if not maes:
            print("No evaluation results to plot yet.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(ts_list, maes, marker="o", label="MAE")
        plt.grid(True)
        plt.title("Prediction Evaluation (MAE)")
        plt.xlabel("Time")
        plt.ylabel("MAE")
        plt.legend()

        # x 軸時間格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        plt.gcf().autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig("../prediction/evaluation.png")
        plt.close()