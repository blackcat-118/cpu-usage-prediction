import os
import traceback
from collections import defaultdict
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import wandb

wandb_project = "timesfm-cpu-prediction"  # reuse same project
RESULT_FILELIST = ["cpu_usage.png", "evaluation.png", "metrics.csv"]
METRICS_SCRAPE_INTERVAL = 10  # seconds (align with TimesfmPredictor)

class LSTMPredictor:
    """LSTM based predictor with API compatible to TimesfmPredictor (minus finetune)."""

    def __init__(self, pod_name: str, context_len: int = 64, pred_len: int = 8, pretrained_path: str = "pretrained_data_3.csv", use_features: bool = True):
        self.pod_name = pod_name
        self.context_len = 32
        self.pred_len = 6
        self.pretrained_path = pretrained_path
        self.wb_runner = None
        self.cpu_prediction = [] 
        self.cpu_ground_truth = []
        self.scaler = StandardScaler()
        self.log_list = ["src/api.py", "src/lstm_predictor.py"]
        self.log_folder = f"../results/{self.pod_name}/"
        os.makedirs(self.log_folder, exist_ok=True)

        # buffers
        self.cpu_buffer = []  # [(timestamp, value)]
        self.covariate_keys = ["n3", "n4", "n6", "pod_num", "session_count"]
        if not use_features:
            self.covariate_keys = []
        self.covariates_buffer = defaultdict(list)
        self.pred_history = []  # list of dicts: {start_ts, preds, gt, mae}

        # build / train model
        self.model = self._build_model()
        self._wandb_init()
        # self.train_from_pretrained()

    # ------------------------------------------------------------------
    # Setup / training
    # ------------------------------------------------------------------
    def _wandb_init(self):
        self.wb_runner = wandb.init(project=wandb_project, group="exp1_lstm", name=f"lstm-{self.pod_name}", reinit="create_new")
        self.wb_runner.define_metric("mae", summary="last")
        self.wb_runner.define_metric("mse", summary="last")
        self.wb_runner.define_metric("rmse", summary="last")
        self.wb_runner.define_metric("cpu_usage", step_metric="timestamp", summary="mean")
        for k in self.covariate_keys:
            self.wb_runner.define_metric(k, step_metric="timestamp", summary="mean")

    def terminate(self):
        print(f"Releasing resources for pod {self.pod_name}")
        if self.wb_runner is not None:
            for filename in RESULT_FILELIST:
                self.log_list.append(f"results/{self.pod_name}/{filename}")
            self.wb_runner.log_code(root="../", include_fn=lambda path: os.path.relpath(path, start="/home/blackcat/cpu-usage-prediction/") in self.log_list)
            self.wb_runner.finish()

    def _build_model(self):
        """Build and compile LSTM model with multi-feature input.

        Input shape: (context_len, feature_dim), feature_dim = 1 (cpu) + len(covariates).
        Output: pred_len future cpu values (Dense(pred_len)).
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.context_len, len(self.covariate_keys) + 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(self.pred_len)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_from_pretrained(self):
        """Load pretrained CSV and train model once (no future finetuning)."""
        if not os.path.isfile(self.pretrained_path):
            print(f"Pretrained dataset not found at {self.pretrained_path}, skipping training.")
            return
        df = pd.read_csv(self.pretrained_path)
        # ensure required columns exist
        expected_cols = {"timestamp", "cpu_usage", "n3", "n4", "n6", "pod_num", "session_count"}
        missing = expected_cols - set(df.columns)
        if missing:
            print(f"Missing columns in pretrained dataset: {missing}. Training aborted.")
            return

        # normalize numeric covariates except cpu_usage (leave raw scale?)
        covs = [c for c in self.covariate_keys if c in df.columns]
        for c in covs:
            df[c] = df[c].astype(float)
        df['cpu_usage'] = df['cpu_usage'].astype(float)

        # build supervised learning windows
        sequences = []
        targets = []
        feature_cols = ['cpu_usage'] + covs
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        values = df[feature_cols].values
        for i in range(len(values) - self.context_len - self.pred_len):
            past = values[i:i + self.context_len]
            future_cpu = values[i + self.context_len:i + self.context_len + self.pred_len, 0]  # cpu column
            sequences.append(past)
            targets.append(future_cpu)
        if not sequences:
            print("Not enough data to build training windows.")
            return
        X = np.stack(sequences)
        y = np.stack(targets)
        # train/val split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        print(f"Training LSTM on {X_train.shape[0]} sequences; validating on {X_val.shape[0]} sequences.")
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=2, callbacks=[es])
        print(f"Final training loss: {history.history['loss'][-1]:.4f}, val_loss: {history.history['val_loss'][-1]:.4f}")
        if self.wb_runner is not None:
            self.wb_runner.log({"final_train_loss": history.history['loss'][-1], "final_val_loss": history.history['val_loss'][-1]})

    # ------------------------------------------------------------------
    # Runtime metrics ingestion
    # ------------------------------------------------------------------
    def get_metrics_length(self):
        return len(self.cpu_buffer)

    def add_metrics(self, metrics: dict):
        """Append new metrics with structure {timestamp, cpu_usage, n3, n4, n6, pod_num, session_count?}."""
        ts = pd.to_datetime(metrics["timestamp"])
        self.cpu_buffer.append((ts, metrics["cpu_usage"]))
        if self.wb_runner is not None:
            self.wb_runner.log({"cpu_usage": metrics["cpu_usage"], "timestamp": ts.timestamp()})
        for k in self.covariate_keys:
            if k in metrics:
                self.covariates_buffer[k].append(metrics[k])
                if self.wb_runner is not None:
                    self.wb_runner.log({k: metrics[k], "timestamp": ts.timestamp()})

        # persist to CSV
        row = {
            "timestamp": ts,
            "cpu_usage": metrics.get("cpu_usage"),
            "n3": metrics.get("n3"),
            "n4": metrics.get("n4"),
            "n6": metrics.get("n6"),
            "pod_num": metrics.get("pod_num"),
            "session_count": metrics.get("session_count"),
        }
        csv_path = f"{self.log_folder}/metrics.csv"
        file_exists = os.path.isfile(csv_path)
        pd.DataFrame([row]).to_csv(csv_path, mode='a', header=not file_exists, index=False)

    # ------------------------------------------------------------------
    # Prediction / evaluation
    # ------------------------------------------------------------------
    def predict(self, plot: bool = True):
        """Predict next pred_len cpu usage values; returns a scalar (max of horizon)."""
        if len(self.cpu_buffer) < 10:
            raise ValueError("Insufficient data for prediction")
        
        ctn_len = min(self.context_len, len(self.cpu_buffer))
        if ctn_len == 0:
            raise ValueError("No data for prediction")

        # --- Step 1. 準備 feature 矩陣 ---
        cpu_values = [v for _, v in self.cpu_buffer[-ctn_len:]]
        features = [cpu_values]
        for k in self.covariate_keys:
            buf = self.covariates_buffer.get(k, [])
            features.append(buf[-ctn_len:] if len(buf) >= ctn_len else [0.0] * ctn_len)
        features = np.array(features).T  # shape: (ctn_len, feature_dim)

        # --- Step 2. 正規化（用訓練時的 scaler）---
        # 使用 DataFrame 以避免 feature name 警告
        cols = ['cpu_usage'] + self.covariate_keys
        features_df = pd.DataFrame(features, columns=cols)
        features_scaled = self.scaler.transform(features_df)

        # --- Step 3. 確保輸入長度 = context_len ---
        feat_mat = features_scaled
        if ctn_len < self.context_len:
            pad_rows = np.zeros((self.context_len - ctn_len, feat_mat.shape[1]))
            feat_mat = np.vstack([pad_rows, feat_mat])  # recent data at bottom

        input_batch = feat_mat.reshape(1, self.context_len, feat_mat.shape[1])

        # --- Step 4. 模型預測 ---
        try:
            preds = self.model.predict(input_batch, verbose=0)[0]  # shape: (pred_len,)
            print("DEBUG >>> model raw output:", preds)

            # --- Step 5. 反標準化 ---
            raw_preds = np.array(preds).reshape(-1, 1)
            zeros = np.zeros((len(preds), len(self.covariate_keys)))
            tmp = np.hstack([raw_preds, zeros])
            tmp_df = pd.DataFrame(tmp, columns=cols)

            inv = self.scaler.inverse_transform(tmp_df)[:, 0]
            preds = inv

            print("DEBUG >>> inverse transformed output:", preds)

        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Error during LSTM prediction: {e}")

        # --- Step 6. Clamp 非負數 ---
        preds = [max(0, p) for p in preds]

        # --- Step 7. 若有前一次預測，則做 evaluation ---
        if len(self.pred_history) > 0:
            for i in range(len(self.cpu_buffer)-1, 0, -1):
                # print(f"Checking cpu_buffer index {i} with timestamp {self.cpu_buffer[i][0]} against last pred start_ts {self.pred_history[-1]['start_ts']}")
                
                if abs(self.pred_history[-1]["start_ts"].timestamp() - self.cpu_buffer[i][0].timestamp()) < 5:
                    # get ground truth values with the same length as pred_len (previous predictions)
                    gt_ts = self.cpu_buffer[i][0]
                    gt_values = self.cpu_buffer[i][1]
                    self.cpu_ground_truth.append(gt_values)
                    self.cpu_prediction.append(self.pred_history[-1]["preds"][0])  # only keep the next step prediction in the list
                    break
                
            print(f"Ground truth timestamps: {gt_ts}, values: {gt_values}")
            # print(self.cpu_ground_truth, self.cpu_prediction)
            
            last_pred = self.pred_history[-1]
            if last_pred["start_ts"] != gt_ts:
                print("Warning: Ground truth timestamps do not align with last prediction start time.")
                print(f"Last prediction start_ts: {last_pred['start_ts']}, GT timestamps: {gt_ts}")

            # Evaluation
            mae_score = self.evaluate(
                self.cpu_ground_truth[:],
                self.cpu_prediction[:], method="mae")
            mse_score = self.evaluate(
                self.cpu_ground_truth[:],
                self.cpu_prediction[:] , method="mse")
            rmse_score = self.evaluate(
                self.cpu_ground_truth[:],
                self.cpu_prediction[:], method="rmse")
            self.wb_runner.log({"mae": mae_score})
            self.wb_runner.log({"mse": mse_score})
            self.wb_runner.log({"rmse": rmse_score})
            print(f"Evaluation (MAE): {mae_score}")
            print(f"Evaluation (MSE): {mse_score}")
            print(f"Evaluation (RMSE): {rmse_score}")

            self.pred_history[-1]["mae"] = mae_score

        # --- Step 8. 儲存預測結果 ---
        prediction_scalar = max(preds)
        start_ts = self.cpu_buffer[-1][0]  # 最後一個已知點的時間
        self.pred_history.append({
            "start_ts": start_ts + timedelta(seconds=METRICS_SCRAPE_INTERVAL),
            "preds": preds,
            "mae": None,
        })

        # --- Step 9. 繪圖（可選）---
        if plot:
            self.plot()
            self.plot_evaluation()

        return prediction_scalar

    def evaluate(self, ground_truth: list, predictions: list, method: str = "mae"):
        n = min(len(ground_truth), len(predictions))
        if n == 0:
            return None
        gt = np.array(ground_truth[:n])
        pr = np.array(predictions[:n])
        if method == "mae":
            return float(np.mean(np.abs(gt - pr)))
        if method == "rmse":
            return float(np.sqrt(np.mean((gt - pr) ** 2)))
        if method == "mse":
            return float(np.mean((gt - pr) ** 2))
        raise ValueError(f"Unknown evaluation method: {method}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self):
        plt.figure(figsize=(12, 6))
        ts_axis = [ts for ts, _ in self.cpu_buffer]
        values = [v for _, v in self.cpu_buffer]
        plt.plot(ts_axis, values, label="Ground Truth")
        preds_dict = defaultdict(float)
        for ph in self.pred_history:
            if "start_ts" in ph and len(ph["preds"]) > 0:
                for i, p in enumerate(ph["preds"]):
                    preds_dict[ph["start_ts"] + timedelta(seconds=METRICS_SCRAPE_INTERVAL * i)] = p
        if preds_dict:
            sorted_preds = dict(sorted(preds_dict.items()))
            plt.plot(sorted_preds.keys(), sorted_preds.values(), "--", label="Prediction")
        plt.legend(); plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        plt.gcf().autofmt_xdate(rotation=45)
        plt.title("CPU Usage Forecast (LSTM)")
        plt.xlabel("Time"); plt.ylabel("CPU Usage(%)")
        plt.tight_layout()
        plt.savefig(f"{self.log_folder}/cpu_usage.png"); plt.close()

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
        
        plt.ylim(0, 100)

        # x 軸時間格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        plt.gcf().autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{self.log_folder}/evaluation.png")
        plt.close()

