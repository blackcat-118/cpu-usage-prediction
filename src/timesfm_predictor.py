import os
from collections import defaultdict
import numpy as np
import pandas as pd
import timesfm
from timesfm import patched_decoder, data_loader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import time
import gc
from tqdm import tqdm
import jax
from jax import numpy as jnp
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import optimizers
from praxis import schedules
from praxis import base_hyperparams
from praxis import base_layer
from paxml import tasks_lib
from paxml import trainer_lib
from paxml import checkpoints
from paxml import learners
from paxml import checkpoint_types
from sklearn.linear_model import LinearRegression
import traceback
import wandb

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'
wandb_project = "timesfm-cpu-prediction"
use_pod_num=True
RESULT_FILELIST = ["cpu_usage.png", "evaluation.png", "metrics.csv"]
METRICS_SCRAPE_INTERVAL = 20  # seconds

def compute_trend(cpu_values):
    x = np.arange(len(cpu_values)).reshape(-1, 1)
    y = np.array(cpu_values)
    model = LinearRegression().fit(x, y)
    return model.coef_[0]  # slope

class TimesfmPredictor:
    def __init__(self, pod_name, context_len=64, pred_len=8):
        self.pod_name = pod_name
        self.context_len = context_len
        self.pred_len = pred_len
        self.tfm = self._load_pretrained_model(context_len=context_len, horizon_len=pred_len)
        self.wb_runner = None
        self.log_list = ["src/api.py", "src/timesfm_predictor.py"]
        self.log_folder = f"../results/{self.pod_name}/"
        # collected metrics buffer
        self.cpu_buffer = []   # [(timestamp, value)]
        self.covariates_buffer = defaultdict(list)
        self.past_predicted_cpu = []
        self.covariate_keys = ["n3", "n4", "n6", "pod_num", "session_count"]
        # 用來保存每次預測的結果
        self.pred_history = []   # list of dicts: {"preds": [...], "gt": [...]}
        self._set_covariates()
        self._wandb_init()
    
    def terminate(self):
        print(f"Releasing resources for pod {self.pod_name}")
        if self.wb_runner is not None:
            for filename in RESULT_FILELIST:
                self.log_list.append(f"results/{self.pod_name}/{filename}")
            self.wb_runner.log_code(root="../", include_fn=lambda path: os.path.relpath(path, start="/home/blackcat/cpu-usage-prediction/") in self.log_list)
            self.wb_runner.finish()

    def _wandb_init(self):
        self.wb_runner = wandb.init(project=wandb_project, name=self.pod_name, reinit="create_new")
        self.wb_runner.define_metric("mae", summary="mean")
        self.wb_runner.define_metric("mse", summary="mean")
        self.wb_runner.define_metric("rmse", summary="mean")
        self.wb_runner.define_metric("cpu_usage", step_metric="timestamp", summary="mean")
        self.wb_runner.define_metric("predicted_cpu_usage", step_metric="timestamp", summary="max")
        self.wb_runner.define_metric("n3", step_metric="timestamp", summary="mean")
        self.wb_runner.define_metric("n4", step_metric="timestamp", summary="mean")
        self.wb_runner.define_metric("n6", step_metric="timestamp", summary="mean")
        self.wb_runner.define_metric("pod_num", step_metric="timestamp")
        self.wb_runner.define_metric("session_count", step_metric="timestamp", summary="mean")

    def _set_covariates(self):
        padding = list(np.zeros(shape=(self.pred_len,), dtype=float))
        self.covariates_buffer = {k: padding for k in self.covariate_keys}
        print(f"Initialized covariates buffer with keys: {self.covariates_buffer}")
        
    def _load_pretrained_model(self, horizon_len=128, context_len=512):
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=1,
                horizon_len=16,
                num_layers=5,
                use_positional_embedding=False,
                context_len=64,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-jax",
                # local_dir="./checkpoints/pretrained_checkpoint/checkpoint_1301"
            ),
        )
        return tfm

    def get_metrics_length(self):
        return len(self.cpu_buffer)

    def add_metrics(self, metrics: dict):
        """
        Append new metrics to the buffer.
        metrics 必須包含 {"timestamp": ..., "cpu_usage": ...}
        """
        # print(f"Adding metrics: {metrics}")
        ts = pd.to_datetime(metrics["timestamp"])
        self.cpu_buffer.append((ts, metrics["cpu_usage"]))
        self.wb_runner.log({"cpu_usage": metrics["cpu_usage"], "timestamp": ts.timestamp()})

        for k, v in metrics.items():
            if k in self.covariate_keys:
                self.covariates_buffer[k].append(v) # not need timestamp for covariates
                self.wb_runner.log({k: v, "timestamp": ts.timestamp()})
                
        # --- Write metrics to CSV ---
        row = {
            "timestamp": ts,
            "cpu_usage": metrics.get("cpu_usage", None),
            "n3": metrics.get("n3", None),
            "n4": metrics.get("n4", None),
            "n6": metrics.get("n6", None),
            "pod_num": metrics.get("pod_num", None),
            "session_count": metrics.get("session_count", None),
        }

        csv_path = f"{self.log_folder}/metrics.csv"
        file_exists = os.path.isfile(csv_path)

        df = pd.DataFrame([row])
        df.to_csv(
            csv_path,
            mode="a",
            header=not file_exists,  # write header only once
            index=False
        )

    def finetune(self):
        
        start_time = time.time()
        print(f"Starting finetuning for {self.pod_name}. ")
        
        data_path = f"{self.log_folder}/metrics.csv"
        data_df = pd.read_csv(open(data_path, "r"))
        data_df.drop(columns=["pod", "time"], inplace=True, errors="ignore")

        boundaries = [int(len(data_df)*0.7), len(data_df)-2, len(data_df)-1]
        print(f"Data boundaries (train/val/test): {boundaries}")
        
        freq = "10s"
        int_freq = timesfm.freq_map(freq)
        ts_cols = ["cpu_usage"]
        date_col = "timestamp"
        num_cov_cols = ["n3", "n4", "n6", "pod_num", "session_count"]
        cat_cov_cols = None
        num_ts = len(ts_cols)
        
        best_eval_loss = 1e7
        step_count = 0
        patience = 0
        NUM_EPOCHS = 10
        PATIENCE = 10
        TRAIN_STEPS_PER_EVAL = 100
        CHECKPOINT_DIR=f"{self.log_folder}/checkpoints/"

        config = dict(context_len=self.context_len, 
                      pred_len=self.pred_len, 
                      batch_size=2, 
                      NUM_EPOCHS=NUM_EPOCHS,
                      PATIENCE=PATIENCE,
                      TRAIN_STEPS_PER_EVAL=TRAIN_STEPS_PER_EVAL,
                    )
        self.wb_runner.config.update(config)
        
        dtl = data_loader.TimeSeriesdata(
            data_path=data_path,
            datetime_col=date_col,
            num_cov_cols=num_cov_cols,
            cat_cov_cols=cat_cov_cols,
            ts_cols=np.array(ts_cols),
            train_range=[0, boundaries[0]],
            val_range=[boundaries[0], boundaries[1]],
            test_range=[boundaries[1], boundaries[2]],
            hist_len=config["context_len"],
            pred_len=config["pred_len"],
            batch_size=config["batch_size"],
            freq=freq,
            normalize=False,
            epoch_len=None,
            holiday=False,
            permute=False,
        )
        train_batches = dtl.tf_dataset(mode="train", shift=1).batch(config["batch_size"])
        val_batches = dtl.tf_dataset(mode="val", shift=config["pred_len"])
        # PAX shortcuts
        NestedMap = py_utils.NestedMap
        WeightInit = base_layer.WeightInit
        WeightHParams = base_layer.WeightHParams
        InstantiableParams = py_utils.InstantiableParams
        JTensor = pytypes.JTensor
        NpTensor = pytypes.NpTensor
        WeightedScalars = pytypes.WeightedScalars
        instantiate = base_hyperparams.instantiate
        LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
        AuxLossStruct = base_layer.AuxLossStruct

        AUX_LOSS = base_layer.AUX_LOSS
        template_field = base_layer.template_field

        # Standard prng key names
        PARAMS = base_layer.PARAMS
        RANDOM = base_layer.RANDOM

        key = jax.random.PRNGKey(seed=1234)

        model = pax_fiddle.Config(
            patched_decoder.PatchedDecoderFinetuneModel,
            name='patched_decoder_finetune',
            core_layer_tpl=self.tfm.model_p,
        )
        @pax_fiddle.auto_config
        def build_learner() -> learners.Learner:
            return pax_fiddle.Config(
                learners.Learner,
                name='learner',
                loss_name='avg_qloss',
                optimizer=optimizers.Adam(
                    epsilon=1e-7,
                    clip_threshold=1e2,
                    learning_rate=1e-2,
                    lr_schedule=pax_fiddle.Config(
                        schedules.Cosine,
                        initial_value=1e-3,
                        final_value=1e-4,
                        total_steps=4000,
                    ),
                    ema_decay=0.9999,
                ),
                # Linear probing i.e we hold the transformer layers fixed.
                bprop_variable_exclusion=['.*/stacked_transformer_layer/.*'],
            )
        task_p = tasks_lib.SingleTask(
        name='ts-learn',
        model=model,
        train=tasks_lib.SingleTask.Train(
            learner=build_learner(),
            ),
        )
        task_p.model.ici_mesh_shape = [1, 1, 1]
        task_p.model.mesh_axis_names = ['replica', 'data', 'mdl']

        DEVICES = np.array(jax.devices()).reshape([1, 1, 1])
        MESH = jax.sharding.Mesh(DEVICES, ['replica', 'data', 'mdl'])

        num_devices = jax.local_device_count()
        print(f'num_devices: {num_devices}')
        print(f'device kind: {jax.local_devices()[0].device_kind}')

        jax_task = task_p
        key, init_key = jax.random.split(key)

        # To correctly prepare a batch of data for model initialization (now that shape
        # inference is merged), we take one devices*batch_size tensor tuple of data,
        # slice out just one batch, then run the prepare_input_batch function over it.


        def process_train_batch(batch):
            past_ts = batch[0].reshape(config["batch_size"] * num_ts, -1)
            actual_ts = batch[3].reshape(config["batch_size"] * num_ts, -1)
            return NestedMap(input_ts=past_ts, actual_ts=actual_ts)


        def process_eval_batch(batch):
            past_ts = batch[0]
            actual_ts = batch[3]
            return NestedMap(input_ts=past_ts, actual_ts=actual_ts)

        for tbatch in tqdm(train_batches.as_numpy_iterator()):
            break
        print(tbatch[0].shape)
        jax_model_states, _ = trainer_lib.initialize_model_state(
            jax_task,
            init_key,
            process_train_batch(tbatch),
            checkpoint_type=checkpoint_types.CheckpointType.GDA,
        )
        jax_model_states.mdl_vars['params']['core_layer'] = self.tfm._train_state.mdl_vars['params']
        jax_vars = jax_model_states.mdl_vars
        gc.collect()
        jax_task = task_p


        def train_step(states, prng_key, inputs):
            return trainer_lib.train_step_single_learner(
                jax_task, states, prng_key, inputs
            )


        def eval_step(states, prng_key, inputs):
            states = states.to_eval_state()
            return trainer_lib.eval_step_single_learner(
                jax_task, states, prng_key, inputs
            )

        key, train_key, eval_key = jax.random.split(key, 3)
        train_prng_seed = jax.random.split(train_key, num=jax.local_device_count())
        eval_prng_seed = jax.random.split(eval_key, num=jax.local_device_count())

        p_train_step = jax.pmap(train_step, axis_name='batch')
        p_eval_step = jax.pmap(eval_step, axis_name='batch')

        replicated_jax_states = trainer_lib.replicate_model_state(jax_model_states)
        replicated_jax_vars = replicated_jax_states.mdl_vars


        def reshape_batch_for_pmap(batch, num_devices):
            def _reshape(input_tensor):
                bsize = input_tensor.shape[0]
                residual_shape = list(input_tensor.shape[1:])
                nbsize = bsize // num_devices
                return jnp.reshape(input_tensor, [num_devices, nbsize] + residual_shape)

            return jax.tree.map(_reshape, batch)
        for epoch in range(NUM_EPOCHS):
            print(f"__________________Epoch: {epoch}__________________", flush=True)
            train_its = train_batches.as_numpy_iterator()
            if patience >= PATIENCE:
                print("Early stopping.", flush=True)
                break
            for batch in tqdm(train_its):
                train_losses = []
                if patience >= PATIENCE:
                    print("Early stopping.", flush=True)
                    break
                tbatch = process_train_batch(batch)
                tbatch = reshape_batch_for_pmap(tbatch, num_devices)
                replicated_jax_states, step_fun_out = p_train_step(
                    replicated_jax_states, train_prng_seed, tbatch
                )
                train_losses.append(step_fun_out.loss[0])
                if step_count % TRAIN_STEPS_PER_EVAL == 0:
                    print(
                        f"Train loss at step {step_count}: {np.mean(train_losses)}",
                        flush=True,
                    )
                    train_losses = []
                    print("Starting eval.", flush=True)
                    val_its = val_batches.as_numpy_iterator()
                    eval_losses = []
                    for ev_batch in tqdm(val_its):
                        ebatch = process_eval_batch(ev_batch)
                        ebatch = reshape_batch_for_pmap(ebatch, num_devices)
                        _, step_fun_out = p_eval_step(
                            replicated_jax_states, eval_prng_seed, ebatch
                        )
                        eval_losses.append(step_fun_out.loss[0])
                    mean_loss = np.mean(eval_losses)
                    print(f"Eval loss at step {step_count}: {mean_loss}", flush=True)
                    if mean_loss < best_eval_loss or np.isnan(mean_loss):
                        best_eval_loss = mean_loss
                        print("Saving checkpoint.")
                        jax_state_for_saving = py_utils.maybe_unreplicate_for_fully_replicated(
                            replicated_jax_states
                        )
                        checkpoints.save_checkpoint(
                            jax_state_for_saving, CHECKPOINT_DIR, overwrite=False
                        )
                        patience = 0
                        del jax_state_for_saving
                        gc.collect()
                    else:
                        patience += 1
                        print(f"patience: {patience}")
                step_count += 1


        train_state = checkpoints.restore_checkpoint(jax_model_states, CHECKPOINT_DIR)
        print(train_state.step)
        self.tfm._train_state.mdl_vars['params'] = train_state.mdl_vars['params']['core_layer']
        self.tfm.jit_decode()
        print(f"Finetuning completed in {time.time() - start_time:.2f} seconds.")


    def predict(self, plot=True):
        """Predict the next pred_len cpu usage values."""
        cpu_values = []
        ctn_len = min(self.context_len, len(self.cpu_buffer))
        if ctn_len == 0:
            return ValueError("No data for prediction")

        cpu_values = [v for _, v in self.cpu_buffer[-ctn_len:]]
        cpu = np.array(cpu_values).reshape(1, -1)

        try:
            # preds, _ = self.tfm.forecast(cpu)
            preds, _ = self.tfm.forecast_with_covariates(
                cpu,
                dynamic_numerical_covariates={
                    k: [[v for v in self.covariates_buffer[k][- (ctn_len + self.pred_len):]]]
                    for k in self.covariate_keys
                },
                normalize_xreg_target_per_input=False
            )

        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Error during prediction: {e}")
            
        print(f"Predictions: {preds}")
        preds = preds[0].tolist()
        for i, p in enumerate(preds):
            preds[i] = max(0, p)
        
        
        # ground truth: 預測 horizon 的未來段落（如果有的話）
        if len(self.pred_history) > 0:
            gt_ts = []
            gt_values = []
            for i in range(len(self.cpu_buffer)-1, 0, -1):
                # print(f"Checking cpu_buffer index {i} with timestamp {self.cpu_buffer[i][0]} against last pred start_ts {self.pred_history[-1]['start_ts']}")
                
                if abs(self.pred_history[-1]["start_ts"].timestamp() - self.cpu_buffer[i][0].timestamp()) < 5:
                    # get ground truth values with the same length as pred_len (previous predictions)
                    gt_ts = [ts for ts, _ in self.cpu_buffer[i:i + self.pred_len]]
                    gt_values = [v for _, v in self.cpu_buffer[i:i + self.pred_len]]
                    break
                
            # print(f"Ground truth timestamps: {gt_ts}, values: {gt_values}")
            
            last_pred = self.pred_history[-1]
            if last_pred["start_ts"] != gt_ts[0]:
                print("Warning: Ground truth timestamps do not align with last prediction start time.")
                print(f"Last prediction start_ts: {last_pred['start_ts']}, GT timestamps: {gt_ts}")
            mae_score = self.evaluate(gt_values, last_pred["preds"], method="mae")
            mse_score = self.evaluate(gt_values, last_pred["preds"], method="mse")
            rmse_score = self.evaluate(gt_values, last_pred["preds"], method="rmse")
            self.pred_history[-1]["gt"] = gt_values
            self.pred_history[-1]["mae"] = mae_score
            self.wb_runner.log({"mae": mae_score})
            self.wb_runner.log({"mse": mse_score})
            self.wb_runner.log({"rmse": rmse_score})
            print(f"Evaluation (MAE): {mae_score}")
        else:
            gt_values = []
            eval_score = None
            
        # prediction = np.percentile(preds, 75)
        prediction = max(preds)

        # 存預測，帶上 timestamp 範圍
        start_ts = self.cpu_buffer[-1][0]  # 最後一個已知點的時間
        self.pred_history.append({
            "start_ts": start_ts + timedelta(seconds=METRICS_SCRAPE_INTERVAL),
            "preds": preds,
            "gt": [],
            "mae": None,
        })
        
        if plot:
            self.plot()
            self.plot_evaluation()
        
        return prediction

    def evaluate(self, ground_truth: list, predictions: list, method="mae"):
        """Evaluate predictions (e.g., MAE)."""
        n = min(len(ground_truth), len(predictions))
        if n == 0:
            return None
        if method == "mae":
            return float(np.mean(np.abs(np.array(ground_truth[:n]) - np.array(predictions[:n]))))
        elif method == "rmse":
            return np.sqrt(np.mean((np.array(ground_truth[:n]) - np.array(predictions[:n]))**2))
        elif method == "mse":
            return np.mean((np.array(ground_truth[:n]) - np.array(predictions[:n]))**2)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")
        return None

    def plot(self):
        """Plot ground truth and predictions with timestamp x-axis."""
        plt.figure(figsize=(12, 6))
        ts_axis = [ts for ts, _ in self.cpu_buffer]
        values = [v for _, v in self.cpu_buffer]
        plt.plot(ts_axis, values, label="Ground Truth")

        # Use a dictionary to collect all predictions for plotting
        # overwrite preds to avoid overlapping lines
        preds = defaultdict(float)
        for ph in self.pred_history:
            if "start_ts" in ph and len(ph["preds"]) > 0:
                for i, p in enumerate(ph["preds"]):
                    preds[ph["start_ts"] + timedelta(seconds=METRICS_SCRAPE_INTERVAL * i)] = p
                    
        sorted_preds = dict(sorted(preds.items()))
        # print(f"All predictions for plotting: {sorted_preds.keys()}, {sorted_preds.values()}")
        plt.plot(sorted_preds.keys(), sorted_preds.values(), "--", label="Prediction")
        plt.legend()
        plt.grid(True)

        # 設定 x 軸時間格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))  # 最多顯示 10 個 tick
        plt.gcf().autofmt_xdate(rotation=45)

        plt.title("CPU Usage Forecast")
        plt.xlabel("Time")
        plt.ylabel("CPU Usage(%)")
        plt.tight_layout()
        plt.savefig(f"{self.log_folder}/cpu_usage.png")
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
        
        plt.ylim(0, 30)

        # x 軸時間格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        plt.gcf().autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{self.log_folder}/evaluation.png")
        plt.close()
