import os
import numpy as np
import pandas as pd
import timesfm
from timesfm import patched_decoder, data_loader

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'

class CpuUsagePredictor:
    def __init__(self, context_len=256, pred_len=10, checkpoint_dir=None):
        self.context_len = context_len
        self.pred_len = pred_len
        self.checkpoint_dir = checkpoint_dir
        self.tfm = self._load_pretrained_model()
        self.data_buffer = []

    def set_buffer(self, buffer):
        self.data_buffer = list(buffer)

    def _load_pretrained_model(self):
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=128,
                num_layers=50,
                use_positional_embedding=False,
                context_len=512,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-jax"
            ),
        )
        return tfm

    def add_metrics(self, metrics: list):
        """Append new metrics to the buffer."""
        self.data_buffer.extend(metrics)
        # Keep only the latest context_len + pred_len points
        self.data_buffer = self.data_buffer[-(self.context_len + self.pred_len):]

    def finetune(self):
        """Finetune the model using buffered metrics."""
        # This is a placeholder for actual fine-tuning logic.
        # You can implement the full JAX/Praxis training loop here as in your notebook.
        pass

    def predict(self):
        """Predict the next pred_len cpu usage values."""
        if len(self.data_buffer) < self.context_len:
            raise ValueError("Not enough data for prediction")
        context = np.array(self.data_buffer[-self.context_len:]).reshape(1, -1)
        # Dummy covariates for demo; replace with real if available
        covariates = [0] * context.shape[0]
        preds, _ = self.tfm.forecast(list(context), covariates)
        return preds[0, :self.pred_len].tolist()

    def evaluate(self, ground_truth: list, predictions: list):
        """Evaluate predictions (e.g., MAE)."""
        if len(ground_truth) != len(predictions):
            return None
        return float(np.mean(np.abs(np.array(ground_truth) - np.array(predictions))))