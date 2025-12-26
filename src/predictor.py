import os
import re
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from .featurizer import featurize_sample_v2

class PerBucketDNNEnsemble:
    """Loads per-bucket scalers + DNN ensemble and predicts m-heights.

    Expected files inside models_dir:
      - scaler_dnn_k{K}_m{M}.pkl  (dict: scaler, mu_y, std_y, seeds)
      - dnn_k{K}_m{M}_seed{s}.keras (one per seed)
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.bundles = {}  # (k,m) -> dict

        # Load scaler/normalization metadata
        for fname in os.listdir(models_dir):
            if not (fname.startswith("scaler_dnn_k") and fname.endswith(".pkl")):
                continue
            m = re.match(r"scaler_dnn_k(\d+)_m(\d+)\.pkl", fname)
            if not m:
                continue
            k, mv = int(m.group(1)), int(m.group(2))
            with open(os.path.join(models_dir, fname), "rb") as f:
                info = pickle.load(f)

            self.bundles[(k, mv)] = {
                "scaler": info["scaler"],
                "mu_y": float(info["mu_y"]),
                "std_y": float(info["std_y"]),
                "seeds": list(info.get("seeds", [])),
                "models": [],
            }

        # Load models for each bucket
        for (k, mv), bundle in self.bundles.items():
            for s in bundle["seeds"]:
                mname = f"dnn_k{k}_m{mv}_seed{s}.keras"
                mpath = os.path.join(models_dir, mname)
                if os.path.exists(mpath):
                    bundle["models"].append(tf.keras.models.load_model(mpath, compile=False))

    def predict_one(self, n: int, k: int, m: int, P: np.ndarray) -> float:
        bundle = self.bundles.get((k, m))
        if not bundle or not bundle["models"]:
            return float("nan")

        x = featurize_sample_v2(n, k, m, P).reshape(1, -1)
        x_s = bundle["scaler"].transform(x)

        z = [float(model.predict(x_s, verbose=0)[0, 0]) for model in bundle["models"]]
        z_mean = float(np.mean(z))

        y_log2 = bundle["mu_y"] + bundle["std_y"] * z_mean

        # Gentle safety cap (same cap used in your notebook version)
        y_log2 = float(np.clip(y_log2, 0.0, 17.0))  # ~[1, 131072]
        return max(1.0, float(2.0 ** y_log2))

    def predict(self, X_list):
        return np.array([self.predict_one(n,k,m,P) for (n,k,m,P) in X_list], dtype=float)
