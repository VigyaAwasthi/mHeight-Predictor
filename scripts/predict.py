"""Run Project 3 predictor on an input pickle and save predictions for submission.

Example:
  python -m scripts.predict \
    --models_dir models/unzipped_P3 \
    --input_pkl /path/to/PROJECT3-SampleTest-n_k_m_P \
    --output_pkl results/Project3_Test_Results.pkl
"""

import argparse, os
import numpy as np
from src.predictor import PerBucketDNNEnsemble
from src.io_utils import load_pickle, save_pickle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True, help="Directory with per-bucket scaler+keras files")
    ap.add_argument("--input_pkl", required=True, help="Pickle file with list of (n,k,m,P)")
    ap.add_argument("--output_pkl", required=True, help="Where to save list[float] predictions")
    args = ap.parse_args()

    X = list(load_pickle(args.input_pkl))
    model = PerBucketDNNEnsemble(args.models_dir)
    y = model.predict(X)

    os.makedirs(os.path.dirname(args.output_pkl) or ".", exist_ok=True)
    save_pickle(list(map(float, y.tolist())), args.output_pkl)
    print(f"Wrote: {args.output_pkl}  (N={len(y)})")

if __name__ == "__main__":
    main()
