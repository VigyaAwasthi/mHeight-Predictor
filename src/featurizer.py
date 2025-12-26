import numpy as np

P_SCALE = 100.0

def _safe_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    m = np.mean(x); s = np.std(x) + 1e-12
    z = (x - m) / s
    return float(np.mean(z**3))

def _safe_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    m = np.mean(x); s = np.std(x) + 1e-12
    z = (x - m) / s
    return float(np.mean(z**4) - 3.0)

def featurize_sample_v2(n: int, k: int, m: int, P: np.ndarray) -> np.ndarray:
    """Feature vector used in Projects 2/3.

    Args:
        n, k, m: bucket identifiers
        P: generator matrix / input matrix for the sample

    Returns:
        1D float32 numpy array.
    """
    Pn = np.asarray(P, dtype=np.float32) / P_SCALE
    feats = [float(n), float(k), float(m)]

    # Flattened P (padded/truncated to 20)
    pf = Pn.ravel().tolist()
    pf += [0.0] * (20 - len(pf))
    feats.extend(pf[:20])

    # One-hot (k in {4,5,6}, m in {2,3,4,5})
    feats.extend([1.0 if k == t else 0.0 for t in (4, 5, 6)])
    feats.extend([1.0 if m == t else 0.0 for t in (2, 3, 4, 5)])

    # Global stats
    p_abs = np.abs(Pn)
    feats.extend([
        float(np.mean(Pn)), float(np.std(Pn)),
        float(np.min(Pn)),  float(np.max(Pn)),
        float(np.linalg.norm(Pn, 'fro')),
        float(np.sum(p_abs)), float(np.max(p_abs)),
        _safe_skew(Pn), _safe_kurtosis(Pn),
    ])

    # Percentiles
    for q in (10, 25, 50, 75, 90):
        feats.append(float(np.percentile(Pn, q)))

    # Top-5 singular values
    svals = np.sort(np.linalg.svd(Pn, compute_uv=False))[::-1]
    if len(svals) < 5:
        svals = np.pad(svals, (0, 5 - len(svals)), constant_values=0.0)
    feats.extend(svals[:5].tolist())

    # Row/col norm summaries
    feats.extend([
        float(np.mean(np.linalg.norm(Pn, axis=1))),
        float(np.mean(np.linalg.norm(Pn, axis=0))),
    ])

    # Row/col mean+std summaries
    feats.extend([
        float(np.mean(np.mean(Pn, axis=1))),
        float(np.std(np.mean(Pn, axis=1))),
        float(np.mean(np.std(Pn, axis=1))),
        float(np.std(np.std(Pn, axis=1))),
        float(np.mean(np.mean(Pn, axis=0))),
        float(np.std(np.mean(Pn, axis=0))),
        float(np.mean(np.std(Pn, axis=0))),
        float(np.std(np.std(Pn, axis=0))),
    ])

    rn_mean = float(np.mean(np.linalg.norm(Pn, axis=1)))
    feats.extend([
        float(np.mean(Pn) * np.std(Pn)),
        float(np.max(Pn) * rn_mean),
    ])

    return np.asarray(feats, dtype=np.float32)
