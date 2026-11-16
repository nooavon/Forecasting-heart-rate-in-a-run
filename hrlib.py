"""Shared helpers for the HR forecasting notebooks (same logic as the step-4 notebook)."""
import numpy as np, pandas as pd, statsmodels.api as sm

DATA = "run_2026-04-26.csv"
HORIZONS = [1, 5, 10, 20, 30, 60, 120]
BURN = 60
MOM_ALL = (1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45, 60)
EW_ALL = (3, 5, 10, 30, 60, 120)

def load(path=DATA):
    return pd.read_csv(path).set_index("t")

def build_features(frame):
    hr = frame.hr.astype(float); pw = frame.power.astype(float); sp = frame.speed.astype(float)
    F = pd.DataFrame(index=frame.index); F["hr"] = hr
    for k in MOM_ALL: F[f"d{k}"] = hr - hr.shift(k)
    for k in (0, 1, 2, 3, 5, 10, 20, 30, 60): F[f"p{k}"] = pw.shift(k)
    for w in EW_ALL: F[f"pew{w}"] = pw.ewm(span=w).mean()
    F["s0"] = sp; F["sew60"] = sp.ewm(span=60).mean()
    alt = frame.altitude.rolling(5, min_periods=5).mean(); dd = frame.distance - frame.distance.shift(60)
    F["g60"] = (100 * (alt - alt.shift(60)) / dd).where(dd >= 30).clip(-20, 20).ffill().fillna(0)
    F["time"] = frame.index.values / 1000
    F["m30"] = hr.rolling(30, min_periods=1).mean()
    return F

def targets(hr, horizons=HORIZONS):
    return {h: (hr.shift(-h) - hr).values for h in horizons}

def spec(mom, ew=None, level=False, speed=False):
    cols = (["hr"] if level else []) + [f"d{k}" for k in mom]
    if ew is not None: cols += ["p0"] + [f"pew{w}" for w in ew]
    if speed: cols += ["s0", "sew60"]
    return cols

def fit_ridge(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; ym = y.mean()
    b = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return mu, sd, b, ym

def predict_ridge(model, X):
    mu, sd, b, ym = model; return ((X - mu) / sd) @ b + ym

def rolling_preds(F, Y, cols, h, lo, hi, step=60, window=None, lam=1.0, train_from=None):
    """Predicted HR(t+h)-HR(t) for origins lo..hi-1; refit every `step` s on pairs (s, s+h) with s+h <= refit time."""
    X, y = F[cols].values, Y[h]; tf = BURN if train_from is None else max(train_from, BURN)
    out = np.full(hi - lo, np.nan)
    for tb in range(lo, hi, step):
        t_lo = tf if window is None else max(tf, tb - h - window + 1)
        tr = np.arange(t_lo, tb - h + 1); te = np.arange(tb, min(tb + step, hi))
        out[te - lo] = predict_ridge(fit_ridge(X[tr], y[tr], lam), X[te])
    return out

def rmse(e): return float(np.sqrt(np.mean(np.asarray(e) ** 2)))

def dm(e_a, e_b, h):
    """Diebold-Mariano on squared errors; negative mean => A better. Returns (mean diff, HAC p-value)."""
    d = np.asarray(e_a) ** 2 - np.asarray(e_b) ** 2
    f = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": h + 10})
    return f.params[0], f.pvalues[0]

def block_boot(e2_list, L=200, B=2000, seed=0):
    """Circular moving-block bootstrap. e2_list = list of squared-error arrays (same length).
    Returns array (B, len(e2_list)) of bootstrap mean squared errors."""
    rng = np.random.default_rng(seed); n = len(e2_list[0]); nb = int(np.ceil(n / L))
    E = np.vstack(e2_list)                                   # (k, n)
    out = np.empty((B, E.shape[0])); done = 0
    while done < B:
        m = min(500, B - done)
        starts = rng.integers(0, n, size=(m, nb))
        idx = ((starts[:, :, None] + np.arange(L)) % n).reshape(m, -1)[:, :n]
        out[done:done + m] = E[:, idx].mean(axis=2).T
        done += m
    return out
