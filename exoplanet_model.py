import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import joblib
import json
import time

FEATURE_COLS = [
    'period/days', 'duration/hours', 'depth', 'planet_radius',
    'stellar_radius', 'stteff', 'logg', 'tess_mag'
]

def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path, comment='#', na_values=['',' '])
    rename_map = {
        'tid': 'tic_id',
        'toi': 'toi',
        'tfopwg_disp': 'disposition',
        'pl_orbper': 'period/days',
        'pl_tranmid': 'epoch_bjd',
        'pl_trandurh': 'duration/hours',
        'pl_trandep': 'depth',
        'pl_rade': 'planet_radius',
        'st_rad': 'stellar_radius',
        'st_teff': 'stteff',
        'st_logg': 'logg',
        'st_tmag': 'tess_mag',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    needed = ['disposition'] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    keep_cols = list(dict.fromkeys(['tic_id','toi','disposition','epoch_bjd'] + FEATURE_COLS))
    df = df[keep_cols].copy()

    # Imputación simple
    for c in FEATURE_COLS:
        df[c] = df[c].fillna(df[c].median())

    train_df = df[df['disposition'].isin(['CP','KP','FP'])].copy()
    candidates_df = df[df['disposition'] == 'PC'].copy()
    train_df['label'] = (train_df['disposition'].isin(['CP','KP'])).astype(int)
    return train_df, candidates_df

def cross_validate(train_df, params, n_splits=5, calibrate=False):
    X = train_df[FEATURE_COLS]
    y = train_df['label']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=params.get('random_state',42))
    aucs, aps, best_f1s = [], [], []
    for fold, (tr, va) in enumerate(skf.split(X,y),1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        if calibrate:
            cal = CalibratedClassifierCV(model, method='isotonic', cv=3)
            cal.fit(X_tr, y_tr)
            probs = cal.predict_proba(X_va)[:,1]
        else:
            probs = model.predict_proba(X_va)[:,1]
        auc = roc_auc_score(y_va, probs)
        ap = average_precision_score(y_va, probs)
        p,r,t = precision_recall_curve(y_va, probs)
        f1s = 2*p*r/(p+r+1e-9)
        best_f1 = f1s.max()
        aucs.append(auc); aps.append(ap); best_f1s.append(best_f1)
    return {
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "ap_mean": float(np.mean(aps)),
        "ap_std": float(np.std(aps)),
        "best_f1_mean": float(np.mean(best_f1s)),
        "best_f1_std": float(np.std(best_f1s))
    }

def train_full_model(train_df, params, calibrate=False):
    X = train_df[FEATURE_COLS]
    y = train_df['label']
    base_model = LGBMClassifier(**params)
    base_model.fit(X,y)
    if calibrate:
        cal = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
        cal.fit(X,y)
        return cal
    return base_model

def compute_full_metrics(train_df, model):
    X = train_df[FEATURE_COLS]
    y = train_df['label']
    probs = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    p,r,t = precision_recall_curve(y, probs)
    f1s = 2*p*r/(p+r+1e-9)
    best_idx = f1s.argmax()
    best_f1 = float(f1s[best_idx])
    best_th = float(t[best_idx]) if best_idx < len(t) else 0.5
    return {
        "roc_auc_full": float(auc),
        "ap_full": float(ap),
        "best_f1_full": best_f1,
        "best_threshold_full": best_th,
        "precision_at_best_f1": float(p[best_idx]),
        "recall_at_best_f1": float(r[best_idx]),
        "n_train_rows": int(len(train_df)),
        "class_counts": {
            "planet(1)": int((y==1).sum()),
            "fp(0)": int((y==0).sum())
        }
    }

def score_candidates(model, candidates_df):
    if candidates_df.empty:
        return candidates_df.assign(prob_planet=[], rank=[])
    X_cand = candidates_df[FEATURE_COLS]
    probs = model.predict_proba(X_cand)[:,1]
    out = candidates_df.copy()
    out['prob_planet'] = probs
    out = out.sort_values('prob_planet', ascending=False).reset_index(drop=True)
    out['rank'] = np.arange(1, len(out)+1)
    return out

def extract_feature_importances(model):
    base = model
    if hasattr(model, "base_estimator_"):
        base = model.base_estimator_
    if hasattr(base, "feature_importances_"):
        return dict(zip(FEATURE_COLS, base.feature_importances_.tolist()))
    return None

def save_artifacts(model, scored_df, cv_metrics, full_metrics, train_df, outdir="artifacts"):
    outp = Path(outdir)
    outp.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, outp / "lgbm_exoplanet_model.joblib")
    scored_df.to_csv(outp / "candidates_scored.csv", index=False)

    medians = {c: float(train_df[c].median()) for c in FEATURE_COLS}

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_candidates": int(len(scored_df)),
        "top5": scored_df[['toi','prob_planet']].head(5).to_dict(orient='records'),
        "cv_metrics": cv_metrics,
        "full_metrics": full_metrics,
        "feature_importances": extract_feature_importances(model),
        "feature_medians": medians
    }
    with open(outp / "metadata.json","w") as f:
        json.dump(meta, f, indent=2)

def load_model(model_path="artifacts/lgbm_exoplanet_model.joblib"):
    return joblib.load(model_path)

def load_medians(metadata_path="artifacts/metadata.json"):
    try:
        meta = json.loads(Path(metadata_path).read_text())
        return meta.get("feature_medians", {})
    except FileNotFoundError:
        return {}

def predict_single(model, feature_dict, medians=None):
    row = pd.DataFrame([feature_dict])[FEATURE_COLS]
    if medians:
        for c in FEATURE_COLS:
            if row[c].isna().any():
                row[c] = medians.get(c, 0.0)
    else:
        for c in FEATURE_COLS:
            if row[c].isna().any():
                row[c] = row[c].fillna(0)
    prob = model.predict_proba(row)[0,1]
    return prob