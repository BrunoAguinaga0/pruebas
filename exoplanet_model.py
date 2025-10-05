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

from exoplanet_schema import load_schema, auto_map_columns, apply_mapping, columns_needed

def load_and_standardize(csv_path: str,
                        schema_path: str = "schema_mapping.json",
                        interactive_map: dict = None):
    """
    1. Lee el CSV.
    2. Aplica mapping automático (sinónimos + fuzzy si disponible).
    3. Aplica mapping manual adicional (interactive_map) si se proporciona desde la UI.
    4. Verifica columnas canónicas mínimas.
    Devuelve: df, missing_after
    """
    df = pd.read_csv(csv_path, comment="#", na_values=["", " "])
    schema = load_schema(schema_path)

    # Paso 1: mapping automático
    exact_map, unmapped, fuzzy_suggestions = auto_map_columns(list(df.columns), schema, fuzzy=True)

    # Si el usuario pasó un mapping manual (col_original -> canon), se fusiona
    if interactive_map:
        exact_map.update(interactive_map)

    df_std = apply_mapping(df, exact_map)

    required = columns_needed(schema)
    missing_after = [c for c in required if c not in df_std.columns]

    info = {
        "auto_mapped": exact_map,
        "unmapped_original": unmapped,
        "fuzzy_suggestions": fuzzy_suggestions,
        "missing_after": missing_after
    }
    return df_std, info

FEATURE_COLS = [
    'period/days', 'duration/hours', 'depth', 'planet_radius',
    'stellar_radius', 'stteff', 'logg', 'tess_mag'
]

KOI_DISP_MAP = {
    "CONFIRMED": "CP",
    "CANDIDATE": "PC",
    "FALSE POSITIVE": "FP"
}

def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si existen columnas duplicadas (mismo nombre), combina tomando el primer
    valor no nulo por fila y deja una sola columna final.
    """
    if not df.columns.duplicated().any():
        return df

    # Recorremos cada nombre duplicado y consolidamos
    new_cols = []
    to_add = {}
    seen = set()

    for col in df.columns:
        if col in seen:
            continue
        duplicates = df.columns[df.columns == col]
        if len(duplicates) == 1:
            new_cols.append(col)
            seen.add(col)
        else:
            # Sub-dataframe con duplicados
            sub = df.loc[:, duplicates]
            # tomar primer no nulo hacia la derecha
            combined = sub.bfill(axis=1).iloc[:, 0]
            to_add[col] = combined
            # marcar todos como vistos
            for c in duplicates:
                seen.add(c)

    # Construir nuevo DataFrame (dropear duplicados y añadir combinados)
    # Más sencillo: para cada col duplicada, dropear todas y luego añadir una
    result = df.loc[:, ~df.columns.duplicated()].copy()
    for col, series in to_add.items():
        result[col] = series

    # Asegurar no quedaron duplicados
    result = result.loc[:, ~result.columns.duplicated()]
    return result

def load_and_prepare(csv_path: str,
                     schema_path: str = "schema_mapping.json",
                     allow_koi_depth_fraction: bool = True,
                     convert_bkjd_to_bjd: bool = True):
    """
    Carga dataset TESS o KOI y lo estandariza:
    - Mapping columnas (sinónimos/fuzzy)
    - Consolidación de duplicados
    - Normalización de disposition KOI
    - Corrección de depth (fracción → ppm si aplica)
    - Conversión opcional BKJD → BJD en epoch_bjd
    - Imputación
    - Separación train (CP/KP/FP) y candidatos (PC)
    """
    df = pd.read_csv(csv_path, comment='#', na_values=['',' '])

    # 1. Mapping
    schema = load_schema(schema_path)
    exact_map, unmapped, fuzzy_suggestions = auto_map_columns(
        list(df.columns), schema, fuzzy=True
    )
    df = apply_mapping(df, exact_map)

    # 2. Consolidar duplicados (por si koi_disposition y koi_pdisposition quedaron juntos)
    df = _collapse_duplicate_columns(df)

    # 3. Validar requeridos
    needed = ['disposition'] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas requeridas incluso tras mapping automático.\n"
            f"Faltan: {missing}\n"
            f"Sin mapear: {unmapped}\n"
            f"Sugerencias fuzzy: {fuzzy_suggestions}"
        )

    # 4. Normalizar disposición KOI
    df['disposition'] = df['disposition'].replace(KOI_DISP_MAP)

    # Filtrar solo disposiciones que reconocemos
    valid_disp = {'CP','KP','FP','PC'}
    df = df[df['disposition'].isin(valid_disp)].copy()

    # 5. Ajustar depth si parece fracción
    if allow_koi_depth_fraction and df['depth'].median() < 0.01:
        df['depth'] = df['depth'] * 1e6  # convertir a ppm

    # 6. Convertir BKJD -> BJD si corresponde
    if convert_bkjd_to_bjd and 'epoch_bjd' in df.columns:
        # Heurística: si median < 5000 asumimos BKJD (Kepler) y sumamos offset
        if df['epoch_bjd'].dropna().median() < 5000:
            df['epoch_bjd'] = df['epoch_bjd'] + 2454833.0

    # 7. Imputación simple
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())

    keep_cols = list(dict.fromkeys(['tic_id','toi','disposition','epoch_bjd'] + FEATURE_COLS))
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # 8. Separar train y candidatos
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