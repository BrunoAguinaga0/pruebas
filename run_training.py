#!/usr/bin/env python
"""
CLI para entrenamiento binario (CP/KP vs FP) y ranking de PC.

Usa las funciones del módulo exoplanet_model.py para:
1. Cargar y preparar datos.
2. Cross-validation (ROC-AUC, AP, Best-F1).
3. Entrenar modelo final (con opción de calibración isotónica).
4. Calcular métricas sobre todo el set de entrenamiento.
5. Rankear candidatos PC.
6. Guardar artefactos:
   - model: artifacts/lgbm_exoplanet_model.joblib
   - candidates_scored.csv
   - metadata.json (incluye medianas, métricas y top5)

Ejemplo:
    python run_training.py --csv data/TOI_2025.10.04_10.50.38.csv
    python run_training.py --csv data/TOI.csv --calibrate --outdir resultados
"""

import argparse
from pathlib import Path
from exoplanet_model import (
    load_and_prepare,
    cross_validate,
    train_full_model,
    score_candidates,
    save_artifacts,
    compute_full_metrics
)

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.04,
    "num_leaves": 48,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": 42,
    "objective": "binary"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Ruta al CSV de TOIs (TESS)")
    parser.add_argument("--outdir", default="artifacts", help="Carpeta de salida artefactos")
    parser.add_argument("--calibrate", action="store_true", help="Usar calibración isotónica (binario)")
    parser.add_argument("--folds", type=int, default=5, help="Número de folds para CV")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")

    print(f"[1/6] Cargando y preparando datos desde: {csv_path}")
    train_df, candidates_df = load_and_prepare(str(csv_path))
    print(f"   - Filas entrenamiento: {len(train_df)}")
    print(f"   - Candidatos (PC): {len(candidates_df)}")

    print(f"[2/6] Cross-validation ({args.folds} folds)...")
    cv_metrics = cross_validate(train_df, DEFAULT_PARAMS, n_splits=args.folds, calibrate=args.calibrate)
    print("   Métricas CV (promedios):")
    for k,v in cv_metrics.items():
        print(f"   {k}: {v}")

    print("[3/6] Entrenando modelo final...")
    model = train_full_model(train_df, DEFAULT_PARAMS, calibrate=args.calibrate)

    print("[4/6] Métricas sobre todo el set de entrenamiento...")
    full_metrics = compute_full_metrics(train_df, model)
    for k,v in full_metrics.items():
        print(f"   {k}: {v}")

    print("[5/6] Rankeo de candidatos PC...")
    scored = score_candidates(model, candidates_df)
    if scored.empty:
        print("   No hay candidatos PC para rankear.")
    else:
        print("   Top 10 candidatos:")
        print(scored[['toi','prob_planet']].head(10).to_string(index=False))

    print("[6/6] Guardando artefactos...")
    save_artifacts(model, scored, cv_metrics, full_metrics, train_df, outdir=args.outdir)
    print(f"Listo. Artefactos en: {Path(args.outdir).resolve()}")

if __name__ == "__main__":
    main()