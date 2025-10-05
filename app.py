import streamlit as st
import pandas as pd
import json
from pathlib import Path
from exoplanet_model import (
    load_and_prepare, cross_validate, train_full_model,
    score_candidates, save_artifacts, load_model,
    predict_single, compute_full_metrics, load_medians, FEATURE_COLS
)

st.set_page_config(page_title="Exoplanet Predictor", layout="wide")
st.title("Exoplanet Candidate Ranking – Binary Model")

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "lgbm_exoplanet_model.joblib"
CANDS_PATH = ARTIFACTS_DIR / "candidates_scored.csv"
META_PATH = ARTIFACTS_DIR / "metadata.json"
USER_INPUTS = ARTIFACTS_DIR / "user_inputs.csv"

default_params = {
    "n_estimators": 400,
    "learning_rate": 0.04,
    "num_leaves": 48,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": 42,
    "objective": "binary"
}

with st.sidebar:
    st.header("Entrenamiento")
    uploaded = st.file_uploader("Subir CSV TOI", type=["csv"])
    calibrate = st.checkbox("Calibrar probabilidades (isotonic)", value=False)
    train_btn = st.button("Entrenar / Re-entrenar")

    st.markdown("---")
    st.header("Predicción Manual")
    threshold = st.slider("Umbral clasificación (planet-like)", 0.0, 1.0, 0.5, 0.01)

# Cargar modelo y medianas si existen
model = None
medians = {}
if MODEL_PATH.exists():
    try:
        model = load_model(MODEL_PATH)
        st.success("Modelo cargado.")
        medians = load_medians(META_PATH)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")

# Entrenamiento
if train_btn:
    if uploaded is None:
        st.error("Primero sube un CSV.")
        st.stop()
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    temp_csv = ARTIFACTS_DIR / "uploaded_training.csv"
    with open(temp_csv, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Preparando datos...")
    train_df, candidates_df = load_and_prepare(str(temp_csv))
    st.write(f"Filas entrenamiento: {len(train_df)} | Candidatos PC: {len(candidates_df)}")

    st.info("Cross-validation...")
    cv = cross_validate(train_df, default_params, n_splits=5, calibrate=calibrate)
    st.subheader("Métricas CV promedio")
    st.json(cv)

    st.info("Entrenando modelo final...")
    model = train_full_model(train_df, default_params, calibrate=calibrate)

    st.info("Métricas sobre todo el dataset de entrenamiento...")
    full_metrics = compute_full_metrics(train_df, model)
    st.json(full_metrics)

    st.info("Rankeando candidatos (PC)...")
    scored = score_candidates(model, candidates_df)
    if not scored.empty:
        st.dataframe(scored.head(20))
    else:
        st.warning("No había filas PC para rankear.")

    st.info("Guardando artefactos...")
    save_artifacts(model, scored, cv, full_metrics, train_df, outdir=str(ARTIFACTS_DIR))

    st.success("Entrenamiento completo. Recarga para usar medianas actualizadas.")
    st.stop()

# Tabs
tab_ranking, tab_manual, tab_metadata = st.tabs(["Ranking PC", "Ingreso Manual", "Metadatos"])

with tab_ranking:
    st.subheader("Ranking de candidatos (PC)")
    if CANDS_PATH.exists():
        df_cands = pd.read_csv(CANDS_PATH)
        st.dataframe(df_cands.head(50), use_container_width=True)
        st.download_button(
            label="Descargar ranking completo (CSV)",
            data=df_cands.to_csv(index=False),
            file_name="candidates_scored.csv",
            mime="text/csv"
        )
    else:
        st.info("Aún no hay ranking. Entrena primero.")

with tab_manual:
    st.subheader("Ingresar nuevo objeto / planeta")
    if model is None:
        st.warning("Primero entrena o carga un modelo.")
    else:
        with st.form("manual_form"):
            cols = st.columns(4)
            input_data = {}
            for i, feat in enumerate(FEATURE_COLS):
                with cols[i % 4]:
                    input_data[feat] = st.number_input(feat, value=0.0, format="%.5f", key=f"in_{feat}")
            submit_btn = st.form_submit_button("Calcular probabilidad")

        if submit_btn:
            prob = predict_single(model, input_data, medians=medians)
            decision = "PLANET-LIKE" if prob >= threshold else "NOT PLANET-LIKE"
            st.markdown(f"**Probabilidad:** {prob:.6f}")
            st.markdown(f"**Decisión (umbral {threshold:.2f}):** {decision}")

            # Opción guardar
            save_choice = st.checkbox("Guardar este objeto en registro local", value=True)
            if save_choice:
                row = input_data.copy()
                row.update({
                    "prob_planet": prob,
                    "decision": decision,
                    "threshold_used": threshold,
                    "timestamp": pd.Timestamp.utcnow().isoformat()
                })
                if USER_INPUTS.exists():
                    df_prev = pd.read_csv(USER_INPUTS)
                    df_new = pd.concat([df_prev, pd.DataFrame([row])], ignore_index=True)
                else:
                    df_new = pd.DataFrame([row])
                df_new.to_csv(USER_INPUTS, index=False)
                st.success("Guardado en artifacts/user_inputs.csv")

            if USER_INPUTS.exists():
                with st.expander("Ver últimos guardados"):
                    st.dataframe(pd.read_csv(USER_INPUTS).tail(10))

with tab_metadata:
    st.subheader("Metadatos / Métricas")
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        st.json(meta)
        imps = meta.get("feature_importances")
        if imps:
            st.markdown("### Importancia de Features")
            imp_df = (pd.DataFrame(list(imps.items()), columns=["feature","importance"])
                        .sort_values("importance", ascending=False))
            st.bar_chart(imp_df.set_index("feature"))
    else:
        st.info("Sin metadata todavía.")

st.caption("Prototipo – Modelo binario (CP/KP vs FP) con ranking de PC y formulario de ingreso manual.")