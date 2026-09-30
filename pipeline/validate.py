"""
================================================================================
 ETAPA 5 — validate.py : VALIDACIÓN CIEGA 2019 -> CSV de salida
================================================================================
Aplica el modelo entrenado (solo con 2023) a UB4/2019, sin reentrenar, y
produce el CSV `ubinas_v16_ub4_2019.csv` con EXACTAMENTE las 26 columnas que
consumen los scripts de figuras 4-6:

  Time, entropy_val, kurtosis_val, freqidx_val, ssam_val,
  p_alert, p_alert_sm, p_alert_freeze, p_alert_sm_freeze,
  alert_op, alert_you_op, alert_05,
  pred_4c, p_c0, p_c1, p_c2, p_c3,
  pred_4c_hmm_causal, pred_4c_hmm_viterbi,
  p_ge_unrest, p_unrest_sm, unrest_alert,
  ref_4class, ref_bin_amp, ref_bin_op, valid

Las columnas de HMM (pred_4c_hmm_*) y de alerta sostenida se llenan en la
Etapa 6; aquí se dejan como placeholder para mantener el orden de columnas.
Esta etapa produce las columnas de probabilidad cruda y clasificación.

CHECK: el modelo nunca vio 2019 (assert de que se cargó de disco, entrenado
en 2023).

Entrada: intermediate/features_2019.parquet + modelos de train.py
Salida:  intermediate/validation_2019.parquet (base para Etapa 6)
================================================================================
"""
import sys
import numpy as np
import pandas as pd
import joblib
import config as C


class ValidateError(Exception):
    pass


def run():
    print("=" * 70)
    print("ETAPA 5 — VALIDACIÓN CIEGA 2019")
    print("=" * 70)

    # ── Cargar modelos entrenados (solo 2023) ──
    mb_path = C.INTERMEDIATE / "model_binary.joblib"
    mc_path = C.INTERMEDIATE / "model_multiclass.joblib"
    if not mb_path.exists() or not mc_path.exists():
        raise ValidateError("Faltan modelos. Corre train.py (Etapa 4) primero.")
    mb = joblib.load(mb_path)
    mc = joblib.load(mc_path)
    rf_bin, feat_bin = mb["model"], mb["features"]
    rf_4c,  feat_4c  = mc["model"], mc["features"]

    # ── Cargar features 2019 ──
    feat19 = pd.read_parquet(C.INTERMEDIATE / "features_2019.parquet")
    # assert temporal: todo 2019
    if feat19["Time"].dt.year.nunique() != 1 or feat19["Time"].dt.year.iloc[0] != 2019:
        raise ValidateError("features_2019 contiene años != 2019.")

    X = feat19[feat_bin].fillna(0)
    valid = feat19["valid"].values

    # ── Predicciones ──
    print("\n  Aplicando modelo binario (P(Alert))...")
    p_alert = rf_bin.predict_proba(X)[:, 1]
    p_alert[~valid] = np.nan

    print("  Aplicando modelo multiclase (4 estados)...")
    proba_4c = rf_4c.predict_proba(feat19[feat_4c].fillna(0))
    # mapear a p_c0..p_c3 según las clases del modelo
    classes = list(rf_4c.classes_)
    p_c = {f"p_c{s}": np.full(len(feat19), np.nan) for s in [0, 1, 2, 3]}
    for i, cls in enumerate(classes):
        p_c[f"p_c{int(cls)}"] = proba_4c[:, i]
    pred_4c = rf_4c.predict(feat19[feat_4c].fillna(0)).astype(float)
    pred_4c[~valid] = np.nan

    # ── Suavizado causal (trailing) de P(Alert), 60 min ──
    p_alert_sm = pd.Series(p_alert).rolling(60, min_periods=1).mean().values

    # ── P(>= Unrest) = 1 - P(Quiescence) ──
    p_ge_unrest = 1.0 - p_c["p_c0"]
    p_unrest_sm = pd.Series(p_ge_unrest).rolling(60, min_periods=1).mean().values

    # ── Umbrales operacionales (se calibran en Etapa 6; aquí placeholders 0) ──
    # Se dejan las columnas de alerta con la clasificación por th=0.5 como
    # referencia inmediata; los umbrales calibrados llegan en Etapa 6.
    alert_05 = (np.nan_to_num(p_alert, nan=0) >= 0.5).astype(int)

    # ── Referencias (ground truth) ──
    ref_4class = feat19["state_4c"].values
    ref_bin_op = feat19["label_bin_op"].values

    # ── Ensamblar CSV con las 26 columnas EXACTAS y en orden ──
    out = pd.DataFrame({
        "Time": feat19["Time"],
        "entropy_val":  feat19.get("entropy_inst", np.nan),   # valor normalizado inst.
        "kurtosis_val": feat19.get("kurtosis_inst", np.nan),
        "freqidx_val":  feat19.get("freqidx_inst", np.nan),
        "ssam_val":     feat19.get("ssam_inst", np.nan),
        "p_alert": p_alert,
        "p_alert_sm": p_alert_sm,
        "p_alert_freeze": p_alert,          # freeze se implementa en Etapa 6 si se usa
        "p_alert_sm_freeze": p_alert_sm,
        "alert_op": 0,                       # <- Etapa 6 (th_op calibrado)
        "alert_you_op": 0,                   # <- Etapa 6 (th Youden)
        "alert_05": alert_05,
        "pred_4c": pred_4c,
        "p_c0": p_c["p_c0"], "p_c1": p_c["p_c1"],
        "p_c2": p_c["p_c2"], "p_c3": p_c["p_c3"],
        "pred_4c_hmm_causal": np.nan,        # <- Etapa 6
        "pred_4c_hmm_viterbi": np.nan,       # <- Etapa 6
        "p_ge_unrest": p_ge_unrest,
        "p_unrest_sm": p_unrest_sm,
        "unrest_alert": 0,                   # <- Etapa 6
        "ref_4class": ref_4class,
        "ref_bin_amp": (ref_4class >= 1).astype(int),  # alerta amplia (>=unrest)
        "ref_bin_op": ref_bin_op,
        "valid": valid.astype(int),
    })

    out_path = C.INTERMEDIATE / "validation_2019.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n  Guardado: {out_path}  ({len(out):,} filas, {len(out.columns)} columnas)")

    # ── AUC operacional rápido (referencia; el detalle va en Etapa 6) ──
    from sklearn.metrics import roc_auc_score
    m = out["valid"] == 1
    if out.loc[m, "ref_bin_op"].nunique() > 1:
        auc_op = roc_auc_score(out.loc[m, "ref_bin_op"], out.loc[m, "p_alert"].fillna(0))
        print(f"  AUC operacional (referencia): {auc_op:.4f}")

    print("\nETAPA 5 completada.")
    return out


if __name__ == "__main__":
    run()
