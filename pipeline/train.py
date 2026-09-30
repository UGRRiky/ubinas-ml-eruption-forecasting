"""
================================================================================
 ETAPA 4 — train.py : ENTRENAMIENTO RANDOM FOREST + LOSO (robustez espacial)
================================================================================
Entrena los clasificadores con datos 2023 y evalúa transferibilidad ESPACIAL
con Leave-One-Station-Out (LOSO).

DOS PROTECCIONES METODOLÓGICAS:

1. RESUELVE P48 (Jesús): la selección de características se hace DENTRO de cada
   fold LOSO (solo con las 2 estaciones de entrenamiento) -> sin fuga espacial.

2. ANTI-MEMORIZACIÓN (sin cegar el modelo):
   - min_samples_leaf = 20 (config): consenso suficiente para no ajustar al
     ruido de un minuto, pero ágil para eventos cortos de 10-15 min. (Un valor
     como 1000 = 16 h haría al modelo ciego a explosiones breves.)
   - submuestreo 1/5 SOLO de la clase mayoritaria (No-Alerta). La clase
     positiva (Pre/Eruptivo) NUNCA se submuestrea -> el modelo no se vuelve
     ciego ante explosiones.
   Estas dos protecciones rompen la memorización trivial que, sin ellas,
   inflaba el AUC a 1.0 por autocorrelación temporal.

ALCANCE DEL LOSO (declarado explícitamente): las 3 estaciones registran los
MISMOS instantes de tiempo, de modo que el LOSO mide robustez ESPACIAL
(¿generaliza el modelo a una estación con instrumentación distinta?), no
temporal — los folds comparten los mismos minutos. La separación temporal
genuina la aporta la validación 2019 (crisis independiente, >6 meses de
separación de los datos de entrenamiento), que es la MÉTRICA ANCLA del paper.

NARRATIVA (acordada): "En 2019 probamos que el modelo funciona (métrica
principal); en 2023, mediante LOSO, probamos que sobrevive a cambios en la
red — test de estrés de robustez espacial."

Salidas:
  intermediate/model_binary.joblib, model_multiclass.joblib
  intermediate/selected_features.json
  intermediate/loso_results.json      (AUC por estación)
================================================================================
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import config as C


class TrainError(Exception):
    pass


FEATURE_COLS = None


def load_training_data():
    feat = pd.read_parquet(C.INTERMEDIATE / "features_2023.parquet")
    tmin, tmax = feat["Time"].min(), feat["Time"].max()
    if tmin.year != 2023 or tmax.year != 2023:
        raise TrainError(f"Datos de entrenamiento no son solo 2023: {tmin}..{tmax}")
    if "label_bin_op" not in feat.columns:
        raise TrainError("features_2023.parquet no tiene 'label_bin_op'. "
                         "Corre label.py y features.py (con merge) primero.")
    global FEATURE_COLS
    FEATURE_COLS = [c for c in feat.columns
                    if c not in ("Time", "station", "valid", "label_bin_op", "state_4c")]
    feat["Time"] = pd.to_datetime(feat["Time"])
    return feat


def subsample_majority(df, y, seed=0):
    """Submuestrea SOLO la clase mayoritaria (y==0) a 1/SUBSAMPLE_MAJORITY.
    Conserva TODAS las muestras positivas (y==1) -> el modelo no se vuelve
    ciego ante explosiones. Devuelve el df filtrado."""
    rng = np.random.default_rng(seed)
    pos_idx = df.index[y == 1]
    neg_idx = df.index[y == 0]
    keep_neg = rng.choice(neg_idx, size=len(neg_idx) // C.SUBSAMPLE_MAJORITY,
                          replace=False)
    keep = np.concatenate([pos_idx.values, keep_neg])
    return df.loc[np.sort(keep)]


def select_features_fold(X, y, n_final=C.N_FEATURES_FINAL,
                         corr_th=C.CORR_THRESHOLD, n_trees=C.N_TREES_FS, seed=0):
    rf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=C.MIN_LEAF,
                                 n_jobs=-1, random_state=seed, class_weight="balanced")
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    corr = X[imp.index].corr().abs()
    keep, dropped = [], set()
    for f in imp.index:
        if f in dropped:
            continue
        keep.append(f)
        high = corr.index[(corr[f] > corr_th) & (corr.index != f)]
        for h in high:
            if imp[h] <= imp[f]:
                dropped.add(h)
        if len(keep) >= n_final:
            break
    return keep[:n_final]


def run_loso(feat):
    """LOSO con selección de features por fold + submuestreo de la clase
    mayoritaria. Mide robustez ESPACIAL (los folds comparten instantes)."""
    stations = list(feat["station"].unique())
    print(f"\n  LOSO — {len(stations)} folds (robustez ESPACIAL; "
          f"submuestreo mayoritaria 1/{C.SUBSAMPLE_MAJORITY}, "
          f"min_leaf={C.MIN_LEAF}):")
    print(f"    [Nota: los folds comparten los mismos instantes de tiempo; el")
    print(f"     LOSO mide generalización entre estaciones, no temporal. La")
    print(f"     separación temporal la aporta la validación 2019.]")
    results = {}
    score_frames = []  # scores por estación para curvas ROC
    for test_st in stations:
        test_df = feat[(feat["station"] == test_st) & feat["valid"]].copy()
        train_df = feat[(feat["station"] != test_st) & feat["valid"]].copy()

        # submuestreo SOLO de la clase mayoritaria (positivas intactas)
        train_df = subsample_majority(train_df, train_df["label_bin_op"])

        Xtr = train_df[FEATURE_COLS].fillna(0)
        ytr = train_df["label_bin_op"]
        Xte = test_df[FEATURE_COLS].fillna(0)
        yte = test_df["label_bin_op"]

        sel = select_features_fold(Xtr, ytr)
        rf = RandomForestClassifier(n_estimators=C.N_TREES, min_samples_leaf=C.MIN_LEAF,
                                     n_jobs=-1, random_state=0, class_weight="balanced")
        rf.fit(Xtr[sel], ytr)
        p = rf.predict_proba(Xte[sel])[:, 1]
        auc = roc_auc_score(yte, p) if yte.nunique() > 1 else float("nan")
        results[test_st] = {"auc": float(auc), "n_selected": len(sel),
                            "n_train": int(len(train_df)), "n_test": int(len(test_df)),
                            "pos_train": int(ytr.sum()), "pos_test": int(yte.sum())}
        # guardar scores para la curva ROC de esta estación
        score_frames.append(pd.DataFrame({"station": test_st,
                                          "y_true": yte.values, "score": p}))
        print(f"    Test [{test_st}]: AUC={auc:.4f}  "
              f"(train {len(train_df):,}, {int(ytr.sum())} pos | "
              f"test {len(test_df):,}, {int(yte.sum())} pos)")

    aucs = [r["auc"] for r in results.values() if not np.isnan(r["auc"])]
    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    print(f"\n    AUC LOSO medio: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"    (robustez espacial / test de estrés out-of-distribution;")
    print(f"     la métrica principal del paper es el AUC de validación 2019)")
    loso_scores = pd.concat(score_frames, ignore_index=True)
    return results, loso_scores


def export_figure4_data(Xall, y_bin, y_4c, sel_final, loso_scores=None,
                        subsample_curve=3):
    """Exporta los CSV que alimentan la Figura 4:
      - oob_curves.csv          : error/accuracy OOB acumulado vs nº de árboles
      - leafsize_sensitivity.csv: accuracy OOB vs min_samples_leaf
      - loso_scores.csv         : scores LOSO por estación para curvas ROC reales

    subsample_curve: las curvas OOB y de leaf-size se calculan sobre 1 de cada
    `subsample_curve` filas del entrenamiento. La FORMA de las curvas (que es
    lo que la figura muestra) no cambia, pero el tiempo se reduce ~3x. Los
    valores OOB del MODELO FINAL (los que van al texto) NO usan este submuestreo
    — se calculan sobre el conjunto completo en run(). Poner 1 para no submuestrear.
    """
    import csv
    Xc = Xall.iloc[::subsample_curve]
    yb = y_bin.iloc[::subsample_curve] if hasattr(y_bin, "iloc") else y_bin[::subsample_curve]
    y4 = y_4c.iloc[::subsample_curve] if hasattr(y_4c, "iloc") else y_4c[::subsample_curve]
    if subsample_curve > 1:
        print(f"    (curvas OOB/leaf-size sobre 1/{subsample_curve} de las filas; "
              f"forma idéntica, ~{subsample_curve}x más rápido)")

    # ── (a) Curva OOB acumulada (binario y multiclase) ──
    print("    Exportando curva OOB acumulada...")
    tree_grid = list(range(50, C.N_TREES + 1, 50))
    oob_rows = []
    for label, y in [("binary", yb), ("multiclass", y4)]:
        for nt in tree_grid:
            rf = RandomForestClassifier(n_estimators=nt, oob_score=True,
                                        min_samples_leaf=C.MIN_LEAF, n_jobs=-1,
                                        random_state=0, class_weight="balanced",
                                        bootstrap=True)
            rf.fit(Xc[sel_final], y)
            oob_rows.append({"model": label, "n_trees": nt,
                             "oob_error": 1 - rf.oob_score_,
                             "oob_accuracy": rf.oob_score_})
    with open(C.INTERMEDIATE / "oob_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n_trees", "oob_error", "oob_accuracy"])
        w.writeheader(); w.writerows(oob_rows)

    # ── (b) Sensibilidad a min_samples_leaf ──
    print("    Exportando sensibilidad a leaf-size...")
    leaf_grid = [1, 2, 5, 10, 20, 50]
    leaf_rows = []
    for leaf in leaf_grid:
        for label, y in [("binary", yb), ("multiclass", y4)]:
            rf = RandomForestClassifier(n_estimators=C.N_TREES_FS,
                                         min_samples_leaf=leaf, n_jobs=-1,
                                         random_state=0, oob_score=True,
                                         class_weight="balanced", bootstrap=True)
            rf.fit(Xc[sel_final], y)
            leaf_rows.append({"model": label, "min_samples_leaf": leaf,
                              "oob_accuracy": rf.oob_score_})
    with open(C.INTERMEDIATE / "leafsize_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "min_samples_leaf", "oob_accuracy"])
        w.writeheader(); w.writerows(leaf_rows)

    # ── (c) Scores LOSO por estación (para curvas ROC reales) ──
    if loso_scores is not None:
        print("    Exportando scores LOSO para curvas ROC...")
        loso_scores.to_parquet(C.INTERMEDIATE / "loso_scores.parquet", index=False)
    print("    Datos de Figura 4 exportados.")


def run():
    print("=" * 70)
    print("ETAPA 4 — ENTRENAMIENTO + LOSO (robustez espacial)")
    print("=" * 70)

    feat = load_training_data()

    loso, loso_scores = run_loso(feat)
    with open(C.INTERMEDIATE / "loso_results.json", "w") as f:
        json.dump(loso, f, indent=2)

    # ── Modelo final (3 estaciones, submuestreo de la mayoritaria) ──
    print("\n  Entrenando modelo final (3 estaciones)...")
    m = feat["valid"]
    train_final = feat[m].copy()
    train_final = subsample_majority(train_final, train_final["label_bin_op"])
    Xall = train_final[FEATURE_COLS].fillna(0)
    y_bin = train_final["label_bin_op"]
    y_4c = train_final["state_4c"]
    print(f"    Muestras de entrenamiento final: {len(train_final):,} "
          f"({int(y_bin.sum())} positivas conservadas al 100%)")

    sel_final = select_features_fold(Xall, y_bin)
    with open(C.INTERMEDIATE / "selected_features.json", "w") as f:
        json.dump(sel_final, f, indent=2)
    print(f"    Features finales seleccionadas: {len(sel_final)}")

    rf_bin = RandomForestClassifier(n_estimators=C.N_TREES, min_samples_leaf=C.MIN_LEAF,
                                     n_jobs=-1, random_state=0, oob_score=True,
                                     class_weight="balanced", bootstrap=True)
    rf_bin.fit(Xall[sel_final], y_bin)
    print(f"    OOB binario: {100*rf_bin.oob_score_:.2f}%")

    rf_4c = RandomForestClassifier(n_estimators=C.N_TREES, min_samples_leaf=C.MIN_LEAF,
                                    n_jobs=-1, random_state=0, oob_score=True,
                                    class_weight="balanced", bootstrap=True)
    rf_4c.fit(Xall[sel_final], y_4c)
    print(f"    OOB multiclase: {100*rf_4c.oob_score_:.2f}%")

    joblib.dump({"model": rf_bin, "features": sel_final},
                C.INTERMEDIATE / "model_binary.joblib")
    joblib.dump({"model": rf_4c, "features": sel_final},
                C.INTERMEDIATE / "model_multiclass.joblib")
    print(f"    Modelos guardados.")

    # ── Exportar datos para la Figura 4 ──
    print("\n  Exportando datos para la Figura 4...")
    export_figure4_data(Xall, y_bin, y_4c, sel_final, loso_scores)

    print("\nETAPA 4 completada.")
    return loso


if __name__ == "__main__":
    run()
