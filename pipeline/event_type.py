"""
================================================================================
 ETAPA 7 — event_type.py : CLASIFICADOR PROSPECTIVO EXPLOSIÓN/EMISIÓN
================================================================================
Una vez declarada la alerta, discrimina el tipo de episodio esperado
(explosión vs emisión) a partir de las 45 características, EN EL INSTANTE
EXACTO en que se declara la alerta (Opción A, estrictamente prospectivo).

Diseño (acordado con el usuario):
  - Instante de predicción = momento de declaración de alerta (densidad 85% en
    ventana de 6 h, misma función sustained.alert_density con th_op).
  - Entrenamiento: SOLO eventos de 2023 que generaron alerta (coherente con la
    operación real). Features tomadas en el minuto de su declaración.
  - Métrica PRINCIPAL: validación cruzada estratificada sobre 2023 (donde hay
    muestra: 76 eventos, 17 exp / 59 emi).
  - 2019: tabla cualitativa predicho/real sobre los eventos detectados (muestra
    demasiado pequeña para un porcentaje serio; se reporta de forma honesta).

Requisitos previos (en intermediate/):
  - model_binary.joblib, model_multiclass.joblib (Etapa 4)
  - features_2023.parquet, features_2019.parquet (Etapa 3)
  - validation_2019.parquet (Etapa 5) -> para th_op vía postprocess_report.json

Salida:
  - intermediate/event_type_report.json  (métricas 2023-CV + tabla 2019)
  - impresión en consola de la precisión y la tabla
================================================================================
"""
import sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import config as C
import sustained as S


class EventTypeError(Exception):
    pass


def _alert_onset_for_event(times, p_alert, valid, ev_start, th_op,
                           assoc_h=C.ASSOC_WINDOW_H):
    """Devuelve el minuto de DECLARACIÓN de alerta asociado a un evento:
    el onset del intervalo de alerta activo dentro de las assoc_h horas
    previas al evento. None si el evento no generó alerta."""
    declared = S.alert_density(times, p_alert, th_op,
                               persist_min=C.PERSIST_MIN, density_frac=C.DENSITY_FRAC)
    intervals = S.declaration_intervals(times, declared)
    ev_t = pd.Timestamp(ev_start)
    assoc = pd.Timedelta(hours=float(assoc_h))
    cands = [iv for iv in intervals if iv[1] >= ev_t - assoc and iv[0] <= ev_t]
    if not cands:
        return None
    return min(cands, key=lambda x: x[0])[0]   # onset más temprano en la ventana


def _features_at_instant(feat_df, sel, t_instant):
    """Devuelve el vector de 45 features en el minuto t_instant (o el más
    cercano disponible). None si no hay fila válida cercana."""
    ft = feat_df.copy()
    ft["Time"] = pd.to_datetime(ft["Time"])
    idx = ft["Time"].searchsorted(pd.Timestamp(t_instant))
    idx = min(max(idx, 0), len(ft) - 1)
    row = ft.iloc[idx]
    x = row[sel].fillna(0).values.astype(float)
    return x


def run():
    print("=" * 70)
    print("ETAPA 7 — CLASIFICADOR EXPLOSIÓN/EMISIÓN (instante de alerta)")
    print("=" * 70)

    # ── th_op del reporte de post-proceso ──
    rep_path = C.INTERMEDIATE / "postprocess_report.json"
    if not rep_path.exists():
        raise EventTypeError("Falta postprocess_report.json (corre Etapa 6).")
    th_op = json.load(open(rep_path))["th_op"]
    print(f"  th_op = {th_op:.4f}")

    # ── modelo binario y features seleccionadas ──
    mb = joblib.load(C.INTERMEDIATE / "model_binary.joblib")
    rf_bin, sel = mb["model"], mb["features"]

    # ══════════ ENTRENAMIENTO (2023) ══════════
    feat23 = pd.read_parquet(C.INTERMEDIATE / "features_2023.parquet")
    feat23["Time"] = pd.to_datetime(feat23["Time"])
    catalog23 = C.load_catalog_2023()
    print(f"\n  Catálogo 2023: {len(catalog23)} eventos")

    # P(Alert) de 2023 por estación (aplicar el modelo binario a sus features).
    # La declaración de alerta se busca por estación; para el tipo de evento se
    # usa la estación de referencia UB4 (la de validación), o la primera que
    # haya declarado alerta antes del evento.
    X_train, y_train, used = [], [], 0
    stations = list(feat23["station"].unique())
    p_by_station = {}
    for st in stations:
        g = feat23[feat23["station"] == st].sort_values("Time")
        p = rf_bin.predict_proba(g[sel].fillna(0))[:, 1]
        valid = g["valid"].values == 1
        p[~valid] = np.nan
        p_by_station[st] = (g["Time"].values, p, valid, g)

    for ev in catalog23:
        ev_start = ev["start"]
        is_exp = 1 if ev["type"] == "explosion" else 0
        # buscar la primera estación que declaró alerta para este evento
        onset, gref = None, None
        for st in stations:
            times, p, valid, g = p_by_station[st]
            o = _alert_onset_for_event(times, p, valid, ev_start, th_op)
            if o is not None and (onset is None or o < onset):
                onset, gref = o, g
        if onset is None:
            continue  # evento no generó alerta -> no se usa (opción a)
        x = _features_at_instant(gref, sel, onset)
        X_train.append(x); y_train.append(is_exp); used += 1

    X_train = np.array(X_train); y_train = np.array(y_train)
    n_exp = int(y_train.sum()); n_emi = int(len(y_train) - n_exp)
    print(f"  Eventos 2023 con alerta usados para entrenar: {used} "
          f"({n_exp} explosiones, {n_emi} emisiones)")
    if used < 10 or n_exp < 2:
        print("  *** AVISO: muestra de entrenamiento muy pequeña; los resultados "
              "de CV serán poco robustos. ***")

    # ── Validación cruzada estratificada (métrica principal) ──
    rf_type = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     class_weight="balanced", random_state=0, n_jobs=-1)
    k = min(5, n_exp)  # no más folds que explosiones
    if k >= 2:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        y_pred_cv = cross_val_predict(rf_type, X_train, y_train, cv=skf)
        acc = accuracy_score(y_train, y_pred_cv)
        f1 = f1_score(y_train, y_pred_cv, zero_division=0)
        cm = confusion_matrix(y_train, y_pred_cv, labels=[0, 1])
        print(f"\n  === Validación cruzada {k}-fold sobre 2023 (métrica principal) ===")
        print(f"    Precisión (accuracy): {100*acc:.1f}%")
        print(f"    F1 (clase explosión): {f1:.3f}")
        print(f"    Matriz de confusión [filas=real, cols=pred] (0=emi,1=exp):")
        print(f"      {cm.tolist()}")
    else:
        acc, f1, cm = None, None, None
        print("  No hay suficientes explosiones para CV estratificada.")

    # ── Modelo final entrenado con todo 2023 ──
    rf_type.fit(X_train, y_train)

    # ══════════ VALIDACIÓN CUALITATIVA (2019) ══════════
    feat19 = pd.read_parquet(C.INTERMEDIATE / "features_2019.parquet")
    feat19["Time"] = pd.to_datetime(feat19["Time"])
    val19 = pd.read_parquet(C.INTERMEDIATE / "validation_2019.parquet")
    val19["Time"] = pd.to_datetime(val19["Time"])
    t19 = val19["Time"].values
    p19 = val19["p_alert_sm"].fillna(0).values
    v19 = val19["valid"].values == 1

    print(f"\n  === Validación cualitativa sobre 2019 (eventos detectados) ===")
    print(f"  {'Evento':16s} {'Real':10s} {'Predicho':10s} {'P(exp)':>8s}")
    table19 = []
    for ev in C.CATALOG_2019:
        if ev["override"] is not None:
            continue
        real = "explosion" if ev["type"] == "explosion" else "emission"
        onset = _alert_onset_for_event(t19, p19, v19, ev["start"], th_op)
        if onset is None:
            print(f"  {ev['label']:16s} {real:10s} {'(no alerta)':10s} {'—':>8s}")
            table19.append({"event": ev["label"], "real": real,
                            "predicted": None, "p_exp": None})
            continue
        x = _features_at_instant(feat19, sel, onset).reshape(1, -1)
        p_exp = float(rf_type.predict_proba(x)[0, 1])
        pred = "explosion" if p_exp >= 0.5 else "emission"
        ok = "OK" if pred == real else "X"
        print(f"  {ev['label']:16s} {real:10s} {pred:10s} {p_exp:>8.2f}  {ok}")
        table19.append({"event": ev["label"], "real": real,
                        "predicted": pred, "p_exp": round(p_exp, 3)})

    # ── Guardar reporte ──
    report = {
        "th_op": th_op,
        "train_2023": {"n_used": used, "n_explosion": n_exp, "n_emission": n_emi},
        "cv_2023": {"folds": k if k >= 2 else None,
                    "accuracy": round(acc, 3) if acc is not None else None,
                    "f1_explosion": round(f1, 3) if f1 is not None else None,
                    "confusion_matrix": cm.tolist() if cm is not None else None},
        "qualitative_2019": table19,
        "instant_definition": "features taken at the alert-declaration minute "
                              "(density 85% over 6 h with th_op); strictly prospective (Option A)",
    }
    with open(C.INTERMEDIATE / "event_type_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Reporte guardado: {C.INTERMEDIATE / 'event_type_report.json'}")
    print("\nETAPA 7 completada.")
    return report


if __name__ == "__main__":
    run()
