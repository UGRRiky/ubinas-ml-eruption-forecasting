"""
================================================================================
 ETAPA 6 — postprocess.py : HMM CAUSAL + UMBRALES + ALERTA
================================================================================
Toma la salida de validación (Etapa 5) y produce:
  - estados suavizados por HMM (Viterbi causal Y retrospectivo, ambos)
  - umbral operacional th_op calibrado con la quietud de 2019
  - declaraciones de alerta con persistencia medida en TIEMPO DE RELOJ REAL
  - anticipación por evento, distinguiendo detección FRESCA vs. HEREDADA
  - el CSV final ubinas_v16_ub4_2019.csv con las 26 columnas completas

Resuelve puntos de Jesús:
  - P51: Viterbi causal (forward-only) separado del retrospectivo; el causal
    es el que se usa para decisiones; el test confirma que no usa el futuro.
  - P69: matriz de transición con Q<->E prohibido, verificada numéricamente.
  - P70: anticipación distingue alerta fresca de alerta heredada (misma
    racha que cubre varios eventos).
  - persistencia en tiempo real (no en nº de filas), robusta a la
    resolución de muestreo.

Entrada: intermediate/validation_2019.parquet
Salida:  <BASE_DATA>/ubinas_v16_ub4_2019.csv  (26 columnas, para figuras)
         + reporte de umbrales y anticipaciones
================================================================================
"""
import sys
import json
import numpy as np
import pandas as pd
import config as C
import sustained as S


class PostError(Exception):
    pass


# ──────────────────────────────────────────────────────────────────────────
# MATRIZ HMM (simétrica, Q<->E prohibido en ambas direcciones)
# ──────────────────────────────────────────────────────────────────────────
def build_transition_matrix():
    """4x4, alta persistencia, progresión gradual, Q<->E prohibido."""
    A = np.array([
        [0.9990, 0.0009, 0.0001, 0.0000],  # desde Quietud
        [0.0020, 0.9960, 0.0019, 0.0001],  # desde Intranquilidad
        [0.0002, 0.0030, 0.9948, 0.0020],  # desde Pre-eruptivo
        [0.0000, 0.0040, 0.0010, 0.9950],  # desde Eruptivo (E->Q = 0)
    ])
    # verificación: Q->E y E->Q exactamente 0
    assert A[0, 3] == 0.0, "A(Q,E) debe ser 0"
    assert A[3, 0] == 0.0, "A(E,Q) debe ser 0"
    # normalizar filas por si acaso
    A = A / A.sum(axis=1, keepdims=True)
    return A


def viterbi_causal(log_emissions, log_A, log_pi):
    """Viterbi CAUSAL (forward-only): en cada t decide el estado usando solo
    observaciones hasta t. No hace backtracking sobre el futuro.
    log_emissions: (T, S). Devuelve secuencia de estados (T,)."""
    T, S = log_emissions.shape
    delta = log_pi + log_emissions[0]
    states = np.zeros(T, dtype=int)
    states[0] = np.argmax(delta)
    for t in range(1, T):
        # mejor transición desde el delta actual a cada estado
        trans = delta[:, None] + log_A  # (S_prev, S_next)
        best_prev = np.max(trans, axis=0)
        delta = best_prev + log_emissions[t]
        states[t] = np.argmax(delta)
    return states


def viterbi_retrospective(log_emissions, log_A, log_pi):
    """Viterbi clásico (forward-backward): usa toda la serie. Solo para
    comparación descriptiva; NO se usa para decisiones operacionales."""
    T, S = log_emissions.shape
    delta = np.zeros((T, S))
    psi = np.zeros((T, S), dtype=int)
    delta[0] = log_pi + log_emissions[0]
    for t in range(1, T):
        trans = delta[t-1][:, None] + log_A
        psi[t] = np.argmax(trans, axis=0)
        delta[t] = np.max(trans, axis=0) + log_emissions[t]
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]
    return states


def run_hmm(df, emission_temp=None):
    """Aplica HMM causal y retrospectivo sobre las probabilidades multiclase.

    IMPORTANTE — atenuación de emisiones (temperatura): las columnas p_c0..p_c3
    del RF son probabilidades POSTERIORES P(estado|obs), no verosimilitudes
    P(obs|estado). Usadas directamente como emisiones, dominan sobre la matriz
    de transición (un p_c3=0.94 espurio en quietud arrastra al Viterbi pese a
    la penalización de transición). Se atenúan elevándolas a una potencia
    T<1 (temperatura): log_em = T * log(P). Con T pequeño, la matriz de
    transición (persistencia + prohibición Q<->E) recupera su peso y limpia
    los parpadeos espurios. T=1 recupera el comportamiento sin atenuar.
    """
    if emission_temp is None:
        emission_temp = C.HMM_EMISSION_TEMP
    A = build_transition_matrix()
    eps = 1e-12
    log_A = np.log(A + eps)
    log_pi = np.log(np.array([0.97, 0.02, 0.005, 0.005]) + eps)

    P = df[["p_c0", "p_c1", "p_c2", "p_c3"]].values.copy()
    valid = df["valid"].values == 1
    P[~valid] = [1.0, 0.0, 0.0, 0.0]
    P = np.clip(P, eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    log_em = emission_temp * np.log(P)   # atenuación por temperatura

    causal = viterbi_causal(log_em, log_A, log_pi)
    retro  = viterbi_retrospective(log_em, log_A, log_pi)
    causal = causal.astype(float); retro = retro.astype(float)
    causal[~valid] = np.nan
    retro[~valid] = np.nan
    return causal, retro


def test_viterbi_causal():
    """El estado causal en t no debe cambiar si se altera el futuro."""
    rng = np.random.default_rng(0)
    T, S = 2000, 4
    P = rng.random((T, S)); P = P / P.sum(1, keepdims=True)
    A = build_transition_matrix()
    log_A = np.log(A + 1e-12); log_pi = np.log(np.full(S, 1/S))
    s1 = viterbi_causal(np.log(P), log_A, log_pi)
    P2 = P.copy(); P2[1500:] = rng.random((500, S)); P2[1500:] /= P2[1500:].sum(1, keepdims=True)
    s2 = viterbi_causal(np.log(P2), log_A, log_pi)
    if not (s1[:1500] == s2[:1500]).all():
        raise PostError("TEST FALLÓ: Viterbi causal usó el futuro.")
    return True


# ──────────────────────────────────────────────────────────────────────────
# UMBRAL + PERSISTENCIA EN TIEMPO REAL
# ──────────────────────────────────────────────────────────────────────────
def calibrate_th_op(df, far_max=C.FAR_MAX):
    """Calibra th_op con DOS criterios usando la definición ÚNICA de alerta
    sostenida (densidad, sustained.py). Reporta ambos lado a lado:

      - "f1"       : máximo F1 con FAR_quiet <= far_max (clásico).
      - "zero_fa"  : umbral MÁS BAJO con cero declaraciones de densidad falsas
                     durante la quietud pura. Prioriza anticipación temprana.
    """
    m = df["valid"] == 1
    p = df.loc[m, "p_alert_sm"].fillna(0).values
    y = df.loc[m, "ref_bin_op"].values
    times = df.loc[m, "Time"].values
    valid = np.ones(len(p), dtype=bool)  # ya filtrado a válidos
    quiet_mask = (df.loc[m, "Time"] < pd.Timestamp(C.QUIET_PURE_END)).values

    cand_f1 = {"th": 0.5, "f1": -1, "far": 1, "rec": 0, "prec": 0}
    zero_fa_ths = []
    for th in np.linspace(0.05, 0.95, 181):
        pred = p >= th
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        fn = np.sum(~pred & (y == 1)); tn = np.sum(~pred & (y == 0))
        far_quiet = np.sum((p[quiet_mask] >= th)) / max(quiet_mask.sum(), 1)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        if far_quiet <= far_max and f1 > cand_f1["f1"]:
            cand_f1 = {"th": float(th), "f1": f1, "far": far_quiet, "rec": rec, "prec": prec}
        # cero declaraciones de DENSIDAD en quietud
        decl_q = S.alert_density(times[quiet_mask], p[quiet_mask], th,
                                 valid=valid[quiet_mask],
                                 persist_min=C.PERSIST_MIN, density_frac=C.DENSITY_FRAC)
        if not decl_q.any():
            zero_fa_ths.append(float(th))

    cand_zero = {"th": min(zero_fa_ths) if zero_fa_ths else None}
    return {"f1": cand_f1, "zero_fa": cand_zero}


def anticipation_fresh_vs_inherited(df, th_op):
    """Para cada evento: anticipación + fresca/heredada, con SEPARACIÓN LIMPIA
    entre (1) declaración de alerta y (2) asociación evento<->alerta.

    (1) Declaración: array 'declared' por densidad >= DENSITY_FRAC en ventana
        de PERSIST_MIN (cero falsas alarmas en quietud). -> intervalos de
        alerta activa.
    (2) Asociación: un evento se considera ANTICIPADO si algún intervalo de
        alerta estuvo activo dentro de las C.ASSOC_WINDOW_H horas previas al
        evento (o seguía activo al ocurrir). Esto refleja el uso operacional:
        una alerta declarada horas antes sigue vigente aunque la señal tenga
        un valle en el minuto exacto previo.

    FRESCA / HEREDADA:
      - FRESCA: el intervalo de alerta asociado empezó DESPUÉS del evento
        anterior (alerta propia de este evento).
      - HEREDADA: el mismo intervalo ya venía activo desde antes del evento
        anterior (una sola alerta continua cubre varios episodios).
    """
    m = df["valid"] == 1
    dsub = df.loc[m, ["Time", "p_alert_sm"]].copy().reset_index(drop=True)
    times = dsub["Time"].values
    p = dsub["p_alert_sm"].fillna(0).values

    declared = S.alert_density(times, p, th_op, persist_min=C.PERSIST_MIN,
                               density_frac=C.DENSITY_FRAC)
    intervals = S.declaration_intervals(times, declared)  # (onset, fin)
    assoc = pd.Timedelta(hours=float(C.ASSOC_WINDOW_H))

    real_events = [ev for ev in C.CATALOG_2019 if ev["override"] is None]
    results = []
    prev_event_t = None
    for ev in real_events:
        ev_t = pd.Timestamp(ev["start"])
        # intervalos de alerta activos dentro de [ev_t - assoc, ev_t]
        # (fin >= inicio de ventana Y onset <= evento)
        candidates = [iv for iv in intervals
                      if iv[1] >= ev_t - assoc and iv[0] <= ev_t]
        if not candidates:
            results.append({"label": ev["label"], "anticipation_h": None,
                            "declaration": None, "fresh": None})
            prev_event_t = ev_t
            continue
        # el más temprano que aún cae en la ventana -> mayor anticipación
        onset, fin = min(candidates, key=lambda x: x[0])
        ant = (ev_t - onset).total_seconds() / 3600
        if prev_event_t is not None and onset <= prev_event_t:
            fresh = False
        else:
            fresh = True
        results.append({"label": ev["label"], "anticipation_h": round(ant, 1),
                        "declaration": str(onset), "fresh": fresh})
        prev_event_t = ev_t
    return results


def filter_display_states(df, col="pred_4c_hmm_causal",
                          min_erup_minutes=None, require_gradual=True):
    """Genera una columna 'state_display' para la VISUALIZACIÓN del estado
    multiclase, atenuando los falsos eruptivos por transitorios tectónicos.

    NO modifica pred_4c_hmm_causal (que se conserva intacto para las métricas
    y matrices de confusión del paper). Solo produce una versión limpia para
    la figura, con el criterio declarado en el caption.

    Un bloque etiquetado Eruptivo (3) se RECLASIFICA a Intranquilidad (1) en
    la visualización si es un transitorio aislado, es decir si:
      (a) dura menos de `min_erup_minutes` (los sismos tectónicos son cortos
          frente a una fase eruptiva sostenida), O
      (b) `require_gradual` y el bloque NO va precedido de Pre-eruptivo (2)
          dentro de una ventana previa — una erupción real transita
          gradualmente Quietud->Intranq->Pre->Eruptivo; un sismo tectónico
          salta directo a Eruptivo sin pre-eruptivo.

    Los bloques eruptivos que SÍ son sostenidos y con transición gradual
    (la crisis real) se conservan sin cambios.
    """
    if min_erup_minutes is None:
        min_erup_minutes = C.DISPLAY_MIN_ERUP_MIN
    s = df[col].values.copy()
    t = df["Time"].values
    n = len(s)
    display = s.copy()

    # identificar bloques continuos de estado Eruptivo (==3)
    i = 0
    while i < n:
        if s[i] != 3:
            i += 1
            continue
        j = i
        while j < n and s[j] == 3:
            j += 1
        # bloque eruptivo [i, j)
        dur_min = (pd.Timestamp(t[j-1]) - pd.Timestamp(t[i])).total_seconds() / 60
        # ¿hubo pre-eruptivo (2) en la ventana previa (min_erup_minutes)?
        look_start = max(0, i - int(min_erup_minutes))
        had_pre = np.any(s[look_start:i] == 2)
        is_transient = (dur_min < min_erup_minutes) or (require_gradual and not had_pre)
        if is_transient:
            display[i:j] = 1   # reclasificar a Intranquilidad en la figura
        i = j

    df["state_display"] = display
    n_reclass = int((df[col].values == 3).sum() - (display == 3).sum())
    return df, n_reclass


def declare_unrest(df):
    """Declaración de INTRANQUILIDAD (unrest), separada de la alerta eruptiva.

    Usa P(>=unrest) = 1 - P(quiescence), suavizada (p_unrest_sm), con:
      - th_unrest = percentil 99.5 de P(>=unrest) durante la quietud pura de
        2019 (mismo criterio que el paper MATLAB).
      - misma definición de sostenida (densidad 85% en ventana de 6h).
    Devuelve (th_unrest, onset de la primera declaración, array unrest_alert).

    El primer evento eruptivo (24-jun 12:20) sirve de referencia para la
    anticipación; el enjambre VT del IGP (18-jun) para el contraste.
    """
    m = df["valid"] == 1
    t = df.loc[m, "Time"].values
    pu = df.loc[m, "p_unrest_sm"].fillna(0).values
    quiet_mask = (df.loc[m, "Time"] < pd.Timestamp(C.QUIET_PURE_END)).values

    # th_unrest = P99.5 de la quietud pura
    pu_quiet = pu[quiet_mask]
    th_unrest = float(np.percentile(pu_quiet, 99.5)) if len(pu_quiet) else 0.5

    # declaración por densidad (misma definición de sostenida)
    declared = S.alert_density(t, pu, th_unrest, persist_min=C.PERSIST_MIN,
                               density_frac=C.DENSITY_FRAC)
    intervals = S.declaration_intervals(t, declared)

    # falsas declaraciones en quietud pura
    decl_q = S.alert_density(t[quiet_mask], pu[quiet_mask], th_unrest,
                             persist_min=C.PERSIST_MIN, density_frac=C.DENSITY_FRAC)
    n_false_quiet = len(S.declaration_intervals(t[quiet_mask], decl_q))

    # primera declaración fuera de la quietud pura (la operacionalmente real)
    first_onset = None
    for on, off in intervals:
        if on >= pd.Timestamp(C.QUIET_PURE_END):
            first_onset = on
            break
    if first_onset is None and intervals:
        first_onset = intervals[0][0]

    # mapear declared de vuelta al df completo (0 en inválidos)
    unrest_alert_full = np.zeros(len(df), dtype=int)
    unrest_alert_full[m.values] = declared.astype(int)

    return th_unrest, first_onset, unrest_alert_full, n_false_quiet


def run():
    print("=" * 70)
    print("ETAPA 6 — HMM + UMBRALES + ALERTA")
    print("=" * 70)

    print("\n  Test de causalidad de Viterbi...")
    test_viterbi_causal()
    print("  ✓ Viterbi causal confirmado (no usa el futuro).")

    df = pd.read_parquet(C.INTERMEDIATE / "validation_2019.parquet")

    # ── HMM ──
    print("\n  Aplicando HMM (causal + retrospectivo)...")
    causal, retro = run_hmm(df)
    df["pred_4c_hmm_causal"] = causal
    df["pred_4c_hmm_viterbi"] = retro

    # ── Filtro de visualización del estado (transitorios tectónicos) ──
    df, n_reclass = filter_display_states(df)
    print(f"    Filtro de visualización: {n_reclass:,} min eruptivos aislados "
          f"reclasificados a Intranquilidad en la figura (transitorios "
          f"tectónicos; no afecta métricas).")

    # ── Umbral operacional (dos criterios, elección consciente) ──
    print("\n  Calibrando th_op (dos criterios)...")
    cands = calibrate_th_op(df)
    f1c = cands["f1"]
    zfc = cands["zero_fa"]
    print(f"    Criterio F1 (máx F1, FAR<=10%):  th_op = {f1c['th']:.4f}  "
          f"(F1={f1c['f1']:.3f}, Rec={f1c['rec']:.3f}, Prec={f1c['prec']:.3f})")
    if zfc["th"] is not None:
        print(f"    Criterio cero-FA-sostenidas:     th_op = {zfc['th']:.4f}  "
              f"(umbral más bajo sin falsas declaraciones de 6h en quietud)")
    else:
        print(f"    Criterio cero-FA-sostenidas:     ninguno alcanza cero FA")

    # elegir según config
    if C.TH_OP_CRITERION == "zero_fa" and zfc["th"] is not None:
        th_op = zfc["th"]
        print(f"    -> Adoptado (config='zero_fa'): th_op = {th_op:.4f}")
    else:
        th_op = f1c["th"]
        print(f"    -> Adoptado (config='f1'): th_op = {th_op:.4f}")

    best = f1c  # para el reporte de métricas

    # ── Alerta operacional con persistencia real ──
    # alert_op usa la definición de densidad (misma que anticipación)
    _decl = S.alert_density(df["Time"].values, df["p_alert_sm"].fillna(0).values,
                            th_op, valid=(df["valid"].values==1),
                            persist_min=C.PERSIST_MIN, density_frac=C.DENSITY_FRAC)
    df["alert_op"] = _decl.astype(int)

    # ── Anticipación por evento (fresca vs heredada) ──
    print("\n  Anticipación por evento (persistencia 6h, tiempo real):")
    ant = anticipation_fresh_vs_inherited(df, th_op)
    n_fresh = sum(1 for a in ant if a["fresh"])
    n_detected = sum(1 for a in ant if a["anticipation_h"] is not None)
    for a in ant:
        if a["anticipation_h"] is not None:
            tag = "FRESCA" if a["fresh"] else "heredada"
            print(f"    {a['label']:<16s}: {a['anticipation_h']:>6.1f} h  ({tag})")
        else:
            print(f"    {a['label']:<16s}:  no detectada")
    print(f"\n    Detectados: {n_detected}/9 episodios "
          f"({n_fresh} con declaración fresca propia)")

    # ── Declaración de INTRANQUILIDAD (unrest) ──
    print("\n  Declaración de intranquilidad (unrest)...")
    th_unrest, unrest_onset, unrest_alert_full, n_false_u = declare_unrest(df)
    df["unrest_alert"] = unrest_alert_full
    igp_vt = pd.Timestamp(2019, 6, 18, 0, 0)       # enjambre VT del IGP
    first_event = pd.Timestamp(2019, 6, 24, 12, 20)  # primer episodio eruptivo
    print(f"    th_unrest (P99.5 de quietud) = {th_unrest:.4f}")
    if unrest_onset is not None:
        ant_ev = (first_event - unrest_onset).total_seconds() / 3600
        d_igp = (unrest_onset - igp_vt).total_seconds() / 3600
        print(f"    Unrest declarado: {unrest_onset}")
        print(f"      -> {ant_ev:.1f} h antes del primer episodio (24-jun 12:20)")
        print(f"      -> {d_igp:.1f} h respecto al enjambre VT del IGP (18-jun)")
    else:
        print(f"    No se declaró unrest sostenido.")
    print(f"    Falsas declaraciones de unrest en quietud pura: {n_false_u}")

    # ── Guardar CSV final con las 26 columnas ──
    col_order = ["Time", "entropy_val", "kurtosis_val", "freqidx_val", "ssam_val",
                 "p_alert", "p_alert_sm", "p_alert_freeze", "p_alert_sm_freeze",
                 "alert_op", "alert_you_op", "alert_05", "pred_4c",
                 "p_c0", "p_c1", "p_c2", "p_c3",
                 "pred_4c_hmm_causal", "pred_4c_hmm_viterbi",
                 "p_ge_unrest", "p_unrest_sm", "unrest_alert",
                 "ref_4class", "ref_bin_amp", "ref_bin_op", "valid"]
    df_out = df[col_order].copy()
    # columna extra para la figura de estado (no forma parte de las 26
    # canónicas; las figuras la leen por nombre si la necesitan)
    if "state_display" in df.columns:
        df_out["state_display"] = df["state_display"].values
    df_out["Time"] = pd.to_datetime(df_out["Time"]).dt.strftime("%d-%b-%Y %H:%M:%S")

    out_csv = C.BASE_DATA / "ubinas_v16_ub4_2019.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n  Guardado CSV final: {out_csv}  ({len(df_out.columns)} columnas)")

    # reporte de umbrales/anticipaciones
    report = {"th_op": th_op, "th_op_criterion": C.TH_OP_CRITERION,
              "candidates": cands, "anticipations": ant,
              "n_detected": n_detected, "n_fresh": n_fresh,
              "th_unrest": th_unrest,
              "unrest_onset": str(unrest_onset) if unrest_onset is not None else None,
              "unrest_false_quiet": n_false_u}
    with open(C.INTERMEDIATE / "postprocess_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nETAPA 6 completada.")
    return df_out, report


if __name__ == "__main__":
    run()
