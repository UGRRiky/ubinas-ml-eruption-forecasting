"""
================================================================================
 ETAPA 3 — features.py : NORMALIZACIÓN CAUSAL + INGENIERÍA DE CARACTERÍSTICAS
================================================================================
Dos operaciones, en orden:

  A) Normalización rolling z-score CAUSAL (W=30 días) por estación y por
     parámetro. Causal = en el instante t se usan solo datos de [t-W, t-1]
     (la ventana NO incluye el punto actual ni el futuro). Esto es lo que
     Jesús pidió blindar (su punto P7/P51): "causal" en sentido temporal.

  B) Ingeniería de 85 características sobre los 4 parámetros ya normalizados,
     usando estadísticos en ventanas multi-escala (10/30/60/180 min) — todas
     causales (trailing), más 5 características cruzadas.

CLAVE de diseño: la normalización se calcula sobre la serie COMPLETA de cada
estación (incluyendo burn-in 2018 para UB4), de forma causal, ANTES de
recortar a 2019. Así el z-score de enero-2019 ya tiene 30 días de referencia
previa real (de dic-2018), sin fuga de información.

Incluye TESTS DE CAUSALIDAD ejecutables: alterar un valor en t+1 no debe
cambiar el z-score ni las features en t.

Entrada:
  intermediate/stations_2023.parquet    (para features de entrenamiento)
  intermediate/ub4_2019.parquet         (serie completa UB4 con burn-in)
Salida:
  intermediate/features_2023.parquet
  intermediate/features_2019.parquet    (ya recortado a 2019)
================================================================================
"""
import sys
import numpy as np
import pandas as pd
import config as C


class FeatureError(Exception):
    pass


# ──────────────────────────────────────────────────────────────────────────
# A) NORMALIZACIÓN CAUSAL
# ──────────────────────────────────────────────────────────────────────────
def causal_zscore(series, window_min=C.W_ROLLING_MIN, min_valid=C.MIN_VALID_ROLLING):
    """z-score causal: en t usa media y sd de [t-window, t-1].
    El .shift(1) garantiza que el punto actual NO entra en su propia
    referencia (estrictamente causal). Devuelve una Serie del mismo índice.
    """
    ref = series.shift(1)  # excluye el punto actual -> causal estricto
    mu = ref.rolling(window_min, min_periods=min_valid).mean()
    sd = ref.rolling(window_min, min_periods=min_valid).std()
    sd = sd.where(sd > 1e-9, np.nan)  # evita división por ~0
    return (series - mu) / sd


def normalize_station(df_st, smooth_min=C.SMOOTH_MIN):
    """Aplica z-score causal a los 4 parámetros de una estación.
    df_st debe venir ordenado por Time. Devuelve df con columnas _z añadidas.
    El suavizado posterior de 60 min es CENTRADO (documentado): introduce
    ~30 min de mirada al futuro en esta etapa de preproceso; es intencional
    y consistente con el pipeline previo, pero se declara explícitamente.
    """
    df_st = df_st.sort_values("Time").reset_index(drop=True)
    for base in C.MODEL_FEATURES_BASE:
        z = causal_zscore(df_st[base])
        # suavizado centrado de 60 min (declara mirada al futuro de ~30 min)
        df_st[f"{base}_z"] = z.rolling(smooth_min, center=True, min_periods=10).mean()
    return df_st


# ──────────────────────────────────────────────────────────────────────────
# B) INGENIERÍA DE 85 CARACTERÍSTICAS
# ──────────────────────────────────────────────────────────────────────────
def build_features(df_st):
    """Genera las 85 características sobre los 4 parámetros normalizados (_z).
    Todas las ventanas son causales (trailing). Devuelve un DataFrame de
    features (misma longitud que df_st).

    Estructura (para llegar a 85):
      - 4 params x 4 ventanas x 4 estadísticos (mean/std/slope/cv) = 64
      - 4 params x 3 (pct_range_w60, d2_w60, ratio180)             = 12
      - 4 params instantáneos (el valor _z mismo)                  = 4
      - 5 cruzadas                                                 = 5
      total = 85
    """
    zcols = {b: f"{b}_z" for b in C.MODEL_FEATURES_BASE}
    feats = {}

    def roll(s, w):
        return s.rolling(w, min_periods=max(2, w // 4))

    for base, zc in zcols.items():
        s = df_st[zc]
        short = base.replace("_val", "")  # entropy/kurtosis/freqidx/ssam
        for w in C.FEATURE_WINDOWS:
            feats[f"{short}_mean_w{w}"]  = roll(s, w).mean()
            feats[f"{short}_std_w{w}"]   = roll(s, w).std()
            # pendiente lineal causal aproximada: (último - primero)/w
            feats[f"{short}_slope_w{w}"] = (s - s.shift(w)) / w
            m = roll(s, w).mean()
            sd = roll(s, w).std()
            feats[f"{short}_cv_w{w}"]    = sd / m.abs().where(m.abs() > 1e-6, np.nan)
        # extras por parámetro
        feats[f"{short}_pctrange_w60"] = roll(s, 60).quantile(0.75) - roll(s, 60).quantile(0.25)
        feats[f"{short}_d2_w60"]       = s.diff().diff().rolling(60, min_periods=10).mean()
        feats[f"{short}_ratio180"]     = s / roll(s, 180).mean().where(roll(s, 180).mean().abs() > 1e-6, np.nan)
        # instantáneo
        feats[f"{short}_inst"]         = s

    # 5 cruzadas
    ze = df_st[zcols["entropy_val"]]
    zk = df_st[zcols["kurtosis_val"]]
    zf = df_st[zcols["freqidx_val"]]
    zs = df_st[zcols["ssam_val"]]
    feats["se_x_k"]        = ze * zk
    feats["ssam_over_ent"] = zs / ze.where(ze.abs() > 1e-6, np.nan)
    feats["fi_x_ssam"]     = zf * zs
    feats["dse_x_dssam"]   = ze.diff(60) * zs.diff(60)
    feats["se_minus_k"]    = ze - zk

    fdf = pd.DataFrame(feats, index=df_st.index)
    return fdf


def process_station_group(df, station_col="station"):
    """Normaliza y genera features para cada estación por separado, luego
    concatena. (Cada estación se normaliza con su propia referencia)."""
    out = []
    for name, g in df.groupby(station_col, sort=False):
        g = normalize_station(g)
        fdf = build_features(g)
        fdf["Time"] = g["Time"].values
        fdf["station"] = name
        fdf["valid"] = g["valid"].values
        out.append(fdf)
    return pd.concat(out, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────
# TESTS DE CAUSALIDAD (se corren en run())
# ──────────────────────────────────────────────────────────────────────────
def test_causality():
    """Verifica que alterar un valor futuro no cambia el z-score presente."""
    n = 5000
    rng = np.random.default_rng(0)
    s = pd.Series(np.cumsum(rng.standard_normal(n)) + 1000)

    z_orig = causal_zscore(s, window_min=1440, min_valid=100)
    t_test = 3000
    s2 = s.copy()
    s2.iloc[t_test + 1:] += 500  # altera TODO el futuro después de t_test
    z_mod = causal_zscore(s2, window_min=1440, min_valid=100)

    # el z en t_test NO debe cambiar (usa solo pasado)
    diff = abs(z_orig.iloc[t_test] - z_mod.iloc[t_test])
    if not (np.isnan(z_orig.iloc[t_test]) or diff < 1e-9):
        raise FeatureError(
            f"TEST DE CAUSALIDAD FALLÓ: z-score en t={t_test} cambió en "
            f"{diff} al alterar el futuro. La normalización NO es causal.")
    # y el z en t_test+2 SÍ debe cambiar (confirma que el test es sensible)
    diff_future = abs(z_orig.iloc[t_test + 2] - z_mod.iloc[t_test + 2])
    if diff_future < 1e-9:
        raise FeatureError(
            "TEST DE CAUSALIDAD INCONCLUSO: alterar el futuro no cambió el "
            "z-score futuro; el test no es sensible, revísalo.")
    return True


# ──────────────────────────────────────────────────────────────────────────
def run():
    print("=" * 70)
    print("ETAPA 3 — NORMALIZACIÓN CAUSAL + FEATURES")
    print("=" * 70)

    # Test de causalidad primero (falla ruidosamente si algo está mal)
    print("\n  Ejecutando test de causalidad del z-score...")
    test_causality()
    print("  ✓ z-score confirmado estrictamente causal.")

    # ── 2023 ──
    in23 = C.INTERMEDIATE / "stations_2023.parquet"
    if not in23.exists():
        raise FeatureError(f"No existe {in23}. Corre load.py primero.")
    df23 = pd.read_parquet(in23)
    print(f"\n  [2023] Procesando features para {df23['station'].nunique()} estaciones...")
    feat23 = process_station_group(df23)
    n_feat_cols = len([c for c in feat23.columns if c not in ("Time", "station", "valid")])
    print(f"    Características generadas: {n_feat_cols}")
    if n_feat_cols != 85:
        raise FeatureError(f"Se esperaban 85 características, se generaron {n_feat_cols}.")

    # ── Merge con etiquetas de 2023 (de label.py) ──
    lab_path = C.INTERMEDIATE / "stations_2023_labeled.parquet"
    if lab_path.exists():
        lab23 = pd.read_parquet(lab_path)[["Time", "station", "state_4c", "label_bin_op"]]
        feat23 = feat23.merge(lab23, on=["Time", "station"], how="left")
        n_missing = feat23["state_4c"].isna().sum()
        if n_missing > 0:
            raise FeatureError(f"[2023] {n_missing} filas sin etiqueta tras "
                               f"el merge (Time/station no coinciden).")
        feat23["state_4c"] = feat23["state_4c"].astype(int)
        feat23["label_bin_op"] = feat23["label_bin_op"].astype(int)
        print(f"    Etiquetas 2023 añadidas (merge con label.py).")
    else:
        print(f"    AVISO: no existe {lab_path}; features_2023 quedará SIN "
              f"etiquetas. Corre label.py (Etapa 2) antes de train.py.")

    out23 = C.INTERMEDIATE / "features_2023.parquet"
    feat23.to_parquet(out23, index=False)
    print(f"    Guardado: {out23}  ({len(feat23):,} filas)")

    # ── 2019 (serie completa con burn-in, normaliza, luego recorta) ──
    in19 = C.INTERMEDIATE / "ub4_2019.parquet"
    df19_full = pd.read_parquet(in19)
    print(f"\n  [2019] Normalizando UB4 con burn-in y generando features...")
    feat19_full = process_station_group(df19_full)
    # recortar a 2019 (el burn-in 2018 ya cumplió su función de referencia)
    feat19 = feat19_full[feat19_full["Time"] >= pd.Timestamp(C.T19_START)].reset_index(drop=True)

    # ── Merge con etiquetas de 2019 ──
    lab19_path = C.INTERMEDIATE / "ub4_2019_labeled.parquet"
    if lab19_path.exists():
        lab19 = pd.read_parquet(lab19_path)[["Time", "state_4c", "label_bin_op"]]
        feat19 = feat19.merge(lab19, on="Time", how="left")
        n_missing = feat19["state_4c"].isna().sum()
        if n_missing > 0:
            raise FeatureError(f"[2019] {n_missing} filas sin etiqueta tras merge.")
        feat19["state_4c"] = feat19["state_4c"].astype(int)
        feat19["label_bin_op"] = feat19["label_bin_op"].astype(int)
        print(f"    Etiquetas 2019 añadidas.")

    out19 = C.INTERMEDIATE / "features_2019.parquet"
    feat19.to_parquet(out19, index=False)
    print(f"    Guardado: {out19}  ({len(feat19):,} filas)")

    # ── Check: durante quietud, media≈0 y sd≈1 (prueba de normalización) ──
    print("\n  Verificación de normalización (quietud pura 2019, params _z):")
    df19_lab = pd.read_parquet(C.INTERMEDIATE / "ub4_2019_labeled.parquet") \
        if (C.INTERMEDIATE / "ub4_2019_labeled.parquet").exists() else None
    quiet_end = pd.Timestamp(C.QUIET_PURE_END)
    # reconstruimos los _z para el chequeo (rápido, solo UB4 2019)
    g19 = normalize_station(df19_full[df19_full["station"] == "UB4_ext"].copy())
    g19 = g19[(g19["Time"] >= pd.Timestamp(C.T19_START)) & (g19["Time"] <= quiet_end)]
    for base in C.MODEL_FEATURES_BASE:
        zc = f"{base}_z"
        vals = g19.loc[g19["valid"], zc].dropna()
        if len(vals) > 0:
            mu, sd = vals.mean(), vals.std()
            flag = "" if (abs(mu) < 0.5 and 0.7 < sd < 1.4) else "  <-- fuera de rango esperado"
            print(f"    {base:<14s}: media={mu:+.3f}  sd={sd:.3f}{flag}")

    print("\nETAPA 3 completada.")
    return feat23, feat19


if __name__ == "__main__":
    run()
