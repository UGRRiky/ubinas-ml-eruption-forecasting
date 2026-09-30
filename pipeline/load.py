"""
================================================================================
 ETAPA 1 — load.py : CARGA Y NORMALIZACIÓN
================================================================================
Lee los CSV crudos y produce series limpias en grilla de 1 minuto, con una
columna 'valid' booleana.

CLAVE — manejo de timestamps: tus datos vienen con el segundo en :59
(ej. "01-Jan-2023 00:00:59"). NO es duplicación (confirmado: cero filas en
otros segundos). Se normaliza cada timestamp al minuto que le corresponde
haciendo floor a minuto, SIN descartar ninguna fila. Este es exactamente el
punto donde el pipeline anterior se rompió (un filtro second()==0 mal puesto
descartaba todo).

Salidas:
  intermediate/stations_2023.parquet   (las 3 estaciones apiladas)
  intermediate/ub4_2019.parquet        (UB4 extendido 2018-2019)

Checks que fallan ruidosamente:
  - resolución real = 1 min
  - nº de filas coherente con el rango de fechas
  - % de válidos por estación dentro de rango razonable
  - sin timestamps duplicados tras normalizar
================================================================================
"""
import sys
import pandas as pd
import numpy as np
import config as C


class LoadError(Exception):
    """Error de carga que debe detener el pipeline ruidosamente."""
    pass


def load_one_station(name, filepath, t_start, t_end, verbose=True):
    """Carga un CSV crudo, normaliza a grilla de 1 min, marca validez.

    Devuelve un DataFrame con:
      columnas: Time (datetime, en grilla de minuto), entropy_val,
                kurtosis_val, freqidx_val, ssam_val, station, valid
    """
    if verbose:
        print(f"  --- {name} ---")
    if not filepath.exists():
        raise LoadError(f"[{name}] Archivo no encontrado: {filepath}")

    # Leer solo las columnas que necesitamos (más rápido, menos memoria)
    usecols = [C.TIME_COL] + C.PARAM_COLS + [C.GAP_COL]
    df = pd.read_csv(filepath, usecols=lambda c: c in usecols)

    # ── Parseo de fecha con el formato exacto de tus CSV ──
    df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL], format=C.TIME_FORMAT,
                                    errors="coerce")
    n_unparsed = df[C.TIME_COL].isna().sum()
    if n_unparsed > 0:
        raise LoadError(
            f"[{name}] {n_unparsed} timestamps no se pudieron parsear con "
            f"formato '{C.TIME_FORMAT}'. Revisa el formato de fecha del CSV.")

    # ── Normalización de timestamps :59 -> grilla de minuto ──
    # floor a minuto: 00:00:59 -> 00:00:00, sin descartar filas.
    seconds_present = df[C.TIME_COL].dt.second.value_counts()
    df["Time"] = df[C.TIME_COL].dt.floor("min")
    if verbose:
        top_sec = seconds_present.index[0]
        print(f"      Segundo dominante en timestamps: :{top_sec:02d} "
              f"({100*seconds_present.iloc[0]/len(df):.1f}%) "
              f"-> normalizado a grilla de minuto")

    # ── Detección de duplicados REALES (mismo minuto, más de una fila) ──
    n_dup = df["Time"].duplicated().sum()
    if n_dup > 0:
        # No deberían existir según tu confirmación; si aparecen, avisamos
        # y nos quedamos con la primera (comportamiento documentado).
        print(f"      AVISO [{name}]: {n_dup:,} minutos con más de una fila; "
              f"se conserva la primera de cada minuto.")
        df = df.drop_duplicates(subset="Time", keep="first")

    # ── Recorte al rango temporal de interés ──
    df = df[(df["Time"] >= t_start) & (df["Time"] <= t_end)].copy()

    # ── Renombrado a nombres internos ──
    df = df.rename(columns=C.RENAME_MAP)

    # ── Validez: inválido si Has_Gaps==1 o si algún parámetro es NaN ──
    param_nan = df[C.MODEL_FEATURES_BASE].isna().any(axis=1)
    gap_flag = df[C.GAP_COL] == 1 if C.GAP_COL in df.columns else False
    df["valid"] = ~(param_nan | gap_flag)

    # ── Reindexar a grilla completa de 1 min (rellena huecos con NaN) ──
    full_grid = pd.date_range(t_start, t_end, freq="1min")
    df = df.set_index("Time").reindex(full_grid)
    df.index.name = "Time"
    # las filas nuevas (huecos) quedan con valid=NaN -> False
    # (fillna sobre bool + astype evita el FutureWarning de downcasting)
    df["valid"] = df["valid"].astype("object").where(df["valid"].notna(), False).astype(bool)
    df["station"] = name

    df = df.reset_index()

    # ── CHECKS que fallan ruidosamente ──
    n_expected = len(full_grid)
    if len(df) != n_expected:
        raise LoadError(
            f"[{name}] nº de filas ({len(df)}) != minutos esperados "
            f"({n_expected}) en el rango.")

    # resolución real = 1 min
    diffs = df["Time"].diff().dropna()
    if not (diffs == pd.Timedelta(minutes=1)).all():
        bad = (diffs != pd.Timedelta(minutes=1)).sum()
        raise LoadError(f"[{name}] {bad} saltos distintos de 1 min tras "
                        f"reindexar (no debería pasar).")

    pct_valid = 100 * df["valid"].mean()
    if verbose:
        print(f"      Registros: {len(df):,}  Válidos: {df['valid'].sum():,} "
              f"({pct_valid:.1f}%)")

    # rango de validez razonable (alerta, no error, si sale de rango)
    if pct_valid < 50:
        print(f"      *** ALERTA [{name}]: solo {pct_valid:.1f}% de muestras "
              f"válidas. Revisa si es esperado (¿estación con muchos gaps?) "
              f"o si hay un problema de carga. ***")

    return df[["Time", "station", "valid"] + C.MODEL_FEATURES_BASE]


def run():
    print("=" * 70)
    print("ETAPA 1 — CARGA Y NORMALIZACIÓN")
    print("=" * 70)

    # ── 2023: las 3 estaciones ──
    print("\n[2023 — entrenamiento]")
    frames_23 = []
    for st in C.STATIONS_2023:
        dfs = load_one_station(st["name"], st["file"], C.T23_START, C.T23_END)
        frames_23.append(dfs)
    stations_2023 = pd.concat(frames_23, ignore_index=True)

    out_23 = C.INTERMEDIATE / "stations_2023.parquet"
    stations_2023.to_parquet(out_23, index=False)
    print(f"\n  Guardado: {out_23}  ({len(stations_2023):,} filas)")

    # ── 2019: UB4 extendido (con burn-in desde 2018) ──
    print("\n[2018-2019 — validación UB4 extendido]")
    ub4_2019 = load_one_station("UB4_ext", C.UB4_EXT_FILE,
                                C.T19_LOAD_START, C.T19_END)
    out_19 = C.INTERMEDIATE / "ub4_2019.parquet"
    ub4_2019.to_parquet(out_19, index=False)
    print(f"\n  Guardado: {out_19}  ({len(ub4_2019):,} filas)")

    # resumen del período de análisis real (2019, sin burn-in)
    analysis = ub4_2019[ub4_2019["Time"] >= C.T19_START]
    print(f"  Período análisis 2019: {len(analysis):,} filas, "
          f"{analysis['valid'].sum():,} válidas "
          f"({100*analysis['valid'].mean():.1f}%)")

    print("\nETAPA 1 completada.")
    return stations_2023, ub4_2019


if __name__ == "__main__":
    run()
