"""
================================================================================
 ETAPA 2 — label.py : ETIQUETADO DE ESTADOS VOLCÁNICOS (2019 y 2023)
================================================================================
Asigna a cada minuto uno de los 4 estados y una etiqueta binaria operacional,
según el catálogo correspondiente. Misma lógica para 2019 (validación) y 2023
(entrenamiento), parametrizada por año.

CHECK CRÍTICO: cada evento real del catálogo DEBE producir > 0 minutos
eruptivos, o el pipeline se detiene nombrando el evento.
================================================================================
"""
from datetime import datetime
import pandas as pd
import numpy as np
import config as C


class LabelError(Exception):
    pass


def label_series(times, catalog, unrest_on, calm_resume):
    """Etiqueta un vector de tiempos según un catálogo.
    Devuelve (state_4c array, per_event_counts list)."""
    T = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    state = np.full(len(T), C.STATE_QUIET, dtype=int)

    unrest_mask = (T >= pd.Timestamp(unrest_on)) & (T < pd.Timestamp(calm_resume))
    state[unrest_mask.values] = C.STATE_UNREST
    state[(T >= pd.Timestamp(calm_resume)).values] = C.STATE_UNREST

    per_event = []
    for ev in catalog:
        ev_start = pd.Timestamp(ev["start"])
        ev_end   = pd.Timestamp(ev["end"])
        if ev["pre_h"] > 0:
            pre_start = ev_start - pd.Timedelta(hours=ev["pre_h"])
            pre_mask = ((T >= pre_start) & (T < ev_start)).values & (state != C.STATE_ERUPT)
            state[pre_mask] = C.STATE_PRE
        er_mask = ((T >= ev_start) & (T <= ev_end)).values
        if ev["override"] is not None:
            state[er_mask] = ev["override"]
            n_er = 0
        else:
            state[er_mask] = C.STATE_ERUPT
            n_er = int(er_mask.sum())
        per_event.append({"label": ev["label"], "type": ev["type"],
                          "pre_h": ev["pre_h"], "n_eruptive": n_er,
                          "override": ev["override"]})
    return state, per_event


def _report_and_check(df_lab, per_event, year_label):
    print(f"\n  [{year_label}] Etiquetado por evento:")
    print(f"  {'Evento':<22s} {'Tipo':<10s} {'pre_h':>5s} {'N_eruptivo':>11s}")
    zero_events = []
    n_shown = 0
    for e in per_event:
        is_zero = (e["override"] is None and e["n_eruptive"] == 0)
        if is_zero:
            zero_events.append(e["label"])
        if n_shown < 12 or is_zero:
            marker = "  <-- CERO! ERROR" if is_zero else ""
            ov = f" (override={e['override']})" if e["override"] is not None else ""
            print(f"  {e['label']:<22s} {e['type']:<10s} {e['pre_h']:>5d} "
                  f"{e['n_eruptive']:>11d}{marker}{ov}")
            n_shown += 1
    if len(per_event) > 12:
        print(f"  ... ({len(per_event)-12} eventos mas)")

    if zero_events:
        raise LabelError(
            f"[{year_label}] ETIQUETADO EN CERO para: " + ", ".join(zero_events) +
            "\nEl catalogo no cae dentro del rango de datos o las fechas no "
            "coinciden. Pipeline detenido a proposito.")

    total = len(df_lab)
    print(f"\n  [{year_label}] Distribucion de estados:")
    for s in [0, 1, 2, 3]:
        n = int((df_lab["state_4c"] == s).sum())
        print(f"    {C.STATE_NAMES[s]:<16s}: {n:>9,} ({100*n/total:.2f}%)")
    for s in [C.STATE_PRE, C.STATE_ERUPT]:
        if (df_lab["state_4c"] == s).sum() == 0:
            raise LabelError(f"[{year_label}] Clase '{C.STATE_NAMES[s]}' con "
                             f"CERO muestras totales.")


def run():
    print("=" * 70)
    print("ETAPA 2 — ETIQUETADO DE ESTADOS (2019 y 2023)")
    print("=" * 70)

    # ── 2019 (validacion) ──
    p19 = C.INTERMEDIATE / "ub4_2019.parquet"
    if not p19.exists():
        raise LabelError(f"No existe {p19}. Corre load.py primero.")
    df19 = pd.read_parquet(p19)
    df19 = df19[df19["Time"] >= pd.Timestamp(C.T19_START)].copy().reset_index(drop=True)
    st19, pe19 = label_series(df19["Time"], C.CATALOG_2019,
                              datetime(2019, 6, 18, 0, 0), C.T19_CALM_RESUME)
    df19["state_4c"] = st19
    df19["label_bin_op"] = np.isin(st19, [C.STATE_PRE, C.STATE_ERUPT]).astype(int)
    _report_and_check(df19, pe19, "2019")
    out19 = C.INTERMEDIATE / "ub4_2019_labeled.parquet"
    df19.to_parquet(out19, index=False)
    print(f"  Guardado: {out19}")

    # ── 2023 (entrenamiento) ──
    p23 = C.INTERMEDIATE / "stations_2023.parquet"
    if not p23.exists():
        raise LabelError(f"No existe {p23}. Corre load.py primero.")
    df23 = pd.read_parquet(p23)
    catalog23 = C.load_catalog_2023()
    print(f"\n  Catalogo 2023 cargado: {len(catalog23)} eventos "
          f"({sum(1 for e in catalog23 if e['type']=='explosion')} exp, "
          f"{sum(1 for e in catalog23 if e['type']=='emision')} emi)")

    labeled_parts = []
    pe23_ref = None
    first_st = df23["station"].iloc[0]
    for st, g in df23.groupby("station", sort=False):
        g = g.sort_values("Time").reset_index(drop=True)
        st23, pe23 = label_series(g["Time"], catalog23,
                                  C.T23_UNREST_ON, C.T23_CALM_RESUME)
        g["state_4c"] = st23
        g["label_bin_op"] = np.isin(st23, [C.STATE_PRE, C.STATE_ERUPT]).astype(int)
        labeled_parts.append(g)
        if pe23_ref is None:
            pe23_ref = pe23
    df23_lab = pd.concat(labeled_parts, ignore_index=True)
    _report_and_check(df23_lab[df23_lab["station"] == first_st], pe23_ref, "2023")
    out23 = C.INTERMEDIATE / "stations_2023_labeled.parquet"
    df23_lab.to_parquet(out23, index=False)
    print(f"  Guardado: {out23}")

    print("\nETAPA 2 completada — 2019 y 2023 etiquetados, sin clases en cero.")
    return df19, df23_lab


if __name__ == "__main__":
    run()
