"""
Barrido de th_op — VERSIÓN UNIFICADA con sustained.py

Usa EXACTAMENTE la misma definición de alerta que el pipeline (postprocess.py):
  - Declaración de alerta: densidad >= DENSITY_FRAC en ventana de PERSIST_MIN
    (criterio de densidad, tolera micro-huecos, cero falsas alarmas en quietud).
  - Anticipación de evento: alerta activa dentro de ASSOC_WINDOW_H horas
    previas al evento.

Antes este script usaba su propia lógica de "racha continua", que divergía del
pipeline. Ahora comparten sustained.py -> los números coinciden siempre.

Requiere que config.py y sustained.py estén en el mismo directorio (o en el
PYTHONPATH). Uso:
    python sweep_threshold_v2.py ubinas_v16_ub4_2019.csv
"""
import sys
import os
import numpy as np
import pandas as pd

# importar config y sustained del pipeline (mismo directorio que este script)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import sustained as S

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "ubinas_v16_ub4_2019.csv"
COL_TIME = "Time"
COL_PALERT = "p_alert_sm"
COL_VALID = "valid"
COL_REF = "ref_bin_op"

# eventos reales del catálogo (excluye override)
EVENTS = [(ev["label"], pd.Timestamp(ev["start"]))
          for ev in C.CATALOG_2019 if ev["override"] is None]
QUIET_END = pd.Timestamp(C.QUIET_PURE_END)

print(f"Cargando {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
try:
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], format="%d-%b-%Y %H:%M:%S")
except ValueError:
    df[COL_TIME] = pd.to_datetime(df[COL_TIME])
df = df.sort_values(COL_TIME).reset_index(drop=True)

if COL_VALID in df.columns:
    df = df[df[COL_VALID] == 1].reset_index(drop=True)
print(f"Filas válidas: {len(df):,}")

t = df[COL_TIME].values
p = df[COL_PALERT].fillna(0).values
quiet_mask = (df[COL_TIME] < QUIET_END).values
y = df[COL_REF].values if COL_REF in df.columns else None
if y is None:
    print(f"AVISO: no hay columna '{COL_REF}'; F1/Prec/Rec quedarán vacíos.")

assoc = pd.Timedelta(hours=C.ASSOC_WINDOW_H)
print(f"Config: th por densidad {C.DENSITY_FRAC:.0%} en ventana {C.PERSIST_MIN} min, "
      f"asociación evento±alerta = {C.ASSOC_WINDOW_H} h")


def metrics_at(th):
    if y is None:
        return None
    pred = p >= th
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1)); tn = np.sum(~pred & (y == 0))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    far  = fp / (fp + tn) if (fp + tn) else 0.0
    return dict(F1=f1, Prec=prec, Rec=rec, FAR=far)


def alert_intervals(th, mask=None):
    """Intervalos de alerta declarada (densidad) para un umbral, opcionalmente
    restringido a una máscara temporal (p.ej. quietud)."""
    tt = t if mask is None else t[mask]
    pp = p if mask is None else p[mask]
    if len(tt) == 0:
        return []
    declared = S.alert_density(tt, pp, th, persist_min=C.PERSIST_MIN,
                               density_frac=C.DENSITY_FRAC)
    return S.declaration_intervals(tt, declared)


def false_alarms_in_quiescence(th):
    return len(alert_intervals(th, mask=quiet_mask))


def anticipation_coverage(th):
    """Cobertura de eventos: alerta activa dentro de ASSOC_WINDOW_H previas."""
    intervals = alert_intervals(th)
    out = []
    for _, ev_t in EVENTS:
        cands = [iv for iv in intervals if iv[1] >= ev_t - assoc and iv[0] <= ev_t]
        if cands:
            onset = min(cands, key=lambda x: x[0])[0]
            out.append((ev_t - onset).total_seconds() / 3600)
        else:
            out.append(None)
    n_ok = sum(1 for x in out if x is not None)
    med = float(np.median([x for x in out if x is not None])) if n_ok else np.nan
    return n_ok, med, out


# ── Barrido ──
candidates = sorted(set(np.round(np.arange(0.05, 0.90, 0.025), 4)))
print(f"\n{'th_op':>7s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} {'FAR':>7s} "
      f"{'FA_quiet':>9s} {'N_ok/9':>7s} {'mediana_h':>10s}")
rows = []
for th in candidates:
    m = metrics_at(th)
    fa = false_alarms_in_quiescence(th)
    n_ok, med, _ = anticipation_coverage(th)
    f1s  = f"{m['F1']:.3f}"  if m else "   -   "
    precs= f"{m['Prec']:.3f}" if m else "   -   "
    recs = f"{m['Rec']:.3f}" if m else "   -   "
    fars = f"{m['FAR']:.3f}" if m else "   -   "
    rows.append((th, m, fa, n_ok, med))
    med_s = f"{med:9.1f}" if not np.isnan(med) else "        -"
    print(f"{th:7.3f} {f1s:>7s} {precs:>7s} {recs:>7s} {fars:>7s} "
          f"{fa:9d} {n_ok:>4d}/9   {med_s}")

# ── Candidatos ──
print("\n=== Candidato A: umbral mínimo con CERO falsas alarmas en quietud ===")
zero_fa = [r for r in rows if r[2] == 0]
if zero_fa:
    th_a = min(zero_fa, key=lambda r: r[0])
    print(f"th_op = {th_a[0]:.4f}  ->  cobertura {th_a[3]}/9 eventos, "
          f"mediana {th_a[4]:.1f} h" + (f", F1={th_a[1]['F1']:.3f}" if th_a[1] else ""))
else:
    print("Ningún candidato del grid tiene cero falsas alarmas.")

print("\n=== Candidato B: máximo F1 ===")
with_f1 = [r for r in rows if r[1] is not None]
if with_f1:
    th_b = max(with_f1, key=lambda r: r[1]['F1'])
    print(f"th_op = {th_b[0]:.4f}  ->  F1={th_b[1]['F1']:.3f}, "
          f"cobertura {th_b[3]}/9 eventos, mediana {th_b[4]:.1f} h")

# ── Detalle por evento para el candidato A ──
if zero_fa:
    print(f"\n=== Detalle por evento con th_op={th_a[0]:.4f} (Candidato A) ===")
    _, _, detail = anticipation_coverage(th_a[0])
    for (name, ev_t), h in zip(EVENTS, detail):
        s = f"{h:.1f} h" if h is not None else "NO detectado"
        print(f"  {name:<14s} {ev_t}  ->  {s}")

out_df = pd.DataFrame([(r[0], r[2], r[3], r[4]) for r in rows],
                      columns=['th_op', 'FA_quiescence', 'N_events_ok', 'median_h'])
out_df.to_csv("threshold_sweep_result_v3.csv", index=False)
print("\nResultado exportado a threshold_sweep_result_v3.csv")
