"""
Diagnostico: por que el criterio de densidad detecta 24-jun/23-jul pero no
18-jul/19-jul. Corre sobre el CSV real. Muestra, para cada evento, la
densidad maxima de superacion del umbral en las 6h previas, para ver si el
85% es demasiado estricto.

Uso: python diag_densidad.py ubinas_v16_ub4_2019.csv
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import config as C
import sustained as S

csv = sys.argv[1] if len(sys.argv) > 1 else "ubinas_v16_ub4_2019.csv"
df = pd.read_csv(csv)
df['Time'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S')
df = df[df['valid']==1].reset_index(drop=True)
th = 0.6200  # el th_op adoptado

t = df['Time'].values
p = df['p_alert_sm'].fillna(0).values

print(f"th_op = {th}, densidad requerida = {C.DENSITY_FRAC}, ventana = {C.PERSIST_MIN} min")
print(f"\n{'Evento':<16s} {'dens_max_6h_previa':>18s} {'p_max_6h':>10s} {'p_med_6h':>10s} {'min>th':>8s}")

real_events = [ev for ev in C.CATALOG_2019 if ev['override'] is None]
for ev in real_events:
    ev_t = pd.Timestamp(ev['start'])
    # ventana de 6h antes del evento
    win_start = ev_t - pd.Timedelta(hours=6)
    mask = (df['Time'] >= win_start) & (df['Time'] < ev_t)
    if mask.sum() == 0:
        print(f"{ev['label']:<16s} {'sin datos':>18s}")
        continue
    pw = p[mask.values]
    above = pw >= th
    dens = above.mean()  # fraccion de superacion en la ventana de 6h
    n_above = above.sum()
    print(f"{ev['label']:<16s} {dens:>18.2%} {pw.max():>10.3f} {np.median(pw):>10.3f} "
          f"{n_above:>5d}/{len(pw)}")

# Ademas: barrido de density_frac para ver cuantos eventos se detectan con
# cada nivel de tolerancia, manteniendo th=0.62
print(f"\n--- Sensibilidad al umbral de densidad (th fijo={th}) ---")
for dfrac in [0.85, 0.70, 0.60, 0.50, 0.40, 0.30]:
    decl = S.alert_density(t, p, th, persist_min=C.PERSIST_MIN, density_frac=dfrac)
    intervals = S.declaration_intervals(t, decl)
    # contar cuantos eventos tienen una declaracion activa antes
    n_det = 0
    for ev in real_events:
        ev_t = pd.Timestamp(ev['start'])
        prev = [iv for iv in intervals if iv[0] < ev_t and iv[1] >= ev_t - pd.Timedelta(hours=6)]
        if prev: n_det += 1
    # falsas en quietud
    qmask = df['Time'] < pd.Timestamp(C.QUIET_PURE_END)
    decl_q = S.alert_density(t[qmask.values], p[qmask.values], th,
                             persist_min=C.PERSIST_MIN, density_frac=dfrac)
    n_false = len(S.declaration_intervals(t[qmask.values], decl_q))
    print(f"  densidad {dfrac:.0%}: {n_det}/9 eventos detectados, "
          f"{n_false} falsas declaraciones en quietud")
