"""
Barrido de la temperatura de emisión del HMM. Para cada T, calcula:
  - falsos eruptivos en quietud (deben BAJAR mucho)
  - oscilaciones de estado totales (deben bajar ~90%)
  - saltos prohibidos Q<->E (deben bajar)
  - que NO destruya la deteccion de los eventos reales (estado 3 durante eventos)

Elige el T que limpia la quietud sin borrar la señal eruptiva real.

Uso: python hmm_temp_sweep.py ubinas_v16_ub4_2019.csv
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import config as C
import postprocess as PP

csv = sys.argv[1] if len(sys.argv) > 1 else "ubinas_v16_ub4_2019.csv"
df = pd.read_csv(csv)
df['Time'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S')

quiet_end = pd.Timestamp(C.QUIET_PURE_END)
# indices de eventos reales (para verificar que no borramos la señal)
ev_mask = df['ref_4class'] == 3   # eruptivo real segun ground truth

def count_transitions(states):
    s = states[~np.isnan(states)]
    return int((np.diff(s) != 0).sum())

def prohibited_jumps(states):
    s = states[~np.isnan(states)]
    d = np.abs(np.diff(s))
    # salto directo Q<->E = cambio entre estado 0 y 3
    jumps = 0
    for i in range(len(s)-1):
        a,b = int(s[i]), int(s[i+1])
        if (a==0 and b==3) or (a==3 and b==0):
            jumps += 1
    return jumps

print(f"{'T':>6s} {'falsos_erup_quiet':>18s} {'transic':>10s} {'saltos_QE':>10s} "
      f"{'recall_erup_real':>17s}")
# referencia RF crudo
rf = df['pred_4c'].values.astype(float)
rf[df['valid'].values==0] = np.nan
q = df['Time'] < quiet_end
rf_false = int(((df['pred_4c']==3) & q).sum())
rf_trans = count_transitions(rf)
rf_recall = ((df['pred_4c']==3) & ev_mask).sum() / max(ev_mask.sum(),1)
print(f"{'RF':>6s} {rf_false:>18,} {rf_trans:>10,} {prohibited_jumps(rf):>10,} "
      f"{rf_recall:>17.3f}")

for T in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
    causal, _ = PP.run_hmm(df, emission_temp=T)
    dfc = df.copy(); dfc['hmm'] = causal
    false_e = int(((dfc['hmm']==3) & q).sum())
    trans = count_transitions(causal)
    jumps = prohibited_jumps(causal)
    recall = ((dfc['hmm']==3) & ev_mask).sum() / max(ev_mask.sum(),1)
    print(f"{T:>6.2f} {false_e:>18,} {trans:>10,} {jumps:>10,} {recall:>17.3f}")

print("\nBuscar: T que MINIMIZA falsos_erup_quiet y transiciones,")
print("manteniendo recall_erup_real alto (cerca del RF, no lo destruye).")
