"""
Confirma que los falsos eruptivos de quietud son TRANSITORIOS EXTERNOS
(un solo parametro disparado) y no actividad volcanica (varios parametros
coherentes). Prueba un criterio de coherencia multiparametrica.

Uso: python diag_transitorios.py ubinas_v16_ub4_2019.csv
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import config as C

csv = sys.argv[1] if len(sys.argv) > 1 else "ubinas_v16_ub4_2019.csv"
df = pd.read_csv(csv)
df['Time'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S')
df = df[df['valid']==1].reset_index(drop=True)
params = ['entropy_val','kurtosis_val','freqidx_val','ssam_val']

# Para cada minuto: cuantos de los 4 parametros estan "activos" (|z|>2)
# Actividad volcanica real -> varios a la vez. Transitorio -> uno solo.
Z = df[params].abs()
df['n_activos'] = (Z > 2).sum(axis=1)

q = df['Time'] < pd.Timestamp(C.QUIET_PURE_END)
crisis = (df['Time'] >= pd.Timestamp('2019-06-18')) & (df['Time'] <= pd.Timestamp('2019-09-25'))

print("=== ¿Cuantos parametros se activan juntos (|z|>2)? ===")
print("Hipotesis: en falsos eruptivos de quietud, casi siempre 1 solo param.")
print("           en eventos reales de crisis, varios a la vez.\n")

# minutos ERUPTIVOS falsos en quietud
falsos = df[q & (df['pred_4c']==3)]
print(f"Minutos eruptivos FALSOS en quietud: {len(falsos)}")
print(f"  Distribucion de n_activos:")
for k in range(5):
    n = (falsos['n_activos']==k).sum()
    print(f"    {k} parametros activos: {n:>6,} ({100*n/max(len(falsos),1):.1f}%)")

# minutos ERUPTIVOS reales en crisis
reales = df[crisis & (df['pred_4c']==3) & (df['ref_4class']==3)]
print(f"\nMinutos eruptivos REALES en crisis: {len(reales)}")
print(f"  Distribucion de n_activos:")
for k in range(5):
    n = (reales['n_activos']==k).sum()
    print(f"    {k} parametros activos: {n:>6,} ({100*n/max(len(reales),1):.1f}%)")

# Prueba del criterio: exigir >=2 parametros activos para permitir eruptivo
print("\n=== Efecto de exigir >=2 parametros coherentes para estado eruptivo ===")
# cuantos falsos eruptivos se eliminarian
falsos_1param = (falsos['n_activos'] < 2).sum()
reales_perdidos = (reales['n_activos'] < 2).sum()
print(f"  Falsos eruptivos eliminados (quietud): {falsos_1param}/{len(falsos)} "
      f"({100*falsos_1param/max(len(falsos),1):.0f}%)")
print(f"  Eruptivos reales perdidos (crisis): {reales_perdidos}/{len(reales)} "
      f"({100*reales_perdidos/max(len(reales),1):.1f}%)")
print("\n  Bueno si: elimina MUCHOS falsos y pierde POCOS reales.")
