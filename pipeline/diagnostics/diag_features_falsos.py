"""
Diagnostico decisivo: ¿que features hacen que el modelo diga 'eruptivo' en
minutos de quietud donde los parametros instantaneos son normales?

Carga el modelo multiclase y las features 2019, y examina los minutos
eruptivos falsos: que features estan en valores extremos, y compara con la
importancia de features del modelo.

Corre en la carpeta del pipeline (necesita intermediate/ y los modelos).
Uso: python diag_features_falsos.py
"""
import sys, numpy as np, pandas as pd, json
sys.path.insert(0, '.')
import config as C
import joblib

# cargar features 2019 (tienen las 85 columnas + Time + valid + labels)
feat = pd.read_parquet(C.INTERMEDIATE / "features_2019.parquet")
feat['Time'] = pd.to_datetime(feat['Time'])
val = pd.read_parquet(C.INTERMEDIATE / "validation_2019.parquet")
val['Time'] = pd.to_datetime(val['Time'])

# unir la prediccion multiclase con las features
m = val[['Time','pred_4c']].merge(feat, on='Time', how='inner')

# modelo multiclase y sus 45 features
mc = joblib.load(C.INTERMEDIATE / "model_multiclass.joblib")
sel = mc['features']
imp = pd.Series(mc['model'].feature_importances_, index=sel).sort_values(ascending=False)
print("=== Top 10 features mas importantes del modelo multiclase ===")
print(imp.head(10).to_string())

# minutos eruptivos falsos: quietud (ene-17jun) + pred==3
q = m['Time'] < pd.Timestamp(C.QUIET_PURE_END)
falsos = m[q & (m['pred_4c']==3)]
# minutos quietud correctos (pred==0) para comparar
correctos = m[q & (m['pred_4c']==0)]
print(f"\nMinutos eruptivos FALSOS en quietud: {len(falsos)}")
print(f"Minutos quietud correctos: {len(correctos)}")

# para las top features, comparar su valor medio en falsos vs correctos
print(f"\n=== Valor medio de las top-10 features: falsos_erup vs quietud_ok ===")
print(f"{'feature':<28s} {'falsos':>10s} {'quietud_ok':>12s} {'ratio':>8s}")
for f in imp.head(10).index:
    if f in falsos.columns:
        vf = falsos[f].mean()
        vc = correctos[f].mean()
        ratio = vf/vc if abs(vc)>1e-6 else np.inf
        print(f"{f:<28s} {vf:>10.3f} {vc:>12.3f} {ratio:>8.2f}")
