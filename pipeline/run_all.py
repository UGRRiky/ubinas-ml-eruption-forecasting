"""
================================================================================
 run_all.py — Ejecuta el pipeline completo (etapas 0-6) en secuencia.
================================================================================
Uso:
    1. Edita config.py -> BASE_DATA con la carpeta de tus datos:
       - los 4 CSV crudos (Parametros_7params_UB?_...)
       - el catálogo 2023 (Ubinas_Periodos_Clasificados_2023_final__4.csv)
    2. Ejecuta:  python run_all.py

Cada etapa deja su salida en intermediate/. Si una falla, se detiene ahí con
un mensaje claro. El CSV final (ubinas_v16_ub4_2019.csv) se escribe en
BASE_DATA con las 26 columnas que consumen los scripts de figuras 4-6.

Requisitos: pandas numpy scikit-learn joblib pyarrow
    pip install pandas numpy scikit-learn joblib pyarrow
================================================================================
"""
import sys, traceback, time

STAGES = [
    ("Etapa 1 — Carga",        "load"),
    ("Etapa 2 — Etiquetado",   "label"),
    ("Etapa 3 — Features",     "features"),
    ("Etapa 4 — Entrenamiento","train"),
    ("Etapa 5 — Validación",   "validate"),
    ("Etapa 6 — Post-proceso", "postprocess"),
    ("Etapa 7 — Clasificador explosión/emisión", "event_type"),
]

def main():
    print("#" * 70)
    print("# PIPELINE UBINAS — COMPLETO (etapas 0-6)")
    print("#" * 70)
    try:
        import config as C
        print(f"\n[Etapa 0] config OK — catálogo 2019: {len(C.CATALOG_2019)} entradas "
              f"({C._N_REAL} reales)")
    except Exception as e:
        print(f"[Etapa 0] FALLÓ config: {e}"); return 1

    for title, mod_name in STAGES:
        print(f"\n{'='*70}\n{title}\n{'='*70}")
        t0 = time.time()
        try:
            mod = __import__(mod_name)
            mod.run()
            print(f"  [{title}] completada en {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"\n[{title}] FALLÓ:\n{e}")
            traceback.print_exc()
            return 1

    print("\n" + "#" * 70)
    print("# PIPELINE COMPLETO — TODAS LAS ETAPAS OK")
    print("#" * 70)
    print("\nSalida final: ubinas_v16_ub4_2019.csv (en BASE_DATA)")
    print("Listo para los scripts de figuras 4-6.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
