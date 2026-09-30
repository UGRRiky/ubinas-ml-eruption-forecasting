"""
================================================================================
 run_stages_0_1_2.py — Ejecuta las etapas 0, 1 y 2 en secuencia.
================================================================================
Uso:
    1. Edita config.py -> BASE_DATA con la carpeta de tus 4 CSV crudos,
       y confirma los nombres de archivo en STATIONS_2023 / UB4_EXT_FILE.
    2. Ejecuta:  python run_stages_0_1_2.py

Cada etapa deja su salida en intermediate/ (Parquet). Si una etapa falla,
se detiene ahí con un mensaje claro y NO continúa con datos corruptos.

Requisitos: pandas, numpy, pyarrow
    pip install pandas numpy pyarrow
================================================================================
"""
import sys
import traceback

def main():
    print("\n" + "#" * 70)
    print("# PIPELINE UBINAS — ETAPAS 0-1-2")
    print("#" * 70)

    # ── Etapa 0: config (se auto-verifica al importar) ──
    try:
        import config as C
        print("\n[Etapa 0] config.py cargado y verificado.")
        print(f"  Catálogo 2019: {len(C.CATALOG_2019)} entradas "
              f"({C._N_EXP} explosiones, {C._N_EMI} emisiones, "
              f"{C._N_REAL} episodios reales)")
    except Exception as e:
        print(f"\n[Etapa 0] FALLÓ config.py:\n{e}")
        return 1

    # ── Etapa 1: carga ──
    try:
        import load
        load.run()
    except Exception as e:
        print(f"\n[Etapa 1] FALLÓ la carga:\n{e}")
        traceback.print_exc()
        return 1

    # ── Etapa 2: etiquetado ──
    try:
        import label
        label.run()
    except Exception as e:
        print(f"\n[Etapa 2] FALLÓ el etiquetado:\n{e}")
        traceback.print_exc()
        return 1

    print("\n" + "#" * 70)
    print("# ETAPAS 0-1-2 COMPLETADAS")
    print("#" * 70)
    print("\nSalidas generadas en intermediate/:")
    print("  stations_2023.parquet       (3 estaciones apiladas, para entrenar)")
    print("  ub4_2019.parquet            (UB4 extendido, series limpias)")
    print("  ub4_2019_labeled.parquet    (UB4 2019 con estados etiquetados)")
    print("\nRevisa los reportes de arriba:")
    print("  - ¿El % de válidos por estación coincide con lo que esperas?")
    print("  - ¿Los N_eruptivo por evento coinciden con las duraciones reales")
    print("    de tus eventos (start..end del catálogo)?")
    print("  - ¿La distribución de las 4 clases es razonable?")
    print("\nSi todo cuadra, seguimos con las etapas 3-4-5-6 (features,")
    print("entrenamiento, validación, post-proceso).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
