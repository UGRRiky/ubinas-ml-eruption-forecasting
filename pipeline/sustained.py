"""
================================================================================
 sustained.py — DEFINICIÓN ÚNICA de "alerta sostenida" (criterio de densidad)
================================================================================
Una sola función compartida por TODO el pipeline (calibración de th_op,
anticipación por evento, y el sweep de umbral). Antes había tres mediciones
distintas de "persistencia 6h" que daban resultados incompatibles; esto lo
unifica.

CRITERIO DE DENSIDAD DE ALERTA (Opción 2, acordada):
  Se declara Alarma Operacional en el instante t si, dentro de la ventana
  móvil de PERSIST_MIN minutos que TERMINA en t (trailing, causal), el umbral
  fue superado en al menos DENSITY_FRAC de las muestras VÁLIDAS de esa
  ventana.

  Ej.: ventana 360 min, densidad 0.85 -> se declara si P(Alert) >= th en al
  menos 306 de los ~360 minutos previos. Tolera micro-huecos (glitches, caídas
  breves de entropía) sin reiniciar el reloj a cero — como haría un analista
  real en una sala de monitoreo.

Ventajas frente a "racha estrictamente continua":
  - No falla por una caída de 5 min (el error clásico de exigir continuidad
    matemática perfecta sobre una señal geofísica).
  - Es causal (solo mira hacia atrás) -> no introduce lag artificial como
    haría suavizar la señal a 6h.
  - Es un cálculo único y auditable: densidad en ventana, sin máquina de
    estados ambigua.
================================================================================
"""
import numpy as np
import pandas as pd


def alert_density(times, p_alert, th, valid=None,
                  persist_min=360, density_frac=0.85):
    """Devuelve un array booleano 'declared' del mismo largo que times: True
    en cada instante t donde la densidad de superación del umbral en la
    ventana trailing de persist_min minutos alcanza density_frac.

    Implementación vectorizada con suma acumulada sobre grilla de minutos:
    reindexa a grilla de 1 min, calcula la fracción de superación en la
    ventana trailing con diferencia de sumas acumuladas. O(n) total, no
    O(n) por llamada-en-bucle.

    times:    array de datetime64 (grilla ~1 min, puede tener huecos)
    p_alert:  array float de P(Alert) (puede tener NaN)
    th:       umbral
    valid:    array bool opcional; si se da, solo cuentan muestras válidas
    persist_min: ancho de la ventana (min)
    density_frac: fracción mínima de superación (0-1)
    """
    t = pd.to_datetime(pd.Series(times)).values.astype("datetime64[m]")
    p = np.asarray(p_alert, dtype=float)
    above = np.nan_to_num(p, nan=-np.inf) >= th
    if valid is None:
        valid = ~np.isnan(p)
    else:
        valid = np.asarray(valid, dtype=bool)

    n = len(t)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # índice de minuto (entero) desde el primer instante
    minute_idx = ((t - t[0]) / np.timedelta64(1, "m")).astype(np.int64)
    span_total = int(minute_idx[-1]) + 1

    # colocar en grilla densa de minutos: cuántas válidas y cuántas above por minuto
    grid_valid = np.zeros(span_total, dtype=np.float64)
    grid_above = np.zeros(span_total, dtype=np.float64)
    np.add.at(grid_valid, minute_idx, valid.astype(np.float64))
    np.add.at(grid_above, minute_idx, (valid & above).astype(np.float64))

    # sumas acumuladas para ventana trailing de persist_min
    cs_valid = np.concatenate([[0], np.cumsum(grid_valid)])
    cs_above = np.concatenate([[0], np.cumsum(grid_above)])
    W = int(persist_min)

    # para cada minuto g, ventana [g-W+1, g]
    g = np.arange(span_total)
    lo = np.maximum(0, g - W + 1)
    win_valid = cs_valid[g + 1] - cs_valid[lo]
    win_above = cs_above[g + 1] - cs_above[lo]
    span = g - lo + 1  # cobertura temporal de la ventana (min)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(win_valid > 0, win_above / win_valid, 0.0)
    # declara si: densidad suficiente, cobertura temporal >= media ventana,
    # y suficientes muestras válidas reales
    declared_grid = ((frac >= density_frac) &
                     (span >= persist_min * 0.5) &
                     (win_valid >= persist_min * density_frac * 0.5))

    # mapear de vuelta a las posiciones originales
    return declared_grid[minute_idx]


def declaration_onsets(times, declared):
    """Dado el array booleano 'declared', devuelve los instantes de INICIO de
    cada episodio de declaración (transiciones False->True). Cada onset es el
    momento en que la alarma se activa tras haber estado inactiva."""
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    d = np.asarray(declared, dtype=bool)
    onsets = []
    prev = False
    for i in range(len(d)):
        if d[i] and not prev:
            onsets.append(pd.Timestamp(t.iloc[i]))
        prev = d[i]
    return onsets


def declaration_intervals(times, declared):
    """Devuelve intervalos (onset, fin) de cada episodio continuo de alarma
    activa. Útil para distinguir alerta fresca vs. heredada por evento."""
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    d = np.asarray(declared, dtype=bool)
    intervals = []
    onset = None
    for i in range(len(d)):
        if d[i] and onset is None:
            onset = pd.Timestamp(t.iloc[i])
        elif not d[i] and onset is not None:
            intervals.append((onset, pd.Timestamp(t.iloc[i-1])))
            onset = None
    if onset is not None:
        intervals.append((onset, pd.Timestamp(t.iloc[-1])))
    return intervals


# ── Tests con casos controlados ──
if __name__ == "__main__":
    print("TEST 1: racha continua de 6h supera densidad")
    t = pd.date_range("2019-01-01", periods=1000, freq="1min").values
    p = np.zeros(1000); p[100:600] = 0.9  # 500 min continuos altos
    d = alert_density(t, p, th=0.5, persist_min=360, density_frac=0.85)
    onsets = declaration_onsets(t, d)
    print(f"  Declaraciones: {len(onsets)}, primera en índice "
          f"{np.argmax(d) if d.any() else 'ninguna'}")
    # debería declarar en algún punto tras acumular 306/360 min altos
    assert d.any(), "FALLO: no declaró con racha continua larga"
    print("  ✓ declara con racha continua")

    print("\nTEST 2: micro-hueco de 5 min NO cancela la alarma")
    p2 = np.zeros(1000); p2[100:600] = 0.9
    p2[350:355] = 0.1  # caída de 5 min en medio
    d2 = alert_density(t, p2, th=0.5, persist_min=360, density_frac=0.85)
    assert d2.any(), "FALLO: el micro-hueco canceló la alarma"
    # con 5 min de hueco en 360, densidad = 355/360 = 0.986 > 0.85 -> OK
    print(f"  ✓ tolera micro-hueco (declaraciones: {len(declaration_onsets(t, d2))})")

    print("\nTEST 3: señal nerviosa (50% arriba) NO declara")
    p3 = np.zeros(1000)
    p3[100:600:2] = 0.9  # alterna arriba/abajo -> densidad ~50%
    d3 = alert_density(t, p3, th=0.5, persist_min=360, density_frac=0.85)
    assert not d3.any(), "FALLO: declaró con densidad de solo 50%"
    print("  ✓ NO declara con densidad 50% (< 85%)")

    print("\nTEST 4: causalidad (alterar futuro no cambia el pasado)")
    p4 = np.zeros(1000); p4[100:600] = 0.9
    d4a = alert_density(t, p4, th=0.5)
    p4b = p4.copy(); p4b[700:] = 0.9
    d4b = alert_density(t, p4b, th=0.5)
    assert (d4a[:650] == d4b[:650]).all(), "FALLO: no es causal"
    print("  ✓ causal (el futuro no altera declaraciones pasadas)")

    print("\nTodos los tests OK.")
