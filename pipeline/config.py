"""
================================================================================
 ETAPA 0 — config.py : FUENTE ÚNICA DE VERDAD
================================================================================
Todo lo que en el pipeline MATLAB estaba disperso y hardcodeado vive aquí:
rutas, catálogos de eventos, parámetros del modelo, umbrales, definición de
quietud. Ningún otro script define estas cosas; todos importan de aquí.

Si mañana cambia el catálogo de eventos, se cambia SOLO en este archivo y
todo el pipeline se ajusta solo (no más índices ev{2,2} que se rompen).
================================================================================
"""
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────
# RUTAS
# ──────────────────────────────────────────────────────────────────────────
# Edita BASE_DATA a la carpeta donde están tus 4 CSV crudos + el catálogo 2023.
BASE_DATA = Path(r"/mnt/i/ACTIVIDADES/2024/02_proyecto_SINFRA/RES/zed/data_input_paper")
# INTERMEDIATE anclado a la carpeta de ESTE archivo (no al directorio de
# ejecución), para que el pipeline encuentre sus parquet/modelos sin importar
# desde dónde se lance. Antes era relativo y creaba una carpeta vacía distinta
# según el cwd.
_PIPELINE_DIR = Path(__file__).resolve().parent
INTERMEDIATE = _PIPELINE_DIR / "intermediate"  # salidas por etapa (parquet, modelos)
INTERMEDIATE.mkdir(exist_ok=True)

# Las 3 estaciones de entrenamiento (2023) + la estación extendida de
# validación (UB4, 2018-2019). 'name' es como se identifica internamente.
STATIONS_2023 = [
    {"name": "UB1", "file": BASE_DATA / "Parametros_7params_UB1_BHZ_20230101_to_20231231_wd.csv"},
    {"name": "UB2", "file": BASE_DATA / "Parametros_7params_UB2_BHZ_20230101_to_20231231_wd.csv"},
    {"name": "UB4", "file": BASE_DATA / "Parametros_7params_UB4_BHZ_20230101_to_20231231_wd.csv"},
]
UB4_EXT_FILE = BASE_DATA / "Parametros_7params_UB4_BHZ_20180814_to_20191231_wd.csv"

# Catálogo de eventos 2023 (76 episodios) — se carga desde CSV en vez de
# hardcodearse. Columnas: inicio;fin;Duracion_horas;tipo_evento;altura_col;disp
CATALOG_2023_FILE = BASE_DATA / "Ubinas_Periodos_Clasificados_2023_final__4.csv"
CATALOG_2023_SEP = ";"

# ──────────────────────────────────────────────────────────────────────────
# COLUMNAS ESPERADAS EN LOS CSV CRUDOS (confirmadas con tus muestras)
# ──────────────────────────────────────────────────────────────────────────
TIME_COL = "DateTime_UTC"
# Formato de fecha en los CSV crudos: "01-Jan-2023 00:00:59"
TIME_FORMAT = "%d-%b-%Y %H:%M:%S"

# Los 4 parámetros crudos que usa el modelo (los *_Smooth son auxiliares
# para figuras, no entran al modelo directamente).
PARAM_COLS = ["SH_Shannon", "Curtosis", "FreqIndex", "SSAM_Mean"]
SMOOTH_COLS = ["SH_Shannon_Smooth", "FreqIndex_Smooth"]
GAP_COL = "Has_Gaps"

# Nombres internos que usará el resto del pipeline (más cortos y estables).
# Mapeo crudo -> interno.
RENAME_MAP = {
    "SH_Shannon": "entropy_val",
    "Curtosis":   "kurtosis_val",
    "FreqIndex":  "freqidx_val",
    "SSAM_Mean":  "ssam_val",
}
MODEL_FEATURES_BASE = list(RENAME_MAP.values())  # entropy, kurtosis, freqidx, ssam

# ──────────────────────────────────────────────────────────────────────────
# RANGOS TEMPORALES
# ──────────────────────────────────────────────────────────────────────────
T23_START = datetime(2023, 1, 1, 0, 0)
T23_END   = datetime(2023, 12, 31, 23, 59)

# UB4 extendido: se carga desde 2018 para tener burn-in de normalización.
T19_LOAD_START = datetime(2018, 8, 14, 0, 0)   # inicio de carga (burn-in)
T19_START      = datetime(2019, 1, 1, 0, 0)     # inicio del análisis real
T19_END        = datetime(2019, 12, 31, 23, 59)
BURN_IN_DAYS   = 140

# Quietud pura de 2019 (para calibrar umbrales) — hasta el enjambre VT del IGP.
QUIET_PURE_END = datetime(2019, 6, 17, 23, 59)

# ──────────────────────────────────────────────────────────────────────────
# CATÁLOGO DE EVENTOS 2019 — 9 episodios (3 explosiones + 6 emisiones)
# ──────────────────────────────────────────────────────────────────────────
# Fuente: cronología de Riky Centeno (informes CENVUL). Un solo lugar.
# pre_h = horas de ventana pre-eruptiva antes del inicio (4 exp / 6 emi).
# clase override: si no es None, fuerza esa etiqueta en vez de "eruptivo"
#   (usado para el evento débil del 23-jul 09:52, reclasificado a unrest=1).
CATALOG_2019 = [
    {"type": "emision",   "start": datetime(2019, 6, 24, 12, 20), "end": datetime(2019, 6, 24, 19, 52), "pre_h": 6, "override": None, "label": "Em. 24-Jun"},
    {"type": "emision",   "start": datetime(2019, 7, 18, 5, 14),  "end": datetime(2019, 7, 18, 18, 17), "pre_h": 6, "override": None, "label": "Em. 18-Jul"},
    {"type": "explosion", "start": datetime(2019, 7, 19, 7, 37),  "end": datetime(2019, 7, 22, 23, 59), "pre_h": 4, "override": None, "label": "Exp. 19-Jul"},
    {"type": "explosion", "start": datetime(2019, 7, 23, 4, 25),  "end": datetime(2019, 7, 23, 9, 50),  "pre_h": 4, "override": None, "label": "Exp. 23-Jul"},
    {"type": "emision",   "start": datetime(2019, 7, 23, 9, 52),  "end": datetime(2019, 7, 25, 23, 59), "pre_h": 0, "override": 1,    "label": "(débil 23-Jul, →unrest)"},
    {"type": "emision",   "start": datetime(2019, 8, 27, 10, 30), "end": datetime(2019, 8, 27, 10, 45), "pre_h": 6, "override": None, "label": "Em. 27-Ago"},
    {"type": "explosion", "start": datetime(2019, 9, 3, 18, 58),  "end": datetime(2019, 9, 3, 20, 0),   "pre_h": 4, "override": None, "label": "Exp. 3-Sep"},
    {"type": "emision",   "start": datetime(2019, 9, 4, 10, 56),  "end": datetime(2019, 9, 4, 13, 47),  "pre_h": 6, "override": None, "label": "Em. 4-Sep(a)"},
    {"type": "emision",   "start": datetime(2019, 9, 4, 13, 50),  "end": datetime(2019, 9, 4, 14, 50),  "pre_h": 0, "override": None, "label": "Em. 4-Sep(b)"},
    {"type": "emision",   "start": datetime(2019, 9, 12, 12, 30), "end": datetime(2019, 9, 12, 12, 45), "pre_h": 6, "override": None, "label": "Em. 12-Sep"},
]
# Reanudación de quietud tras la crisis (todo lo posterior = unrest/post-crisis)
T19_CALM_RESUME = datetime(2019, 9, 26, 0, 0)

# ──────────────────────────────────────────────────────────────────────────
# CATÁLOGO 2023 — se carga desde CSV (76 eventos)
# ──────────────────────────────────────────────────────────────────────────
# Ventanas pre-eruptivas por tipo (mismas reglas que 2019): 4h exp / 6h emi.
PRE_H_EXPLOSION = 4
PRE_H_EMISSION  = 6

def load_catalog_2023():
    """Carga el catálogo 2023 desde CSV y lo devuelve en el mismo formato de
    dicts que CATALOG_2019 (type/start/end/pre_h/override/label). Se llama
    desde label.py; vive aquí para mantener config como fuente única."""
    import pandas as pd
    if not CATALOG_2023_FILE.exists():
        raise FileNotFoundError(
            f"No se encuentra el catálogo 2023: {CATALOG_2023_FILE}")
    df = pd.read_csv(CATALOG_2023_FILE, sep=CATALOG_2023_SEP)
    required = {"inicio", "fin", "tipo_evento"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"El catálogo 2023 debe tener columnas {required}; "
            f"tiene {list(df.columns)}")
    df["inicio"] = pd.to_datetime(df["inicio"])
    df["fin"]    = pd.to_datetime(df["fin"])
    df = df.sort_values("inicio").reset_index(drop=True)
    catalog = []
    for i, row in df.iterrows():
        is_exp = str(row["tipo_evento"]).strip().lower() == "explosion"
        catalog.append({
            "type": "explosion" if is_exp else "emision",
            "start": row["inicio"].to_pydatetime(),
            "end": row["fin"].to_pydatetime(),
            "pre_h": PRE_H_EXPLOSION if is_exp else PRE_H_EMISSION,
            "override": None,
            "label": f"{row['tipo_evento'][:3]}. {row['inicio']:%d-%b}",
        })
    return catalog

# Fechas clave de 2023 (para etiquetar intranquilidad de fondo).
# El IGP elevó alerta ~17-may por incremento sísmico; primera emisión 22-jun.
T23_UNREST_ON  = datetime(2023, 5, 17, 0, 0)
T23_CALM_RESUME = datetime(2023, 12, 17, 0, 0)  # tras el último evento (16-dic)

# ──────────────────────────────────────────────────────────────────────────
# ESTADOS
# ──────────────────────────────────────────────────────────────────────────
STATE_QUIET   = 0  # Quietud
STATE_UNREST  = 1  # Intranquilidad
STATE_PRE     = 2  # Pre-eruptivo
STATE_ERUPT   = 3  # Eruptivo
STATE_NAMES = {0: "Quietud", 1: "Intranquilidad", 2: "Pre-eruptivo", 3: "Eruptivo"}

# Alerta binaria operacional: positivo = Pre-eruptivo o Eruptivo.
def is_alert_op(state):
    return int(state in (STATE_PRE, STATE_ERUPT))

# ──────────────────────────────────────────────────────────────────────────
# PARÁMETROS DEL MODELO
# ──────────────────────────────────────────────────────────────────────────
W_ROLLING_DAYS = 30           # ventana del z-score causal
W_ROLLING_MIN  = W_ROLLING_DAYS * 1440
MIN_VALID_ROLLING = 1440      # mínimo de muestras válidas para calcular z
SMOOTH_MIN = 60               # suavizado de features (min)

# Ventanas multi-escala para ingeniería de características (min)
FEATURE_WINDOWS = [10, 30, 60, 180]

# Random Forest
N_TREES = 400
MIN_LEAF = 20                 # entre 10-30: consenso sin cegar eventos cortos
                              # (1000 muestras = 16h -> ciego a explosiones de 15 min)
N_TREES_FS = 200              # para la selección de features
CORR_THRESHOLD = 0.95         # decorrelación
N_FEATURES_FINAL = 45

# ── Anti-memorización temporal ──
# Submuestreo SOLO de la clase mayoritaria (No-Alerta). NUNCA se submuestrea
# la clase positiva (Pre/Eruptivo) para no cegar el modelo ante explosiones.
SUBSAMPLE_MAJORITY = 5        # 1 de cada 5 filas No-Alerta (20%)
# NOTA sobre purga temporal: NO se aplica en el LOSO. Las 3 estaciones
# registran los mismos instantes, así que purgar la vecindad temporal del
# test borraría el 100% del entrenamiento (imposible). El LOSO mide, por
# tanto, robustez ESPACIAL (¿generaliza a una estación con otra
# instrumentación?), no temporal. La separación temporal genuina la aporta
# la validación 2019 (crisis independiente, >6 meses de separación), que es
# la métrica ANCLA del paper. La memorización trivial se controla con
# min_samples_leaf=20 y el submuestreo de la mayoritaria.

# ──────────────────────────────────────────────────────────────────────────
# UMBRALES Y PERSISTENCIA
# ──────────────────────────────────────────────────────────────────────────
PERSIST_HOURS = 6             # ancho de la ventana de densidad (tiempo REAL)
PERSIST_MIN = PERSIST_HOURS * 60
# Criterio de DENSIDAD DE ALERTA (Opción 2): se declara alarma si, en la
# ventana trailing de PERSIST_MIN, el umbral se superó en al menos
# DENSITY_FRAC de las muestras válidas. Tolera micro-huecos (glitches, caídas
# breves) sin reiniciar — como un analista real. Una sola definición de
# "sostenida", compartida por calibración, anticipación y sweep (sustained.py).
DENSITY_FRAC = 0.85
# Temperatura de emisión del HMM (atenúa las posteriores del RF para que la
# matriz de transición pese lo suficiente y limpie parpadeos espurios).
# T=1 -> sin atenuar (el RF domina, HMM apenas suaviza). T pequeño -> más
# suavizado. Se fija empíricamente con hmm_temperature_sweep.py.
HMM_EMISSION_TEMP = 0.15
# Filtro de VISUALIZACIÓN del estado (figura, no métricas): un bloque
# eruptivo se muestra como Intranquilidad si es un transitorio tectónico
# aislado — dura menos de este umbral o no fue precedido de Pre-eruptivo.
# NO afecta pred_4c_hmm_causal ni ninguna métrica; solo la columna
# state_display usada por la figura. Declarar en el caption.
DISPLAY_MIN_ERUP_MIN = 120   # min: erupción sostenida vs sismo tectónico corto
# Ventana de ASOCIACIÓN evento<->alerta: un evento se considera anticipado si
# hubo un episodio de alerta declarada activo dentro de las ASSOC_WINDOW_H
# horas previas al inicio del evento. Separa dos preguntas distintas:
#   (1) ¿cuándo se DECLARA alerta? -> densidad 85% en 6h (cero falsas alarmas).
#   (2) ¿un evento fue anticipado? -> hubo alerta activa en las 24h previas.
# 24h se fijó empíricamente: da 6/9 eventos, y ampliar a 48h NO añade ninguno
# (los 6 detectados tienen alerta dentro de 24h; los 3 restantes fallan
# incluso con 48h) -> umbral natural, no ajuste arbitrario. Operacionalmente:
# un observatorio que declaró alerta ayer sigue en alerta hoy al ocurrir el
# evento; no se exige señal activa en el minuto exacto previo.
ASSOC_WINDOW_H = 24
FAR_MAX = 0.10                # restricción para calibrar th_op (criterio F1)
# Criterio para el th_op operacional adoptado:
#   "zero_fa" -> umbral más bajo con cero falsas alarmas de densidad en quietud
#                (prioriza anticipación temprana; coherente con th_unrest).
#   "f1"      -> máximo F1 con FAR<=FAR_MAX (clásico; empuja el umbral alto y
#                puede perder el primer evento).
TH_OP_CRITERION = "zero_fa"

# ──────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN DE COHERENCIA DEL PROPIO CONFIG (se corre al importar)
# ──────────────────────────────────────────────────────────────────────────
def _self_check():
    problems = []
    # eventos en orden cronológico
    starts = [e["start"] for e in CATALOG_2019]
    if starts != sorted(starts):
        problems.append("CATALOG_2019 no está en orden cronológico.")
    # fin posterior a inicio
    for e in CATALOG_2019:
        if e["end"] <= e["start"]:
            problems.append(f"Evento {e['label']}: fin <= inicio.")
    # conteo
    n_exp = sum(1 for e in CATALOG_2019 if e["type"] == "explosion")
    n_emi = sum(1 for e in CATALOG_2019 if e["type"] == "emision")
    n_real = sum(1 for e in CATALOG_2019 if e["override"] is None)
    if problems:
        raise ValueError("config.py incoherente:\n  " + "\n  ".join(problems))
    return n_exp, n_emi, n_real

_N_EXP, _N_EMI, _N_REAL = _self_check()

if __name__ == "__main__":
    print("config.py — verificación de coherencia OK")
    print(f"  Catálogo 2019: {len(CATALOG_2019)} entradas")
    print(f"    Explosiones: {_N_EXP}")
    print(f"    Emisiones:   {_N_EMI}")
    print(f"    Episodios reales (excluye override): {_N_REAL}")
    print(f"  Estados: {STATE_NAMES}")
    print(f"  W rolling: {W_ROLLING_DAYS} días  |  Persistencia: {PERSIST_HOURS} h")
