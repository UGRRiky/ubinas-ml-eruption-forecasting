#!/usr/bin/env python3
"""
Ubinas Volcano Monitoring (2023) — SPLIT FIGURES (v4)

Divide la figura original de 7 paneles en dos figuras nuevas, preservando
colores, fuentes y convenciones de estilo del script original
(plot_fig2_paper_v3_.py):

FIGURA 2 — "Comportamiento de los datos y efecto de normalización"
    4 filas x 1 columna (filas 3 y 4 divididas en 2 columnas):
    a) Shannon Entropy cruda (3 estaciones)              [= antiguo panel b]
    b) Decay Ratio (3 estaciones)                        [= antiguo panel c]
       -> unica fila con ticks mayores (mes) y menores (semana) visibles
    c) Zoom SE cruda,        17-abr a 17-jun-2023
    d) Zoom SE normalizada,  17-abr a 17-jun-2023
    e) Zoom SE cruda,        17-oct a 17-dic-2023
    f) Zoom SE normalizada,  17-oct a 17-dic-2023

FIGURA 3 — igual a la figura original, eliminando los paneles b) y c)
    5 paneles:
    a) Imagenes + barra de actividad                     [sin cambios]
    b) SE normalizada (z-score)      [= antiguo panel d]  -> lleva la leyenda
    c) FI normalizada (z-score)      [= antiguo panel e]
    d) Kurtosis normalizada (z-score)[= antiguo panel f]
    e) SSAM normalizada (z-score)    [= antiguo panel g]  -> ticks mes/semana
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import glob
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import matplotlib.path as mpath
from datetime import timedelta
import seaborn as sns

# ===== CONFIGURATION (sin cambios respecto al script original) =====
sns.set_style("ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.switch_backend('Agg')

# ===== FILE PATHS =====
BASE_PATH = "/mnt/i/ACTIVIDADES/2024/02_proyecto_SINFRA"

images_folder = f"{BASE_PATH}/BIN/GMT/ubinas_eruptive_events"
periods_file = f"{BASE_PATH}/RES/zed/To_GMT/Ubinas_Periodos_Clasificados_2023_final__4.csv"
external_events_file = f"{BASE_PATH}/DATA/others_fount5.csv"

file_ub1 = f"{BASE_PATH}/RES/zed/UB1_10m_smoothed_parameters.csv"
file_ub2 = f"{BASE_PATH}/RES/zed/UB2_10m_smoothed_parameters.csv"
file_ub4 = f"{BASE_PATH}/RES/zed/UB4_10m_smoothed_parameters.csv"

OUT_DIR = f"{BASE_PATH}/RES/zed"

# ===== COLORS (sin cambios) =====
ACTIVITY_LEVELS = [
    {"inicio": pd.Timestamp("2022-12-31"), "fin": pd.Timestamp("2023-05-16 23:59:59"), "color": "#70AD47", "label": "Green"},
    {"inicio": pd.Timestamp("2023-05-17"), "fin": pd.Timestamp("2023-06-30 23:59:59"), "color": "#FFDF27", "label": "Yellow"},
    {"inicio": pd.Timestamp("2023-07-01"), "fin": pd.Timestamp("2023-11-05 23:59:59"), "color": "#ED7D31", "label": "Orange"},
    {"inicio": pd.Timestamp("2023-11-06"), "fin": pd.Timestamp("2024-01-01"),          "color": "#FFDF27", "label": "Yellow"},
]

COLORS = {
    'explosion_marker': '#FF3333',
    'emission_marker': '#FFC107',
    'emission_line': '#FFC107',
    'eruptive_period': '#B5BBC7',
    'arrow': (0.3, 0.3, 0.35),
    'reference_line': '#000000',
    'lahar': '#540905',
    'vt_sismo': '#0000F7',
}

C_UB1 = '#1F77B4'
C_UB2 = '#FF7F0E'
C_UB4 = '#2CA02C'

REFERENCE_DATE = pd.Timestamp('2023-05-17')
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2024-01-01')

# ===== SE0 REFERENCE PERIOD =====
SE0_REF_START = pd.Timestamp('2023-01-01')
SE0_REF_END   = pd.Timestamp('2023-04-30 23:59:59')
DR_THRESHOLD  = 70  # percent

# ===== ZOOM WINDOWS (NUEVO — paneles c,d,e,f de la Figura 2) =====
ZOOM1_START = pd.Timestamp('2023-04-17')
ZOOM1_END   = pd.Timestamp('2023-06-17')
ZOOM2_START = pd.Timestamp('2023-10-17')
ZOOM2_END   = pd.Timestamp('2023-12-17')

# ===== HELPER FUNCTIONS (idénticas al script original) =====
def create_volcano_marker():
    verts = np.array([(-0.5, -0.5), (0.5, -0.5), (0.3, 0.5), (0.1, 0.3), (-0.1, 0.3), (-0.3, 0.5), (-0.5, -0.5)])
    return mpath.Path(verts)

def add_activity_bar(ax, y_pos, height):
    for level in ACTIVITY_LEVELS:
        start = mdates.date2num(level['inicio'])
        end = mdates.date2num(level['fin'])
        rect = Rectangle((start, y_pos), end - start, height,
                         facecolor=level['color'], alpha=0.9, edgecolor='none', zorder=2)
        ax.add_patch(rect)

def add_eruptive_periods(ax, periods_df, y_pos, height):
    for _, period in periods_df.iterrows():
        start = mdates.date2num(period['inicio'])
        end = mdates.date2num(period['fin'])
        width = end - start
        rect = Rectangle((start, y_pos), width, height,
                         facecolor=COLORS['eruptive_period'], alpha=0.6,
                         edgecolor='none', transform=ax.transData,
                         clip_on=True, zorder=3)
        ax.add_patch(rect)

def add_eruptive_bands(ax, periods_df):
    for _, period in periods_df.iterrows():
        ax.axvspan(period['inicio'], period['fin'], color=COLORS['eruptive_period'], alpha=0.60, zorder=1, linewidth=0)

def configure_panel(ax, ylabel, is_last=False, label_x=-0.07):
    ax.tick_params(direction='out', length=8, labelsize=14, which='major')
    ax.tick_params(direction='out', length=4, which='minor')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelsize=14)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator([1, 8, 15, 22]))
    ax.set_ylabel(ylabel, fontsize=14)
    ax.yaxis.set_label_coords(label_x, 0.5)
    if not is_last:
        ax.tick_params(labelbottom=False)
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlabel('Month', fontsize=14, fontweight='normal')

def configure_month_week_ticks(ax, xlabel='Month'):
    """NUEVO: aplica ticks mayores (mes) + menores (semana), con etiquetas
    visibles. Replica exactamente lo que el script original hacia en el
    ultimo panel (antiguo panel g / SSAM) para dejarlo consistente en
    cualquier panel que deba mostrar el eje x."""
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    ax.tick_params(axis='x', which='major', bottom=True, direction='out',
                    length=8, width=0.8, color='black')
    ax.tick_params(axis='x', which='minor', bottom=True, direction='out',
                    length=4, width=0.6, color='black')
    locator = mdates.MonthLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(labelbottom=True, labelsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, fontweight='normal')

def add_explosion_lines(ax, explosions_df):
    for _, exp in explosions_df.iterrows():
        ax.axvline(exp['inicio'], color=COLORS['explosion_marker'], linewidth=0.85, alpha=0.85, linestyle='--', zorder=2)

def add_external_event_lines(ax, external_events_df):
    for _, event in external_events_df.iterrows():
        event_type = event['EventType'].lower()
        if 'lahar' in event_type:
            color = COLORS['lahar']
        elif 'vt' in event_type or 'sismo' in event_type or 'earthquake' in event_type:
            color = COLORS['vt_sismo']
        else:
            continue
        ax.axvline(event['DateTime'], color=color, linewidth=0.85, alpha=0.85, linestyle='--', zorder=2)

def add_reference_line(ax, date, dashed=True):
    ls = '--' if dashed else '-'
    ax.axvline(date, color=COLORS['reference_line'], linewidth=1.0, alpha=0.8, linestyle=ls, zorder=2)

def parse_image_datetime(filename):
    parts = filename.split('_')
    if len(parts) >= 4:
        date_part, time_part = parts[2], parts[3]
        year, month, day = int(date_part[:4]), int(date_part[4:6]), int(date_part[6:8])
        hour, minute = int(time_part[:2]), int(time_part[2:4])
        return pd.Timestamp(year, month, day, hour, minute)
    return None

def add_panel_label(ax, label):
    ax.text(-0.032, 1.03, label, transform=ax.transAxes, fontsize=18,
            fontweight='normal', va='top', ha='left',
            clip_on=False)

# ===== LOAD DATA (idéntico al script original) =====
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

periods_df = pd.read_csv(periods_file, sep=';', parse_dates=['inicio', 'fin'])
explosions_df = periods_df[periods_df['tipo_evento'] == 'explosion']
emissions_df = periods_df[periods_df['tipo_evento'] == 'emision']

try:
    external_events = pd.read_csv(external_events_file, parse_dates=['DateTime'])
    external_events = external_events[(external_events['DateTime'] >= start_date) & (external_events['DateTime'] <= end_date)]
    lahars_df = external_events[external_events['EventType'].str.lower().str.contains('lahar', na=False)]
    vt_sismo_df = external_events[external_events['EventType'].str.lower().str.contains('vt|sismo|earthquake', na=False, regex=True)]
except Exception as e:
    print(f"WARNING: Could not load external events: {e}")
    external_events, lahars_df, vt_sismo_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_ub1 = pd.read_csv(file_ub1, parse_dates=['DateTime_UTC'])
df_ub2 = pd.read_csv(file_ub2, parse_dates=['DateTime_UTC'])
df_ub4 = pd.read_csv(file_ub4, parse_dates=['DateTime_UTC'])

# ===== COMPUTE DECAY RATIO (idéntico) =====
print("\nComputing Decay Ratio (SE0 from Jan-Apr 2023)...")

def compute_decay_ratio(df, se0_start, se0_end, col='SH_Shannon_Smooth'):
    mask_ref = (df['DateTime_UTC'] >= se0_start) & (df['DateTime_UTC'] <= se0_end)
    se0 = df.loc[mask_ref, col].mean()
    dr = 100.0 * (1.0 - df[col] / se0)
    dr = dr.clip(-50, 100)
    return dr, se0

dr_ub1, se0_ub1 = compute_decay_ratio(df_ub1, SE0_REF_START, SE0_REF_END)
dr_ub2, se0_ub2 = compute_decay_ratio(df_ub2, SE0_REF_START, SE0_REF_END)
dr_ub4, se0_ub4 = compute_decay_ratio(df_ub4, SE0_REF_START, SE0_REF_END)

# ===== COMPUTE NORMALIZED SE (causal rolling z-score, W=30d) — idéntico =====
print("\nComputing normalized Shannon Entropy (causal z-score)...")
W_DAYS = 30
W_MIN = W_DAYS * 1440
SM_Z = 60

def causal_rolling_zscore(series, window):
    rw = series.shift(1).rolling(window=window, min_periods=1440)
    mu = rw.mean()
    sd = rw.std().replace(0, np.nan)
    return (series - mu) / sd

z_ub1 = causal_rolling_zscore(df_ub1['SH_Shannon_Smooth'], W_MIN)
z_ub2 = causal_rolling_zscore(df_ub2['SH_Shannon_Smooth'], W_MIN)
z_ub4 = causal_rolling_zscore(df_ub4['SH_Shannon_Smooth'], W_MIN)
z_ub1_sm = z_ub1.rolling(window=SM_Z, min_periods=1, center=True).mean()
z_ub2_sm = z_ub2.rolling(window=SM_Z, min_periods=1, center=True).mean()
z_ub4_sm = z_ub4.rolling(window=SM_Z, min_periods=1, center=True).mean()

def zscore_station(df_st, col):
    z = causal_rolling_zscore(df_st[col], W_MIN)
    return z.rolling(window=SM_Z, min_periods=1, center=True).mean()

zfi_ub1 = zscore_station(df_ub1, 'FreqIndex_Smooth')
zfi_ub2 = zscore_station(df_ub2, 'FreqIndex_Smooth')
zfi_ub4 = zscore_station(df_ub4, 'FreqIndex_Smooth')

zk_ub1 = zscore_station(df_ub1, 'Curtosis_Acum_Smooth')
zk_ub2 = zscore_station(df_ub2, 'Curtosis_Acum_Smooth')
zk_ub4 = zscore_station(df_ub4, 'Curtosis_Acum_Smooth')

zss_ub1 = zscore_station(df_ub1, 'SSAM_Acum_Smooth')
zss_ub2 = zscore_station(df_ub2, 'SSAM_Acum_Smooth')
zss_ub4 = zscore_station(df_ub4, 'SSAM_Acum_Smooth')

_qm_ub1 = df_ub1['DateTime_UTC'] < pd.Timestamp('2023-05-17')
_qm_ub2 = df_ub2['DateTime_UTC'] < pd.Timestamp('2023-05-17')
_qm_ub4 = df_ub4['DateTime_UTC'] < pd.Timestamp('2023-05-17')
print(f"  UBI1 quiet z: mu={z_ub1[_qm_ub1].mean():.3f} sd={z_ub1[_qm_ub1].std():.3f}")
print(f"  UBI2 quiet z: mu={z_ub2[_qm_ub2].mean():.3f} sd={z_ub2[_qm_ub2].std():.3f}")
print(f"  UBI4 quiet z: mu={z_ub4[_qm_ub4].mean():.3f} sd={z_ub4[_qm_ub4].std():.3f}")
print(f"  SE0 UBI1 = {se0_ub1:.1f} Bits")
print(f"  SE0 UBI2 = {se0_ub2:.1f} Bits")
print(f"  SE0 UBI4 = {se0_ub4:.1f} Bits")
print(f"  DR threshold = {DR_THRESHOLD}%")

for name, dr_series, df_st in [('UBI1', dr_ub1, df_ub1), ('UBI2', dr_ub2, df_ub2), ('UBI4', dr_ub4, df_ub4)]:
    above = dr_series > DR_THRESHOLD
    n_above = above.sum()
    pct = 100 * n_above / len(dr_series)
    quiet_mask = df_st['DateTime_UTC'] < pd.Timestamp('2023-05-17')
    fa = (dr_series[quiet_mask] > DR_THRESHOLD).sum()
    print(f"  {name}: {n_above} samples above {DR_THRESHOLD}% ({pct:.1f}%), false alarms in quiescence: {fa}")

print("Loading eruptive event images...")
explosion_images = sorted(glob.glob(os.path.join(images_folder, "UBI_EXP_*_circle.png")))
emission_images = sorted(glob.glob(os.path.join(images_folder, "UBI_EMI_*_circle.png")))

image_info = []
for img_path in explosion_images:
    img_datetime = parse_image_datetime(os.path.basename(img_path))
    if img_datetime:
        image_info.append({'path': img_path, 'datetime': img_datetime, 'label': img_datetime.strftime('%Y-%m-%d %H:%M'), 'type': 'explosion', 'color': COLORS['explosion_marker']})

for img_path in emission_images:
    img_datetime = parse_image_datetime(os.path.basename(img_path))
    if img_datetime:
        image_info.append({'path': img_path, 'datetime': img_datetime, 'label': img_datetime.strftime('%Y-%m-%d %H:%M'), 'type': 'emission', 'color': COLORS['emission_marker']})

image_info.sort(key=lambda x: x['datetime'])

xlim_full = (start_date, end_date)
volcano_marker = create_volcano_marker()
line_args = {'linewidth': 1.2, 'alpha': 0.85}

# Elementos de leyenda compartidos (idénticos a la leyenda del script original)
legend_elements = [
    Line2D([0], [0], color=C_UB1, lw=1.2, label='UBI1'),
    Line2D([0], [0], color=C_UB2, lw=1.2, label='UBI2'),
    Line2D([0], [0], color=C_UB4, lw=1.2, label='UBI4'),
    Rectangle((0, 0), 1, 1, facecolor=COLORS['eruptive_period'], alpha=0.3, label='Eruptive Period'),
    Line2D([0], [0], marker=volcano_marker, color='w', markerfacecolor=COLORS['explosion_marker'], markeredgecolor='darkred', markersize=10, label='Explosions'),
    Line2D([0], [0], marker=volcano_marker, color='w', markerfacecolor=COLORS['emission_marker'], markeredgecolor='black', markersize=10, label='Emissions'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor=COLORS['vt_sismo'], markeredgecolor='black', markersize=8, label='VT swarm'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor=COLORS['lahar'], markeredgecolor='black', markersize=8, label='Wind noise'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['vt_sismo'], markeredgecolor='black', markersize=10, label='M5.4 Eq.(Chivay)')
]


# ══════════════════════════════════════════════════════════════════════
#  FIGURA 2 (NUEVA) — Comportamiento de los datos y efecto de normalización
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CREATING FIGURE 2 (6 paneles: a, b, c, d, e, f)")
print("=" * 60)

fig2 = plt.figure(figsize=(16, 20), dpi=300)
gs2 = fig2.add_gridspec(4, 2, height_ratios=[1.5, 1.5, 1.3, 1.3],
                         hspace=0.28, wspace=0.28)

with sns.axes_style("darkgrid"):
    ax_a = fig2.add_subplot(gs2[0, :])   # a) SE cruda (full width)
    ax_b = fig2.add_subplot(gs2[1, :])   # b) Decay Ratio (full width)
    ax_c = fig2.add_subplot(gs2[2, 0])   # c) Zoom SE cruda (abr-jun)
    ax_d = fig2.add_subplot(gs2[2, 1])   # d) Zoom SE normalizada (abr-jun)
    ax_e = fig2.add_subplot(gs2[3, 0])   # e) Zoom SE cruda (oct-dic)
    ax_f = fig2.add_subplot(gs2[3, 1])   # f) Zoom SE normalizada (oct-dic)

# ---- Panel a) Shannon Entropy cruda [= antiguo panel b] ----
print("Panel a: Shannon Entropy (raw)...")
ax_a.set_xlim(xlim_full)
add_eruptive_bands(ax_a, periods_df)
ax_a.plot(df_ub1['DateTime_UTC'], df_ub1['SH_Shannon_Smooth'], color=C_UB1, label='UBI1', **line_args)
ax_a.plot(df_ub2['DateTime_UTC'], df_ub2['SH_Shannon_Smooth'], color=C_UB2, label='UBI2', **line_args)
ax_a.plot(df_ub4['DateTime_UTC'], df_ub4['SH_Shannon_Smooth'], color=C_UB4, label='UBI4', **line_args)
add_external_event_lines(ax_a, external_events)
add_reference_line(ax_a, REFERENCE_DATE, dashed=True)
ax_a.set_ylim(-150, 2900)
ax_a.set_yticks([0, 1000, 2000])
mid_date = SE0_REF_START + (SE0_REF_END - SE0_REF_START) / 2
c_f = 2670
ax_a.annotate('', xy=(mid_date, c_f), xytext=(SE0_REF_START, c_f),
              arrowprops=dict(arrowstyle='<-', color='#4CAF50', lw=1.0))
ax_a.annotate('', xy=(mid_date, c_f), xytext=(SE0_REF_END, c_f),
              arrowprops=dict(arrowstyle='<-', color='#4CAF50', lw=1.0))
ax_a.annotate('SE$_0$ ref.\nperiod', xy=(pd.Timestamp('2023-03-01'), 2550),
              fontsize=9, color='#2E7D32', ha='center', style='italic',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#4CAF50'))
configure_panel(ax_a, 'Shannon\nEntropy [Bits]', is_last=False)
add_panel_label(ax_a, 'a)')
# Leyenda general en el panel a (misma leyenda que antes iba en panel b)
ax_a.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
            ncol=3, fontsize=10)

# ---- Panel b) Decay Ratio [= antiguo panel c] — CON ticks mes/semana ----
print("Panel b: Decay Ratio...")
ax_b.set_xlim(xlim_full)
ax_b.set_yticks([-50, 0, 50])
ax_b.set_ylim(-60, 100)
add_eruptive_bands(ax_b, periods_df)
ax_b.plot(df_ub1['DateTime_UTC'], dr_ub1, color=C_UB1, label=f'UB1 (SE$_0$={se0_ub1:.0f})', **line_args)
ax_b.plot(df_ub2['DateTime_UTC'], dr_ub2, color=C_UB2, label=f'UB2 (SE$_0$={se0_ub2:.0f})', **line_args)
ax_b.plot(df_ub4['DateTime_UTC'], dr_ub4, color=C_UB4, label=f'UB4 (SE$_0$={se0_ub4:.0f})', **line_args)
ax_b.axhline(y=DR_THRESHOLD, color='#C62828', linewidth=1.5, linestyle='-', alpha=0.9, zorder=5)
ax_b.text(pd.Timestamp('2023-04-01'), DR_THRESHOLD + 2, f'DR = {DR_THRESHOLD}%',
          fontsize=11, color='#C62828', fontweight='bold', ha='left', va='bottom')
ax_b.axhline(y=0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)
add_reference_line(ax_b, REFERENCE_DATE, dashed=True)
add_external_event_lines(ax_b, external_events)
ax_b.annotate('Different SE$_0$ per station\n\u2192 incomparable DR scales',
              xy=(pd.Timestamp('2023-03-01'), 80),
              fontsize=8.5, color='#555555', ha='center', style='italic',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#FBC02D', alpha=0.9))
add_panel_label(ax_b, 'b)')
configure_panel(ax_b, 'Decay\nRatio [%]', is_last=False)  # ticks base (mes, sin label)
configure_month_week_ticks(ax_b, xlabel='Month')          # NUEVO: mes+semana visibles
ax_b.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02), ncol=1, framealpha=0.9, fontsize=10)

# ---- Helper para paneles de zoom (c,d,e,f) ----
def plot_zoom_raw(ax, t0, t1, letter):
    """Zoom de SE cruda para las 3 estaciones, con ejes X e Y etiquetados."""
    ax.set_xlim(t0, t1)
    add_eruptive_bands(ax, periods_df)
    ax.plot(df_ub1['DateTime_UTC'], df_ub1['SH_Shannon_Smooth'], color=C_UB1, label='UBI1', **line_args)
    ax.plot(df_ub2['DateTime_UTC'], df_ub2['SH_Shannon_Smooth'], color=C_UB2, label='UBI2', **line_args)
    ax.plot(df_ub4['DateTime_UTC'], df_ub4['SH_Shannon_Smooth'], color=C_UB4, label='UBI4', **line_args)
    add_external_event_lines(ax, external_events)
    add_reference_line(ax, REFERENCE_DATE, dashed=True)
    ax.set_ylabel('Shannon\nEntropy [Bits]', fontsize=13)
    configure_month_week_ticks(ax, xlabel='Date')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    add_panel_label(ax, letter)

def plot_zoom_norm(ax, t0, t1, letter):
    """Zoom de SE normalizada (z-score) para las 3 estaciones."""
    ax.set_xlim(t0, t1)
    add_eruptive_bands(ax, periods_df)
    ax.axhspan(-3, 3, color='#DDDDDD', alpha=0.25, zorder=0)
    ax.axhline(3, color='#C62828', lw=0.9, alpha=0.8, dashes=(8, 5), zorder=1)
    ax.axhline(-3, color='#C62828', lw=0.9, alpha=0.8, dashes=(8, 5), zorder=1)
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5, zorder=1)
    ax.plot(df_ub1['DateTime_UTC'], z_ub1_sm, color=C_UB1, label='UB1', **line_args)
    ax.plot(df_ub2['DateTime_UTC'], z_ub2_sm, color=C_UB2, label='UB2', **line_args)
    ax.plot(df_ub4['DateTime_UTC'], z_ub4_sm, color=C_UB4, label='UB4', **line_args)
    add_reference_line(ax, REFERENCE_DATE, dashed=True)
    add_external_event_lines(ax, external_events)
    ax.set_ylabel('Normalized SE\n(z-score) [\u03c3]', fontsize=13)
    configure_month_week_ticks(ax, xlabel='Date')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    add_panel_label(ax, letter)

# ---- Panel c) Zoom SE cruda, 17-abr a 17-jun-2023 ----
print("Panel c: Zoom SE cruda (17-abr a 17-jun)...")
plot_zoom_raw(ax_c, ZOOM1_START, ZOOM1_END, 'c)')

# ---- Panel d) Zoom SE normalizada, 17-abr a 17-jun-2023 ----
print("Panel d: Zoom SE normalizada (17-abr a 17-jun)...")
plot_zoom_norm(ax_d, ZOOM1_START, ZOOM1_END, 'd)')

# ---- Panel e) Zoom SE cruda, 17-oct a 17-dic-2023 ----
print("Panel e: Zoom SE cruda (17-oct a 17-dic)...")
plot_zoom_raw(ax_e, ZOOM2_START, ZOOM2_END, 'e)')

# ---- Panel f) Zoom SE normalizada, 17-oct a 17-dic-2023 ----
print("Panel f: Zoom SE normalizada (17-oct a 17-dic)...")
plot_zoom_norm(ax_f, ZOOM2_START, ZOOM2_END, 'f)')

plt.tight_layout()
fig2.savefig(f"{OUT_DIR}/Fig2_split_new.png", dpi=300, bbox_inches='tight', facecolor='white')
fig2.savefig(f"{OUT_DIR}/Fig2_split_new.pdf", bbox_inches='tight', facecolor='white')
fig2.savefig(f"{OUT_DIR}/Fig2_split_new.svg", bbox_inches='tight', facecolor='white')
print("Saved: Fig2_split_new.{png,pdf,svg}")
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════
#  FIGURA 3 (NUEVA) — Igual a la figura original SIN paneles b) y c)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CREATING FIGURE 3 (5 paneles: a, b, c, d, e)")
print("=" * 60)

fig3 = plt.figure(figsize=(16, 18), dpi=300)
gs3 = fig3.add_gridspec(5, 1, height_ratios=[2.5, 1.4, 1.4, 1.4, 1.4], hspace=0.041)

ax_images_bar = fig3.add_subplot(gs3[0])

with sns.axes_style("darkgrid"):
    ax_znorm     = fig3.add_subplot(gs3[1])   # b) SE normalizada [antiguo d] -> lleva leyenda
    ax_freqindex = fig3.add_subplot(gs3[2])   # c) FI normalizada [antiguo e]
    ax_kurtosis  = fig3.add_subplot(gs3[3])   # d) Kurtosis normalizada [antiguo f]
    ax_ssam      = fig3.add_subplot(gs3[4])   # e) SSAM normalizada [antiguo g]

# ---- Panel a) IMAGENES + BARRA DE ACTIVIDAD (idéntico al original) ----
print("Panel a: Images + Activity Bar...")
ax_images_bar.set_xlim(xlim_full)
ax_images_bar.set_ylim(0, 100)

bar_y = 0
bar_height = 6

add_activity_bar(ax_images_bar, bar_y, bar_height)
add_eruptive_periods(ax_images_bar, periods_df, bar_y, bar_height)

outline_start = mdates.date2num(pd.Timestamp("2023-01-01 01:59:59"))
outline_end = mdates.date2num(pd.Timestamp("2023-12-31 17:59:59"))
outline_rect = Rectangle((outline_start, bar_y), outline_end - outline_start, bar_height,
                          facecolor='none', edgecolor='#555555', linewidth=0.8,
                          transform=ax_images_bar.transData, clip_on=True, zorder=6)
ax_images_bar.add_patch(outline_rect)

for _, exp in explosions_df.iterrows():
    x_pos = mdates.date2num(exp['inicio'])
    ax_images_bar.plot([x_pos, x_pos], [bar_y, bar_y + bar_height], color=COLORS['explosion_marker'], linewidth=0.8, alpha=0.9, zorder=5)

marker_y = bar_y + bar_height + 2

for _, period in emissions_df.iterrows():
    ax_images_bar.scatter(period['inicio'], marker_y, marker=volcano_marker, s=60, color=COLORS['emission_marker'], edgecolor='black', linewidth=0.3, zorder=6)

for _, exp in explosions_df.iterrows():
    ax_images_bar.scatter(exp['inicio'], marker_y, marker=volcano_marker, s=60, color=COLORS['explosion_marker'], edgecolor='darkred', linewidth=0.3, zorder=7)

for _, event in lahars_df.iterrows():
    x_pos = mdates.date2num(event['DateTime'])
    ax_images_bar.plot([x_pos, x_pos], [bar_y, bar_y + bar_height], color=COLORS['lahar'], linewidth=0.8, alpha=0.9, zorder=5)
    ax_images_bar.scatter(event['DateTime'], marker_y, marker='v', s=60, color=COLORS['lahar'], edgecolor='black', linewidth=0.3, zorder=8)

for _, event in vt_sismo_df.iterrows():
    event_type = event['EventType'].lower()
    marker_style = 'v' if 'vt' in event_type else '*'
    marker_size = 60 if 'vt' in event_type else 100
    x_pos = mdates.date2num(event['DateTime'])
    ax_images_bar.plot([x_pos, x_pos], [bar_y, bar_y + bar_height], color=COLORS['vt_sismo'], linewidth=0.8, alpha=0.9, zorder=5)
    ax_images_bar.scatter(event['DateTime'], marker_y, marker=marker_style, s=marker_size, color=COLORS['vt_sismo'], edgecolor='black', linewidth=0.3, zorder=8)

if image_info:
    x_range = end_date - start_date
    padding = x_range * 0.02
    plot_start = start_date + padding
    plot_end = end_date - padding
    total_width = plot_end - plot_start

    top_row_y, bottom_row_y = 78, 42
    n_images = len(image_info)
    top_count = (n_images + 1) // 2
    bottom_count = n_images - top_count

    top_spacing = total_width / (top_count - 1) if top_count > 1 else total_width
    bottom_spacing = total_width / (bottom_count - 1) if bottom_count > 1 else total_width

    top_x = [plot_start + i * top_spacing for i in range(top_count)]
    bottom_x = [plot_start + i * bottom_spacing for i in range(bottom_count)]

    for i, img_data in enumerate(image_info):
        row_y, x_pos = (top_row_y, top_x[i]) if i < top_count else (bottom_row_y, bottom_x[i - top_count])
        try:
            img = plt.imread(img_data['path'])
            imagebox = OffsetImage(img, zoom=0.14)
            ab = AnnotationBbox(imagebox, (mdates.date2num(x_pos), row_y), frameon=False, pad=0, zorder=10)
            ax_images_bar.add_artist(ab)

            circle = plt.Circle((mdates.date2num(x_pos), row_y), 4.0, fill=False, edgecolor=img_data['color'], linewidth=2.0, zorder=9, transform=ax_images_bar.transData)
            ax_images_bar.add_patch(circle)

            text = ax_images_bar.text(mdates.date2num(x_pos), row_y + 9, img_data['label'], fontsize=9, fontweight='normal', ha='center', va='bottom', color='black', rotation=25, zorder=15)
            text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white'), path_effects.Normal()])

            arrow = FancyArrowPatch((mdates.date2num(x_pos), row_y - 8), (mdates.date2num(img_data['datetime']), marker_y + 3), connectionstyle="arc3,rad=0.15", arrowstyle="-|>", mutation_scale=6, linewidth=0.5, color=img_data['color'], alpha=0.8, zorder=8)
            ax_images_bar.add_patch(arrow)
        except Exception as e:
            print(f"Error with image {img_data['path']}: {e}")

ax_images_bar.set_xlim(mdates.date2num(start_date), mdates.date2num(end_date))
ax_images_bar.xaxis.set_major_locator(mdates.MonthLocator())
ax_images_bar.xaxis.set_major_formatter(plt.NullFormatter())
ax_images_bar.tick_params(axis='x', direction='out', length=7, which='major', labelbottom=False, color='#555555')
ax_images_bar.tick_params(axis='y', left=False, labelleft=False)

ax_images_bar.spines['bottom'].set_visible(True)
ax_images_bar.spines['top'].set_visible(False)
ax_images_bar.spines['left'].set_visible(False)
ax_images_bar.spines['right'].set_visible(False)
ax_images_bar.spines['bottom'].set_color('#555555')
ax_images_bar.spines['bottom'].set_linewidth(0.8)

add_panel_label(ax_images_bar, 'a)')

# ---- Panel b) SE normalizada [antiguo panel d] — AHORA lleva la leyenda ----
print("Panel b: Normalized Shannon Entropy (con leyenda)...")
ax_znorm.set_xlim(xlim_full)
ax_znorm.set_yticks([-3, 0, 3])
ax_znorm.set_ylim(-6, 4.5)
add_eruptive_bands(ax_znorm, periods_df)
ax_znorm.axhspan(-3, 3, color='#DDDDDD', alpha=0.25, zorder=0)
ax_znorm.axhline(3, color='#C62828', lw=0.9, alpha=0.8, dashes=(8, 5), zorder=1)
ax_znorm.axhline(-3, color='#C62828', lw=0.9, alpha=0.8, dashes=(8, 5), zorder=1)
ax_znorm.axhline(0, color='gray', lw=0.4, alpha=0.5, zorder=1)
ax_znorm.plot(df_ub1['DateTime_UTC'], z_ub1_sm, color=C_UB1, label='UB1', **line_args)
ax_znorm.plot(df_ub2['DateTime_UTC'], z_ub2_sm, color=C_UB2, label='UB2', **line_args)
ax_znorm.plot(df_ub4['DateTime_UTC'], z_ub4_sm, color=C_UB4, label='UB4', **line_args)
ax_znorm.text(pd.Timestamp('2023-04-01'), 3.3, 'z = \u00b13', fontsize=11,
              color='#C62828', fontweight='normal', ha='left', va='bottom')
add_reference_line(ax_znorm, REFERENCE_DATE, dashed=True)
add_external_event_lines(ax_znorm, external_events)
configure_panel(ax_znorm, 'Normalized SE\n(z-score) [\u03c3]', is_last=False)
add_panel_label(ax_znorm, 'b)')
# Leyenda general (la misma que antes estaba en el antiguo panel b, ahora eliminado)
ax_znorm.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                 ncol=3, fontsize=10)

def plot_znorm_panel(ax, z1, z2, z4, ylabel, letter, is_last=False):
    ax.set_xlim(xlim_full)
    add_eruptive_bands(ax, periods_df)
    ax.axhspan(-3, 3, color='#DDDDDD', alpha=0.25, zorder=0)
    ax.axhline(3, color='#C62828', lw=0.9, alpha=0.8, dashes=(8, 5), zorder=1)
    ax.axhline(-3, color='#C62828', lw=0.9, alpha=0.8, dashes=(8, 5), zorder=1)
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5, zorder=1)
    ax.plot(df_ub1['DateTime_UTC'], z1, color=C_UB1, label='UB1', **line_args)
    ax.plot(df_ub2['DateTime_UTC'], z2, color=C_UB2, label='UB2', **line_args)
    ax.plot(df_ub4['DateTime_UTC'], z4, color=C_UB4, label='UB4', **line_args)
    add_reference_line(ax, REFERENCE_DATE, dashed=True)
    add_external_event_lines(ax, external_events)
    configure_panel(ax, ylabel, is_last=is_last)
    add_panel_label(ax, letter)

# ---- Panel c) FI normalizada [antiguo panel e] ----
print("Panel c: Normalized Frequency Index...")
ax_freqindex.set_yticks([-3, 0, 3])
ax_freqindex.set_ylim(-4, 4.5)
plot_znorm_panel(ax_freqindex, zfi_ub1, zfi_ub2, zfi_ub4, 'Normalized FI\n(z-score) [\u03c3]', 'c)')

# ---- Panel d) Kurtosis normalizada [antiguo panel f] ----
print("Panel d: Normalized Kurtosis...")
plot_znorm_panel(ax_kurtosis, zk_ub1, zk_ub2, zk_ub4, 'Normalized K\n(z-score) [\u03c3]', 'd)')
ax_kurtosis.set_yticks([-3, 0, 3, 6])

# ---- Panel e) SSAM normalizada [antiguo panel g] — ultimo, con ticks mes/semana ----
print("Panel e: Normalized SSAM...")
plot_znorm_panel(ax_ssam, zss_ub1, zss_ub2, zss_ub4, 'Normalized SSAM\n(z-score) [\u03c3]', 'e)', is_last=True)
configure_month_week_ticks(ax_ssam, xlabel=None)  # is_last=True ya puso 'Month'; solo agrega semana
ax_ssam.set_yticks([0, 6, 12])

plt.tight_layout()
fig3.savefig(f"{OUT_DIR}/Fig3_split_new.png", dpi=300, bbox_inches='tight', facecolor='white')
fig3.savefig(f"{OUT_DIR}/Fig3_split_new.pdf", bbox_inches='tight', facecolor='white')
fig3.savefig(f"{OUT_DIR}/Fig3_split_new.svg", bbox_inches='tight', facecolor='white')
print("Saved: Fig3_split_new.{png,pdf,svg}")
plt.close(fig3)

print("\n" + "=" * 60)
print("PROCESS COMPLETED — Figura 2 y Figura 3 generadas.")
print("=" * 60)
