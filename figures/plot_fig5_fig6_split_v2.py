"""
========================================================================
 FIGURES 5 & 6 (SPLIT) — Temporal Validation: UBI4/2019 (model trained 2023)
 Split from the original 6-panel figure (gen_fig6_final_v5.py) into two
 new figures, with the 2019 eruptive catalog corrected to 9 documented
 episodes (3 explosions + 6 emissions) and th_op updated to the
 definitive value (0.6140).

 FIGURE 5 (4 panels, 1 column):
   a) Chronology bar only (phase bar + event windows), NO photos/arrows/markers
   b) Predicted volcanic state (dominant-state bands)      [= old panel c]
   c) Normalized seismic parameters (SE/FI/K/SSAM offset)  [= old panel b]
   d) P(>=Unrest) + persistence counter                    [= old panel e]
   -> all 4 panels show: IGP VT-swarm onset (18-Jun) as a vertical line,
      and the onset of ash activity as a red dashed vertical line.

 FIGURE 6 (4 rows x 1 col; row 4 split into 2 columns):
   a) Chronology + event photos, NO phase-level bar; markers show only
      the explosion/emission symbol                        [= old panel a]
   b) Alert probability P(Alert)                            [= old panel d]
   c) Persistence counter + per-event anticipation labels   [= old panel f]
   d) Zoom of panel c, 15-30 June 2019 (with exact declaration instant)
   e) Zoom of panel c, 15-30 July 2019 (with exact declaration instant)

 CATALOG CORRECTIONS APPLIED (vs. gen_fig6_final_v5.py):
   - 24-Jun: start corrected 10:43 -> 12:20, end 11:39 -> 19:52
   - 17-Jul: REMOVED from the catalog (reclassified as precursor /
     Intranquilidad — water-vapor-only emission, <600 m, no ash)
   - 18-Jul: kept as a real cataloged emission (05:14 -> 18:17)
   - 19-Jul: start corrected 07:23 -> 07:37 (tail kept to 22-Jul 23:59)
   - 3-Sep : NEW explosion added (18:58 -> 20:00) — previously missing
   - 4-Sep : kept split into two distinct emissions (a: 10:56-13:47,
     b: 13:50-14:50), matching the real chronology (two separate pulses,
     1500 m and 2000 m respectively)
   - TH_OP : 0.4667 -> 0.6140 (definitive operational threshold)
   - Stale per-cycle detection markers (det_jun/det_jul, computed under
     the old threshold/catalog) were REMOVED rather than guessed; the
     per-event anticipation shown in Figure 6 panels c/d/e is computed
     directly from the persistence counter at run time, so it will be
     correct automatically once this script runs against the CSV
     regenerated with the corrected catalog and th_op.
========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as path_effects
import seaborn as sns
from datetime import datetime
import glob
import os

# ── Paths ────────────────────────────────────────────────────────
basepath = '/mnt/i/ACTIVIDADES/2024/02_proyecto_SINFRA/RES/zed/To_GMT'
csv_file = f'{basepath}/ubinas_v16_ub4_2019.csv'
csv_smooth_b = '/mnt/i/ACTIVIDADES/2024/02_proyecto_SINFRA/RES/zed/UB4_2019_10m_smoothed_parameters.csv'
images_folder = '/mnt/i/ACTIVIDADES/2024/02_proyecto_SINFRA/BIN/GMT/ubinas_eruptive_events_2019'

# ── Style ────────────────────────────────────────────────────────
sns.set_style("ticks")
plt.rcParams.update({
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# ── Colors (identical to original script) ─────────────────────────
C_AREA_EP = '#4E4E4E'
LABELS_4  = ['Quiescence', 'Unrest', 'Pre-eruptive', 'Eruptive']

C_ENT  = '#1565C0'
C_KURT = '#C62828'
C_FI   = '#2E7D32'
C_SSAM = '#6A1B9A'

C_ALERT   = '#D32F2F'
C_TREND   = '#0D2240'
C_COUNTER = '#1565C0'
C_SIGNAL  = '#7BAFD4'
C_DET     = '#2CA02C'

C_UB1 = '#1F77B4'
C_UB2 = '#FF7F0E'
C_UB4 = '#2CA02C'
FIG2_ERUPTIVE_PERIOD = '#B5BBC7'
FIG2_GREEN  = '#70AD47'
FIG2_YELLOW = '#FFDF27'
FIG2_ORANGE = '#ED7D31'
C_QUIET   = FIG2_GREEN
C_UNREST  = FIG2_YELLOW
C_PREERUP = FIG2_ORANGE
C_ERUPT   = '#C0392B'
COLORS_4  = [C_QUIET, C_UNREST, C_PREERUP, C_ERUPT]

C_EXP_MARKER = '#FF3333'
C_EMI_MARKER = '#FFC107'

# NEW: color for the IGP VT-swarm reference line (distinct from the red
# dashed ash-onset line already used by add_datelines)
C_IGP_LINE = '#000000'

# ── v16 parameters — DEFINITIVE VALUES ─────────────────────────────
TH_OP  = 0.6200          # criterio cero-falsas-alarmas (pipeline Python final)
TH_YOU = 0.1709
TH_UNREST = 0.9024       # P99.5 de quietud (pipeline Python final)
N_PERSIST     = 360
N_PERSIST_JUN = 60
W_ROLLING         = 30 * 1440
MIN_VALID_ROLLING = 1440
STEP = 5

# ── Load data ────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(csv_file)
time_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()][0]
df['Time'] = pd.to_datetime(df[time_col])
df = df.set_index('Time').sort_index()

T_PLOT_START = pd.Timestamp('2019-01-01 00:00:00')
df = df[df.index >= T_PLOT_START]

for col in df.columns:
    if col != 'valid':
        df.loc[df['valid'] == 0, col] = np.nan

print("Gap-day masking disabled (using raw 'valid' flags from CSV only)")

n = len(df)
T = df.index

# ── Load better-smoothed parameters for panel (c, normalized params) ──
print("Loading smoothed parameters for the normalized-parameters panel...")
dfb = pd.read_csv(csv_smooth_b)
_tcol = [c for c in dfb.columns if 'time' in c.lower() or 'date' in c.lower()][0]
dfb['Time'] = pd.to_datetime(dfb[_tcol])
dfb = dfb.set_index('Time').sort_index()

def _causal_z(series, window=W_ROLLING, minp=MIN_VALID_ROLLING):
    mu = series.shift(1).rolling(window, min_periods=minp).mean()
    sd = series.shift(1).rolling(window, min_periods=minp).std().clip(lower=1e-6)
    return (series - mu) / sd

zb = {}
_smooth_cols = {
    'SE':   'SH_Shannon_Smooth',
    'FI':   'FreqIndex_Smooth',
    'K':    'Curtosis_Acum_Smooth',
    'SSAM': 'SSAM_Acum_Smooth',
}
for _lab, _col in _smooth_cols.items():
    if _col in dfb.columns:
        zb[_lab] = _causal_z(dfb[_col]).rolling(60, center=True, min_periods=10).mean()
    else:
        print(f"  WARNING: column {_col} not found in smoothed CSV")
        zb[_lab] = pd.Series(np.nan, index=dfb.index)
print(f"  Smoothed panel data: {len(dfb):,} rows")

# ── Derived signals ──────────────────────────────────────────────
sm = 60
print("  Computing rolling z-score (W=30d, causal)...")
for col in ['entropy_val', 'kurtosis_val', 'freqidx_val', 'ssam_val']:
    mu_roll = df[col].shift(1).rolling(W_ROLLING, min_periods=MIN_VALID_ROLLING).mean()
    sd_roll = df[col].shift(1).rolling(W_ROLLING, min_periods=MIN_VALID_ROLLING).std().clip(lower=1e-6)
    z = (df[col] - mu_roll) / sd_roll
    df[f'{col}_z'] = z.rolling(sm, center=True, min_periods=10).mean()

df['p_alert_24h'] = df['p_alert_sm'].rolling(1440, center=True, min_periods=60).mean()

# ── Persistence counter (uses the corrected TH_OP) ─────────────────
p_sm_full  = df['p_alert_sm'].values
valid_full = df['valid'].values
consec_counter = np.zeros(n)
count = 0
for i in range(n):
    if valid_full[i] == 1 and not np.isnan(p_sm_full[i]) and p_sm_full[i] > TH_OP:
        count += 1
    else:
        count = 0
    consec_counter[i] = count
consec_hours = consec_counter / 60.0
alert_360 = (consec_counter >= N_PERSIST).astype(float)

# ── Unrest signals: P(>=Unrest) = 1 - P(Quiescence) ──
if 'p_unrest_sm' in df.columns:
    p_unrest_sm_full = df['p_unrest_sm'].values
elif 'p_ge_unrest' in df.columns:
    p_unrest_sm_full = df['p_ge_unrest'].rolling(60, min_periods=1).mean().values
else:
    p_ge = 1.0 - df['p_c0']
    p_unrest_sm_full = p_ge.rolling(60, min_periods=1).mean().values

consec_unrest = np.zeros(n)
cu = 0
for i in range(n):
    if valid_full[i] == 1 and not np.isnan(p_unrest_sm_full[i]) and p_unrest_sm_full[i] > TH_UNREST:
        cu += 1
    else:
        cu = 0
    consec_unrest[i] = cu
consec_unrest_hours = consec_unrest / 60.0

# Model unrest declaration (v16) — UNCHANGED, confirmed stable across
# both catalog versions (23-Jun 15:48)
det_unrest = datetime(2019, 6, 22, 9, 52)   # 22-jun 09:52 (pipeline Python final)
igp_unrest = datetime(2019, 6, 18, 0, 0)

# ── Chronology — CORRECTED ──────────────────────────────────────────
T_START, T_END = T[0], T[-1]
T_QUIET_END  = pd.Timestamp('2019-06-17 23:59:00')
T_UNREST_ON  = pd.Timestamp('2019-06-18 00:00:00')
T_ERUPTIVE   = pd.Timestamp('2019-06-24 12:20:00')   # corrected from 12:54
T_CRISIS_END = pd.Timestamp('2019-09-25 23:59:00')
T_POST_END   = pd.Timestamp('2019-12-31 23:59:00')

bg_phases = [
    ('Quiescence',      T_START,      T_QUIET_END,  C_QUIET,  0.06),
    ('Unrest',          T_UNREST_ON,  T_ERUPTIVE,   C_UNREST, 0.08),
    ('Eruptive period', T_ERUPTIVE,   T_CRISIS_END, C_ERUPT,  0.05),
    ('Post-crisis',     T_CRISIS_END, T_POST_END,   C_UNREST, 0.05),
]

# CORRECTED 2019 CATALOG — 9 episodes (3 explosions, 6 emissions).
# 17-Jul removed (water-vapor-only precursor, no ash, reclassified as
# Intranquilidad). 4-Sep kept as two separate emissions. 3-Sep explosion
# added. Times corrected against the field chronology.
ev_type  = ['emission', 'emission', 'explosion', 'explosion',
            'emission', 'explosion', 'emission', 'emission', 'emission']
ev_start = [datetime(2019, 6, 24, 12, 20), datetime(2019, 7, 18, 5, 14),
            datetime(2019, 7, 19, 7, 37),  datetime(2019, 7, 23, 4, 25),
            datetime(2019, 8, 27, 10, 30), datetime(2019, 9, 3, 18, 58),
            datetime(2019, 9, 4, 10, 56),  datetime(2019, 9, 4, 13, 50),
            datetime(2019, 9, 12, 12, 30)]
ev_end   = [datetime(2019, 6, 24, 19, 52), datetime(2019, 7, 18, 18, 17),
            datetime(2019, 7, 22, 23, 59), datetime(2019, 7, 23, 9, 50),
            datetime(2019, 8, 27, 10, 45), datetime(2019, 9, 3, 20, 0),
            datetime(2019, 9, 4, 13, 47),  datetime(2019, 9, 4, 14, 50),
            datetime(2019, 9, 12, 12, 45)]
ev_label = ['Em. 24-Jun', 'Em. 18-Jul', 'Exp. 19-Jul', 'Exp. 23-Jul',
            'Em. 27-Aug', 'Exp. 3-Sep', 'Em. 4-Sep(a)', 'Em. 4-Sep(b)',
            'Em. 12-Sep']

# NOTE: the old per-cycle detection markers (det_jun, ev_jun, det_jul,
# ev_jul) computed under the previous catalog/threshold were removed.
# They are not reused anywhere below; per-event anticipation is instead
# computed directly from consec_hours at run time (see
# `compute_event_anticipations()` below), so it is automatically correct
# for the new catalog and TH_OP once this script runs against the CSV
# regenerated with both corrections applied upstream.


# ══════════════════ HELPERS (identical to original + 1 new) ══════════════════
def add_phase_bands(ax, f=1.0):
    for _, t0, t1, col, a in bg_phases:
        ax.axvspan(t0, t1, alpha=a * f, color=col, zorder=0, lw=0)

def add_event_shading(ax, alpha=0.10):
    for i in range(len(ev_start)):
        col = C_ERUPT if ev_type[i] == 'explosion' else C_PREERUP
        ax.axvspan(ev_start[i], ev_end[i], alpha=alpha, color=col, zorder=1, lw=0)

def add_datelines(ax):
    """Red dashed line = onset of ash activity (T_ERUPTIVE)."""
    ax.axvline(T_QUIET_END, color='k', dashes=(8, 5), lw=0.6, alpha=0.74, zorder=2)
    ax.axvline(T_ERUPTIVE, color=C_ERUPT, dashes=(8, 5), lw=0.6, alpha=0.74, zorder=2)
    ax.axvline(T_CRISIS_END, color=C_AREA_EP, dashes=(8, 5), lw=0.6, alpha=0.4, zorder=2)

def add_igp_line(ax):
    """NEW: vertical line marking the IGP-reported VT-swarm onset (18-Jun)."""
    #ax.axvline(igp_unrest, color=C_IGP_LINE, lw=0.8, ls='--', alpha=0.75, zorder=3)

def add_panel_label(ax, label, x=-0.032, y=1.03):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=18,
            fontweight='normal', va='top', ha='left', clip_on=False)

def configure_panel_fig6(ax, ylabel, is_last=False, label_x=-0.055):
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
        ax.set_xlabel('2019', fontsize=13)

def parse_image_datetime_v2(filename):
    parts = filename.split('_')
    if len(parts) >= 4:
        dp, tp = parts[2], parts[3]
        try:
            return pd.Timestamp(int(dp[:4]), int(dp[4:6]), int(dp[6:8]),
                                 int(tp[:2]), int(tp[2:4]))
        except Exception:
            return None
    return None

def create_volcano_marker():
    import matplotlib.path as mpath
    verts = np.array([(-0.5, -0.5), (0.5, -0.5), (0.3, 0.5), (0.1, 0.3),
                       (-0.1, 0.3), (-0.3, 0.5), (-0.5, -0.5)])
    return mpath.Path(verts)

def dominant_state_bands(times, states, window_hours=6):
    dt_min = window_hours * 60
    bands = []; i = 0; nn = len(times)
    while i < nn:
        t0 = times[i]; j = i
        while j < nn and (times[j] - t0) < np.timedelta64(dt_min, 'm'):
            j += 1
        chunk = states[i:j]; cv = chunk[~np.isnan(chunk)]
        if len(cv) > 0:
            vals, counts = np.unique(cv.astype(int), return_counts=True)
            bands.append((times[i], times[min(j, nn - 1)], vals[np.argmax(counts)]))
        i = j
    return bands

def compute_event_anticipations():
    """Per-event anticipation, computed from the (corrected) persistence
    counter at run time — walks backward from each event's start to find
    the onset of the sustained-alert streak that precedes it (or that is
    still active at the event's start). Returns a list of dicts, one per
    event, each with 'onset' (Timestamp or None) and 'hours' (float or None).
    This replaces the old hardcoded det_jun/det_jul markers."""
    sustained_full = consec_hours >= (N_PERSIST / 60.0)
    T_full = df.index
    out = []
    for i in range(len(ev_start)):
        ev_t = pd.Timestamp(ev_start[i])
        ev_idx = T_full.get_indexer([ev_t], method='nearest')[0]
        ant_hours = None
        onset_t = None
        if sustained_full[ev_idx]:
            j = ev_idx
            while j > 0 and sustained_full[j - 1]:
                j -= 1
            onset_t = T_full[j]
            ant_hours = (ev_t - onset_t).total_seconds() / 3600.0
        else:
            window_start = max(0, ev_idx - 48 * 60)
            recent = sustained_full[window_start:ev_idx]
            if recent.any():
                last_true = window_start + np.where(recent)[0][-1]
                j = last_true
                while j > 0 and sustained_full[j - 1]:
                    j -= 1
                onset_t = T_full[j]
                ant_hours = (ev_t - onset_t).total_seconds() / 3600.0
        out.append({'onset': onset_t, 'hours': ant_hours})
    return out

EV_ANTICIPATION = compute_event_anticipations()
print("Per-event anticipation (recomputed with corrected catalog + th_op):")
for i, r in enumerate(EV_ANTICIPATION):
    h = f"{r['hours']:.1f} h" if r['hours'] is not None else "no alert"
    print(f"  {ev_label[i]:14s} {ev_start[i]}  ->  {h}")


# ── Load 2019 event photos (used only in Figure 6, panel a) ──
exp_images_2019 = sorted(glob.glob(os.path.join(images_folder, "UBI_EXP_2019*_circle.png")))
emi_images_2019 = sorted(glob.glob(os.path.join(images_folder, "UBI_EMI_2019*_circle.png")))
image_info = []
for _ip in exp_images_2019:
    _dt = parse_image_datetime_v2(os.path.basename(_ip))
    if _dt:
        image_info.append({'path': _ip, 'datetime': _dt,
                            'label': _dt.strftime('%Y-%m-%d %H:%M'),
                            'type': 'explosion', 'color': C_EXP_MARKER})
for _ip in emi_images_2019:
    _dt = parse_image_datetime_v2(os.path.basename(_ip))
    if _dt:
        image_info.append({'path': _ip, 'datetime': _dt,
                            'label': _dt.strftime('%Y-%m-%d %H:%M'),
                            'type': 'emission', 'color': C_EMI_MARKER})
image_info.sort(key=lambda x: x['datetime'])
print(f"  Found {len(image_info)} images for 2019 events")
volcano_marker = create_volcano_marker()

T_arr = T.values
pred_4c = df['pred_4c'].values.astype(float)
pred_4c[valid_full == 0] = np.nan
if 'state_display' in df.columns:
    # versión limpia para visualización (transitorios tectónicos atenuados);
    # pred_4c_hmm_causal se conserva para métricas, pero la figura muestra
    # state_display. Ver caption / Sección de métodos.
    pred_hmm = df['state_display'].values.astype(float)
    pred_hmm[valid_full == 0] = np.nan
elif 'pred_4c_hmm_causal' in df.columns:
    pred_hmm = df['pred_4c_hmm_causal'].values.astype(float)
    pred_hmm[valid_full == 0] = np.nan
else:
    pred_hmm = None

T_sub = T_arr[::STEP]
p_sm_sub  = p_sm_full[::STEP]
ch_sub    = consec_hours[::STEP]
alert_sub = alert_360[::STEP]
p_unrest_sub  = p_unrest_sm_full[::STEP]
cu_hours_sub  = consec_unrest_hours[::STEP]


# ════════════════════════════════════════════════════════════════════
#  FIGURE 5 — 4 panels, 1 column
# ════════════════════════════════════════════════════════════════════
print("\nCreating Figure 5 (4 panels: a, b, c, d)...")
fig5 = plt.figure(figsize=(16, 16))
gs5 = gridspec.GridSpec(4, 1, height_ratios=[0.1, 0.36, 1.95, 0.65],
                         hspace=0.08, figure=fig5)
ax5 = [fig5.add_subplot(gs5[0])]
with sns.axes_style("ticks"):
    for i in range(1, 4):
        ax5.append(fig5.add_subplot(gs5[i]))

# ── Panel a) Chronology bar ONLY (no photos / arrows / symbols) ──
ax = ax5[0]
ax.set_xlim(mdates.date2num(T_START), mdates.date2num(T_END))
ax.set_ylim(0, 12)
bar_y, bar_height = 0, 12
for label, t0, t1, col, _ in bg_phases:
    s = mdates.date2num(max(t0, T_START)); e = mdates.date2num(t1)
    if e <= mdates.date2num(T_START):
        continue
    ax.add_patch(mpatches.Rectangle((s, bar_y), e - s, bar_height,
                 facecolor=col, alpha=0.85, edgecolor='none', zorder=2))
for i in range(len(ev_start)):
    s = mdates.date2num(ev_start[i]); e = mdates.date2num(ev_end[i])
    ax.add_patch(mpatches.Rectangle((s, bar_y), e - s, bar_height,
                 facecolor=C_ERUPT, alpha=0.7, edgecolor='none', zorder=3))
for i in range(len(ev_start)):
    if ev_type[i] == 'explosion':
        x_pos = mdates.date2num(ev_start[i])
        ax.plot([x_pos, x_pos], [bar_y, bar_y + bar_height],
                color=C_EXP_MARKER, linewidth=0.8, alpha=0.9, zorder=5)
outline_s = mdates.date2num(T_START + pd.Timedelta(hours=3))
outline_e = mdates.date2num(T_END - pd.Timedelta(hours=3))
ax.add_patch(mpatches.Rectangle((outline_s, bar_y), outline_e - outline_s, bar_height,
             facecolor='none', edgecolor='#555555', linewidth=0.8, zorder=6))
phase_labels_pos = [
    ('Quiescence',      max(T_START, T_START) + (T_QUIET_END - max(T_START, T_START)) / 2, 'black'),
    ('Unrest',          T_UNREST_ON + (T_ERUPTIVE - T_UNREST_ON) / 2, 'black'),
    ('Eruptive period', T_ERUPTIVE + (T_CRISIS_END - T_ERUPTIVE) / 2, 'black'),
    ('Post-crisis',     T_CRISIS_END + (T_POST_END - T_CRISIS_END) / 2, 'black'),
]
for lab, t_pos, txtcol in phase_labels_pos:
    ax.text(mdates.date2num(t_pos), bar_y + bar_height / 2, lab,
            ha='center', va='center', fontsize=12, color=txtcol, zorder=10)
ax.axvline(T_ERUPTIVE, color=C_ERUPT, dashes=(8, 5), lw=0.8, alpha=0.85, zorder=7)
add_igp_line(ax)
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator([1, 8, 15, 22]))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.tick_params(axis='x', direction='out', length=8, which='major', labelbottom=False, color='#555555')
ax.tick_params(axis='x', direction='out', length=4, which='minor', labelbottom=False, color='#555555')
ax.tick_params(axis='y', left=False, labelleft=False)
for sp in ['top', 'right', 'left']:
    ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(0.8)
ax.spines['bottom'].set_color('#555555')
add_panel_label(ax, 'a)')

# ── Panel b) Predicted state [= old panel c] ──
ax = ax5[1]
draw_state_row_bands_top = dominant_state_bands(T_arr, pred_4c, window_hours=6)
def draw_state_row(ax, states, y_center, height=0.54):
    bands = dominant_state_bands(T_arr, states, window_hours=6)
    for t0, t1, st in bands:
        t0d = pd.Timestamp(t0).to_pydatetime(); t1d = pd.Timestamp(t1).to_pydatetime()
        ax.barh(y_center, (t1d - t0d).total_seconds() / 86400, left=mdates.date2num(t0d),
                height=height, color=COLORS_4[int(st)], alpha=0.85, edgecolor='none', zorder=2)
draw_state_row(ax, pred_4c, 1.0)
if pred_hmm is not None:
    draw_state_row(ax, pred_hmm, 0.0)
    ax.set_yticks([0.0, 1.0]); ax.set_yticklabels(['HMM\nsmoothed', 'RF raw'], fontsize=8.5)
    ax.set_ylim(-0.5, 1.5)
else:
    ax.set_yticks([1.0]); ax.set_yticklabels(['RF raw'], fontsize=8.5)
    ax.set_ylim(0.5, 1.5)
for i in range(len(ev_start)):
    ax.axvspan(ev_start[i], ev_end[i], alpha=0.15, color=C_ERUPT, zorder=0, lw=0)
add_datelines(ax)
add_igp_line(ax)
ax.set_ylabel('Predicted state', fontsize=13)
ax.yaxis.set_label_coords(-0.082, 0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator([1, 8, 15, 22]))
#ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.tick_params(direction='out', length=8, which='major', labelsize=12)
ax.tick_params(direction='out', length=4, which='minor')
ax.tick_params(labelbottom=False)
add_panel_label(ax, 'b)')
lg_st = [mpatches.Patch(color=COLORS_4[i], label=LABELS_4[i]) for i in range(4)]
# ax.legend(handles=lg_st, loc='upper left', ncol=4, fontsize=9, framealpha=0.9,
#           borderpad=0.3, handlelength=1.1, columnspacing=0.7)

# ── Panel c) Normalized parameters [= old panel b] ──
ax = ax5[2]
add_phase_bands(ax)
add_datelines(ax)
add_igp_line(ax)
zb_order  = ['SE', 'FI', 'K', 'SSAM']
zb_colors = {'SE': C_ENT, 'FI': C_FI, 'K': C_KURT, 'SSAM': C_SSAM}
OFFSET, CLIP = 8, 7
x_text = pd.Timestamp('2019-01-10')
yticks_pos = []
for k, lab in enumerate(zb_order):
    base = (len(zb_order) - 1 - k) * OFFSET
    yticks_pos.append(base)
    s = zb[lab].reindex(dfb.index)
    vals = np.clip(s.values.astype(float), -CLIP, CLIP)
    ax.axhline(base, color='gray', lw=0.8, alpha=0.65, zorder=1)
    ax.axhline(base + 3, color=zb_colors[lab], lw=0.7, dashes=(8, 5), alpha=0.75, zorder=1)
    ax.axhline(base - 3, color=zb_colors[lab], lw=0.7, dashes=(8, 5), alpha=0.75, zorder=1)
    ax.plot(dfb.index, base + vals, color=zb_colors[lab], lw=0.8, alpha=0.9, zorder=3)
    ax.text(x_text, base + 3.05, 'z = \u00b13', fontsize=12, fontweight='normal',
            color=zb_colors[lab], va='bottom', ha='left')
ax.set_ylim(-2, 29)
ax.set_yticks(yticks_pos)
ax.set_yticklabels(zb_order, fontsize=13)
configure_panel_fig6(ax, 'Normalized parameters\n(z-score) [\u03c3]')
add_panel_label(ax, 'c)')

# ── Panel d) P(>=Unrest) + persistence counter [= old panel e] ──
ax = ax5[3]
add_phase_bands(ax)
add_datelines(ax)
add_igp_line(ax)
vmu = ~np.isnan(p_unrest_sub)
C_UNREST_CURVE = '#E2B47B'
C_UNREST_HI    = '#D98629'
ax.fill_between(T_sub[vmu], 0, p_unrest_sub[vmu], color=C_UNREST_CURVE, alpha=0.10, lw=0, zorder=1)
line_pu, = ax.plot(T_sub, p_unrest_sub, color=C_UNREST_CURVE, lw=0.6, alpha=0.85, zorder=3,
                    label='P($\\geq$ Unrest), 60-min trailing mean')
p_u_above = np.where(np.nan_to_num(p_unrest_sub, nan=0) >= TH_UNREST, p_unrest_sub, np.nan)
line_hi, = ax.plot(T_sub, p_u_above, color=C_UNREST_HI, lw=1.1, zorder=4, label='Above th$_{unrest}$')
ax.axhline(TH_UNREST, color=C_UNREST_HI, ls='-', lw=1.0, alpha=0.75, zorder=4)
ax.text(pd.Timestamp('2019-03-01'), TH_UNREST + 0.02,
        f'th$_{{unrest}}$ = {TH_UNREST:.3f}  (P$_{{99.5}}$ of quiescence)',
        fontsize=11, color=C_UNREST_HI, fontstyle='italic')
ax.axvline(det_unrest, color=C_DET, dashes=(8, 5), lw=0.6, alpha=0.95, zorder=4)
ax.annotate(f'Unrest declared (model)\n{det_unrest:%d %b %H:%M} (\u2265 6 h)',
            xy=(det_unrest, 0.15), xytext=(pd.Timestamp('2019-12-05'), 1.055),
            fontsize=8.5, color='#1B5E20', fontweight='normal', ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color=C_DET, lw=0.6),
            bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec=C_DET, alpha=0.92, lw=0.6))
ax_d2 = ax.twinx()
for side in ['left', 'right', 'top', 'bottom']:
    ax_d2.spines[side].set_visible(False)
ax_d2.tick_params(axis='y', labelcolor=C_UNREST_HI, labelsize=14, length=8, width=0.8, direction='out')
ax_d2.fill_between(T_sub, 0, cu_hours_sub, color='#9CC3E3', alpha=0.75, lw=0, zorder=0)
line_cnt, = ax_d2.plot(T_sub, cu_hours_sub, color='#5791CB', lw=1.1, zorder=1,
                        label='Consec. hours above th$_{unrest}$')
ax_d2.axhline(N_PERSIST / 60, color=C_UNREST_HI, dashes=[5, 2], lw=0.9, alpha=0.7)
ax_d2.scatter(det_unrest, [N_PERSIST / 60], color=C_DET, s=45, zorder=6,
              edgecolor='white', linewidth=0.8)
ax_d2.set_ylabel('Consec. hours\nabove th$_{unrest}$', fontsize=14, color='#5791CB')
ax_d2.tick_params(axis='y', labelcolor='#5791CB', labelsize=14)
ax_d2.set_ylim(-2, 58.214)
ax_d2.set_yticks([0, 8, 16, 24, 32, 40])
ax.set_ylim(-0.05, 1.45)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(handles=[line_pu, line_hi, line_cnt], loc='upper left', fontsize=10,
          framealpha=0.9, borderpad=0.3, handlelength=1.5)
configure_panel_fig6(ax, 'P($\\geq$ Unrest)', is_last=True)
ax.text(igp_unrest, 1.0, 'IGP unrest\n18 Jun (VT swarm)', fontsize=8, va='bottom', ha='right',
        multialignment='right', color='k')
add_panel_label(ax, 'd)')

for a in ax5:
    a.set_xlim(T_START, T_END)
plt.figure(fig5.number)
plt.tight_layout(rect=[0, 0.02, 1, 1])
out5_png = os.path.join(basepath, 'Fig5_split_new.png')
out5_pdf = os.path.join(basepath, 'Fig5_split_new.pdf')
plt.savefig(out5_png, dpi=300); plt.savefig(out5_pdf)
plt.close(fig5)
print(f"  Saved: {out5_png}\n  Saved: {out5_pdf}")


# ════════════════════════════════════════════════════════════════════
#  FIGURE 6 — 4 rows x 1 col (row 4 split into 2 columns)
# ════════════════════════════════════════════════════════════════════
print("\nCreating Figure 6 (5 panels: a, b, c, d, e)...")
fig6 = plt.figure(figsize=(16, 18))
# Espaciados pedidos: a-b PEGADOS (hspace casi 0), b-c separación mínima.
# Se usan dos gridspec anidados para controlar el espaciado por bloque:
#  - bloque superior (a, b, c) con hspace muy pequeño
#  - fila inferior (d, e) separada del bloque superior
gs6_outer = gridspec.GridSpec(2, 1, height_ratios=[2.85, 1.0], hspace=0.14,
                              figure=fig6)
gs6_top = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs6_outer[0],
                                           height_ratios=[0.9, 1.1, 0.85],
                                           hspace=0.035)  # a-b pegados, b-c mínimo
gs6_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs6_outer[1],
                                           wspace=0.18)
ax_a = fig6.add_subplot(gs6_top[0])
with sns.axes_style("darkgrid"):
    ax_b = fig6.add_subplot(gs6_top[1])
    ax_c = fig6.add_subplot(gs6_top[2])
    ax_d = fig6.add_subplot(gs6_bot[0])
    ax_e = fig6.add_subplot(gs6_bot[1])

# ── Panel a) Photos + arrows, event markers only (NO phase-level bar) ──
ax = ax_a
start_date, end_date = T[0], T[-1]
ax.set_xlim(mdates.date2num(start_date), mdates.date2num(end_date))
ax.set_ylim(0, 100)
marker_y = 14
for i in range(len(ev_start)):
    if ev_type[i] == 'explosion':
        ax.scatter(mdates.date2num(ev_start[i]), marker_y, marker=volcano_marker,
                   s=70, color=C_EXP_MARKER, edgecolor='darkred', linewidth=0.4, zorder=7)
    else:
        ax.scatter(mdates.date2num(ev_start[i]), marker_y, marker=volcano_marker,
                   s=70, color=C_EMI_MARKER, edgecolor='black', linewidth=0.4, zorder=6)
if image_info:
    x_range = end_date - start_date
    padding = x_range * 0.05
    plot_start = start_date + padding
    plot_end = end_date - padding
    total_width = plot_end - plot_start
    n_images = len(image_info)
    img_row_y = 62
    img_spacing = total_width / (n_images - 1) if n_images > 1 else total_width
    img_x_positions = [plot_start + i * img_spacing for i in range(n_images)]
    for i, img_data in enumerate(image_info):
        x_pos = img_x_positions[i]
        try:
            img = plt.imread(img_data['path'])
            imagebox = OffsetImage(img, zoom=0.16)
            ab = AnnotationBbox(imagebox, (mdates.date2num(x_pos), img_row_y),
                                 frameon=False, pad=0, zorder=10)
            ax.add_artist(ab)
            text = ax.text(mdates.date2num(x_pos), img_row_y + 14, img_data['label'],
                            fontsize=8, ha='center', va='bottom', color='black',
                            rotation=20, zorder=15)
            text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white'),
                                    path_effects.Normal()])
            arrow = FancyArrowPatch((mdates.date2num(x_pos), img_row_y - 12),
                                     (mdates.date2num(img_data['datetime']), marker_y + 3),
                                     connectionstyle="arc3,rad=0.15", arrowstyle="-|>",
                                     mutation_scale=7, linewidth=0.6, color=img_data['color'],
                                     alpha=0.8, zorder=8)
            ax.add_patch(arrow)
        except Exception as e:
            print(f"  photo error {img_data['path']}: {e}")
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.tick_params(axis='x', direction='out', length=7, which='major', labelbottom=False, color='#555555')
ax.tick_params(axis='y', left=False, labelleft=False)
for sp in ['top', 'right', 'left']:
    ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(0.8)
ax.spines['bottom'].set_color('#555555')
add_panel_label(ax, 'a)')
legend_ev = [
    Line2D([0], [0], marker=volcano_marker, color='w', markerfacecolor=C_EXP_MARKER,
           markeredgecolor='darkred', markersize=10, label='Explosion'),
    Line2D([0], [0], marker=volcano_marker, color='w', markerfacecolor=C_EMI_MARKER,
           markeredgecolor='black', markersize=10, label='Emission'),
]
ax.legend(handles=legend_ev, loc='upper right', ncol=2, fontsize=10, framealpha=0.9)

# ── Panel b) P(Alert) [= old panel d] ──
ax = ax_b
add_phase_bands(ax); add_event_shading(ax); add_datelines(ax)
vm = ~np.isnan(p_sm_sub)
ax.fill_between(T_sub[vm], 0, p_sm_sub[vm], color=C_SIGNAL, alpha=0.6, lw=0, zorder=1)
ax.plot(T_sub, p_sm_sub, color=C_SIGNAL, lw=0.3, alpha=0.3, zorder=2,
        label='P(Alert), 60-min trailing mean')
p_above = np.where(np.nan_to_num(p_sm_sub, nan=0) >= TH_OP, p_sm_sub, np.nan)
ax.plot(T_sub, p_above, color=C_ALERT, lw=1.1, zorder=4, label='Above th$_{op}$')
mask_360 = alert_sub == 1
if np.any(mask_360):
    ax.fill_between(T_sub, 0, 1, where=mask_360, alpha=0.04, color=C_DET, zorder=0)
ax.axhline(TH_OP, color=C_ALERT, ls='-', lw=1.0, alpha=0.7, zorder=4)
ax.text(T[int(n * 0.003)], TH_OP - 0.08,
        f'th$_{{op}}$ = {TH_OP:.4f}  (definitive, 2019-calibrated)',
        fontsize=11, color=C_ALERT, fontstyle='italic')
ax.set_ylim(-0.05, 1.15)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9, borderpad=0.3, handlelength=1.5)
configure_panel_fig6(ax, 'P(Alert)')
add_panel_label(ax, 'b)')

# ── Panel c) Persistence counter + per-event anticipation [= old panel f] ──
ax = ax_c
add_phase_bands(ax, f=0.7); add_datelines(ax)
ax.plot(T_sub, ch_sub, color=C_COUNTER, lw=1.0, zorder=2, label='Counter above th$_{op}$')
above_6h = np.where(ch_sub >= N_PERSIST / 60, ch_sub, np.nan)
ax.plot(T_sub, above_6h, color=C_DET, lw=1.0, zorder=3, label='Criterion met (\u2265 6 h)')
ax.axhline(N_PERSIST / 60, color=C_DET, ls='--', lw=1.0, alpha=0.7, zorder=4)
ax.text(T[int(n * 0.003)], N_PERSIST / 60 + 0.3, f'Persistence criterion = {N_PERSIST} min (6 h)',
        fontsize=11, color=C_DET, fontweight='normal', fontstyle='italic')

y_max = max(15, np.nanmax(consec_hours) * 1.1)
_stagger_levels = [0.82, 0.60, 0.38]
for i in range(len(ev_start)):
    col = C_ERUPT if ev_type[i] == 'explosion' else C_PREERUP
    ax.axvline(ev_start[i], color=col, lw=0.6, alpha=0.5, ls=':', zorder=1)
    ax.axvspan(ev_start[i], ev_end[i], alpha=0.06, color=col, zorder=0, lw=0)
    ant_hours = EV_ANTICIPATION[i]['hours']
    if ant_hours is not None:
        lbl = f'{ant_hours:.1f} h' if ant_hours < 48 else f'{ant_hours / 24:.1f} d'
        boxc = col
    else:
        lbl = 'no alert'
        boxc = '#999999'
    y_lbl = y_max * _stagger_levels[i % 3]
    ax.annotate(lbl, xy=(ev_start[i], y_lbl), fontsize=6.5, fontweight='bold',
                ha='center', va='center',
                color=('white' if col == C_ERUPT and ant_hours is not None else 'black'),
                bbox=dict(boxstyle='round,pad=0.2', fc=boxc, ec=boxc, alpha=0.85, lw=0.5))
ax.set_ylim(-0.25, y_max)
ax.set_yticks([0, 10, 20, 30])
ax.legend(loc='upper left', fontsize=11, framealpha=0.9, ncol=1, borderpad=0.3, handlelength=1.5)
configure_panel_fig6(ax, 'Consecutive hours\nabove th$_{op}$', is_last=True)
ax.tick_params(axis='x', which='major', bottom=True, color='black', length=8, direction='out')
# etiqueta del panel c desplazada a la izquierda para que no la tapen los ticks del eje Y
add_panel_label(ax, 'c)', x=-0.055)

# ── Panels d, e) Zoom de PARÁMETROS NORMALIZADOS (= panel c de la Fig 5) ──
#    Junio 15-30 y Julio 15-30, con periodos eruptivos catalogados y el
#    instante EXACTO de declaración de alerta + anotación de anticipación (h).
zb_order  = ['SE', 'FI', 'K', 'SSAM']
zb_colors = {'SE': C_ENT, 'FI': C_FI, 'K': C_KURT, 'SSAM': C_SSAM}

def plot_params_zoom(ax, t0, t1, letter):
    ax.set_xlim(t0, t1)
    add_phase_bands(ax, f=0.7)
    add_datelines(ax)
    OFFSET, CLIP = 8, 7
    yticks_pos = []
    for k, lab in enumerate(zb_order):
        base = (len(zb_order) - 1 - k) * OFFSET
        yticks_pos.append(base)
        s = zb[lab].reindex(dfb.index)
        vals = np.clip(s.values.astype(float), -CLIP, CLIP)
        ax.axhline(base, color='gray', lw=0.8, alpha=0.65, zorder=1)
        ax.axhline(base + 3, color=zb_colors[lab], lw=0.7, dashes=(8, 5), alpha=0.6, zorder=1)
        ax.axhline(base - 3, color=zb_colors[lab], lw=0.7, dashes=(8, 5), alpha=0.6, zorder=1)
        ax.plot(dfb.index, base + vals, color=zb_colors[lab], lw=0.9, alpha=0.9, zorder=3)
    ax.set_ylim(-6, 29)
    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(zb_order, fontsize=12)

    # periodos eruptivos catalogados dentro del zoom + instante de alerta
    # (escalonado vertical de anotaciones para evitar solapes cuando dos
    #  eventos están próximos, p.ej. 17/18/19-jul)
    _n_shown = 0
    _alert_levels = [24.5, 21.0, 17.5]   # alturas escalonadas para las flechas
    _label_levels = [27.0, 14.0, 10.5]   # alturas para el texto "Alert"
    for i in range(len(ev_start)):
        ev_t = pd.Timestamp(ev_start[i])
        if not (t0 <= ev_t <= t1):
            continue
        col = C_ERUPT if ev_type[i] == 'explosion' else C_PREERUP
        ax.axvspan(ev_t, pd.Timestamp(ev_end[i]), alpha=0.12, color=col, zorder=0, lw=0)
        ax.axvline(ev_t, color=col, lw=1.0, alpha=0.7, ls=':', zorder=2)
        onset = EV_ANTICIPATION[i]['onset']
        ant_h = EV_ANTICIPATION[i]['hours']
        if onset is not None and ant_h is not None and t0 <= pd.Timestamp(onset) <= t1:
            y_arrow = _alert_levels[_n_shown % 3]
            y_label = _label_levels[_n_shown % 3]
            ax.axvline(onset, color='#1B5E20', lw=1.4, ls='-', alpha=0.9, zorder=5)
            ax.scatter([onset], [y_arrow + 1.5], color='#1B5E20', s=55, zorder=6,
                       edgecolor='white', linewidth=0.9, marker='*')
            # flecha horizontal desde la alerta hasta el evento, con las horas
            ax.annotate('', xy=(ev_t, y_arrow), xytext=(pd.Timestamp(onset), y_arrow),
                        arrowprops=dict(arrowstyle='<->', color='#1B5E20', lw=1.2),
                        zorder=6)
            t_mid = pd.Timestamp(onset) + (ev_t - pd.Timestamp(onset)) / 2
            ax.annotate(f'+{ant_h:.1f} h', xy=(t_mid, y_arrow), xytext=(0, 3),
                        textcoords='offset points', fontsize=8.5, fontweight='bold',
                        color='#1B5E20', ha='center', va='bottom', zorder=7,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec='#1B5E20', alpha=0.9, lw=0.6))
            ax.annotate(f'{pd.Timestamp(onset):%d-%b %H:%M}',
                        xy=(pd.Timestamp(onset), y_label), xytext=(-3, 0),
                        textcoords='offset points', fontsize=7, color='#1B5E20',
                        ha='right', va='center', rotation=90, zorder=7)
            _n_shown += 1

    configure_month_week_short(ax)
    ax.set_ylabel('Normalized parameters (z-score) [\u03c3]', fontsize=11)
    # etiqueta del panel: arriba-izquierda pero SIN pegarse a la esquina
    add_panel_label(ax, letter, x=-0.10, y=1.06)

def configure_month_week_short(ax):
    """Zoom-panel tick style: visible day/week ticks + date labels."""
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.tick_params(direction='out', length=8, which='major', labelsize=11)
    ax.tick_params(direction='out', length=4, which='minor')
    ax.tick_params(labelbottom=True, labelsize=11)
    ax.set_xlabel('Date (2019)', fontsize=12)

plot_params_zoom(ax_d, pd.Timestamp('2019-06-15'), pd.Timestamp('2019-06-30'), 'd)')
plot_params_zoom(ax_e, pd.Timestamp('2019-07-15'), pd.Timestamp('2019-07-30'), 'e)')

for a in [ax_a, ax_b, ax_c]:
    a.set_xlim(T_START, T_END)

plt.figure(fig6.number)
# no usar tight_layout con gridspec anidado (rompe los hspace personalizados);
# el layout ya está controlado por gs6_outer/gs6_top/gs6_bot
out6_png = os.path.join(basepath, 'Fig6_split_new.png')
out6_pdf = os.path.join(basepath, 'Fig6_split_new.pdf')
plt.savefig(out6_png, dpi=300); plt.savefig(out6_pdf)
plt.close(fig6)
print(f"  Saved: {out6_png}\n  Saved: {out6_pdf}")

print("\nDone — Figure 5 and Figure 6 (split) generated with the corrected 2019 catalog.")
