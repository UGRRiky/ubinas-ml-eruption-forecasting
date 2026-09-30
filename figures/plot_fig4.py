"""
================================================================================
 FIGURA 4 — Rendimiento del Random Forest (pipeline Python)
================================================================================
6 paneles:
  a) Convergencia del error OOB vs nº de árboles (binario y multiclase)
  b) Sensibilidad de la exactitud OOB al min_samples_leaf
  c) Curvas ROC: LOSO por estación (2023) + validación 2019 (operacional/amplia)
  d) Matriz de confusión binaria (2019) con th_op
  e) Matriz de confusión multiclase de 4 estados (2019)
  f) Precisión / recall / F1 por clase (2019)

Lee del pipeline:
  intermediate/oob_curves.csv
  intermediate/leafsize_sensitivity.csv
  intermediate/loso_scores.parquet
  <BASE_DATA>/ubinas_v16_ub4_2019.csv   (para las matrices y ROC 2019)

Requiere: config.py y sustained.py del pipeline en el PYTHONPATH.
Uso: python plot_fig4.py
================================================================================
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             precision_recall_fscore_support)
import seaborn as sns

# importar config del pipeline
PIPE_DIR = os.environ.get("PIPE_DIR", ".")
sys.path.insert(0, PIPE_DIR)
import config as C

sns.set_style("ticks")
plt.rcParams.update({'axes.linewidth': 0.8, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight', 'pdf.fonttype': 42})

INTER = C.INTERMEDIATE
CSV_2019 = C.BASE_DATA / "ubinas_v16_ub4_2019.csv"
OUT_DIR = C.BASE_DATA

# ── Colores (consistentes con las otras figuras) ──
C_BIN = '#1565C0'
C_4C  = '#C62828'
C_UB1 = '#1F77B4'; C_UB2 = '#FF7F0E'; C_UB4 = '#2CA02C'
C_OP  = '#D32F2F'; C_AMP = '#7B1FA2'
STATE_NAMES = ['Quiescence', 'Unrest', 'Pre-erup.', 'Eruptive']

# ── Cargar datos ──
print("Cargando datos de la Figura 4...")
oob = pd.read_csv(INTER / "oob_curves.csv")
leaf = pd.read_csv(INTER / "leafsize_sensitivity.csv")
loso = pd.read_parquet(INTER / "loso_scores.parquet")
df19 = pd.read_csv(CSV_2019)
df19['Time'] = pd.to_datetime(df19['Time'], format='%d-%b-%Y %H:%M:%S')
df19 = df19[df19['valid'] == 1].reset_index(drop=True)

# ── Figura ──
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.32, wspace=0.30, figure=fig)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])
ax_d = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])

def panel_label(ax, letter):
    ax.text(-0.15, 1.05, letter, transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left')

# ── Panel a) Convergencia OOB ──
for label, col in [('binary', C_BIN), ('multiclass', C_4C)]:
    d = oob[oob['model'] == label]
    ax_a.plot(d['n_trees'], 100 * d['oob_accuracy'], '-o', color=col, ms=3,
              lw=1.3, label=('Binary' if label == 'binary' else 'Multiclass'))
ax_a.axvline(C.N_TREES, color='gray', ls='--', lw=0.8, alpha=0.7)
ax_a.text(C.N_TREES, ax_a.get_ylim()[0] + 2, f' {C.N_TREES} trees',
          fontsize=8, color='gray', rotation=90, va='bottom')
ax_a.set_xlabel('Number of trees', fontsize=11)
ax_a.set_ylabel('OOB accuracy [%]', fontsize=11)
ax_a.legend(fontsize=9, loc='lower right')
ax_a.grid(alpha=0.3)
panel_label(ax_a, 'a)')

# ── Panel b) Sensibilidad a leaf-size ──
for label, col in [('binary', C_BIN), ('multiclass', C_4C)]:
    d = leaf[leaf['model'] == label].sort_values('min_samples_leaf')
    ax_b.plot(d['min_samples_leaf'], 100 * d['oob_accuracy'], '-s', color=col,
              ms=4, lw=1.3, label=('Binary' if label == 'binary' else 'Multiclass'))
ax_b.axvline(C.MIN_LEAF, color='gray', ls='--', lw=0.8, alpha=0.7)
ax_b.text(C.MIN_LEAF, ax_b.get_ylim()[0] + 2, f' leaf={C.MIN_LEAF}',
          fontsize=8, color='gray', rotation=90, va='bottom')
ax_b.set_xscale('log')
ax_b.set_xlabel('min_samples_leaf', fontsize=11)
ax_b.set_ylabel('OOB accuracy [%]', fontsize=11)
ax_b.legend(fontsize=9, loc='lower left')
ax_b.grid(alpha=0.3)
panel_label(ax_b, 'b)')

# ── Panel c) Curvas ROC ──
# LOSO por estación
st_colors = {'UB1': C_UB1, 'UB2': C_UB2, 'UB4': C_UB4}
for st in loso['station'].unique():
    d = loso[loso['station'] == st]
    if d['y_true'].nunique() < 2:
        continue
    fpr, tpr, _ = roc_curve(d['y_true'], d['score'])
    a = auc(fpr, tpr)
    ax_c.plot(fpr, tpr, color=st_colors.get(st, 'gray'), lw=1.2,
              label=f'LOSO {st} (AUC={a:.3f})')
# 2019 operacional (pre+eruptivo) y amplia (>=unrest)
y_op = df19['ref_bin_op'].values
y_amp = df19['ref_bin_amp'].values
p = df19['p_alert'].fillna(0).values
for y, lab, col, ls in [(y_op, '2019 operational', C_OP, '-'),
                        (y_amp, '2019 broad', C_AMP, '--')]:
    if len(np.unique(y)) < 2:
        continue
    fpr, tpr, _ = roc_curve(y, p)
    a = auc(fpr, tpr)
    ax_c.plot(fpr, tpr, color=col, lw=1.8, ls=ls, label=f'{lab} (AUC={a:.3f})')
ax_c.plot([0, 1], [0, 1], 'k:', lw=0.6, alpha=0.5)
ax_c.set_xlabel('False positive rate', fontsize=11)
ax_c.set_ylabel('True positive rate', fontsize=11)
ax_c.legend(fontsize=7.5, loc='lower right')
ax_c.grid(alpha=0.3)
panel_label(ax_c, 'c)')

# ── Panel d) Matriz de confusión binaria (th_op) ──
th_op = C.__dict__.get('TH_OP', 0.62)  # si no está en config, usar 0.62
try:
    import json
    rep = json.load(open(INTER / "postprocess_report.json"))
    th_op = rep.get("th_op", th_op)
except Exception:
    pass
pred_bin = (p >= th_op).astype(int)
cmb = confusion_matrix(y_op, pred_bin)
cmb_norm = cmb / cmb.sum(axis=1, keepdims=True)
blues = LinearSegmentedColormap.from_list('b', ['#FFFFFF', C_BIN])
im = ax_d.imshow(cmb_norm, cmap=blues, vmin=0, vmax=1, aspect='auto')
for i in range(2):
    for j in range(2):
        ax_d.text(j, i, f'{cmb_norm[i,j]:.2f}\n({cmb[i,j]:,})', ha='center',
                  va='center', fontsize=9,
                  color='white' if cmb_norm[i, j] > 0.5 else 'black')
ax_d.set_xticks([0, 1]); ax_d.set_xticklabels(['No-Alert', 'Alert'], fontsize=9)
ax_d.set_yticks([0, 1]); ax_d.set_yticklabels(['No-Alert', 'Alert'], fontsize=9)
ax_d.set_xlabel('Predicted', fontsize=11); ax_d.set_ylabel('True', fontsize=11)
ax_d.set_title(f'Binary (th$_{{op}}$={th_op:.2f})', fontsize=10)
panel_label(ax_d, 'd)')

# ── Panel e) Matriz de confusión multiclase (HMM causal) ──
pred_col = 'pred_4c_hmm_causal' if 'pred_4c_hmm_causal' in df19.columns else 'pred_4c'
mask4 = df19[pred_col].notna()
y4 = df19.loc[mask4, 'ref_4class'].astype(int)
p4 = df19.loc[mask4, pred_col].astype(int)
cm4 = confusion_matrix(y4, p4, labels=[0, 1, 2, 3])
cm4_norm = cm4 / np.maximum(cm4.sum(axis=1, keepdims=True), 1)
reds = LinearSegmentedColormap.from_list('r', ['#FFFFFF', C_4C])
ax_e.imshow(cm4_norm, cmap=reds, vmin=0, vmax=1, aspect='auto')
for i in range(4):
    for j in range(4):
        ax_e.text(j, i, f'{cm4_norm[i,j]:.2f}', ha='center', va='center',
                  fontsize=7.5, color='white' if cm4_norm[i, j] > 0.5 else 'black')
ax_e.set_xticks(range(4)); ax_e.set_xticklabels(STATE_NAMES, fontsize=7, rotation=35, ha='right')
ax_e.set_yticks(range(4)); ax_e.set_yticklabels(STATE_NAMES, fontsize=7)
ax_e.set_xlabel('Predicted', fontsize=11); ax_e.set_ylabel('True', fontsize=11)
ax_e.set_title('Multiclass (4 states)', fontsize=10)
panel_label(ax_e, 'e)')

# ── Panel f) Precisión / recall / F1 por clase ──
prec, rec, f1, sup = precision_recall_fscore_support(y4, p4, labels=[0, 1, 2, 3],
                                                     zero_division=0)
x = np.arange(4); w = 0.25
ax_f.bar(x - w, prec, w, label='Precision', color='#1565C0')
ax_f.bar(x, rec, w, label='Recall', color='#2E7D32')
ax_f.bar(x + w, f1, w, label='F1', color='#EF6C00')
for i in range(4):
    ax_f.text(i, 1.02, f'n={sup[i]:,}', ha='center', fontsize=6.5, color='gray')
ax_f.set_xticks(x); ax_f.set_xticklabels(STATE_NAMES, fontsize=7, rotation=35, ha='right')
ax_f.set_ylabel('Score', fontsize=11); ax_f.set_ylim(0, 1.12)
ax_f.legend(fontsize=8, loc='upper right', ncol=3)
ax_f.grid(alpha=0.3, axis='y')
panel_label(ax_f, 'f)')

plt.savefig(os.path.join(OUT_DIR, 'Fig4_performance.png'), dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'Fig4_performance.pdf'))
print(f"Guardado: {os.path.join(OUT_DIR, 'Fig4_performance.png')}")
plt.close(fig)
