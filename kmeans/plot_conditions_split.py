import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

COLUMNS = [
    'unit_number','time_cycles',
    'op_setting_1','op_setting_2','op_setting_3',
    'T2','T24','T30','T50','P2','P15','P30',
    'Nf','Nc','epr','Ps30','phi',
    'NRf','NRc','BPR','farB','htBleed',
    'Nf_dmd','PCNfR_dmd','W31','W32'
]

train = pd.read_csv('/tmp/train_FD002.txt', sep=r'\s+',
                    header=None, names=COLUMNS)
ops = train[['op_setting_1','op_setting_2','op_setting_3']].values
km  = KMeans(n_clusters=6, random_state=42, n_init=10)
km.fit(ops)
centers = km.cluster_centers_
cids    = km.labels_

ops_plot         = ops.copy()
ops_plot[:,0]    = ops_plot[:,0] * 1000
centers_plot     = centers.copy()
centers_plot[:,0] = centers_plot[:,0] * 1000

def round_alt(v):
    return round(v / 1000) * 1000

COLORS = ['#0F6E56','#BA7517','#534AB7','#D85A30','#185FA5','#444441']
PHASES = {
    (0,    0.00, 100): 'sea level\ntakeoff',
    (10000, 0.25, 100): 'climb',
    (20000, 0.70, 100): 'cruise\nfull thrust',
    (25000, 0.62,  60): 'cruise\nreduced',
    (35000, 0.84, 100): 'high alt\ncruise',
    (42000, 0.84, 100): 'top of\nclimb',
}
def get_phase(c):
    key = (round_alt(c[0]), round(c[1],2), round(c[2]))
    return PHASES.get(key, '—')

fmt_k = matplotlib.ticker.FuncFormatter(lambda x,_: f'{int(x):,}')

STYLE = dict(facecolor='white')

order = np.argsort(centers_plot[:,0])

# ── Chart A: Altitude vs Mach ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5), **STYLE)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.add_patch(plt.Rectangle((0,0),42000,0.84,
    fill=False, edgecolor='#d3d1c7', lw=1, linestyle='--', zorder=0))
for ci in range(6):
    mask = cids==ci
    ax.scatter(ops_plot[mask,0], ops_plot[mask,1],
               color=COLORS[ci], s=6, alpha=0.2, zorder=2)
for i,c in enumerate(centers_plot):
    ax.scatter(c[0], c[1], color=COLORS[i], s=180,
               edgecolors='white', linewidths=1, zorder=5)
    ax.annotate(f'C{i+1}\n{get_phase(c)}', (c[0], c[1]),
                textcoords='offset points', xytext=(10,0),
                fontsize=8, fontweight='bold', color=COLORS[i],
                va='center')
ax.set_xlabel('Altitude (ft)', fontsize=11)
ax.set_ylabel('Mach number', fontsize=11)
ax.set_title('Chart A — Altitude vs Mach number\nFD002 train set, 53,759 cycles',
             fontsize=11, pad=10)
ax.set_xlim(-2000, 50000)
ax.set_ylim(-0.05, 0.95)
ax.xaxis.set_major_formatter(fmt_k)
ax.grid(True, color='#f1efe8', linewidth=0.6)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(labelsize=9)
fig.text(0.5, 0.01,
    'C6 and C1 share Mach=0.84 — they cannot be separated in this projection alone.',
    ha='center', fontsize=8, color='#888780')
plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig('/home/claude/cond_A_alt_mach.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart A saved")

# ── Chart B: Altitude vs TRA ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5), **STYLE)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.add_patch(plt.Rectangle((0,20),42000,80,
    fill=False, edgecolor='#d3d1c7', lw=1, linestyle='--', zorder=0))
for ci in range(6):
    mask = cids==ci
    ax.scatter(ops_plot[mask,0], ops_plot[mask,2],
               color=COLORS[ci], s=6, alpha=0.2, zorder=2)
for i,c in enumerate(centers_plot):
    offset = (10, 8) if c[2] < 80 else (10, -14)
    ax.scatter(c[0], c[2], color=COLORS[i], s=180,
               edgecolors='white', linewidths=1, zorder=5)
    ax.annotate(f'C{i+1}  {get_phase(c)}', (c[0], c[2]),
                textcoords='offset points', xytext=offset,
                fontsize=8, fontweight='bold', color=COLORS[i])
ax.set_xlabel('Altitude (ft)', fontsize=11)
ax.set_ylabel('TRA (degrees)', fontsize=11)
ax.set_title('Chart B — Altitude vs TRA (throttle angle)\nFD002 train set, 53,759 cycles',
             fontsize=11, pad=10)
ax.set_xlim(-2000, 50000)
ax.set_ylim(10, 115)
ax.xaxis.set_major_formatter(fmt_k)
ax.grid(True, color='#f1efe8', linewidth=0.6)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(labelsize=9)
fig.text(0.5, 0.01,
    'C3 is the only condition at TRA=60° — all others are at full throttle (100°).',
    ha='center', fontsize=8, color='#888780')
plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig('/home/claude/cond_B_alt_tra.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart B saved")

# ── Chart C: Mach vs TRA ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5), **STYLE)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.add_patch(plt.Rectangle((0,20),0.84,80,
    fill=False, edgecolor='#d3d1c7', lw=1, linestyle='--', zorder=0))
for ci in range(6):
    mask = cids==ci
    ax.scatter(ops_plot[mask,1], ops_plot[mask,2],
               color=COLORS[ci], s=6, alpha=0.2, zorder=2)
for i,c in enumerate(centers_plot):
    offset = (8, 6) if c[2] < 80 else (8, -14)
    ax.scatter(c[1], c[2], color=COLORS[i], s=180,
               edgecolors='white', linewidths=1, zorder=5)
    ax.annotate(f'C{i+1}  {get_phase(c)}', (c[1], c[2]),
                textcoords='offset points', xytext=offset,
                fontsize=8, fontweight='bold', color=COLORS[i])
ax.set_xlabel('Mach number', fontsize=11)
ax.set_ylabel('TRA (degrees)', fontsize=11)
ax.set_title('Chart C — Mach number vs TRA\nFD002 train set, 53,759 cycles',
             fontsize=11, pad=10)
ax.set_xlim(-0.05, 0.95)
ax.set_ylim(10, 115)
ax.grid(True, color='#f1efe8', linewidth=0.6)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(labelsize=9)
fig.text(0.5, 0.01,
    'C6 and C1 overlap here (both Mach=0.84, TRA=100°) — altitude is needed to separate them.',
    ha='center', fontsize=8, color='#888780')
plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig('/home/claude/cond_C_mach_tra.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart C saved")

# ── Chart D: Centroid table ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4), **STYLE)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')

phase_full = {
    (0,    0.00, 100): 'sea level takeoff',
    (10000, 0.25, 100): 'climb',
    (20000, 0.70, 100): 'cruise, full thrust',
    (25000, 0.62,  60): 'cruise, reduced thrust',
    (35000, 0.84, 100): 'high altitude cruise',
    (42000, 0.84, 100): 'top of climb',
}
def get_phase_full(c):
    key = (round_alt(c[0]), round(c[1],2), round(c[2]))
    return phase_full.get(key, '—')

table_data = []
for idx in order:
    c  = centers_plot[idx]
    cr = centers[idx]
    table_data.append([
        f'C{idx+1}',
        f'{round_alt(c[0]):,}',
        f'{cr[1]:.2f}',
        f'{cr[2]:.0f}°',
        get_phase_full(c),
        'Only condition at reduced thrust' if round(cr[2])==60
        else ('Shares Mach with C1' if round_alt(c[0])==35000
        else ('Shares Mach with C6' if round_alt(c[0])==42000
        else '—')),
    ])

tbl = ax.table(
    cellText=table_data,
    colLabels=['Cond','Alt (ft)','Mach','TRA','Flight phase','Note'],
    cellLoc='center', loc='center',
    colWidths=[0.06,0.12,0.08,0.07,0.22,0.28]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)

for j in range(6):
    tbl[(0,j)].set_facecolor('#2C4A7C')
    tbl[(0,j)].set_text_props(color='white', fontweight='bold')
for row_i, data_idx in enumerate(order, start=1):
    tbl[(row_i,0)].set_facecolor(COLORS[data_idx])
    tbl[(row_i,0)].set_text_props(color='white', fontweight='bold')
    for j in range(1,6):
        tbl[(row_i,j)].set_facecolor(
            '#f9f8f4' if row_i%2==0 else 'white')

ax.set_title('Chart D — Six flight condition centroids (FD002 / FD004)',
             fontsize=11, pad=12, loc='left')
plt.tight_layout()
plt.savefig('/home/claude/cond_D_table.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart D saved")
