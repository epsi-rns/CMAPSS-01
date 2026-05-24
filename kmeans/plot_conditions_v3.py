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

km = KMeans(n_clusters=6, random_state=42, n_init=10)
km.fit(ops)
centers = km.cluster_centers_
cids    = km.labels_

# scale altitude to actual feet
ops_plot     = ops.copy()
ops_plot[:,0] = ops_plot[:,0] * 1000

centers_plot     = centers.copy()
centers_plot[:,0] = centers_plot[:,0] * 1000

# round altitude to nearest 1000 for table display
def round_alt(v):
    return round(v / 1000) * 1000

colors = ['#0F6E56','#BA7517','#534AB7','#D85A30','#185FA5','#444441']
phases = {
    0: 'sea level takeoff',
    1: 'cruise, full thrust',
    2: 'cruise, reduced',
    3: 'sea level takeoff',
    4: 'climb',
    5: 'high alt cruise',
}
phase_map = {
    (0,    0.00, 100): 'sea level takeoff',
    (10000, 0.25, 100): 'climb',
    (20000, 0.70, 100): 'cruise, full thrust',
    (25000, 0.62,  60): 'cruise, reduced',
    (35000, 0.84, 100): 'high alt cruise',
    (42000, 0.84, 100): 'top of climb',
}
def get_phase(c):
    key = (round_alt(c[0]), round(c[1],2), round(c[2]))
    return phase_map.get(key, '—')

fig = plt.figure(figsize=(13, 9))
fig.patch.set_facecolor('white')

fmt_k = matplotlib.ticker.FuncFormatter(
    lambda x,_: f'{int(x):,}')

# ── 3D ──────────────────────────────────────────────────────────────
ax3d = fig.add_subplot(221, projection='3d')
ax3d.set_facecolor('white')

alt_r=[0,42000]; mach_r=[0,0.84]; tra_r=[20,100]
for a in alt_r:
    for m in mach_r:
        ax3d.plot([a,a],[m,m],tra_r,color='#d3d1c7',lw=0.5)
for a in alt_r:
    for t in tra_r:
        ax3d.plot([a,a],mach_r,[t,t],color='#d3d1c7',lw=0.5)
for m in mach_r:
    for t in tra_r:
        ax3d.plot(alt_r,[m,m],[t,t],color='#d3d1c7',lw=0.5)

for ci in range(6):
    mask = cids==ci
    ax3d.scatter(ops_plot[mask,0],ops_plot[mask,1],ops_plot[mask,2],
                 color=colors[ci],s=2,alpha=0.15)
for i,c in enumerate(centers_plot):
    ax3d.scatter(c[0],c[1],c[2],color=colors[i],s=120,
                 edgecolors='white',linewidths=0.8,zorder=5)
    ax3d.text(c[0],c[1],c[2]+2,f'C{i+1}',
              fontsize=8,fontweight='bold',color=colors[i])

ax3d.set_xlabel('Altitude (ft)',fontsize=8,labelpad=6)
ax3d.set_ylabel('Mach',fontsize=8,labelpad=4)
ax3d.set_zlabel('TRA (deg)',fontsize=8,labelpad=4)
ax3d.set_title('3D view',fontsize=10,pad=8)
ax3d.tick_params(labelsize=7)
ax3d.xaxis.set_major_formatter(fmt_k)

# ── Altitude vs Mach ────────────────────────────────────────────────
ax1 = fig.add_subplot(222)
ax1.set_facecolor('white')
ax1.add_patch(plt.Rectangle((0,0),42000,0.84,
    fill=False,edgecolor='#d3d1c7',lw=1,linestyle='--'))
for ci in range(6):
    mask=cids==ci
    ax1.scatter(ops_plot[mask,0],ops_plot[mask,1],
                color=colors[ci],s=4,alpha=0.2)
for i,c in enumerate(centers_plot):
    ax1.scatter(c[0],c[1],color=colors[i],s=140,
                edgecolors='white',linewidths=0.8,zorder=5)
    ax1.annotate(f'C{i+1}',(c[0],c[1]),
                 textcoords='offset points',xytext=(6,4),
                 fontsize=8,fontweight='bold',color=colors[i])
ax1.set_xlabel('Altitude (ft)',fontsize=9)
ax1.set_ylabel('Mach number',fontsize=9)
ax1.set_title('Altitude vs Mach',fontsize=10)
ax1.set_xlim(-2000,46000)
ax1.set_ylim(-0.05,0.95)
ax1.xaxis.set_major_formatter(fmt_k)
ax1.grid(True,color='#f1efe8',linewidth=0.5)
ax1.spines[['top','right']].set_visible(False)
ax1.tick_params(labelsize=8)

# ── Altitude vs TRA ─────────────────────────────────────────────────
ax2 = fig.add_subplot(223)
ax2.set_facecolor('white')
ax2.add_patch(plt.Rectangle((0,20),42000,80,
    fill=False,edgecolor='#d3d1c7',lw=1,linestyle='--'))
for ci in range(6):
    mask=cids==ci
    ax2.scatter(ops_plot[mask,0],ops_plot[mask,2],
                color=colors[ci],s=4,alpha=0.2)
for i,c in enumerate(centers_plot):
    ax2.scatter(c[0],c[2],color=colors[i],s=140,
                edgecolors='white',linewidths=0.8,zorder=5)
    ax2.annotate(f'C{i+1}',(c[0],c[2]),
                 textcoords='offset points',xytext=(6,4),
                 fontsize=8,fontweight='bold',color=colors[i])
ax2.set_xlabel('Altitude (ft)',fontsize=9)
ax2.set_ylabel('TRA (degrees)',fontsize=9)
ax2.set_title('Altitude vs TRA',fontsize=10)
ax2.set_xlim(-2000,46000)
ax2.set_ylim(10,110)
ax2.xaxis.set_major_formatter(fmt_k)
ax2.grid(True,color='#f1efe8',linewidth=0.5)
ax2.spines[['top','right']].set_visible(False)
ax2.tick_params(labelsize=8)

# ── Table ────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(224)
ax4.set_facecolor('white')
ax4.axis('off')

order = np.argsort(centers_plot[:,0])
table_data = []
for idx in order:
    c  = centers_plot[idx]
    cr = centers[idx]
    table_data.append([
        f'C{idx+1}',
        f'{round_alt(c[0]):,}',
        f'{cr[1]:.2f}',
        f'{cr[2]:.0f}°',
        get_phase(c),
    ])

tbl = ax4.table(
    cellText=table_data,
    colLabels=['Cond','Alt (ft)','Mach','TRA','Flight phase'],
    cellLoc='center',
    loc='center',
    colWidths=[0.08,0.18,0.10,0.09,0.26]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1,1.6)

for j in range(5):
    tbl[(0,j)].set_facecolor('#2C4A7C')
    tbl[(0,j)].set_text_props(color='white',fontweight='bold')
for row_i, data_idx in enumerate(order,start=1):
    tbl[(row_i,0)].set_facecolor(colors[data_idx])
    tbl[(row_i,0)].set_text_props(color='white',fontweight='bold')
    for j in range(1,5):
        tbl[(row_i,j)].set_facecolor(
            '#f9f8f4' if row_i%2==0 else 'white')

ax4.set_title('Six flight condition centroids\n(KMeans on FD002 train set)',
              fontsize=10,pad=8)

fig.text(0.5,0.01,
    'Paper range: altitude 0–42,000 ft  |  Mach 0–0.84  |  '
    'TRA 20–100°  (dashed box)  |  large dots = centroids',
    ha='center',fontsize=8,color='#888780')

plt.suptitle('C-MAPSS FD002 — Operating conditions',
             fontsize=12,fontweight='bold',y=1.01)
plt.tight_layout(rect=[0,0.03,1,1])
plt.savefig('/home/claude/conditions_chart.png',
            dpi=150,bbox_inches='tight',facecolor='white')
print("saved")
