import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
print("=" * 60)

df = pd.read_csv(r'C:\Users\shani\Desktop\FEBRUARY ACT\COMPUT-SCI\customer_segmentation_5000.csv')

print(f"\n[DATA OVERVIEW]")
print(f"  Total Customers : {len(df):,}")
print(f"  Features        : {list(df.columns)}")
print(f"\n[DESCRIPTIVE STATISTICS]")
print(df[['Annual_Income', 'Spending_Score']].describe().round(2).to_string())

# ─────────────────────────────────────────────
# 2. PREPROCESSING — Normalize for K-Means
# ─────────────────────────────────────────────
features = ['Annual_Income', 'Spending_Score']
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 3. ELBOW METHOD — Find Optimal K
#    (per kMeans.pdf: plot WCSS vs K, pick elbow)
# ─────────────────────────────────────────────
print(f"\n[ELBOW METHOD] Computing WCSS for K = 1..10...")
wcss = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    print(f"  K={k:2d}  WCSS={km.inertia_:.2f}")

# ─────────────────────────────────────────────
# 4. SILHOUETTE SCORES — Validation metric
# ─────────────────────────────────────────────
print(f"\n[SILHOUETTE SCORES] (higher = better separation)")
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)
    print(f"  K={k:2d}  Silhouette={score:.4f}")

optimal_k = 5  # visually confirmed from elbow + silhouette
print(f"\n  → Optimal K selected: {optimal_k}")

# ─────────────────────────────────────────────
# 5. FINAL K-MEANS MODEL
# ─────────────────────────────────────────────
print(f"\n[TRAINING K-MEANS] K={optimal_k}...")
final_km = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = final_km.fit_predict(X_scaled)

centroids_scaled = final_km.cluster_centers_
centroids_orig = scaler.inverse_transform(centroids_scaled)

# ─────────────────────────────────────────────
# 6. CLUSTER ANALYSIS & LABELING
# ─────────────────────────────────────────────
cluster_stats = df.groupby('Cluster')[features + ['Age']].mean().round(2)
cluster_stats['Count'] = df.groupby('Cluster').size()
cluster_stats['Pct'] = (cluster_stats['Count'] / len(df) * 100).round(1)

# Assign meaningful labels based on income/spending profile
def label_cluster(row):
    inc = row['Annual_Income']
    spd = row['Spending_Score']
    mid_inc = df['Annual_Income'].median()
    mid_spd = df['Spending_Score'].median()
    if inc >= mid_inc and spd >= mid_spd:
        return "High Income, High Spenders"
    elif inc >= mid_inc and spd < mid_spd:
        return "High Income, Low Spenders"
    elif inc < mid_inc and spd >= mid_spd:
        return "Low Income, High Spenders"
    elif inc < mid_inc and spd < mid_spd:
        return "Low Income, Low Spenders"
    else:
        return "Middle Segment"

cluster_stats['Label'] = cluster_stats.apply(label_cluster, axis=1)

print(f"\n[CLUSTER PROFILES]")
print(cluster_stats.to_string())

# ─────────────────────────────────────────────
# 7. EVALUATION METRICS (from ML_Evaluation_Blueprint)
# ─────────────────────────────────────────────
final_labels = df['Cluster'].values
sil  = silhouette_score(X_scaled, final_labels)
db   = davies_bouldin_score(X_scaled, final_labels)
ch   = calinski_harabasz_score(X_scaled, final_labels)
wcss_final = final_km.inertia_

print(f"\n[EVALUATION METRICS]")
print(f"  Silhouette Score        : {sil:.4f}  (range -1 to 1, higher = better)")
print(f"  Davies-Bouldin Index    : {db:.4f}  (lower = better separation)")
print(f"  Calinski-Harabasz Index : {ch:.2f}  (higher = more compact clusters)")
print(f"  Final WCSS (Inertia)    : {wcss_final:.2f}")
print(f"  Cluster Count           : {optimal_k}")

# Interpret silhouette
if sil > 0.5:
    interp = "Strong cluster structure"
elif sil > 0.25:
    interp = "Reasonable cluster structure"
else:
    interp = "Weak cluster structure — consider revisiting K"
print(f"  Interpretation          : {interp}")

# ─────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────
PALETTE = ['#E63946', '#2A9D8F', '#E9C46A', '#457B9D', '#A8DADC']
DARK_BG  = '#0D1117'
CARD_BG  = '#161B22'
TEXT_COL = '#E6EDF3'
GRID_COL = '#30363D'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor'  : CARD_BG,
    'axes.edgecolor'  : GRID_COL,
    'axes.labelcolor' : TEXT_COL,
    'xtick.color'     : TEXT_COL,
    'ytick.color'     : TEXT_COL,
    'text.color'      : TEXT_COL,
    'grid.color'      : GRID_COL,
    'grid.alpha'      : 0.4,
    'font.family'     : 'DejaVu Sans',
})

# ══════════════════════════════════════════════
# FIGURE 1 — Optimal K Selection (Elbow + Silhouette + Metrics)
# ══════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 3, figsize=(22, 7), facecolor=DARK_BG)
fig1.suptitle('Customer Segmentation — K Selection & Evaluation Metrics',
              fontsize=18, fontweight='bold', color=TEXT_COL, y=1.02)
plt.subplots_adjust(wspace=0.4)

# ── Plot 1: Elbow Curve ──
ax1 = axes1[0]
ax1.plot(list(k_range), wcss, 'o-', color='#E9C46A', lw=2.5, ms=8)
ax1.axvline(x=optimal_k, color='#E63946', ls='--', lw=1.8, alpha=0.8, label=f'Optimal K={optimal_k}')
ax1.set_title('Elbow Method — Optimal K', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('WCSS (Inertia)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True)

# ── Plot 2: Silhouette Scores ──
ax2 = axes1[1]
k_vals_sil = list(range(2, 11))
bar_colors = ['#E63946' if k == optimal_k else '#457B9D' for k in k_vals_sil]
bars = ax2.bar(k_vals_sil, sil_scores, color=bar_colors, edgecolor=GRID_COL, width=0.6)
ax2.set_title('Silhouette Scores per K', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_xticks(k_vals_sil)
for bar, score in zip(bars, sil_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{score:.3f}', ha='center', va='bottom', fontsize=10, color=TEXT_COL)
ax2.grid(True, axis='y')

# ── Plot 3: Evaluation Metrics Table ──
ax3 = axes1[2]
ax3.axis('off')
metric_data = [
    ['Silhouette Score',     f'{sil:.4f}',        'Higher = Better'],
    ['Davies-Bouldin Index', f'{db:.4f}',          'Lower = Better'],
    ['Calinski-Harabasz',    f'{ch:.1f}',          'Higher = Better'],
    ['Final WCSS',           f'{wcss_final:.0f}',  'Lower = Better'],
    ['Optimal K',            f'{optimal_k}',        'Elbow + Silhouette'],
    ['Total Customers',      f'{len(df):,}',        '5,000 Records'],
]
table = ax3.table(
    cellText=metric_data,
    colLabels=['Metric', 'Value', 'Interpretation'],
    loc='center', cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.2)
for (r, c), cell in table.get_celld().items():
    cell.set_facecolor('#21262D' if r == 0 else CARD_BG)
    cell.set_edgecolor(GRID_COL)
    cell.set_text_props(color=TEXT_COL if r > 0 else '#F0C040',
                        fontweight='bold' if r == 0 else 'normal')
ax3.set_title('Evaluation Summary', fontsize=14, fontweight='bold', pad=12)

# ══════════════════════════════════════════════
# FIGURE 2 — Cluster Scatter + Donut
# ══════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8), facecolor=DARK_BG,
                            gridspec_kw={'width_ratios': [2.5, 1]})
fig2.suptitle('Customer Cluster Map & Distribution',
              fontsize=18, fontweight='bold', color=TEXT_COL, y=1.02)
plt.subplots_adjust(wspace=0.3)

# ── Plot 4: Main Cluster Scatter ──
ax4 = axes2[0]
for i in range(optimal_k):
    mask = df['Cluster'] == i
    ax4.scatter(df.loc[mask, 'Annual_Income'], df.loc[mask, 'Spending_Score'],
                c=PALETTE[i], alpha=0.55, s=20, label=f'Cluster {i}', edgecolors='none')
ax4.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
            c='white', marker='*', s=400, zorder=10, edgecolors='#E63946', linewidths=1.5,
            label='Centroids')
ax4.set_title('Customer Clusters — Annual Income vs Spending Score',
              fontsize=14, fontweight='bold', pad=12)
ax4.set_xlabel('Annual Income (₱)', fontsize=12)
ax4.set_ylabel('Spending Score (1–100)', fontsize=12)
ax4.legend(fontsize=11, facecolor=CARD_BG, edgecolor=GRID_COL)
ax4.grid(True)

# ── Plot 5: Cluster Size Donut ──
ax5 = axes2[1]
sizes  = cluster_stats['Count'].values
labels = [f'C{i}  {cluster_stats.loc[i,"Count"]:,}' for i in cluster_stats.index]
wedges, texts, autotexts = ax5.pie(
    sizes, labels=labels, colors=PALETTE[:optimal_k],
    autopct='%1.1f%%', startangle=140,
    wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2),
    textprops={'color': TEXT_COL, 'fontsize': 11}
)
for at in autotexts:
    at.set_color(DARK_BG)
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax5.set_title('Cluster Distribution', fontsize=14, fontweight='bold', pad=12)

# ══════════════════════════════════════════════
# FIGURE 3 — Box Plots + Centroid Bar Chart
# ══════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 3, figsize=(22, 14), facecolor=DARK_BG)
fig3.suptitle('Cluster Distribution Analysis & Centroid Profiles',
              fontsize=18, fontweight='bold', color=TEXT_COL, y=1.01)
plt.subplots_adjust(hspace=0.45, wspace=0.35)

# ── Plot 6: Income boxplot ──
ax6 = axes3[0, 0]
data_by_cluster = [df.loc[df['Cluster'] == i, 'Annual_Income'].values for i in range(optimal_k)]
bp = ax6.boxplot(data_by_cluster, patch_artist=True,
                 medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for element in ['whiskers', 'caps', 'fliers']:
    for item in bp[element]: item.set_color(GRID_COL)
ax6.set_title('Annual Income per Cluster', fontsize=13, fontweight='bold', pad=10)
ax6.set_xlabel('Cluster', fontsize=12)
ax6.set_ylabel('Annual Income (₱)', fontsize=12)
ax6.grid(True, axis='y')

# ── Plot 7: Spending boxplot ──
ax7 = axes3[0, 1]
data_spd = [df.loc[df['Cluster'] == i, 'Spending_Score'].values for i in range(optimal_k)]
bp2 = ax7.boxplot(data_spd, patch_artist=True,
                  medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp2['boxes'], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for element in ['whiskers', 'caps', 'fliers']:
    for item in bp2[element]: item.set_color(GRID_COL)
ax7.set_title('Spending Score per Cluster', fontsize=13, fontweight='bold', pad=10)
ax7.set_xlabel('Cluster', fontsize=12)
ax7.set_ylabel('Spending Score', fontsize=12)
ax7.grid(True, axis='y')

# ── Plot 8: Age boxplot ──
ax8 = axes3[0, 2]
data_age = [df.loc[df['Cluster'] == i, 'Age'].values for i in range(optimal_k)]
bp3 = ax8.boxplot(data_age, patch_artist=True,
                  medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp3['boxes'], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for element in ['whiskers', 'caps', 'fliers']:
    for item in bp3[element]: item.set_color(GRID_COL)
ax8.set_title('Age Distribution per Cluster', fontsize=13, fontweight='bold', pad=10)
ax8.set_xlabel('Cluster', fontsize=12)
ax8.set_ylabel('Age', fontsize=12)
ax8.grid(True, axis='y')

# ── Plot 9: Centroid bar chart (spans full bottom row) ──
ax9 = axes3[1, :]
# merge bottom row into one axis
for a in axes3[1, 1:]: a.set_visible(False)
ax9 = fig3.add_subplot(2, 1, 2)
ax9.set_facecolor(CARD_BG)

x_pos    = np.arange(optimal_k)
width    = 0.35
inc_vals = cluster_stats['Annual_Income'].values
spd_vals = cluster_stats['Spending_Score'].values
inc_norm = inc_vals / inc_vals.max() * 100

bars1 = ax9.bar(x_pos - width/2, inc_norm, width,
                label='Annual Income (normalized to 100)',
                color=[PALETTE[i] for i in range(optimal_k)],
                alpha=0.85, edgecolor=GRID_COL)
bars2 = ax9.bar(x_pos + width/2, spd_vals, width,
                label='Spending Score',
                color=[PALETTE[i] for i in range(optimal_k)],
                alpha=0.45, edgecolor=GRID_COL, hatch='//')

for bar, val in zip(bars1, inc_vals):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'₱{val/1000:.0f}k', ha='center', va='bottom', fontsize=11, color=TEXT_COL)
for bar, val in zip(bars2, spd_vals):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{val:.0f}', ha='center', va='bottom', fontsize=11, color=TEXT_COL)

ax9.set_title('Cluster Centroid Profiles — Income vs Spending Score',
              fontsize=14, fontweight='bold', pad=12)
ax9.set_xlabel('Cluster', fontsize=12)
ax9.set_ylabel('Score / Normalized Value', fontsize=12)
ax9.set_xticks(x_pos)
ax9.set_xticklabels(
    [f'Cluster {i}\n({cluster_stats.loc[i,"Count"]:,} customers)' for i in range(optimal_k)],
    fontsize=11)
ax9.legend(fontsize=11, facecolor=CARD_BG, edgecolor=GRID_COL)
ax9.grid(True, axis='y')

print(f"\n[OUTPUT] 3 chart windows will open — close each to proceed.")

# ─────────────────────────────────────────────
# 9. FINAL REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLUSTER ANALYSIS REPORT")
print("=" * 60)
for i in range(optimal_k):
    grp = df[df['Cluster'] == i]
    print(f"\n  Cluster {i} — {cluster_stats.loc[i, 'Label']}")
    print(f"    Customers    : {len(grp):,}  ({len(grp)/len(df)*100:.1f}%)")
    print(f"    Avg Income   : ₱{grp['Annual_Income'].mean():,.0f}")
    print(f"    Avg Spending : {grp['Spending_Score'].mean():.1f}")
    print(f"    Avg Age      : {grp['Age'].mean():.1f} years")
    print(f"    Income Range : ₱{grp['Annual_Income'].min():,} – ₱{grp['Annual_Income'].max():,}")

print("\n" + "=" * 60)
print("  BUSINESS RECOMMENDATIONS")
print("=" * 60)
print("""
  High Income + High Spending  → VIP loyalty programs, premium products
  High Income + Low Spending   → Re-engagement campaigns, exclusive offers
  Low Income + High Spending   → Budget deals, installment options
  Low Income + Low Spending    → Awareness campaigns, entry-level products
  Middle Segment               → Upsell strategies, personalized promotions
""")
print("  ✓ K-Means Clustering successfully applied to 5,000 customer records.")
print(f"  ✓ Silhouette Score: {sil:.4f} — {interp}")
print("=" * 60)

fig3, ax_action = plt.subplots(figsize=(9, 7))
fig3.patch.set_facecolor('white')
ax_action.set_facecolor('white')

ACTION_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(optimal_k):
    mask = df['Cluster'] == i
    pts  = df.loc[mask, ['Annual_Income', 'Spending_Score']].values
    ax_action.scatter(pts[:, 0], pts[:, 1],
                      c=ACTION_COLORS[i], s=12, alpha=0.6, edgecolors='none')

ax_action.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
                  c='black', marker='*', s=250, zorder=10)

ax_action.set_title('K-Means in action', fontsize=16, fontweight='normal',
                    color='black', pad=14, loc='left')
ax_action.set_xlabel('Annual Income (₱)', color='black')
ax_action.set_ylabel('Spending Score', color='black')
ax_action.tick_params(colors='black')
for spine in ax_action.spines.values():
    spine.set_edgecolor('#cccccc')
ax_action.grid(False)
plt.tight_layout()

plt.show()