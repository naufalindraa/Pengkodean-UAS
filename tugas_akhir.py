# ANALISIS KOMPREHENSIF: DAMPAK TAX HOLIDAY DAN METODE DEPRESIASI
# Analisis Profesional untuk Presentasi Klien
# =================================================

# Install required packages untuk Google Colab
!pip install plotly pandas numpy seaborn matplotlib scipy -q

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style untuk matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("ANALISIS KOMPREHENSIF DAMPAK TAX HOLIDAY & METODE DEPRESIASI")
print("="*60)

# =============================================
# 1. LOAD DAN CLEANING DATA
# =============================================

# Data Simulasi Laba Bersih
laba_data = {
    'tahun': [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024, 2025, 2025],
    'skenario': ['normal', 'tax_holiday', 'normal', 'tax_holiday', 'normal', 'tax_holiday', 'normal', 'tax_holiday', 'normal', 'tax_holiday'],
    'pendapatan': [1000000000, 1000000000, 1200000000, 1200000000, 1500000000, 1500000000, 1800000000, 1800000000, 2000000000, 2000000000],
    'beban_operasional': [600000000, 600000000, 700000000, 700000000, 800000000, 800000000, 900000000, 900000000, 1000000000, 1000000000],
    'penyusutan': [50000000, 50000000, 60000000, 60000000, 70000000, 70000000, 80000000, 80000000, 90000000, 90000000],
    'laba_operasional': [350000000, 350000000, 440000000, 440000000, 630000000, 630000000, 820000000, 820000000, 910000000, 910000000],
    'tax_rate': [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22],
    'pph_badan': [77000000, 77000000, 96800000, 96800000, 138600000, 0, 180400000, 0, 200200000, 0],
    'laba_bersih': [273000000, 273000000, 343200000, 343200000, 491400000, 630000000, 639600000, 820000000, 709800000, 910000000]
}

# Data Aset dan Depresiasi
aset_data = {
    'aset_id': ['A001', 'A002', 'A003', 'A004', 'A005'],
    'kategori': ['Mesin', 'Kendaraan', 'Bangunan', 'Mesin', 'Kendaraan'],
    'nilai_perolehan': [500000000, 300000000, 1000000000, 600000000, 200000000],
    'umur_ekonomis': [5, 4, 20, 6, 3],
    'depresiasi_tahunan': [100000000, 150000000, 50000000, 200000000, 66666667],
    'metode': ['garis_lurus', 'saldo_menurun_berganda', 'garis_lurus', 'saldo_menurun_berganda', 'garis_lurus']
}

# Convert to DataFrame
df_laba = pd.DataFrame(laba_data)
df_aset = pd.DataFrame(aset_data)

print("\n1. DATA OVERVIEW")
print("-" * 30)
print(f"📊 Total periode analisis: {df_laba['tahun'].nunique()} tahun ({df_laba['tahun'].min()}-{df_laba['tahun'].max()})")
print(f"📈 Jumlah skenario: {df_laba['skenario'].nunique()} skenario")
print(f"🏭 Total aset dianalisis: {len(df_aset)} aset")
print(f"💰 Rata-rata pendapatan tahunan: Rp {df_laba['pendapatan'].mean():,.0f}")

# =============================================
# 2. DETEKSI DAN REMOVAL OUTLIERS
# =============================================

print("\n2. DETEKSI OUTLIERS")
print("-" * 30)

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check outliers untuk laba bersih
outliers_laba, lb_laba, ub_laba = detect_outliers_iqr(df_laba, 'laba_bersih')
outliers_aset, lb_aset, ub_aset = detect_outliers_iqr(df_aset, 'nilai_perolehan')

print(f"🔍 Outliers pada laba bersih: {len(outliers_laba)} data point")
print(f"🔍 Outliers pada nilai aset: {len(outliers_aset)} data point")

# Karena data ini adalah proyeksi yang realistis, kita akan mempertahankan semua data
print("✅ Semua data point dianggap valid untuk analisis bisnis")

# =============================================
# 3. KALKULASI METRICS KUNCI
# =============================================

print("\n3. PERHITUNGAN METRICS KUNCI")
print("-" * 30)

# Pivot data untuk analisis perbandingan
df_pivot = df_laba.pivot(index='tahun', columns='skenario', values=['laba_bersih', 'pph_badan'])

# Hitung penghematan pajak
df_laba['penghematan_pajak'] = 0
for tahun in df_laba['tahun'].unique():
    normal_pph = df_laba[(df_laba['tahun'] == tahun) & (df_laba['skenario'] == 'normal')]['pph_badan'].iloc[0]
    holiday_pph = df_laba[(df_laba['tahun'] == tahun) & (df_laba['skenario'] == 'tax_holiday')]['pph_badan'].iloc[0]
    df_laba.loc[(df_laba['tahun'] == tahun) & (df_laba['skenario'] == 'tax_holiday'), 'penghematan_pajak'] = normal_pph - holiday_pph

# Summary statistics
total_penghematan = df_laba[df_laba['skenario'] == 'tax_holiday']['penghematan_pajak'].sum()
total_laba_normal = df_laba[df_laba['skenario'] == 'normal']['laba_bersih'].sum()
total_laba_holiday = df_laba[df_laba['skenario'] == 'tax_holiday']['laba_bersih'].sum()

print(f"💰 Total penghematan pajak (2023-2025): Rp {total_penghematan:,.0f}")
print(f"📊 Peningkatan laba bersih: Rp {total_laba_holiday - total_laba_normal:,.0f}")
print(f"📈 ROI Tax Holiday: {((total_laba_holiday - total_laba_normal) / total_laba_normal * 100):.1f}%")

# =============================================
# 4. ANALISIS DEPRESIASI
# =============================================

print("\n4. ANALISIS METODE DEPRESIASI")
print("-" * 30)

# Analisis berdasarkan metode depresiasi
depresiasi_summary = df_aset.groupby('metode').agg({
    'nilai_perolehan': ['count', 'sum', 'mean'],
    'depresiasi_tahunan': ['sum', 'mean']
}).round(0)

print("📋 Ringkasan per Metode Depresiasi:")
for metode in df_aset['metode'].unique():
    subset = df_aset[df_aset['metode'] == metode]
    total_nilai = subset['nilai_perolehan'].sum()
    total_depresiasi = subset['depresiasi_tahunan'].sum()
    rata_depresiasi = (total_depresiasi / total_nilai) * 100

    print(f"  {metode}:")
    print(f"    - Jumlah aset: {len(subset)}")
    print(f"    - Total nilai: Rp {total_nilai:,.0f}")
    print(f"    - Depresiasi tahunan: Rp {total_depresiasi:,.0f}")
    print(f"    - Rate depresiasi: {rata_depresiasi:.1f}%")

# =============================================
# 5. VISUALISASI INTERAKTIF
# =============================================

print("\n5. MEMBUAT VISUALISASI INTERAKTIF")
print("-" * 30)

# ===== GRAFIK 1: PERBANDINGAN LABA BERSIH =====
fig1 = go.Figure()

# Data untuk normal dan tax holiday
tahun_list = sorted(df_laba['tahun'].unique())
laba_normal = [df_laba[(df_laba['tahun'] == t) & (df_laba['skenario'] == 'normal')]['laba_bersih'].iloc[0] for t in tahun_list]
laba_holiday = [df_laba[(df_laba['tahun'] == t) & (df_laba['skenario'] == 'tax_holiday')]['laba_bersih'].iloc[0] for t in tahun_list]

fig1.add_trace(go.Scatter(
    x=tahun_list, y=laba_normal,
    mode='lines+markers',
    name='Skenario Normal',
    line=dict(color='#FF6B6B', width=4),
    marker=dict(size=10, symbol='circle'),
    hovertemplate='<b>Tahun:</b> %{x}<br><b>Laba Bersih:</b> Rp %{y:,.0f}<br><extra></extra>'
))

fig1.add_trace(go.Scatter(
    x=tahun_list, y=laba_holiday,
    mode='lines+markers',
    name='Tax Holiday',
    line=dict(color='#4ECDC4', width=4),
    marker=dict(size=10, symbol='diamond'),
    hovertemplate='<b>Tahun:</b> %{x}<br><b>Laba Bersih:</b> Rp %{y:,.0f}<br><extra></extra>'
))

fig1.update_layout(
    title={
        'text': '<b>📈 PERBANDINGAN LABA BERSIH: NORMAL vs TAX HOLIDAY</b>',
        'x': 0.5,
        'font': {'size': 20, 'color': '#2C3E50'}
    },
    xaxis_title='<b>Tahun</b>',
    yaxis_title='<b>Laba Bersih (Rp)</b>',
    font=dict(size=12),
    hovermode='x unified',
    plot_bgcolor='rgba(248,249,250,0.8)',
    paper_bgcolor='white',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig1.show()

# ===== GRAFIK 2: PENGHEMATAN PAJAK 3D =====
years_3d = [2023, 2024, 2025]
penghematan_3d = [138600000, 180400000, 200200000]
laba_increase_3d = [138600000, 180400000, 200200000]

fig2 = go.Figure(data=[go.Scatter3d(
    x=years_3d,
    y=penghematan_3d,
    z=laba_increase_3d,
    mode='markers+lines',
    marker=dict(
        size=15,
        color=penghematan_3d,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Penghematan Pajak (Rp)")
    ),
    line=dict(color='darkblue', width=6),
    hovertemplate='<b>Tahun:</b> %{x}<br><b>Penghematan Pajak:</b> Rp %{y:,.0f}<br><b>Peningkatan Laba:</b> Rp %{z:,.0f}<extra></extra>'
)])

fig2.update_layout(
    title={
        'text': '<b>💰 DAMPAK 3D TAX HOLIDAY: PENGHEMATAN & PENINGKATAN LABA</b>',
        'x': 0.5,
        'font': {'size': 18, 'color': '#2C3E50'}
    },
    scene=dict(
        xaxis_title='Tahun',
        yaxis_title='Penghematan Pajak (Rp)',
        zaxis_title='Peningkatan Laba (Rp)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    paper_bgcolor='white'
)

fig2.show()

# ===== GRAFIK 3: ANALISIS DEPRESIASI METODE =====
fig3 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Distribusi Nilai Aset per Metode', 'Depresiasi Tahunan per Metode',
                   'Perbandingan Rate Depresiasi', 'Komposisi Aset per Kategori'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"type": "pie"}]]
)

# Subplot 1: Bar chart nilai aset
metode_colors = {'garis_lurus': '#FF9999', 'saldo_menurun_berganda': '#66B2FF'}
for i, metode in enumerate(df_aset['metode'].unique()):
    subset = df_aset[df_aset['metode'] == metode]
    fig3.add_trace(
        go.Bar(x=subset['aset_id'], y=subset['nilai_perolehan'],
               name=f'{metode}', marker_color=metode_colors.get(metode, '#999999'),
               hovertemplate='<b>Aset:</b> %{x}<br><b>Nilai:</b> Rp %{y:,.0f}<extra></extra>'),
        row=1, col=1
    )

# Subplot 2: Scatter plot depresiasi
for metode in df_aset['metode'].unique():
    subset = df_aset[df_aset['metode'] == metode]
    fig3.add_trace(
        go.Scatter(x=subset['nilai_perolehan'], y=subset['depresiasi_tahunan'],
                  mode='markers', name=f'{metode}',
                  marker=dict(size=12, color=metode_colors.get(metode, '#999999')),
                  hovertemplate='<b>Metode:</b> %{fullData.name}<br><b>Nilai Aset:</b> Rp %{x:,.0f}<br><b>Depresiasi:</b> Rp %{y:,.0f}<extra></extra>'),
        row=1, col=2
    )

# Subplot 3: Comparison chart
metode_summary = df_aset.groupby('metode').agg({
    'nilai_perolehan': 'sum',
    'depresiasi_tahunan': 'sum'
}).reset_index()
metode_summary['rate_depresiasi'] = (metode_summary['depresiasi_tahunan'] / metode_summary['nilai_perolehan']) * 100

fig3.add_trace(
    go.Bar(x=metode_summary['metode'], y=metode_summary['rate_depresiasi'],
           name='Rate Depresiasi (%)', marker_color='#FFD93D',
           hovertemplate='<b>Metode:</b> %{x}<br><b>Rate:</b> %{y:.1f}%<extra></extra>'),
    row=2, col=1
)

# Subplot 4: Pie chart kategori aset
kategori_counts = df_aset['kategori'].value_counts()
fig3.add_trace(
    go.Pie(labels=kategori_counts.index, values=kategori_counts.values,
           name="Kategori Aset", hole=0.4,
           hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>'),
    row=2, col=2
)

fig3.update_layout(
    title={
        'text': '<b>🏭 ANALISIS KOMPREHENSIF METODE DEPRESIASI</b>',
        'x': 0.5,
        'font': {'size': 18, 'color': '#2C3E50'}
    },
    height=800,
    showlegend=True,
    paper_bgcolor='white'
)

fig3.show()

# ===== GRAFIK 4: DASHBOARD SUMMARY ANIMASI =====
fig4 = go.Figure()

# Animasi untuk setiap tahun
for i, tahun in enumerate(sorted(df_laba['tahun'].unique())):
    subset = df_laba[df_laba['tahun'] <= tahun]

    # Data kumulatif
    normal_cumulative = subset[subset['skenario'] == 'normal']['laba_bersih'].sum()
    holiday_cumulative = subset[subset['skenario'] == 'tax_holiday']['laba_bersih'].sum()

    fig4.add_trace(go.Bar(
        x=['Normal', 'Tax Holiday'],
        y=[normal_cumulative, holiday_cumulative],
        name=f'Hingga {tahun}',
        visible=False,
        marker_color=['#FF6B6B', '#4ECDC4'],
        text=[f'Rp {normal_cumulative:,.0f}', f'Rp {holiday_cumulative:,.0f}'],
        textposition='auto',
        hovertemplate='<b>Skenario:</b> %{x}<br><b>Laba Kumulatif:</b> Rp %{y:,.0f}<extra></extra>'
    ))

# Buat animasi visible
fig4.data[0].visible = True

# Animation buttons
steps = []
for i in range(len(fig4.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig4.data)},
              {"title": f"📊 Laba Kumulatif Hingga {2021 + i}"}],
        label=f"{2021 + i}"
    )
    step["args"][0]["visible"][i] = True
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Tahun: "},
    pad={"t": 50},
    steps=steps
)]

fig4.update_layout(
    title={
        'text': '<b>📊 DASHBOARD LABA KUMULATIF (ANIMASI)</b>',
        'x': 0.5,
        'font': {'size': 20, 'color': '#2C3E50'}
    },
    sliders=sliders,
    yaxis_title='<b>Laba Kumulatif (Rp)</b>',
    font=dict(size=12),
    plot_bgcolor='rgba(248,249,250,0.8)',
    paper_bgcolor='white',
    height=600
)

fig4.show()

# =============================================
# 6. KESIMPULAN DAN REKOMENDASI
# =============================================

print("\n" + "="*60)
print("📋 EXECUTIVE SUMMARY & REKOMENDASI STRATEGIS")
print("="*60)

print(f"""
🎯 KEY FINDINGS:

1. DAMPAK TAX HOLIDAY (2023-2025):
   • Total penghematan pajak: Rp {total_penghematan:,.0f}
   • Peningkatan laba bersih: {((total_laba_holiday - total_laba_normal) / total_laba_normal * 100):.1f}%
   • Rata-rata penghematan per tahun: Rp {total_penghematan/3:,.0f}

2. EFISIENSI METODE DEPRESIASI:
   • Garis Lurus: Rate depresiasi {(df_aset[df_aset['metode']=='garis_lurus']['depresiasi_tahunan'].sum() / df_aset[df_aset['metode']=='garis_lurus']['nilai_perolehan'].sum() * 100):.1f}%
   • Saldo Menurun: Rate depresiasi {(df_aset[df_aset['metode']=='saldo_menurun_berganda']['depresiasi_tahunan'].sum() / df_aset[df_aset['metode']=='saldo_menurun_berganda']['nilai_perolehan'].sum() * 100):.1f}%

3. PROYEKSI PERTUMBUHAN:
   • CAGR Pendapatan: {(((2000000000/1000000000)**(1/4)) - 1)*100:.1f}%
   • Margin laba operasional stabil: ~{(df_laba['laba_operasional'].mean()/df_laba['pendapatan'].mean()*100):.0f}%

💡 REKOMENDASI STRATEGIS:

✅ IMPLEMENTASI TAX HOLIDAY:
   • Manfaatkan tax holiday untuk investasi ekspansi
   • Reinvestasi penghematan pajak untuk R&D dan teknologi
   • Optimalisasi cash flow untuk pertumbuhan organik

✅ OPTIMASI DEPRESIASI:
   • Pertimbangkan metode saldo menurun untuk aset produktif
   • Evaluasi ulang umur ekonomis aset secara berkala
   • Alignment dengan strategi pajak perusahaan

✅ STRATEGIC PLANNING:
   • Persiapkan strategi post tax holiday
   • Diversifikasi portofolio investasi
   • Strengthen operational efficiency untuk maintain margin
""")

print("\n🔍 ANALISIS SELESAI - SIAP UNTUK PRESENTASI KLIEN")
print("="*60)