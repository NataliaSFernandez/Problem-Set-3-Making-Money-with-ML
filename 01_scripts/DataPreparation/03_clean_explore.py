"""
Script: 03_clean__explore.py
=================================
Limpieza y exploración de datos OSM, text y de properaty para Chapinero.

Inputs:
    - train_osm.csv / test_osm.csv  → Base + OSM spatial features + text-derived features

Output:
    - train_final.csv / test_final.csv guardados en /00_data/processed  Merged y limpios 
    - Outputs guardadas en /02_outputs/DataPreparation
"""
# =============================================================================
# 0. LIBRERÍAS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Colores para las figuras
PALETTE_MAIN   = "#3a5e8c"   
PALETTE_CHAP   = "#e05c1a"   
PALETTE_ACCENT = "#2A9D8F"  
 
#Estilo para las figuras 
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "figure.figsize": (10, 5)})
 
# Carpeta de salida para figuras
FIG_DIR = Path("02_outputs/DataPreparation")
PRICE_COL = "price"
# =============================================================================
# 1. CARGA  DE BASES DE DATOS
# =============================================================================

print("1. CARGA Y MERGE")
print("=" * 60)

# --- 1.1 Cargar bases --------------------------------------------------------
train  = pd.read_csv("00_data/processed/train_osm.csv")      
test = pd.read_csv("00_data/processed/test_osm.csv")

 
print(f"Train base:  {train.shape[0]:,} filas × {train.shape[1]} cols")
print(f"Test base:   {test.shape[0]:,} filas × {test.shape[1]} cols")

# =============================================================================
# 2. LIMPIEZA DE DATOS
# =============================================================================
print("\n" + "=" * 60)
print("2. LIMPIEZA")
print("=" * 60)

#    2. 1 Variables a eliminar completamente --------------------------------
# city        todas son Bogotá (sin varianza)
# operation_type  todas son ventas (sin varianza)
# title       texto largo, no usable directamente en regresión
# description texto largo, no usable directamente en regresión
DROP_COLS = ["city", "operation_type", "title", "description"]
 
# Variables de texto extraídas de descripción (binarias / conteos) 
# NAs = la palabra no apareció en la descripción , se reemplazan por 0
TEXT_VARS = [
    "remodelado",
    "gimnasio",
    "vista_panoramica",
    "piso_txt",
    "conjunto_cerrado",
    "deposito",
    "amenidades",
    "num_amenidades",
    "balcon_terraza",
    "parqueaderos_txt",
    "tfidf_premium",
]

# ── Variables categórica ───────────────────────────────────────────────────────
# month → efectos estacionales (no es continua ni ordinal lineal)
CAT_VARS = ["month","year"]

STRUCTURAL_VARS = [
    "surface_total",
    "surface_covered",
    "rooms",
    "bedrooms",
    "bathrooms",  
]

# ── Variables continuas OSM 
OSM_VARS = [
    "dist_cbd_km",
    "dist_transmilenio_m",
    "dist_via_arterial_m",
    "dist_hospital_m",
    "dist_centro_com_m",
    "dist_parque_m",
    "n_restaurantes_500m",
    "n_bancos_500m",
    "walkability_score",
    "densidad_vial",
]

# ── Variable objetivo ─────────────────────────────────────────────────────────
PRICE_COL = "price"

# ── Identificadores / espaciales (no entran como features) ────────────────────
ID_VARS = ["property_id", "lat", "lon"]
 
print(f"  Eliminar           ({len(DROP_COLS)}): {DROP_COLS}")
print(f"  Variables texto    ({len(TEXT_VARS)}): {TEXT_VARS}")
print(f"  Variables cat.     ({len(CAT_VARS)}): {CAT_VARS}")
print(f"  Vars estructurales ({len(STRUCTURAL_VARS)}): {STRUCTURAL_VARS}")
print(f"  Vars OSM           ({len(OSM_VARS)}): {OSM_VARS}")

# =============================================================================
# 3. LIMPIEZA
# =============================================================================
print("\n" + "=" * 60)
print("3. LIMPIEZA")
print("=" * 60)

# ── 3.1 Diagnóstico de NAs -----------------------------------------
def na_report(df, name):
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].reset_index()
    miss.columns = ["variable", "pct_na"]
    miss["n_na"] = (miss["pct_na"] / 100 * len(df)).astype(int)
    print(f"\n--- NAs en {name} (variables con >0%) ---")
    print(miss.to_string(index=False))
    return miss
 
miss_train_pre = na_report(train, "TRAIN )")
miss_test_pre  = na_report(test,  "TEST ")

# ── 3.2 Figura: NAs antes de limpiar ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, miss, title in zip(
    axes,
    [miss_train_pre, miss_test_pre],
    ["Train — Bogotá (excl. Chapinero)", "Test — Chapinero"]
):
    top = miss.head(20)
    colors = [PALETTE_CHAP if v > 50 else PALETTE_MAIN for v in top["pct_na"]]
    ax.barh(top["variable"][::-1], top["pct_na"][::-1], color=colors[::-1])
    ax.set_xlabel("% de observaciones faltantes")
    ax.set_title(title)
    ax.axvline(50, color="grey", lw=0.8, ls="--", alpha=0.6)
 
plt.suptitle("Valores faltantes por variable (antes de limpieza)", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_missing_pre.png"); plt.close()
print("\n  Figura 01_missing_pre.png guardada")

# ── 3.3 Eliminar columnas sin varianza ---------------------------------------
cols_to_drop = [c for c in DROP_COLS if c in train.columns]
train.drop(columns=cols_to_drop, inplace=True)
test.drop(columns=[c for c in DROP_COLS if c in test.columns], inplace=True)
print(f"\nColumnas eliminadas: {cols_to_drop}")

# ── 3.4 Variables de texto: NAs → 0 -------------------------------------------
for col in TEXT_VARS:
    if col in train.columns:
        train[col] = train[col].fillna(0)
    if col in test.columns:
        test[col] = test[col].fillna(0)
 
print(f"Variables de texto con NAs → 0: {TEXT_VARS}")

# ── 3.5 Variables continuas: NAs → mediana del train ─────────────────────────
# Para variables estructurales y OSM con pocos NAs se imputa la mediana.
# Se calcula la mediana sobre train para no filtrar información del test.
IMPUTE_COLS = STRUCTURAL_VARS + OSM_VARS
impute_medians = {}
 
for col in IMPUTE_COLS:
    if col in train.columns:
        med = train[col].median()
        impute_medians[col] = med
        train[col] = train[col].fillna(med)
        if col in test.columns:
            test[col] = test[col].fillna(med)
 
print(f"\nImputación por mediana (train) en {len(impute_medians)} variables continuas.")

# 3.6 Variable categórica: month y año a string (para pasar a dummies)
for col in CAT_VARS:
    if col in train.columns:
        train[col] = train[col].astype(str).str.zfill(2)   # "01", "02", ...
    if col in test.columns:
        test[col] = test[col].astype(str).str.zfill(2)

print(f"Categóricas convertidas a string: {CAT_VARS}")

# ── 3.9 Guardar datasets limpios -----------------------------------------------
train.to_csv("00_data/processed/train_final.csv", index=False)
test.to_csv("00_data/processed/test_final.csv",   index=False)
print(f" train_final.csv guardado: {train.shape[0]:,} × {train.shape[1]}")
print(f"test_final.csv  guardado: {test.shape[0]:,}  × {test.shape[1]}")

# =============================================================================
# 4. EXPLORACIÓN DE DATOS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("4. EDA")
print("=" * 60)

# ------------------------------------------------------------------------------
# 4.1 Distribución del precio (train)
# ------------------------------------------------------------------------------
if PRICE_COL in train.columns:
    price_m  = train[PRICE_COL] / 1e6
    log_price = np.log(train[PRICE_COL])
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
 
    # Panel A: precio en millones COP
    bw = 3.5 * price_m.std() / len(price_m) ** (1/3)
    n_bins = max(25, int((price_m.max() - price_m.min()) / bw))
    axes[0].hist(price_m, bins=n_bins, color=PALETTE_MAIN,
                 edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Precio (millones COP)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución del precio")
    axes[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
 
    # Panel B: log-precio
    axes[1].hist(log_price, bins=50, color=PALETTE_ACCENT,
                 edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("Log(precio)")
    axes[1].set_title("Distribución del log-precio")
    med_log = log_price.median()
    axes[1].axvline(med_log, color=PALETTE_CHAP, lw=2, ls="--",
                    label=f"Mediana = {med_log:.2f}")
    axes[1].legend()
 
    plt.suptitle("Precio de vivienda — Train (Bogotá) ",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_price_distribution.png"); plt.close()
    print(" Figura 02_price_distribution.png guardada")
 
 # ─────────────────────────────────────────────────────────────────────────────
# 4.2 Estadísticas descriptivas — 
# ─────────────────────────────────────────────────────────────────────────────
desc_cols = ([PRICE_COL] if PRICE_COL in train.columns else []) + \
            [c for c in STRUCTURAL_VARS + OSM_VARS if c in train.columns]
 
desc = train[desc_cols].describe(percentiles=[.10, .25, .5, .75, .90]).T.round(2)
desc.index.name = "Variable"
print("\n--- Estadísticas descriptivas  ---")
print(desc.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4.3 Correlaciones con el precio

if PRICE_COL in train.columns:
    numeric_feats = [c for c in STRUCTURAL_VARS + OSM_VARS + TEXT_VARS
                     if c in train.columns]
    corr_price = (
        train[numeric_feats + [PRICE_COL]]
        .corr()[PRICE_COL]
        .drop(PRICE_COL)
        .sort_values(key=abs, ascending=False)
        .head(20)
    )
 
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE_CHAP if v < 0 else PALETTE_MAIN for v in corr_price.values]
    ax.barh(corr_price.index[::-1], corr_price.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Correlación de Pearson con log-precio")
    ax.set_title("Top 20 variables más correlacionadas con el precio\n"
                 "(azul = positiva, naranja = negativa)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_correlaciones_precio.png"); plt.close()
    print("Figura 03_correlaciones_precio.png guardada")
 
# ─────────────────────────────────────────────────────────────────────────────
# 4.4 Scatter: superficie total vs. precio

if "surface_total" in train.columns:
    df_sc = train[["surface_total", PRICE_COL]].dropna()
    # Recortar outliers extremos (P1 – P99)
    p1_s  = df_sc["surface_total"].quantile(0.01)
    p99_s = df_sc["surface_total"].quantile(0.99)
    p99_p = df_sc[PRICE_COL].quantile(0.99)
    df_sc = df_sc[
        df_sc["surface_total"].between(p1_s, p99_s) &
        (df_sc[PRICE_COL] <= p99_p)
    ].copy()
 
    log_s = np.log(df_sc["surface_total"])
    log_p = np.log(df_sc[PRICE_COL])
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    # Panel A: escala original (recortada)
    axes[0].scatter(df_sc["surface_total"], df_sc[PRICE_COL] / 1e6,
                    alpha=0.18, s=6, color=PALETTE_MAIN)
    m0, b0 = np.polyfit(df_sc["surface_total"], df_sc[PRICE_COL] / 1e6, 1)
    xs0 = np.linspace(df_sc["surface_total"].min(), df_sc["surface_total"].max(), 200)
    axes[0].plot(xs0, m0*xs0 + b0, color=PALETTE_CHAP, lw=2, label="Tendencia")
    axes[0].set_xlabel("Superficie total (m²)")
    axes[0].set_ylabel("Precio (millones COP)")
    axes[0].set_title("Escala original (P1–P99)")
    axes[0].legend()
 
    # Panel B: log-log — relación lineal más clara
    axes[1].scatter(log_s, log_p, alpha=0.18, s=6, color=PALETTE_ACCENT)
    m1, b1 = np.polyfit(log_s, log_p, 1)
    xs1 = np.linspace(log_s.min(), log_s.max(), 200)
    axes[1].plot(xs1, m1*xs1 + b1, color=PALETTE_CHAP, lw=2,
                 label=f"Elasticidad ≈ {m1:.2f}")
    axes[1].set_xlabel("Log(Superficie total)")
    axes[1].set_ylabel("Log(Precio)")
    axes[1].set_title("Escala log-log (elasticidad precio-superficie)")
    axes[1].legend()
 
    plt.suptitle("Superficie total vs. Precio ",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_scatter_superficie_precio.png"); plt.close()
 
# ─────────────────────────────────────────────────────────────────────────────
# 4.5 Variables OSM: distribución de distancias

if OSM_VARS:
    n_plots = len(OSM_VARS)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()
 
    for i, col in enumerate(OSM_VARS):
        data = train[col].dropna()
        axes[i].hist(data, bins=40, color=PALETTE_MAIN,
                     edgecolor="white", linewidth=0.2)
        axes[i].set_title(col, fontsize=9)
        unit = "m" if "_m" in col else ("km" if "_km" in col else "")
        axes[i].set_xlabel(unit)
        axes[i].tick_params(labelsize=8)
 
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
 
    plt.suptitle("Distribución de variables OSM (accesibilidad y servicios) — Train",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_dist_osm.png"); plt.close()
    print(" Figura 05_dist_osm.png guardada")
 
# ─────────────────────────────────────────────────────────────────────────────
# 4.6 Precio mediano por variable de texto (prima de amenidades)

if TEXT_VARS and PRICE_COL in train.columns:
    medians = {}
    for col in TEXT_VARS:
        d = train[[col, PRICE_COL]].dropna()
        # Binarizar: presencia (>0) vs. ausencia (==0)
        d = d.copy()
        d["tiene"] = (d[col] > 0).astype(int)
        g = d.groupby("tiene")[PRICE_COL].median() / 1e6
        if 0 in g.index and 1 in g.index:
            medians[col] = g[1] - g[0]   # diferencia Con - Sin
 
    if medians:
        med_df = pd.Series(medians).sort_values()
        fig, ax = plt.subplots(figsize=(9, max(3, len(med_df) * 0.45)))
        colors = [PALETTE_CHAP if v < 0 else PALETTE_ACCENT for v in med_df.values]
        ax.barh(med_df.index, med_df.values, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Diferencia en precio mediano (millones COP)\nCon amenidad vs. sin amenidad")
        ax.set_title("Prima de precio por amenidades extraídas del texto\n"
                     "(verde = sube precio, naranja = baja precio)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "06_prima_amenidades_texto.png"); plt.close()
        print("Figura 06_prima_amenidades_texto.png guardada")
 
# ─────────────────────────────────────────────────────────────────────────────
# 4.7 Precio mediano por mes (estacionalidad)

if "month" in train.columns and PRICE_COL in train.columns:
    month_med = (
        train.groupby("month")[PRICE_COL]
        .median()
        .div(1e6)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(month_med.index, month_med.values, color=PALETTE_MAIN,
           edgecolor="white")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Precio mediano (millones COP)")
    ax.set_title("Precio mediano por mes — estacionalidad (Train)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_precio_por_mes.png"); plt.close()
    print(" Figura 07_precio_por_mes.png guardada")
 
# ─────────────────────────────────────────────────────────────────────────────
# 4.8 Train (Bogotá) vs. Test (Chapinero) — comparación de features

compare_cols = [c for c in STRUCTURAL_VARS + OSM_VARS
                if c in train.columns and c in test.columns]
 
if compare_cols:
    n = len(compare_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()
 
    for i, col in enumerate(compare_cols):
        t_data  = train[col].dropna()
        te_data = test[col].dropna()
        axes[i].hist(t_data,  bins=30, alpha=0.55, color=PALETTE_MAIN,
                     label="Train (Bogotá)", density=True)
        axes[i].hist(te_data, bins=30, alpha=0.55, color=PALETTE_CHAP,
                     label="Test (Chapinero)", density=True)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=8)
        if i == 0:
            axes[i].legend(fontsize=8)
 
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
 
    plt.suptitle("Train (Bogotá) vs. Test (Chapinero) — distribución de features\n"
                 "Diferencias revelan el desafío de generalización espacial",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_train_vs_test.png"); plt.close()
    print(" Figura 08_train_vs_test.png guardada")
 
# ─────────────────────────────────────────────────────────────────────────────

 # 4.9 Mapa espacial: precio por coordenadas (train)
# ─────────────────────────────────────────────────────────────────────────────
if all(c in train.columns for c in ["lat", "lon", PRICE_COL]):
 
    # Construir GeoDataFrames en WGS84 y reproyectar a Web Mercator (epsg:3857)
    # para contextily
    def to_gdf(df, lat="lat", lon="lon"):
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=[Point(x, y) for x, y in zip(df[lon], df[lat])],
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")
        return gdf
 
    p98 = train[PRICE_COL].quantile(0.98)
    train_plot = train[train[PRICE_COL] <= p98].copy()
 
    gdf_train = to_gdf(train_plot)
    gdf_test  = to_gdf(test)
 
    fig, ax = plt.subplots(figsize=(9, 10))
 
    # Puntos de train coloreados por precio
    sc = ax.scatter(
        gdf_train.geometry.x, gdf_train.geometry.y,
        c=gdf_train[PRICE_COL] / 1e6,
        cmap="YlOrRd", s=5, alpha=0.55, zorder=3, label="Train (Bogotá)"
    )
 
    # Puntos de test (Chapinero) con borde azul
    ax.scatter(
        gdf_test.geometry.x, gdf_test.geometry.y,
        facecolors="none", edgecolors=PALETTE_MAIN,
        s=12, linewidths=0.6, alpha=0.7, zorder=4, label="Test (Chapinero)"
    )
 
    # Fondo OSM
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12, alpha=0.85)
 
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Precio (millones COP)")
 
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=9,
              handles=[
                  mpatches.Patch(color="#e8a87c", label="Train (Bogotá) — color = precio"),
                  mpatches.Patch(facecolor="none", edgecolor=PALETTE_MAIN,
                                 label="Test (Chapinero)")
              ])
    ax.set_title("Distribución espacial del precio — Train vs. Test (Chapinero)\n"
                 "Los precios más altos se concentran al noreste (Chapinero/Usaquén)",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_mapa_precio_osm.png", bbox_inches="tight"); plt.close()
    print("Fig 09 con mapa OSM guardada")
# ─────────────────────────────────────────────────────────────────────────────
# 4.10 Heatmap de correlaciones entre todas las features
# ─────────────────────────────────────────────────────────────────────────────
all_feat = [c for c in STRUCTURAL_VARS + OSM_VARS + TEXT_VARS
            if c in train.columns]
 
corr_df = (
    train[all_feat + [PRICE_COL]]
    .corr()[PRICE_COL]
    .drop(PRICE_COL)
    .sort_values()
    .reset_index()
)
corr_df.columns = ["variable", "corr"]
 
# Etiquetas de grupo para colorear
def grupo(v):
    if v in TEXT_VARS:    return "Texto"
    if v in OSM_VARS:     return "OSM"
    return "Estructural"
 
corr_df["grupo"] = corr_df["variable"].apply(grupo)
color_map = {"Texto": PALETTE_ACCENT, "OSM": PALETTE_CHAP, "Estructural": PALETTE_MAIN}
 
fig, ax = plt.subplots(figsize=(9, 7))
for _, row in corr_df.iterrows():
    col = color_map[row["grupo"]]
    ax.plot([0, row["corr"]], [row["variable"], row["variable"]],
            color=col, lw=1.8, alpha=0.8)
    ax.scatter(row["corr"], row["variable"],
               color=col, s=55, zorder=5)
 
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Correlación de Pearson con precio")
ax.set_title("Correlación de cada variable con el precio\npor fuente de información",
             fontweight="bold")
 
legend_handles = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
ax.legend(handles=legend_handles, loc="lower right")
plt.tight_layout()
plt.savefig(FIG_DIR / "10_lollipop_correlaciones.png"); plt.close()
print("Fig 10 lollipop correlaciones guardada")

# ─────────────────────────────────────────────────────────────────────────────
# 4.11 Boxplot precio por número de baños  (feature más correlacionado)
# ─────────────────────────────────────────────────────────────────────────────
if "bathrooms" in train.columns:
    df_box = train[["bathrooms", PRICE_COL]].dropna()
    df_box = df_box[df_box["bathrooms"].between(1, 6)].copy()
    df_box["Baños"] = df_box["bathrooms"].astype(int).astype(str)
 
    fig, ax = plt.subplots(figsize=(9, 5))
    order = [str(i) for i in sorted(df_box["bathrooms"].astype(int).unique())]
    sns.boxplot(data=df_box, x="Baños", y=PRICE_COL,
                order=order, color=PALETTE_MAIN,
                flierprops=dict(marker=".", alpha=0.3, markersize=3), ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:,.0f}M"))
    ax.set_xlabel("Número de baños")
    ax.set_ylabel("Precio (millones COP)")
    ax.set_title("Distribución del precio por número de baños\n"
                 "(variable estructural con mayor correlación)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_boxplot_precio_banos.png"); plt.close()
    print("Fig 11 boxplot precio por baños guardada")


 
# =============================================================================
# 4.13 — Comparación Chapinero vs Bogotá: precio/m² por barrio (proxy lat)
#         Muestra en qué segmento del mercado opera Chapinero
# =============================================================================
if "surface_total" in train.columns and "surface_total" in test.columns:
    # Precio/m² en train (donde lo tenemos)
    df_pm2_tr = train[["lat", PRICE_COL, "surface_total"]].dropna()
    df_pm2_tr = df_pm2_tr[df_pm2_tr["surface_total"] > 0]
    df_pm2_tr["precio_m2"] = df_pm2_tr[PRICE_COL] / df_pm2_tr["surface_total"] / 1e6
 
    # Binear por latitud para ver gradiente norte-sur
    df_pm2_tr["lat_bin"] = pd.cut(df_pm2_tr["lat"], bins=15)
    grad = df_pm2_tr.groupby("lat_bin", observed=True)["precio_m2"].median().dropna()
    lat_centers = [iv.mid for iv in grad.index]
 
    # Rango de latitud de Chapinero (aproximado)
    CHAP_LAT_MIN, CHAP_LAT_MAX = 4.62, 4.68
 
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = [
        PALETTE_CHAP if CHAP_LAT_MIN <= c <= CHAP_LAT_MAX else PALETTE_MAIN
        for c in lat_centers
    ]
    ax.bar([str(round(c, 3)) for c in lat_centers], grad.values,
           color=bar_colors, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Latitud (norte →)")
    ax.set_ylabel("Precio mediano/m² (millones COP)")
    ax.set_title("Gradiente norte-sur del precio por m²\n"
                 "(naranja = zona Chapinero)",
                 fontweight="bold")
    plt.xticks(rotation=45, fontsize=7)
    legend_h = [
        mpatches.Patch(color=PALETTE_CHAP, label="Zona Chapinero"),
        mpatches.Patch(color=PALETTE_MAIN, label="Resto de Bogotá"),
    ]
    ax.legend(handles=legend_h)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_gradiente_latitud_pm2.png"); plt.close()
    print(" Fig 12 gradiente norte-sur guardada")
 
# =============================================================================
# 4.14 — Comparación distribución superficie: Train vs Test (Chapinero)
#         Muestra que Chapinero tiene apartamentos más pequeños y más caros
# =============================================================================
if "surface_total" in train.columns and "surface_total" in test.columns:
    tr_s = train["surface_total"].dropna()
    tr_s = tr_s[tr_s <= tr_s.quantile(0.99)]
    te_s = test["surface_total"].dropna()
    te_s = te_s[te_s <= te_s.quantile(0.99)]
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
 
    # Superficie
    axes[0].hist(tr_s, bins=40, alpha=0.6, color=PALETTE_MAIN,
                 label="Train (Bogotá)", density=True)
    axes[0].hist(te_s, bins=40, alpha=0.6, color=PALETTE_CHAP,
                 label="Test (Chapinero)", density=True)
    axes[0].axvline(tr_s.median(), color=PALETTE_MAIN,
                    lw=2, ls="--", label=f"Mediana Train: {tr_s.median():.0f} m²")
    axes[0].axvline(te_s.median(), color=PALETTE_CHAP,
                    lw=2, ls="--", label=f"Mediana Test: {te_s.median():.0f} m²")
    axes[0].set_xlabel("Superficie total (m²)")
    axes[0].set_ylabel("Densidad")
    axes[0].set_title("Distribución de superficie")
    axes[0].legend(fontsize=8)
 
    # Distancia al CBD
    if "dist_cbd_km" in train.columns and "dist_cbd_km" in test.columns:
        tr_cbd = train["dist_cbd_km"].dropna()
        te_cbd = test["dist_cbd_km"].dropna()
        axes[1].hist(tr_cbd, bins=40, alpha=0.6, color=PALETTE_MAIN,
                     label="Train (Bogotá)", density=True)
        axes[1].hist(te_cbd, bins=40, alpha=0.6, color=PALETTE_CHAP,
                     label="Test (Chapinero)", density=True)
        axes[1].axvline(tr_cbd.median(), color=PALETTE_MAIN, lw=2, ls="--",
                        label=f"Mediana Train: {tr_cbd.median():.1f} km")
        axes[1].axvline(te_cbd.median(), color=PALETTE_CHAP, lw=2, ls="--",
                        label=f"Mediana Test: {te_cbd.median():.1f} km")
        axes[1].set_xlabel("Distancia al CBD (km)")
        axes[1].set_ylabel("Densidad")
        axes[1].set_title("Distribución distancia al CBD")
        axes[1].legend(fontsize=8)
 
    plt.suptitle("Chapinero vs. Bogotá — Diferencias estructurales clave\n"
                 "El modelo debe generalizar a un mercado más céntrico y denso",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_chapinero_vs_bogota_clave.png"); plt.close()
    print("Fig 13 Chapinero vs Bogotá (superficie + CBD) guardada")
 
# =============================================================================
# 4.15— Heatmap compacto: solo variables con |corr| > 0.05 con precio
# =============================================================================
top_vars = corr_df[corr_df["corr"].abs() > 0.05]["variable"].tolist()
if len(top_vars) >= 3:
    hm_cols = [PRICE_COL] + top_vars
    hm_corr = train[hm_cols].corr()
 
    fig, ax = plt.subplots(figsize=(max(7, len(hm_cols)*0.6),
                                    max(6, len(hm_cols)*0.55)))
    mask = np.triu(np.ones_like(hm_corr, dtype=bool))
    sns.heatmap(hm_corr, mask=mask, cmap="coolwarm", center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8})
    ax.set_title("Correlaciones entre variables predictoras\n"
                 "(solo variables con |corr precio| > 0.05)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "14_heatmap_compacto.png"); plt.close()
    print("Fig 14 heatmap compacto guardado")
 
# =============================================================================
# 5. RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 60)
print("5. RESUMEN FINAL")
print("=" * 60)
 
print(f"\nTrain final : {len(train):,} obs × {train.shape[1]} variables")
print(f"Test final  : {len(test):,}  obs × {test.shape[1]} variables")
print(f"\n Archivos guardados en 00_data/processed/")
print("\n Stage 1 completado.")