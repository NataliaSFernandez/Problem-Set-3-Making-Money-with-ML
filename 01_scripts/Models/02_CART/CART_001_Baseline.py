"""
CART — CART_001 BASELINE
=========================
Baseline mínimo: features estructurales de Properati (modelo más simple)
(m², cuartos, baños, mes, año) con los hiperparámetros default de sklearn.

  ¿Cuánto predice el CART usando únicamente las variables básicas de
  Properati, sin ninguna ingeniería de features adicional?

  Hiperparámetros default de sklearn:
    max_depth        = None  (árbol crece hasta pureza)
    min_samples_leaf = 1     (default sklearn)
    ccp_alpha        = 0.0   (sin poda por costo-complejidad)

Pipeline
--------
1. Carga  train_final.csv / test_final.csv
2. Features: solo STRUCTURAL + es_apartamento
3. CV aleatorio 5-fold  (KFold)      → MAE_rand
4. CV espacial  5-fold  (GroupKFold) → MAE_esp
5. Modelo final sobre todo el train
6. Diagnósticos: feature importance, árbol top3, residuos, comparación CV
7. Submission → 03_submissions/
8. Registro   → 02_outputs/model_registry.xlsx

Estructura de carpetas generada automáticamente al correr el script:
  <repo>/
  ├── 00_data/processed/             ← train_final.csv, test_final.csv (input)
  ├── 01_scripts/Models/02_CART/     ← este script vive aquí
  ├── 02_outputs/
  │   ├── Models/02_CART/CART_001/   ← gráficos del modelo
  │   └── model_registry.xlsx        ← registro de todos los modelos
  └── 03_submissions/                ← CSV para subir a Kaggle
"""

# =============================================================================
# SECCIÓN 0: IMPORTACIONES
# =============================================================================

import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl  # noqa: F401
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree

warnings.filterwarnings("ignore")


# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN
# =============================================================================

AUTOR        = "Dani"
MODEL_ID     = "CART_001"
SEED         = 42
CV_FOLDS     = 5
SPATIAL_GRID = 5

# Rutas — el script vive en 01_scripts/Models/02_CART/
BASE        = Path(__file__).parent.parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "02_CART" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

# Crear todas las carpetas necesarias al arrancar
for d in [
    BASE / "02_outputs",
    BASE / "02_outputs" / "Models",
    BASE / "02_outputs" / "Models" / "02_CART",
    DIR_MODEL,
    SUBMISSIONS,
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hiperparámetros
MAX_DEPTH        = None   # árbol crece hasta pureza
MIN_SAMPLES_LEAF = 1      # default sklearn
CCP_ALPHA        = 0.0    # sin poda por costo-complejidad


# =============================================================================
# SECCIÓN 2: FEATURES
# =============================================================================
# Solo features estructurales del dataset de Properati.
# No se incluyen OSM ni texto — es el baseline mínimo.

STRUCTURAL = [
    "surface_total",
    "surface_covered",
    "rooms",
    "bedrooms",
    "bathrooms",
    "month",
    "year",
]


# =============================================================================
# SECCIÓN 3: CARGA Y PREPARACIÓN
# =============================================================================

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED / "train_final.csv")
    test  = pd.read_csv(PROCESSED / "test_final.csv")
    return train, test


def construir_features(df, fit_cols=None):
    d = df.copy()
    d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)
    cols = [c for c in STRUCTURAL + ["es_apartamento"] if c in d.columns]
    X = d[cols]
    if fit_cols is not None:
        X = X.reindex(columns=fit_cols, fill_value=0)
    return X


def construir_grupos_espaciales(df):
    lat   = df["lat"].values
    lon   = df["lon"].values
    lat_n = (lat - lat.min()) / (lat.max() - lat.min() + 1e-9) * SPATIAL_GRID
    lon_n = (lon - lon.min()) / (lon.max() - lon.min() + 1e-9) * SPATIAL_GRID
    fila  = np.floor(lat_n).astype(int).clip(0, SPATIAL_GRID - 1)
    col   = np.floor(lon_n).astype(int).clip(0, SPATIAL_GRID - 1)
    return fila * SPATIAL_GRID + col


# =============================================================================
# SECCIÓN 4: CROSS-VALIDATION
# =============================================================================

def cv_aleatorio(cart, X, y):
    kf   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    maes = []
    for tr, va in kf.split(X):
        cart.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[va], cart.predict(X[va])))
    return float(np.mean(maes)), float(np.std(maes))


def cv_espacial(cart, X, y, grupos):
    gkf  = GroupKFold(n_splits=CV_FOLDS)
    maes = []
    for tr, va in gkf.split(X, y, groups=grupos):
        cart.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[va], cart.predict(X[va])))
    return float(np.mean(maes)), float(np.std(maes))


# =============================================================================
# SECCIÓN 5: DIAGNÓSTICOS
# =============================================================================

def plot_importancia(cart, feature_cols):
    imp = pd.Series(cart.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(3, len(imp) * 0.4)))
    ax.barh(list(imp.index), list(imp.values), color="#e67e22", edgecolor="white")
    ax.set_xlabel("Importancia (reducción media de impureza)")
    ax.set_title(f"Feature Importance — {MODEL_ID}")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/feature_importance.png")


def plot_arbol(cart, feature_cols):
    fig, ax = plt.subplots(figsize=(16, 6))
    plot_tree(
        cart, max_depth=3, feature_names=feature_cols,
        filled=True, rounded=True, fontsize=7, ax=ax,
    )
    ax.set_title(f"Árbol (primeros 3 niveles) — {MODEL_ID}")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "arbol_top3.png"), dpi=120)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/arbol_top3.png")


def plot_residuos(y_true, y_pred):
    residuos = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuos, alpha=0.3, s=5, color="#e67e22")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Predicción log(price)")
    axes[0].set_ylabel("Residuo")
    axes[0].set_title(f"Residuos vs. Predichos — {MODEL_ID}")
    axes[1].hist(residuos, bins=60, color="#e67e22", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("Residuo")
    axes[1].set_title(
        f"Distribución de residuos — {MODEL_ID}\n"
        f"media={residuos.mean():.4f}  std={residuos.std():.4f}"
    )
    pct_sobre = np.mean(residuos < 0) * 100
    fig.suptitle(
        f"Sobrepredicción: {pct_sobre:.1f}% de obs (residuo<0 → riesgo Zillow)",
        fontsize=8, color="gray",
    )
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "residuos.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/residuos.png")
    print(f"  Sobrepredicción: {pct_sobre:.1f}% de observaciones")


def plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["CV Aleatorio", "CV Espacial"]
    means  = [mae_rand, mae_esp]
    stds   = [std_rand, std_esp]
    colors = ["#e67e22", "#c0392b"]
    bars   = ax.bar(labels, means, yerr=stds, color=colors,
                    capsize=6, edgecolor="white", width=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.1,
                f"{val:.5f}", ha="center", va="bottom", fontsize=10)
    sesgo = mae_esp - mae_rand
    ax.set_ylabel("MAE log(price)")
    ax.set_title(f"CV Aleatorio vs. Espacial — {MODEL_ID}\nSesgo Δ = {sesgo:+.5f}")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "cv_comparacion.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/cv_comparacion.png")


# =============================================================================
# SECCIÓN 6: SUBMISSION
# =============================================================================

def generar_submission(test, y_pred_log):
    sub = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),
    })
    depth_str = "dNone" if MAX_DEPTH is None else f"d{MAX_DEPTH}"
    sub_name  = (
        f"CART_{depth_str}"
        f"_leaf{MIN_SAMPLES_LEAF}"
        f"_ccp{str(CCP_ALPHA).replace('.', '')}"
        f"_cv{CV_FOLDS}_{MODEL_ID}.csv"
    )
    sub.to_csv(SUBMISSIONS / sub_name, index=False)
    print(f"  Submission: 03_submissions/{sub_name}  ({len(sub):,} filas)")
    return sub_name


# =============================================================================
# SECCIÓN 7: REGISTRO
# =============================================================================

def registrar(mae_rand, std_rand, mae_esp, std_esp, mae_train,
              n_features, sub_name):
    sesgo = mae_esp - mae_rand
    nueva = {
        "model_id":          MODEL_ID,
        "fecha":             str(date.today()),
        "autor":             AUTOR,
        "algoritmo":         "CART",
        "n_features":        n_features,
        "max_depth":         str(MAX_DEPTH),
        "min_samples_leaf":  MIN_SAMPLES_LEAF,
        "ccp_alpha":         CCP_ALPHA,
        "cv_folds":          CV_FOLDS,
        "cv_mae_log":        round(mae_rand,  5),
        "cv_std_log":        round(std_rand,  5),
        "esp_mae_log":       round(mae_esp,   5),
        "esp_std_log":       round(std_esp,   5),
        "train_mae_log":     round(mae_train, 5),
        "sesgo_delta":       round(sesgo,     5),
        "kaggle_public_MAE": None,
        "features_grupos":   f"structural={len(STRUCTURAL)}, text=0, osm=0, es_apartamento",
        "spatial_grid":      f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "submission_file":   sub_name,
        "notas": (
            f"Baseline: solo features estructurales + es_apartamento. "
            f"Defaults sklearn: depth=None, leaf=1, ccp=0. "
            f"Sin OSM ni texto. Sesgo Δ={sesgo:+.5f}."
        ),
    }
    df_new = pd.DataFrame([nueva])
    if REGISTRY.exists():
        df_old = pd.read_excel(REGISTRY, engine="openpyxl")
        df_old = df_old[df_old["model_id"] != MODEL_ID]
        df_reg = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_reg = df_new
    with pd.ExcelWriter(REGISTRY, engine="openpyxl") as writer:
        df_reg.to_excel(writer, index=False, sheet_name="registry")
        ws = writer.sheets["registry"]
        for col in ws.columns:
            max_len = max(
                len(str(col[0].value)) if col[0].value else 0,
                *(len(str(c.value)) if c.value else 0 for c in col[1:]),
            )
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)
    print(f"  Registry: 02_outputs/model_registry.xlsx  ({len(df_reg)} modelos)")


# =============================================================================
# SECCIÓN 8: MAIN
# =============================================================================

def main():
    print(f"{'='*60}")
    print(f"  CART — {MODEL_ID}  (Baseline estructural)")
    print(f"{'='*60}")

    print("\n[1/7] Cargando datos...")
    train, test = cargar_datos()
    y_train = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    print("\n[2/7] Construyendo features...")
    X_train_df   = construir_features(train)
    feature_cols = list(X_train_df.columns)
    X_test_df    = construir_features(test, fit_cols=feature_cols)
    X_train      = X_train_df.values.astype(float)
    X_test       = X_test_df.values.astype(float)
    print(f"  Features: {len(feature_cols)}  (solo estructurales + es_apartamento)")

    print("\n[3/7] Construyendo grupos espaciales...")
    grupos = construir_grupos_espaciales(train)
    print(f"  Cuadrícula {SPATIAL_GRID}×{SPATIAL_GRID} → {len(np.unique(grupos))} bloques")

    cart = DecisionTreeRegressor(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        ccp_alpha=CCP_ALPHA,
        random_state=SEED,
    )

    print(f"\n[4/7] CV aleatorio ({CV_FOLDS}-fold KFold)...")
    mae_rand, std_rand = cv_aleatorio(cart, X_train, y_train)
    print(f"  MAE_log aleatorio = {mae_rand:.5f} ± {std_rand:.5f}")

    print(f"\n[5/7] CV espacial ({CV_FOLDS}-fold GroupKFold)...")
    mae_esp, std_esp = cv_espacial(cart, X_train, y_train, grupos)
    sesgo = mae_esp - mae_rand
    print(f"  MAE_log espacial  = {mae_esp:.5f} ± {std_esp:.5f}")
    print(f"  Sesgo Δ           = {sesgo:+.5f}")

    print("\n[6/7] Modelo final sobre todo el train...")
    cart.fit(X_train, y_train)
    mae_train   = mean_absolute_error(y_train, cart.predict(X_train))
    profundidad = cart.get_depth()
    n_hojas     = cart.get_n_leaves()
    print(f"  MAE_log train     = {mae_train:.5f}")
    print(f"  Profundidad real  = {profundidad} | Hojas: {n_hojas}")

    print("\n[7/7] Diagnósticos, submission y registro...")
    plot_importancia(cart, feature_cols)
    plot_arbol(cart, feature_cols)
    plot_residuos(y_train, cart.predict(X_train))
    plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp)
    sub_name = generar_submission(test, cart.predict(X_test))
    registrar(mae_rand, std_rand, mae_esp, std_esp, mae_train,
              len(feature_cols), sub_name)

    print(f"\n{'='*60}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"  MAE_log aleatorio = {mae_rand:.5f}")
    print(f"  MAE_log espacial  = {mae_esp:.5f}")
    print(f"  Sesgo Δ           = {sesgo:+.5f}")
    print(f"  Profundidad real  = {profundidad} | Hojas: {n_hojas}")
    print(f"  Submission: 03_submissions/{sub_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
