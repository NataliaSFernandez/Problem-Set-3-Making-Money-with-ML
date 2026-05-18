
"""
Random Forest — RF_005 CV ESPACIAL PARA ELEGIR max_depth
=========================================================

¿Qué cambia respecto a RF_004?
-------------------------------
  En RF_003 y RF_004 se eligio  max_depth=10 de forma arbitraria, sin ningún criterio sistemático
  RF_005 busca los mejores hiperparamtros:
    - Busca max_depth en {8, 10, 15}
    - Usa CV ESPACIAL  como criterio de selección
    - Elige el depth que minimiza el MAE espacial 
    - Entrena el modelo final con ese depth óptimo

  Se mantiene n_estimators=500 y min_samples_leaf=5 de RF_004

Grid search
-----------
  Candidatos: max_depth ∈ {8, 10, 15}
  Criterio:   MAE_espacial mínimo (CV espacial, GroupKFold 5-fold)
  Estrategia: grid search exhaustivo (3 valores × 5 folds = 15 ajustes de RF)

Hiperparámetros del modelo final
----------------------------------
    n_estimators    = 500
    max_depth       = elegido por CV espacial
    min_samples_leaf= 5
    max_features    = "sqrt"
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
import openpyxl   # noqa: F401
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")


# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN
# =============================================================================

AUTOR        = "Equipo"
MODEL_ID     = "RF_005"
SEED         = 42
CV_FOLDS     = 5
SPATIAL_GRID = 5

BASE        = Path(__file__).parent.parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "RandomForest" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

for d in [SUBMISSIONS, DIR_MODEL, BASE / "02_outputs"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Grilla de búsqueda ────────────────────────────────────────────────────────
DEPTH_CANDIDATES = [8, 10, 15]   # valores a evaluar con CV espacial

# ── Hiperparámetros fijos (heredados de RF_004) ───────────────────────────────
N_ESTIMATORS     = 500
MIN_SAMPLES_LEAF = 5
MAX_FEATURES     = "sqrt"
# MAX_DEPTH se determina por CV espacial → se registra tras la búsqueda


# =============================================================================
# SECCIÓN 2: FEATURES (idénticas a RF_002, 003, 004)
# =============================================================================

STRUCTURAL = [
    "surface_total", "surface_covered", "rooms",
    "bedrooms", "bathrooms", "month", "year",
]

TEXT = [
    "remodelado", "vista_panoramica", "deposito", "conjunto_cerrado",
    "balcon_terraza", "tfidf_premium", "parqueaderos_txt", "piso_txt",
    "gimnasio", "amenidades", "num_amenidades",
]

OSM = [
    "dist_cbd_km", "dist_transmilenio_m", "dist_via_arterial_m",
    "dist_hospital_m", "dist_centro_com_m", "dist_parque_m",
    "n_restaurantes_500m", "n_bancos_500m", "walkability_score",
    "densidad_vial",
]


# =============================================================================
# SECCIÓN 3: CARGA Y PREPARACIÓN
# =============================================================================

def cargar_datos():
    train = pd.read_csv(PROCESSED / "train_final.csv")
    test  = pd.read_csv(PROCESSED / "test_final.csv")
    return train, test


def construir_features(df, fit_cols=None):
    d = df.copy()
    d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)
    all_cols = STRUCTURAL + TEXT + OSM + ["es_apartamento"]
    cols = [c for c in all_cols if c in d.columns]
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

def cv_aleatorio(rf, X, y):
    kf   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    maes = []
    for tr, va in kf.split(X):
        rf.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[va], rf.predict(X[va])))
    return float(np.mean(maes)), float(np.std(maes))


def cv_espacial(rf, X, y, grupos):
    gkf  = GroupKFold(n_splits=CV_FOLDS)
    maes = []
    for tr, va in gkf.split(X, y, groups=grupos):
        rf.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[va], rf.predict(X[va])))
    return float(np.mean(maes)), float(np.std(maes))


# =============================================================================
# SECCIÓN 5: GRID SEARCH SOBRE max_depth VÍA CV ESPACIAL
# =============================================================================

def buscar_best_depth(X, y, grupos):
    """
    Evalúa cada candidato de max_depth con CV espacial y devuelve el mejor.

      Para cada depth ∈ DEPTH_CANDIDATES:
        → Ajustar RF con ese depth en 5 folds espaciales
        → Calcular MAE_esp promedio
      → Elegir el depth con menor MAE_esp

    Esto garantiza que el hiperparámetro está optimizado para generalizar
    a zonas geográficas no vistas, no solo para folds aleatorios.
    """
    resultados = []
    print(f"  Evaluando max_depth = {DEPTH_CANDIDATES} con CV espacial...")
    for depth in DEPTH_CANDIDATES:
        rf = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=depth,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            n_jobs=-1,
            random_state=SEED,
        )
        mae, std = cv_espacial(rf, X, y, grupos)
        resultados.append({"max_depth": depth, "mae_esp": mae, "std_esp": std})
        print(f"    depth={depth:>3}  →  MAE_esp={mae:.5f} ± {std:.5f}")

    df_grid    = pd.DataFrame(resultados)
    best_idx   = df_grid["mae_esp"].idxmin()
    best_depth = int(df_grid.loc[best_idx, "max_depth"])
    best_mae   = float(df_grid.loc[best_idx, "mae_esp"])
    best_std   = float(df_grid.loc[best_idx, "std_esp"])
    print(f"\n  ✓ Mejor depth = {best_depth}  (MAE_esp={best_mae:.5f})")
    return best_depth, best_mae, best_std, df_grid


# =============================================================================
# SECCIÓN 6: DIAGNÓSTICOS
# =============================================================================

def plot_grid_search(df_grid, best_depth):
    """
    Curva de validación espacial: MAE_esp vs. max_depth.
    Muestra cómo cambia el error a medida que el árbol se hace más profundo.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        df_grid["max_depth"], df_grid["mae_esp"],
        yerr=df_grid["std_esp"],
        fmt="o-", color="#e74c3c", capsize=5, lw=2,
        label="CV espacial"
    )
    ax.axvline(best_depth, color="black", lw=1.2, linestyle="--",
               label=f"best depth = {best_depth}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("MAE log(price)")
    ax.set_title(f"Grid search max_depth — {MODEL_ID}\n"
                 f"Criterio: CV espacial (GroupKFold {CV_FOLDS}-fold)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "grid_search_depth.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/grid_search_depth.png")


def plot_importancia(rf, feature_cols):
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    top = imp.tail(min(20, len(imp)))
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
    colors = []
    for c in top.index:
        if c in STRUCTURAL or c == "es_apartamento":
            colors.append("#3498db")
        elif c in TEXT:
            colors.append("#e67e22")
        else:
            colors.append("#2ecc71")
    ax.barh(list(top.index), list(top.values), color=colors, edgecolor="white")
    ax.set_xlabel("Importancia (reducción media de impureza)")
    ax.set_title(f"Feature Importance — {MODEL_ID}\n"
                 "Azul=Estructural | Naranja=Texto | Verde=OSM")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/feature_importance.png")


def plot_residuos(y_true, y_pred):
    residuos = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuos, alpha=0.3, s=5, color="#3498db")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Predicción log(price)")
    axes[0].set_ylabel("Residuo")
    axes[0].set_title(f"Residuos vs. Predichos — {MODEL_ID}")
    axes[1].hist(residuos, bins=60, color="#3498db", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("Residuo")
    axes[1].set_title(f"Distribución de residuos — {MODEL_ID}\n"
                      f"media={residuos.mean():.4f}, std={residuos.std():.4f}")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "residuos.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/residuos.png")


def plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["CV Aleatorio", "CV Espacial"]
    means  = [mae_rand, mae_esp]
    stds   = [std_rand, std_esp]
    colors = ["#3498db", "#e74c3c"]
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
# SECCIÓN 7: SUBMISSION
# =============================================================================

def generar_submission(test, y_pred_log, best_depth):
    sub = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),
    })
    # best_depth se conoce solo tras el grid search 
    mf_str   = str(MAX_FEATURES).replace(".", "")
    sub_name = (
        f"RF_ntrees{N_ESTIMATORS}_d{best_depth}"
        f"_leaf{MIN_SAMPLES_LEAF}_mf{mf_str}"
        f"_cv{CV_FOLDS}_gridESP_{MODEL_ID}.csv"
    )
    sub.to_csv(SUBMISSIONS / sub_name, index=False)
    print(f"  Submission: 03_submissions/{sub_name}  ({len(sub):,} filas)")
    return sub_name


# =============================================================================
# SECCIÓN 8: REGISTRO
# =============================================================================

def registrar(mae_rand, std_rand, mae_esp, std_esp, mae_train,
              n_features, best_depth, sub_name):
    sesgo = mae_esp - mae_rand
    nueva = {
        "model_id":          MODEL_ID,
        "fecha":             str(date.today()),
        "autor":             AUTOR,
        "algoritmo":         "RandomForest",
        "n_features":        n_features,
        "n_estimators":      N_ESTIMATORS,
        "max_depth":         str(best_depth),
        "min_samples_leaf":  MIN_SAMPLES_LEAF,
        "max_features":      MAX_FEATURES,
        "cv_folds":          CV_FOLDS,
        "cv_mae_log":        round(mae_rand,  5),
        "cv_std_log":        round(std_rand,  5),
        "esp_mae_log":       round(mae_esp,   5),
        "esp_std_log":       round(std_esp,   5),
        "train_mae_log":     round(mae_train, 5),
        "sesgo_delta":       round(sesgo,     5),
        "kaggle_public_MAE": None,
        "features_grupos":   f"structural={len(STRUCTURAL)}, text={len(TEXT)}, osm={len(OSM)}, es_apartamento",
        "spatial_grid":      f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "submission_file":   sub_name,
        "notas": (
            f"Grid search max_depth={DEPTH_CANDIDATES} via CV espacial. "
            f"Best depth={best_depth}. "
            f"n_estimators={N_ESTIMATORS}, min_samples_leaf={MIN_SAMPLES_LEAF}. "
            f"Features completas. "
            f"Sesgo Δ={sesgo:+.5f}."
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
# SECCIÓN 9: MAIN
# =============================================================================

def main():
    print(f"{'='*60}")
    print(f"  RANDOM FOREST — {MODEL_ID}  (CV espacial elige max_depth)")
    print(f"{'='*60}")

    print("\n[1/8] Cargando datos...")
    train, test = cargar_datos()
    y_train = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    print("\n[2/8] Construyendo features...")
    X_train_df   = construir_features(train)
    feature_cols = list(X_train_df.columns)
    X_test_df    = construir_features(test, fit_cols=feature_cols)
    X_train      = X_train_df.values.astype(float)
    X_test       = X_test_df.values.astype(float)
    print(f"  Features: {len(feature_cols)}")

    print("\n[3/8] Construyendo grupos espaciales...")
    grupos = construir_grupos_espaciales(train)
    print(f"  Cuadrícula {SPATIAL_GRID}×{SPATIAL_GRID} → {len(np.unique(grupos))} bloques")

    # ── Grid search ──────────────────────────────────────────────────────────
    print(f"\n[4/8] Grid search sobre max_depth via CV espacial...")
    best_depth, best_mae_esp, best_std_esp, df_grid = buscar_best_depth(
        X_train, y_train, grupos
    )
    # Guardar tabla del grid search
    df_grid.to_csv(DIR_MODEL / "grid_search_depth.csv", index=False)
    plot_grid_search(df_grid, best_depth)

    # ── CV aleatorio con el depth óptimo ──────────────────────────────────────
    rf_final = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=best_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        n_jobs=-1,
        random_state=SEED,
    )

    print(f"\n[5/8] CV aleatorio con best_depth={best_depth}...")
    mae_rand, std_rand = cv_aleatorio(rf_final, X_train, y_train)
    print(f"  MAE_log aleatorio = {mae_rand:.5f} ± {std_rand:.5f}")

    # El MAE espacial ya viene del grid search (es el del best depth)
    mae_esp  = best_mae_esp
    std_esp  = best_std_esp
    sesgo    = mae_esp - mae_rand
    print(f"\n[6/8] CV espacial (resultado del grid search)...")
    print(f"  MAE_log espacial  = {mae_esp:.5f} ± {std_esp:.5f}")
    print(f"  Sesgo Δ           = {sesgo:+.5f}")

    # ── Modelo final ──────────────────────────────────────────────────────────
    print("\n[7/8] Modelo final sobre todo el train...")
    rf_final.fit(X_train, y_train)
    mae_train = mean_absolute_error(y_train, rf_final.predict(X_train))
    print(f"  MAE_log train     = {mae_train:.5f}")

    print("\n[8/8] Diagnósticos, submission y registro...")
    plot_importancia(rf_final, feature_cols)
    plot_residuos(y_train, rf_final.predict(X_train))
    plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp)
    sub_name = generar_submission(test, rf_final.predict(X_test), best_depth)
    registrar(mae_rand, std_rand, mae_esp, std_esp, mae_train,
              len(feature_cols), best_depth, sub_name)

    print(f"\n{'='*60}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"  best max_depth    = {best_depth}  (elegido por CV espacial)")
    print(f"  MAE_log aleatorio = {mae_rand:.5f} ")
    print(f"  MAE_log espacial  = {mae_esp:.5f} ")
    print(f"  Sesgo Δ           = {sesgo:+.5f}")
    print(f"  Submission: 03_submissions/{sub_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()