"""
CART - CART_002 ALL FEATURES
==============================
Modelo completo: estructurales + OSM + texto, con tuning de hiperparametros
por GridSearchCV usando CV espacial (GroupKFold).

  Hiperparametros (tuning por CV espacial):
    max_depth        en {3, 5, 7, 10, 15, None}
    min_samples_leaf en {5, 10, 20}
    ccp_alpha        en {0.0, 0.0001, 0.0005, 0.001, 0.005}

Pipeline
--------
1. Carga  train_final.csv / test_final.csv
2. Features: STRUCTURAL + OSM + TEXT + es_apartamento
3. GridSearchCV con CV espacial -> mejores hiperparametros
4. CV aleatorio 5-fold (KFold)      -> MAE_rand (optimista)
5. CV espacial  5-fold (GroupKFold) -> MAE_esp  (honesto)
6. Modelo final con mejores hiperparametros
7. Diagnosticos: feature importance (por fuente), arbol top3, residuos, CV
8. Submission -> 03_submissions/
9. Registro   -> 02_outputs/model_registry.xlsx

Estructura de carpetas generada automaticamente al correr el script:
  <repo>/
  ├── 00_data/processed/             <- train_final.csv, test_final.csv
  ├── 01_scripts/Models/02_CART/     <- este script vive aqui
  ├── 02_outputs/
  │   ├── Models/02_CART/CART_002/   <- graficos del modelo
  │   └── model_registry.xlsx
  └── 03_submissions/                <- CSV para Kaggle
"""

# =============================================================================
# SECCION 0: IMPORTACIONES
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
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree

warnings.filterwarnings("ignore")


# =============================================================================
# SECCION 1: CONFIGURACION
# =============================================================================

AUTOR        = "Dani"
MODEL_ID     = "CART_002"
SEED         = 42
CV_FOLDS     = 5
SPATIAL_GRID = 5

# Rutas - el script vive en 01_scripts/Models/02_CART/
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

# Grilla de hiperparametros
PARAM_GRID = {
    "max_depth":        [3, 5, 7, 10, 15, None],
    "min_samples_leaf": [5, 10, 20],
    "ccp_alpha":        [0.0, 0.0001, 0.0005, 0.001, 0.005],
}


# =============================================================================
# SECCION 2: FEATURES
# =============================================================================

STRUCTURAL = [
    "surface_total", "surface_covered", "rooms",
    "bedrooms", "bathrooms", "month", "year",
]

OSM = [
    "dist_cbd_km", "dist_transmilenio_m", "dist_via_arterial_m",
    "dist_hospital_m", "dist_centro_com_m", "dist_parque_m",
    "n_restaurantes_500m", "n_bancos_500m", "walkability_score", "densidad_vial",
]

TEXT = [
    "remodelado", "vista_panoramica", "deposito", "conjunto_cerrado",
    "balcon_terraza", "tfidf_premium", "parqueaderos_txt", "piso_txt",
    "gimnasio", "amenidades", "num_amenidades",
]


# =============================================================================
# SECCION 3: CARGA Y PREPARACION
# =============================================================================

def cargar_datos():
    train = pd.read_csv(PROCESSED / "train_final.csv")
    test  = pd.read_csv(PROCESSED / "test_final.csv")
    return train, test


def construir_features(df, fit_cols=None):
    d = df.copy()
    d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)
    all_cols = STRUCTURAL + OSM + TEXT + ["es_apartamento"]
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
# SECCION 4: TUNING + CROSS-VALIDATION
# =============================================================================

def tuning_espacial(X, y, grupos):
    gkf = GroupKFold(n_splits=CV_FOLDS)
    gs  = GridSearchCV(
        DecisionTreeRegressor(random_state=SEED),
        param_grid=PARAM_GRID,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=False,
        verbose=0,
    )
    gs.fit(X, y, groups=grupos)
    best_params = gs.best_params_
    mae_esp     = -gs.best_score_
    print(f"  Mejores params: {best_params}")
    print(f"  MAE_log espacial (tuning): {mae_esp:.5f}")
    return best_params, mae_esp


def cv_aleatorio(params, X, y):
    kf   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cart = DecisionTreeRegressor(**params, random_state=SEED)
    maes = []
    for tr, va in kf.split(X):
        cart.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[va], cart.predict(X[va])))
    return float(np.mean(maes)), float(np.std(maes))


def cv_espacial(params, X, y, grupos):
    gkf  = GroupKFold(n_splits=CV_FOLDS)
    cart = DecisionTreeRegressor(**params, random_state=SEED)
    maes = []
    for tr, va in gkf.split(X, y, groups=grupos):
        cart.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[va], cart.predict(X[va])))
    return float(np.mean(maes)), float(np.std(maes))


# =============================================================================
# SECCION 5: DIAGNOSTICOS
# =============================================================================

def plot_importancia(cart, feature_cols):
    imp = pd.Series(cart.feature_importances_, index=feature_cols)
    imp = imp[imp > 0].sort_values()
    colores = []
    for f in imp.index:
        if f in OSM:       colores.append("#27ae60")
        elif f in TEXT:    colores.append("#8e44ad")
        else:              colores.append("#e67e22")
    fig, ax = plt.subplots(figsize=(8, max(4, len(imp) * 0.35)))
    ax.barh(list(imp.index), list(imp.values), color=colores, edgecolor="white")
    ax.set_xlabel("Importancia (reduccion media de impureza)")
    ax.set_title(f"Feature Importance — {MODEL_ID}")
    legend = [
        Patch(color="#e67e22", label="Estructural"),
        Patch(color="#27ae60", label="OSM"),
        Patch(color="#8e44ad", label="Texto"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/feature_importance.png")


def plot_arbol(cart, feature_cols):
    fig, ax = plt.subplots(figsize=(16, 6))
    plot_tree(cart, max_depth=3, feature_names=feature_cols,
              filled=True, rounded=True, fontsize=7, ax=ax)
    ax.set_title(f"Arbol (primeros 3 niveles) — {MODEL_ID}")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "arbol_top3.png"), dpi=120)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/arbol_top3.png")


def plot_residuos(y_true, y_pred):
    residuos = y_true - y_pred
    pct_sobre = np.mean(residuos < 0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuos, alpha=0.3, s=5, color="#e67e22")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Prediccion log(price)")
    axes[0].set_ylabel("Residuo")
    axes[0].set_title(f"Residuos vs. Predichos — {MODEL_ID}")
    axes[1].hist(residuos, bins=60, color="#e67e22", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("Residuo")
    axes[1].set_title(
        f"Distribucion de residuos — {MODEL_ID}\n"
        f"media={residuos.mean():.4f}  std={residuos.std():.4f}"
    )
    fig.suptitle(
        f"Sobreprediccion: {pct_sobre:.1f}% de obs (residuo<0 → riesgo Zillow)",
        fontsize=8, color="gray",
    )
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "residuos.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/residuos.png")
    print(f"  Sobreprediccion: {pct_sobre:.1f}% de observaciones")


def plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp):
    fig, ax = plt.subplots(figsize=(6, 4))
    means  = [mae_rand, mae_esp]
    stds   = [std_rand, std_esp]
    bars   = ax.bar(["CV Aleatorio", "CV Espacial"], means, yerr=stds,
                    color=["#e67e22", "#c0392b"], capsize=6,
                    edgecolor="white", width=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.1,
                f"{val:.5f}", ha="center", va="bottom", fontsize=10)
    sesgo = mae_esp - mae_rand
    ax.set_ylabel("MAE log(price)")
    ax.set_title(f"CV Aleatorio vs. Espacial — {MODEL_ID}\nSesgo Delta = {sesgo:+.5f}")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "cv_comparacion.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/cv_comparacion.png")


# =============================================================================
# SECCION 6: SUBMISSION
# =============================================================================

def generar_submission(test, y_pred_log, params):
    sub = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),
    })
    depth_str = "dNone" if params.get("max_depth") is None else f"d{params['max_depth']}"
    sub_name  = (
        f"CART_{depth_str}"
        f"_leaf{params.get('min_samples_leaf', 1)}"
        f"_ccp{str(params.get('ccp_alpha', 0)).replace('.', '')}"
        f"_cv{CV_FOLDS}_{MODEL_ID}.csv"
    )
    sub.to_csv(SUBMISSIONS / sub_name, index=False)
    print(f"  Submission: 03_submissions/{sub_name}  ({len(sub):,} filas)")
    return sub_name


# =============================================================================
# SECCION 7: REGISTRO
# =============================================================================

def registrar(mae_rand, std_rand, mae_esp, std_esp, mae_train,
              n_features, params, sub_name):
    sesgo = mae_esp - mae_rand
    nueva = {
        "model_id":          MODEL_ID,
        "fecha":             str(date.today()),
        "autor":             AUTOR,
        "algoritmo":         "CART",
        "n_features":        n_features,
        "max_depth":         str(params.get("max_depth")),
        "min_samples_leaf":  params.get("min_samples_leaf"),
        "ccp_alpha":         params.get("ccp_alpha"),
        "cv_folds":          CV_FOLDS,
        "cv_mae_log":        round(mae_rand,  5),
        "cv_std_log":        round(std_rand,  5),
        "esp_mae_log":       round(mae_esp,   5),
        "esp_std_log":       round(std_esp,   5),
        "train_mae_log":     round(mae_train, 5),
        "sesgo_delta":       round(sesgo,     5),
        "kaggle_public_MAE": None,
        "features_grupos": (
            f"structural={len(STRUCTURAL)}, osm={len(OSM)}, "
            f"text={len(TEXT)}, es_apartamento"
        ),
        "spatial_grid":    f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "submission_file": sub_name,
        "notas": (
            f"All features: estructurales + OSM + texto. "
            f"GridSearchCV espacial. Params: {params}. "
            f"Sesgo Delta={sesgo:+.5f}."
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
# SECCION 8: MAIN
# =============================================================================

def main():
    print(f"{'='*60}")
    print(f"  CART — {MODEL_ID}  (All Features + Tuning espacial)")
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
    n_str = len([c for c in feature_cols if c in STRUCTURAL + ["es_apartamento"]])
    n_osm = len([c for c in feature_cols if c in OSM])
    n_txt = len([c for c in feature_cols if c in TEXT])
    print(f"  Features: {len(feature_cols)} (estructural={n_str}, OSM={n_osm}, texto={n_txt})")

    print("\n[3/8] Construyendo grupos espaciales...")
    grupos = construir_grupos_espaciales(train)
    print(f"  Cuadricula {SPATIAL_GRID}x{SPATIAL_GRID} -> {len(np.unique(grupos))} bloques")

    print("\n[4/8] Tuning con GridSearchCV espacial...")
    print("  Esto puede tardar varios minutos...")
    best_params, _ = tuning_espacial(X_train, y_train, grupos)

    print(f"\n[5/8] CV aleatorio ({CV_FOLDS}-fold) con mejores params...")
    mae_rand, std_rand = cv_aleatorio(best_params, X_train, y_train)
    print(f"  MAE_log aleatorio = {mae_rand:.5f} +- {std_rand:.5f}")

    print(f"\n[6/8] CV espacial ({CV_FOLDS}-fold) con mejores params...")
    mae_esp, std_esp = cv_espacial(best_params, X_train, y_train, grupos)
    sesgo = mae_esp - mae_rand
    print(f"  MAE_log espacial  = {mae_esp:.5f} +- {std_esp:.5f}")
    print(f"  Sesgo Delta       = {sesgo:+.5f}")

    print("\n[7/8] Modelo final sobre todo el train...")
    cart = DecisionTreeRegressor(**best_params, random_state=SEED)
    cart.fit(X_train, y_train)
    mae_train   = mean_absolute_error(y_train, cart.predict(X_train))
    profundidad = cart.get_depth()
    n_hojas     = cart.get_n_leaves()
    print(f"  MAE_log train    = {mae_train:.5f}")
    print(f"  Profundidad real = {profundidad} | Hojas: {n_hojas}")

    print("\n[8/8] Diagnosticos, submission y registro...")
    plot_importancia(cart, feature_cols)
    plot_arbol(cart, feature_cols)
    plot_residuos(y_train, cart.predict(X_train))
    plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp)
    sub_name = generar_submission(test, cart.predict(X_test), best_params)
    registrar(mae_rand, std_rand, mae_esp, std_esp, mae_train,
              len(feature_cols), best_params, sub_name)

    print(f"\n{'='*60}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"  Params: {best_params}")
    print(f"  MAE_log aleatorio = {mae_rand:.5f}")
    print(f"  MAE_log espacial  = {mae_esp:.5f}")
    print(f"  Sesgo Delta       = {sesgo:+.5f}")
    print(f"  Profundidad real  = {profundidad} | Hojas: {n_hojas}")
    print(f"  Submission: 03_submissions/{sub_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
