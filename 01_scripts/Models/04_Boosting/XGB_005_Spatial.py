"""
XGBoost - XGB_005 SPATIAL TARGET ENCODING
===========================================
Todas las features + target encoding espacial por bloque geografico.

El target encoding espacial captura el efecto de ubicacion sobre el precio
de forma directa: para cada propiedad, calcula el precio promedio de las
propiedades vecinas en el mismo bloque geografico (cuadricula 5x5 sobre Bogota).

Justificacion economica (Rosen, 1974):
  El precio de una propiedad depende de su vecindario. El target encoding
  espacial resume la informacion de ubicacion que las variables OSM no
  capturan directamente — es el "precio de barrio" implicito en los datos.

Implementacion sin data leakage:
  En train: precio promedio del bloque calculado SIN la propiedad misma
  (leave-one-out encoding). Esto evita que el modelo "vea" el precio
  de la propiedad que esta prediciendo.
  En test: precio promedio del bloque calculado sobre todo el train.

Hiperparametros:
  Se usa RandomizedSearchCV espacial con N_ITER=50 sobre el mismo espacio
  de busqueda que XGB_004, para comparacion justa.

Pipeline
--------
1. Carga  train_final.csv / test_final.csv
2. Construir bloques espaciales (cuadricula 5x5)
3. Target encoding espacial leave-one-out (train) y bloque (test)
4. Features: STRUCTURAL + OSM + TEXT + target_espacial + es_apartamento
5. RandomizedSearchCV espacial (50 iter) -> mejores hiperparametros
6. CV aleatorio 5-fold  (KFold)          -> MAE_rand
7. CV espacial  5-fold  (GroupKFold)     -> MAE_esp
8. Modelo final con mejores hiperparametros
9. Diagnosticos: feature importance, residuos, CV
10. Submission -> 03_submissions/
11. Registro   -> 02_outputs/model_registry.xlsx

Estructura de carpetas generada automaticamente al correr el script:
  <repo>/
  ├── 00_data/processed/              <- train_final.csv, test_final.csv
  ├── 01_scripts/Models/04_Boosting/  <- este script vive aqui
  ├── 02_outputs/
  │   ├── Models/04_Boosting/XGB_005/ <- graficos del modelo
  │   └── model_registry.xlsx
  └── 03_submissions/                 <- CSV para Kaggle
"""

# =============================================================================
# SECCION -1: INSTALACION AUTOMATICA DE DEPENDENCIAS
# =============================================================================

import subprocess
import sys

def instalar(paquete):
    subprocess.check_call([sys.executable, "-m", "pip", "install", paquete, "-q"])

try:
    import xgboost
except ImportError:
    print("  Instalando xgboost...")
    instalar("xgboost")

try:
    import openpyxl
except ImportError:
    print("  Instalando openpyxl...")
    instalar("openpyxl")

try:
    import scipy
except ImportError:
    print("  Instalando scipy...")
    instalar("scipy")


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
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# SECCION 1: CONFIGURACION
# =============================================================================

MODEL_ID     = "XGB_005"
SEED         = 42
CV_FOLDS     = 5
SPATIAL_GRID = 5
N_ITER       = 50

BASE        = Path(__file__).parent.parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "04_Boosting" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

for d in [
    BASE / "02_outputs",
    BASE / "02_outputs" / "Models",
    BASE / "02_outputs" / "Models" / "04_Boosting",
    DIR_MODEL,
    SUBMISSIONS,
]:
    d.mkdir(parents=True, exist_ok=True)

PARAM_DIST = {
    "n_estimators":       randint(200, 700),
    "max_depth":          randint(3, 10),
    "learning_rate":      uniform(0.01, 0.19),
    "subsample":          uniform(0.6, 0.4),
    "colsample_bytree":   uniform(0.5, 0.5),
    "min_child_weight":   randint(1, 10),
    "gamma":              uniform(0, 0.5),
    "reg_lambda":         uniform(0.5, 4.5),
    "reg_alpha":          uniform(0, 1.0),
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


def construir_bloques(df, grid_size=SPATIAL_GRID):
    """Asigna cada propiedad a un bloque de cuadricula grid_size x grid_size."""
    lat   = df["lat"].values
    lon   = df["lon"].values
    lat_n = (lat - lat.min()) / (lat.max() - lat.min() + 1e-9) * grid_size
    lon_n = (lon - lon.min()) / (lon.max() - lon.min() + 1e-9) * grid_size
    fila  = np.floor(lat_n).astype(int).clip(0, grid_size - 1)
    col   = np.floor(lon_n).astype(int).clip(0, grid_size - 1)
    return fila * grid_size + col


def target_encoding_espacial(train, test, y_train_log):
    """
    Calcula el target encoding espacial sin data leakage.

    Para cada propiedad en train: precio promedio del bloque calculado
    SIN esa propiedad (leave-one-out). Esto evita que el modelo vea
    el precio que esta prediciendo.

    Para test: precio promedio del bloque sobre todo el train.
    Si una propiedad de test cae en un bloque sin datos de train,
    se usa la media global como fallback.
    """
    train = train.copy()
    test  = test.copy()

    bloques_train = construir_bloques(train)
    bloques_test  = construir_bloques(test)

    train["bloque"]   = bloques_train
    train["log_price"] = y_train_log

    # Media global como fallback para bloques sin datos
    media_global = float(np.mean(y_train_log))

    # Leave-one-out encoding para train
    suma_bloque  = train.groupby("bloque")["log_price"].transform("sum")
    count_bloque = train.groupby("bloque")["log_price"].transform("count")
    # Restar la propiedad actual del calculo
    loo_mean = (suma_bloque - train["log_price"]) / (count_bloque - 1)
    # Si el bloque tiene una sola propiedad, usar media global
    loo_mean = loo_mean.where(count_bloque > 1, media_global)
    train["target_espacial"] = loo_mean.values

    # Encoding para test: media del bloque en todo el train
    media_por_bloque = train.groupby("bloque")["log_price"].mean()
    test["bloque"] = bloques_test
    test["target_espacial"] = test["bloque"].map(media_por_bloque).fillna(media_global)

    print(f"  Bloques con datos en train: {train['bloque'].nunique()}")
    print(f"  Bloques en test sin datos de train: "
          f"{(~test['bloque'].isin(train['bloque'])).sum()}")

    return train, test


def construir_features(train, test, fit_cols=None):
    """Construye la matriz de features X incluyendo el target encoding espacial."""
    all_cols = STRUCTURAL + OSM + TEXT + ["target_espacial", "es_apartamento"]

    def _build(df):
        d = df.copy()
        d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)
        cols = [c for c in all_cols if c in d.columns]
        return d[cols]

    X_train = _build(train)
    feature_cols = list(X_train.columns)
    X_test = _build(test)
    if fit_cols is not None:
        X_test = X_test.reindex(columns=fit_cols, fill_value=0)
    return X_train, X_test, feature_cols


def construir_grupos_espaciales(df):
    return construir_bloques(df)


# =============================================================================
# SECCION 4: TUNING + CROSS-VALIDATION
# =============================================================================

def tuning_espacial(X, y, grupos):
    gkf = GroupKFold(n_splits=CV_FOLDS)
    rs  = RandomizedSearchCV(
        XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=-1,
        refit=False,
        verbose=0,
    )
    rs.fit(X, y, groups=grupos)
    best_params = rs.best_params_
    mae_esp     = -rs.best_score_
    print(f"  Mejores params: {best_params}")
    print(f"  MAE_log espacial (tuning): {mae_esp:.5f}")
    return best_params, mae_esp


def cv_aleatorio(params, X, y):
    kf  = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    xgb = XGBRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=0)
    maes = []
    for tr, va in kf.split(X):
        xgb.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[va], xgb.predict(X[va])))
    return float(np.mean(maes)), float(np.std(maes))


def cv_espacial(params, X, y, grupos):
    gkf = GroupKFold(n_splits=CV_FOLDS)
    xgb = XGBRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=0)
    maes = []
    for tr, va in gkf.split(X, y, groups=grupos):
        xgb.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[va], xgb.predict(X[va])))
    return float(np.mean(maes)), float(np.std(maes))


# =============================================================================
# SECCION 5: DIAGNOSTICOS
# =============================================================================

def plot_importancia(xgb, feature_cols):
    imp = pd.Series(xgb.feature_importances_, index=feature_cols)
    imp = imp[imp > 0].sort_values()
    colores = []
    for f in imp.index:
        if f == "target_espacial": colores.append("#e74c3c")
        elif f in OSM:             colores.append("#27ae60")
        elif f in TEXT:            colores.append("#8e44ad")
        else:                      colores.append("#3498db")
    fig, ax = plt.subplots(figsize=(8, max(4, len(imp) * 0.35)))
    ax.barh(list(imp.index), list(imp.values), color=colores, edgecolor="white")
    ax.set_xlabel("Importancia (gain)")
    ax.set_title(f"Feature Importance — {MODEL_ID}")
    legend = [
        Patch(color="#3498db", label="Estructural"),
        Patch(color="#27ae60", label="OSM"),
        Patch(color="#8e44ad", label="Texto"),
        Patch(color="#e74c3c", label="Target espacial"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/feature_importance.png")


def plot_residuos(y_true, y_pred):
    residuos = y_true - y_pred
    pct_sobre = np.mean(residuos < 0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuos, alpha=0.3, s=5, color="#3498db")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Prediccion log(price)")
    axes[0].set_ylabel("Residuo")
    axes[0].set_title(f"Residuos vs. Predichos — {MODEL_ID}")
    axes[1].hist(residuos, bins=60, color="#3498db", edgecolor="white", linewidth=0.3)
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
                    color=["#3498db", "#1a5276"], capsize=6,
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
    sub_name = (
        f"XGB_spatial_nest{params.get('n_estimators')}"
        f"_d{params.get('max_depth')}"
        f"_lr{str(round(params.get('learning_rate', 0), 3)).replace('.','')}"
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
        "algoritmo":         "XGBoost",
        "n_features":        n_features,
        "n_estimators":      params.get("n_estimators"),
        "max_depth":         params.get("max_depth"),
        "learning_rate":     round(params.get("learning_rate", 0), 4),
        "subsample":         round(params.get("subsample", 0), 3),
        "colsample_bytree":  round(params.get("colsample_bytree", 0), 3),
        "min_child_weight":  params.get("min_child_weight"),
        "gamma":             round(params.get("gamma", 0), 4),
        "reg_lambda":        round(params.get("reg_lambda", 0), 4),
        "reg_alpha":         round(params.get("reg_alpha", 0), 4),
        "n_iter_search":     N_ITER,
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
            f"text={len(TEXT)}, target_espacial=1, es_apartamento"
        ),
        "spatial_grid":    f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "submission_file": sub_name,
        "notas": (
            f"All features + target encoding espacial leave-one-out. "
            f"RandomizedSearchCV {N_ITER} iter, CV espacial. "
            f"Params: {params}. Sesgo Delta={sesgo:+.5f}."
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
    print(f"  XGBOOST — {MODEL_ID}  (Spatial Target Encoding)")
    print(f"{'='*60}")

    print("\n[1/9] Cargando datos...")
    train, test = cargar_datos()
    y_train = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    print("\n[2/9] Construyendo target encoding espacial (leave-one-out)...")
    train, test = target_encoding_espacial(train, test, y_train)

    print("\n[3/9] Construyendo features...")
    X_train_df, X_test_df, feature_cols = construir_features(train, test)
    X_train = X_train_df.values.astype(float)
    X_test  = X_test_df.values.astype(float)
    n_str = len([c for c in feature_cols if c in STRUCTURAL + ["es_apartamento"]])
    n_osm = len([c for c in feature_cols if c in OSM])
    n_txt = len([c for c in feature_cols if c in TEXT])
    print(f"  Features: {len(feature_cols)} "
          f"(estructural={n_str}, OSM={n_osm}, texto={n_txt}, target_espacial=1)")

    print("\n[4/9] Construyendo grupos espaciales...")
    grupos = construir_grupos_espaciales(train)
    print(f"  Cuadricula {SPATIAL_GRID}x{SPATIAL_GRID} -> {len(np.unique(grupos))} bloques")

    print(f"\n[5/9] RandomizedSearchCV espacial ({N_ITER} iter)...")
    print("  Esto puede tardar 15-30 minutos...")
    best_params, _ = tuning_espacial(X_train, y_train, grupos)

    print(f"\n[6/9] CV aleatorio ({CV_FOLDS}-fold) con mejores params...")
    mae_rand, std_rand = cv_aleatorio(best_params, X_train, y_train)
    print(f"  MAE_log aleatorio = {mae_rand:.5f} +- {std_rand:.5f}")

    print(f"\n[7/9] CV espacial ({CV_FOLDS}-fold) con mejores params...")
    mae_esp, std_esp = cv_espacial(best_params, X_train, y_train, grupos)
    sesgo = mae_esp - mae_rand
    print(f"  MAE_log espacial  = {mae_esp:.5f} +- {std_esp:.5f}")
    print(f"  Sesgo Delta       = {sesgo:+.5f}")

    print("\n[8/9] Modelo final sobre todo el train...")
    xgb = XGBRegressor(**best_params, random_state=SEED, n_jobs=-1, verbosity=0)
    xgb.fit(X_train, y_train)
    mae_train = mean_absolute_error(y_train, xgb.predict(X_train))
    print(f"  MAE_log train     = {mae_train:.5f}")

    print("\n[9/9] Diagnosticos, submission y registro...")
    plot_importancia(xgb, feature_cols)
    plot_residuos(y_train, xgb.predict(X_train))
    plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp)
    sub_name = generar_submission(test, xgb.predict(X_test), best_params)
    registrar(mae_rand, std_rand, mae_esp, std_esp, mae_train,
              len(feature_cols), best_params, sub_name)

    print(f"\n{'='*60}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"  Params: {best_params}")
    print(f"  MAE_log aleatorio = {mae_rand:.5f}")
    print(f"  MAE_log espacial  = {mae_esp:.5f}")
    print(f"  Sesgo Delta       = {sesgo:+.5f}")
    print(f"  Submission: 03_submissions/{sub_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
