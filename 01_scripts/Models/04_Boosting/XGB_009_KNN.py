"""
XGBoost - XGB_009 KNN PRECIO VECINOS
======================================
Base: XGB_007 (lat/lon + early stopping espacial)
Feature nueva: precio mediano de los K=30 vecinos mas cercanos.

Justificacion economica (Rosen, 1974):
  El precio de una propiedad depende de su vecindario. La mejor
  medida del vecindario es el precio de las propiedades cercanas.
  Esta feature captura directamente el efecto de ubicacion que
  lat y lon capturan de forma indirecta.

Implementacion sin leakage:
  Train: leave-one-out — para cada propiedad, precio mediano de
  sus K vecinos mas cercanos EXCLUYENDOLA a ella misma. El modelo
  nunca ve su propio precio al calcular esta feature.

  Test: precio mediano de los K vecinos mas cercanos en todo el
  train. El test no tiene precio, no hay riesgo de leakage.

K = 30 fijo:
  No se tunea junto con los hiperparametros para evitar optimismo
  adicional. K=30 es un valor conservador — suficiente para capturar
  el vecindario sin introducir ruido de propiedades lejanas.

Pipeline
--------
1. Carga  train_final.csv / test_final.csv
2. Calcular precio mediano K=30 vecinos (leave-one-out en train)
3. Features: STRUCTURAL + OSM + TEXT + lat + lon + precio_vecinos + es_apartamento
4. RandomizedSearchCV espacial (50 iter)
5. Modelo final con early stopping espacial
6. CV aleatorio 5-fold  (KFold)      -> MAE_rand
7. CV espacial  5-fold  (GroupKFold) -> MAE_esp
8. Diagnosticos: feature importance, curva aprendizaje, residuos, CV
9. Submission -> 03_submissions/
10. Registro  -> 02_outputs/model_registry.xlsx

Estructura de carpetas generada automaticamente al correr el script:
  <repo>/
  ├── 00_data/processed/              <- train_final.csv, test_final.csv
  ├── 01_scripts/Models/04_Boosting/  <- este script vive aqui
  ├── 02_outputs/
  │   ├── Models/04_Boosting/XGB_009/ <- graficos del modelo
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
from sklearn.neighbors import BallTree
import xgboost as xgb
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# SECCION 1: CONFIGURACION
# =============================================================================

MODEL_ID     = "XGB_009"
SEED         = 42
CV_FOLDS     = 5
SPATIAL_GRID = 5
N_ITER       = 50
EARLY_STOP   = 50
MAX_ROUNDS   = 1000
K_VECINOS    = 30   # numero de vecinos para el precio mediano

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

GEO = ["lat", "lon"]


# =============================================================================
# SECCION 3: CARGA Y PREPARACION
# =============================================================================

def cargar_datos():
    train = pd.read_csv(PROCESSED / "train_final.csv")
    test  = pd.read_csv(PROCESSED / "test_final.csv")
    return train, test


def calcular_precio_vecinos(train, test, k=K_VECINOS):
    """
    Calcula el precio mediano de los K vecinos mas cercanos.

    Train (leave-one-out):
      Para cada propiedad i, busca los K+1 vecinos mas cercanos
      (incluyendose a si misma) y toma la mediana de los K restantes.
      Esto garantiza que la propiedad no ve su propio precio.

    Test:
      Para cada propiedad del test, busca los K vecinos mas cercanos
      en el train completo y toma la mediana de sus precios.
      No hay leakage porque el test no tiene precio.

    La distancia se calcula sobre lat/lon usando BallTree con
    metrica haversine (distancia geodesica real en la superficie
    de la tierra) — mas precisa que euclidiana para coordenadas.
    """
    train = train.copy()
    test  = test.copy()

    # Coordenadas en radianes para haversine
    coords_train = np.radians(train[["lat", "lon"]].values)
    coords_test  = np.radians(test[["lat", "lon"]].values)
    log_price    = np.log(train["price"].values)

    # BallTree para busqueda eficiente de vecinos
    tree = BallTree(coords_train, metric="haversine")

    # ── Train: leave-one-out ─────────────────────────────────────────────────
    # Buscar K+1 vecinos (el primero es la propiedad misma, distancia=0)
    _, indices_train = tree.query(coords_train, k=k + 1)

    precio_vecinos_train = np.zeros(len(train))
    for i, vecinos in enumerate(indices_train):
        # Excluir la propiedad misma (primer vecino, distancia 0)
        vecinos_sin_self = vecinos[1:]
        precio_vecinos_train[i] = np.median(log_price[vecinos_sin_self])

    train["precio_vecinos"] = precio_vecinos_train

    # ── Test: vecinos del train completo ────────────────────────────────────
    _, indices_test = tree.query(coords_test, k=k)

    precio_vecinos_test = np.zeros(len(test))
    for i, vecinos in enumerate(indices_test):
        precio_vecinos_test[i] = np.median(log_price[vecinos])

    test["precio_vecinos"] = precio_vecinos_test

    print(f"  precio_vecinos train: media={precio_vecinos_train.mean():.3f}, "
          f"std={precio_vecinos_train.std():.3f}")
    print(f"  precio_vecinos test:  media={precio_vecinos_test.mean():.3f}, "
          f"std={precio_vecinos_test.std():.3f}")

    return train, test


def construir_features(df, fit_cols=None):
    d = df.copy()
    d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)
    all_cols = STRUCTURAL + OSM + TEXT + GEO + ["precio_vecinos", "es_apartamento"]
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
    rs  = RandomizedSearchCV(
        XGBRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, verbosity=0),
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
    xgb_sk = XGBRegressor(
        **params, n_estimators=300,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    maes = []
    for tr, va in kf.split(X):
        xgb_sk.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[va], xgb_sk.predict(X[va])))
    return float(np.mean(maes)), float(np.std(maes))


def cv_espacial(params, X, y, grupos):
    gkf = GroupKFold(n_splits=CV_FOLDS)
    xgb_sk = XGBRegressor(
        **params, n_estimators=300,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    maes = []
    for tr, va in gkf.split(X, y, groups=grupos):
        xgb_sk.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[va], xgb_sk.predict(X[va])))
    return float(np.mean(maes)), float(np.std(maes))


# =============================================================================
# SECCION 5: MODELO FINAL CON EARLY STOPPING ESPACIAL
# =============================================================================

def ajustar_con_early_stopping(X_train, y_train, grupos, params):
    """
    Early stopping usando el bloque espacial mas grande como validacion.
    Consistente con el CV espacial — no usa train_test_split aleatorio.
    """
    bloque_val = np.bincount(grupos).argmax()
    mask_val   = grupos == bloque_val
    mask_tr    = ~mask_val

    X_tr  = X_train[mask_tr]
    y_tr  = y_train[mask_tr]
    X_val = X_train[mask_val]
    y_val = y_train[mask_val]

    print(f"  Bloque espacial de validacion: {bloque_val} "
          f"({mask_val.sum():,} obs = {mask_val.mean()*100:.1f}% del train)")

    dtrain = xgb.DMatrix(X_tr,    label=y_tr)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dfull  = xgb.DMatrix(X_train, label=y_train)

    xgb_params = {
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "max_depth":        params.get("max_depth", 6),
        "learning_rate":    params.get("learning_rate", 0.1),
        "subsample":        params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "min_child_weight": params.get("min_child_weight", 1),
        "gamma":            params.get("gamma", 0),
        "reg_lambda":       params.get("reg_lambda", 1),
        "reg_alpha":        params.get("reg_alpha", 0),
        "seed":             SEED,
        "nthread":          -1,
    }

    evals_result = {}
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        evals_result=evals_result,
        verbose_eval=False,
    )
    best_round = model.best_iteration
    print(f"  Early stopping: mejor ronda = {best_round}")
    print(f"  MAE validacion espacial: {evals_result['val']['mae'][best_round]:.5f}")

    model_final = xgb.train(
        params=xgb_params,
        dtrain=dfull,
        num_boost_round=best_round,
        verbose_eval=False,
    )
    return model_final, best_round, evals_result


# =============================================================================
# SECCION 6: DIAGNOSTICOS
# =============================================================================

def plot_importancia(model, feature_cols):
    scores = model.get_score(importance_type="gain")
    imp = pd.Series(scores)
    if all(k.startswith("f") for k in imp.index):
        mapping = {f"f{i}": feature_cols[i]
                   for i in range(len(feature_cols))
                   if f"f{i}" in imp.index}
        imp.index = [mapping.get(k, k) for k in imp.index]
    imp = imp.sort_values()

    colores = []
    for f in imp.index:
        if f == "precio_vecinos":      colores.append("#e74c3c")
        elif f in GEO:                 colores.append("#f39c12")
        elif f in OSM:                 colores.append("#27ae60")
        elif f in TEXT:                colores.append("#8e44ad")
        else:                          colores.append("#3498db")

    fig, ax = plt.subplots(figsize=(8, max(4, len(imp) * 0.35)))
    ax.barh(list(imp.index), list(imp.values), color=colores, edgecolor="white")
    ax.set_xlabel("Importancia (gain)")
    ax.set_title(f"Feature Importance — {MODEL_ID}")
    legend = [
        Patch(color="#3498db", label="Estructural"),
        Patch(color="#27ae60", label="OSM"),
        Patch(color="#8e44ad", label="Texto"),
        Patch(color="#f39c12", label="Geo (lat/lon)"),
        Patch(color="#e74c3c", label="Precio vecinos K=30"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/feature_importance.png")


def plot_curva_aprendizaje(evals_result):
    mae_tr  = evals_result["train"]["mae"]
    mae_val = evals_result["val"]["mae"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mae_tr,  label="Train",             color="#3498db", lw=1)
    ax.plot(mae_val, label="Validacion espacial", color="#e74c3c", lw=1)
    ax.axvline(len(mae_val) - 1, color="gray", lw=1, linestyle="--",
               label=f"Early stop: {len(mae_val)} rondas")
    ax.set_xlabel("Numero de arboles")
    ax.set_ylabel("MAE (log-precio)")
    ax.set_title(f"Curva de Aprendizaje — {MODEL_ID}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "curva_aprendizaje.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/curva_aprendizaje.png")


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
# SECCION 7: SUBMISSION
# =============================================================================

def generar_submission(test, model, feature_cols):
    X_test = construir_features(test, fit_cols=feature_cols).values.astype(float)
    dtest  = xgb.DMatrix(X_test)
    y_pred_log = model.predict(dtest)
    sub = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),
    })
    sub_name = f"XGB_knn{K_VECINOS}_latlon_earlystop_{MODEL_ID}.csv"
    sub.to_csv(SUBMISSIONS / sub_name, index=False)
    print(f"  Submission: 03_submissions/{sub_name}  ({len(sub):,} filas)")
    return sub_name


# =============================================================================
# SECCION 8: REGISTRO
# =============================================================================

def registrar(mae_rand, std_rand, mae_esp, std_esp, mae_train,
              n_features, params, best_round, sub_name):
    sesgo = mae_esp - mae_rand
    nueva = {
        "model_id":          MODEL_ID,
        "fecha":             str(date.today()),
        "algoritmo":         "XGBoost",
        "n_features":        n_features,
        "n_estimators":      best_round,
        "max_depth":         params.get("max_depth"),
        "learning_rate":     round(params.get("learning_rate", 0), 4),
        "subsample":         round(params.get("subsample", 0), 3),
        "colsample_bytree":  round(params.get("colsample_bytree", 0), 3),
        "min_child_weight":  params.get("min_child_weight"),
        "gamma":             round(params.get("gamma", 0), 4),
        "reg_lambda":        round(params.get("reg_lambda", 0), 4),
        "reg_alpha":         round(params.get("reg_alpha", 0), 4),
        "early_stopping":    EARLY_STOP,
        "k_vecinos":         K_VECINOS,
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
            f"text={len(TEXT)}, geo=2 (lat+lon), "
            f"precio_vecinos=1 (K={K_VECINOS} LOO), es_apartamento"
        ),
        "spatial_grid":    f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "submission_file": sub_name,
        "notas": (
            f"Base XGB_007 + precio mediano K={K_VECINOS} vecinos (haversine, LOO). "
            f"Early stopping espacial bloque mas grande. "
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
# SECCION 9: MAIN
# =============================================================================

def main():
    print(f"{'='*60}")
    print(f"  XGBOOST — {MODEL_ID}  (KNN precio vecinos K={K_VECINOS} + lat/lon)")
    print(f"{'='*60}")

    print("\n[1/9] Cargando datos...")
    train, test = cargar_datos()
    y_train = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    print(f"\n[2/9] Calculando precio mediano K={K_VECINOS} vecinos (leave-one-out)...")
    train, test = calcular_precio_vecinos(train, test, k=K_VECINOS)

    print("\n[3/9] Construyendo features...")
    X_train_df   = construir_features(train)
    feature_cols = list(X_train_df.columns)
    X_train      = X_train_df.values.astype(float)
    n_str = len([c for c in feature_cols if c in STRUCTURAL + ["es_apartamento"]])
    n_osm = len([c for c in feature_cols if c in OSM])
    n_txt = len([c for c in feature_cols if c in TEXT])
    n_geo = len([c for c in feature_cols if c in GEO])
    print(f"  Features: {len(feature_cols)} "
          f"(estructural={n_str}, OSM={n_osm}, texto={n_txt}, "
          f"geo={n_geo}, precio_vecinos=1)")

    print("\n[4/9] Construyendo grupos espaciales...")
    grupos = construir_grupos_espaciales(train)
    print(f"  Cuadricula {SPATIAL_GRID}x{SPATIAL_GRID} -> {len(np.unique(grupos))} bloques")

    print(f"\n[5/9] RandomizedSearchCV espacial ({N_ITER} iter)...")
    best_params, _ = tuning_espacial(X_train, y_train, grupos)

    print(f"\n[6/9] CV aleatorio ({CV_FOLDS}-fold)...")
    mae_rand, std_rand = cv_aleatorio(best_params, X_train, y_train)
    print(f"  MAE_log aleatorio = {mae_rand:.5f} +- {std_rand:.5f}")

    print(f"\n[7/9] CV espacial ({CV_FOLDS}-fold)...")
    mae_esp, std_esp = cv_espacial(best_params, X_train, y_train, grupos)
    sesgo = mae_esp - mae_rand
    print(f"  MAE_log espacial  = {mae_esp:.5f} +- {std_esp:.5f}")
    print(f"  Sesgo Delta       = {sesgo:+.5f}")

    print("\n[8/9] Modelo final con early stopping espacial...")
    model_final, best_round, evals_result = ajustar_con_early_stopping(
        X_train, y_train, grupos, best_params
    )
    dtrain_full  = xgb.DMatrix(X_train)
    y_pred_train = model_final.predict(dtrain_full)
    mae_train    = mean_absolute_error(y_train, y_pred_train)
    print(f"  MAE_log train     = {mae_train:.5f}")

    print("\n[9/9] Diagnosticos, submission y registro...")
    plot_importancia(model_final, feature_cols)
    plot_curva_aprendizaje(evals_result)
    plot_residuos(y_train, y_pred_train)
    plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp)
    sub_name = generar_submission(test, model_final, feature_cols)
    registrar(mae_rand, std_rand, mae_esp, std_esp, mae_train,
              len(feature_cols), best_params, best_round, sub_name)

    print(f"\n{'='*60}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"  Params: {best_params}")
    print(f"  Early stopping: mejor ronda = {best_round}")
    print(f"  MAE_log aleatorio = {mae_rand:.5f}")
    print(f"  MAE_log espacial  = {mae_esp:.5f}")
    print(f"  Sesgo Delta       = {sesgo:+.5f}")
    print(f"  Submission: 03_submissions/{sub_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
