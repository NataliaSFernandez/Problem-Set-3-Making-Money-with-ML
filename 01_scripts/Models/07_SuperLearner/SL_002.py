
"""
SuperLearner — SL_002  (Mejores Hiperparámetros del Model Registry)
====================================================================
Problem Set 3 — MECA 4107 (Big Data & ML para Economía Aplicada)

-----------------
  A diferencia de SL_001 (defaults), aquí cada modelo base usa los
  MEJORES hiperparámetros encontrados en las corridas individuales,
  según el score de MAE espacial en el model_registry.xlsx.

    - Cada algoritmo (EN, CART, RF, NN, Boost) ya fue tuneado en scripts
      separados. El model_registry registra qué configuración ganó.
    - SL_002 configura cada base con los mejores hiperparámetros 
      el SuperLearner parte de modelos ya optimizados.

  LPM   → LR_001   (esp=0.31016)
  EN    → EN_001   (esp=0.29860) — alpha=0.00352, l1_ratio=0.7
  CART  → CART_002 (esp=0.22825) — depth=10, leaf=20, ccp=0.0001
  RF    → RF_005   (esp=0.16414) — 500 árboles, depth=15, leaf=5
  NN    → NN_003   (esp=0.13739) — arquitectura (256,256,128,64), PyTorch
                                    aproximado con MLPRegressor sklearn
  Boost → XGB_010  (esp=0.13865) — XGBoost con early stopping

Pipeline  
--------
1.  Carga  train_final.csv / test_final.csv
2.  Ingeniería de features (idéntica a todos los demás scripts)
3.  StandardScaler ajustado SOLO sobre train
4.  Generación de predicciones OOF con CV aleatorio 5-fold
5.  CV del SL: MAE_rand
6.  CV espacial del SL: MAE_esp (proxy Chapinero)
7.  Modelo final: entrena todos los bases + meta-aprendiz
8.  Diagnósticos: pesos, correlación, comparación CV, bases vs SL
9.  Submission → 03_submissions/
10. Registro → 02_outputs/model_registry.xlsx

Outputs
-------
  02_outputs/Models/SuperLearner/SL_002/pesos_meta.png
  02_outputs/Models/SuperLearner/SL_002/correlacion_bases.png
  02_outputs/Models/SuperLearner/SL_002/cv_comparacion.png
  02_outputs/Models/SuperLearner/SL_002/mae_individuales_vs_sl.png
  02_outputs/Models/SuperLearner/SL_002/registry_best_params.png
  03_submissions/submission_SL_002_YYYYMMDD.csv
  02_outputs/model_registry.xlsx
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
import openpyxl          # noqa: F401
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================

AUTOR    = "Equipo"   # ← ajusta con el nombre del autor del commit
MODEL_ID = "SL_002"
SEED     = 42

CV_FOLDS     = 5
SPATIAL_GRID = 5

BASE        = Path(__file__).parent.parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "SuperLearner" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

for d in [SUBMISSIONS, DIR_MODEL, BASE / "02_outputs"]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECCIÓN 2: MEJORES HIPERPARÁMETROS DEL MODEL REGISTRY
# =============================================================================

BEST_PARAMS = {

    # ── LPM (OLS) — LR_001 ────────────────────────────────────────────────

    "LPM": {
        "tipo":            "LinearRegression",
        "params":          {},
        "model_id_origen": "LR_001",
        "esp_mae_log":     0.31016,
    },

    # ── Elastic Net — EN_001 ───────────────────────────────────────────────
    "EN": {
        "tipo": "ElasticNet",
        "params": {
            "alpha":        0.0194,
            "l1_ratio":     1,
            "max_iter":     5000,
            "random_state": SEED,
        },
        "model_id_origen": "EN_002",
        "esp_mae_log":     0.29860,
    },

    "CART": {
        "tipo": "DecisionTreeRegressor",
        "params": {
            "max_depth":        10,
            "min_samples_leaf": 20,
            "ccp_alpha":        0.0001,
            "random_state":     SEED,
        },
        "model_id_origen": "CART_002",
        "esp_mae_log":     0.22825,
    },

    "RF": {
        "tipo": "RandomForestRegressor",
        "params": {
            "n_estimators":     100,
            "max_depth":        20,
            "min_samples_leaf":  1,
            "max_features":     "sqrt",
            "random_state":     SEED,
            "n_jobs":           -1,
        },
        "model_id_origen": "RF_002",
        "esp_mae_log":     0.16414,
    },

    "NN": {
        "tipo": "MLPRegressor",
        "params": {
            "hidden_layer_sizes":  (256, 256, 128, 64),
            "activation":          "relu",
            "alpha":               5e-5,
            "max_iter":            300,
            "early_stopping":      True,
            "validation_fraction": 0.2,
            "n_iter_no_change":    30,
            "random_state":        SEED,
        },
        "model_id_origen": "NN_003",
        "esp_mae_log":     0.13739,
    },

    "Boost": {
        "tipo": "XGBRegressor",
        "params": {
            "n_estimators":      265,
            "max_depth":           6,
            "learning_rate":    0.026,
            "subsample":        0.749,
            "colsample_bytree": 0.89,
            "min_child_weight":    6,
            "gamma":            0.321,
            "reg_lambda":       3.1131,
            "reg_alpha":        0.669,
            "random_state":      SEED,
            "n_jobs":              -1,
            "verbosity":           0,
        },
        "params_fallback": {
            "n_estimators":    265,
            "max_depth":         6,
            "learning_rate":  0.026,
            "subsample":      0.749,
            "min_samples_leaf":  2,
            "random_state":   SEED,
        },
        "model_id_origen": "XGB_009",
        "esp_mae_log":     0.13865,
    },
}


# =============================================================================
# SECCIÓN 3: CONSTRUCCIÓN DE MODELOS BASE CON HIPERPARÁMETROS ÓPTIMOS
# =============================================================================

def construir_modelos_base_optimos(best_params: dict = BEST_PARAMS) -> dict:
    """
    Construye los estimadores sklearn/xgboost con los mejores hiperparámetros
    definidos en BEST_PARAMS.

    Retorna
    -------
    dict {nombre: estimador} listo para usar en el pipeline del SuperLearner.
    """
    modelos = {}

    modelos["LPM"]  = LinearRegression(**best_params["LPM"]["params"])
    modelos["EN"]   = ElasticNet(**best_params["EN"]["params"])
    modelos["CART"] = DecisionTreeRegressor(**best_params["CART"]["params"])
    modelos["RF"]   = RandomForestRegressor(**best_params["RF"]["params"])
    modelos["NN"]   = MLPRegressor(**best_params["NN"]["params"])

    try:
        from xgboost import XGBRegressor
        modelos["Boost"] = XGBRegressor(**best_params["Boost"]["params"])
        print("  Boost → XGBRegressor (xgboost) ✓")
    except ImportError:
        print("  ⚠  xgboost no instalado. Usando GradientBoostingRegressor "
              "(sklearn) como fallback para Boost.")
        modelos["Boost"] = GradientBoostingRegressor(
            **best_params["Boost"]["params_fallback"]
        )

    return modelos


# =============================================================================
# SECCIÓN 4: FEATURES  (idénticas a EN, RF, NN, Boost y SL_001)
# =============================================================================

STRUCTURAL = [
    "surface_total",
    "surface_covered",
    "rooms",
    "bedrooms",
    "bathrooms",
    "month",
    "year",
]

TEXT = [
    "tiene_parqueadero",
    "tiene_piscina",
    "tiene_gimnasio",
    "tiene_vigilancia",
    "tiene_terraza",
    "tiene_deposito",
    "n_palabras_titulo",
    "n_palabras_desc",
]

OSM = [
    "dist_metro",
    "dist_parque",
    "dist_colegio",
    "dist_hospital",
    "dist_supermercado",
    "n_restaurantes_500m",
    "n_bancos_500m",
    "dist_via_principal",
]


# =============================================================================
# SECCIÓN 5: CARGA DE DATOS
# =============================================================================

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga train_final.csv y test_final.csv desde 00_data/processed/.
    Estos archivos ya contienen las variables OSM y de texto construidas
    en Stage 1 del PS.
    """
    train = pd.read_csv(PROCESSED / "train_final.csv")
    test  = pd.read_csv(PROCESSED / "test_final.csv")
    return train, test


# =============================================================================
# SECCIÓN 6: INGENIERÍA DE FEATURES
# =============================================================================

def construir_features(df: pd.DataFrame,
                       fit_cols: list | None = None) -> pd.DataFrame:
    """
    Construye la matriz de features X a partir del DataFrame df.
    Idéntica a SL_001 y al resto de scripts  

    1. Toma las columnas STRUCTURAL, TEXT y OSM que existan en df.
    2. Agrega es_apartamento como dummy.
    3. Crea dummies para month y year (captura estacionalidad no lineal).
    4. Imputa NaN con la mediana de la columna.
    5. Alinea con fit_cols si se proporciona.
    """
    out = pd.DataFrame(index=df.index)

    available_struct = [c for c in STRUCTURAL if c in df.columns]
    available_text   = [c for c in TEXT        if c in df.columns]
    available_osm    = [c for c in OSM         if c in df.columns]

    for col in available_struct + available_text + available_osm:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    if "property_type" in df.columns:
        out["es_apartamento"] = (df["property_type"] == "Apartamento").astype(int)

    if "month" in df.columns:
        month_dummies = pd.get_dummies(df["month"], prefix="mes", drop_first=True)
        out = pd.concat([out, month_dummies], axis=1)
        out.drop(columns=["month"], errors="ignore", inplace=True)

    if "year" in df.columns:
        year_dummies = pd.get_dummies(df["year"], prefix="anio", drop_first=True)
        out = pd.concat([out, year_dummies], axis=1)
        out.drop(columns=["year"], errors="ignore", inplace=True)

    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    if fit_cols is not None:
        for col in fit_cols:
            if col not in out.columns:
                out[col] = 0
        out = out[fit_cols]

    return out


# =============================================================================
# SECCIÓN 7: GRUPOS ESPACIALES (cuadrícula 5×5)
# =============================================================================

def asignar_grupos_espaciales(df: pd.DataFrame,
                               n: int = SPATIAL_GRID) -> np.ndarray:

    lat_col = next((c for c in ["lat", "latitude", "latitud"] if c in df.columns), None)
    lon_col = next((c for c in ["lon", "longitude", "longitud"] if c in df.columns), None)

    if lat_col is None or lon_col is None:
        print("   Sin lat/lon. Usando grupos secuenciales como fallback.")
        return np.arange(len(df)) % (n * n)

    lat = df[lat_col].values
    lon = df[lon_col].values
    lat_bins = np.linspace(lat.min(), lat.max() + 1e-9, n + 1)
    lon_bins = np.linspace(lon.min(), lon.max() + 1e-9, n + 1)
    lat_idx  = np.digitize(lat, lat_bins) - 1
    lon_idx  = np.digitize(lon, lon_bins) - 1
    return lat_idx * n + lon_idx


# =============================================================================
# SECCIÓN 8: OOF Y CV DEL SUPERLEARNER
# =============================================================================

def generar_oof(modelos: dict,
                X: np.ndarray,
                y: np.ndarray,
                cv_folds: int = CV_FOLDS,
                seed: int = SEED) -> np.ndarray:
    """
    Genera la meta-matriz Z de predicciones  con KFold.

    Para cada fold v:
      - Entrena cada modelo base en los folds diferente a v 

    Retorna Z: array de forma (n_train, n_modelos).
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    Z  = np.zeros((len(y), len(modelos)))

    print(f"  Generando OOF ({cv_folds} folds)...")
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X)):
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X[tr_idx])
        X_val_sc = scaler.transform(X[val_idx])
        y_tr     = y[tr_idx]

        for k, (nombre, modelo) in enumerate(modelos.items()):
            m = clone(modelo)
            m.fit(X_tr_sc, y_tr)
            Z[val_idx, k] = m.predict(X_val_sc)

        print(f"    Fold {fold_idx + 1}/{cv_folds} completado")

    return Z


def cv_aleatorio_sl(modelos: dict,
                    X: np.ndarray,
                    y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Estima el MAE del SuperLearner con CV aleatorio (KFold 5-fold).

    Retorna: (mae_medio, std_mae, Z_oof)
    """
    kf   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    Z    = generar_oof(modelos, X, y)
    maes = []

    print(f"\n  CV del meta-aprendiz ({CV_FOLDS} folds)...")
    for tr_idx, val_idx in kf.split(Z):
        meta = LinearRegression(positive=False)
        meta.fit(Z[tr_idx], y[tr_idx])
        maes.append(mean_absolute_error(y[val_idx], meta.predict(Z[val_idx])))

    return float(np.mean(maes)), float(np.std(maes)), Z


def cv_espacial_sl(modelos: dict,
                   X: np.ndarray,
                   y: np.ndarray,
                   grupos: np.ndarray) -> tuple[float, float]:
    """
    Estima el MAE del SuperLearner con CV espacial

    En cada fold externo se excluyen bloques geográficos completos, simulando
    predecir en una zona no vista como Chapinero

    Retorna: (mae_medio, std_mae)
    """
    gkf  = GroupKFold(n_splits=CV_FOLDS)
    maes = []

    print(f"\n  CV espacial del SuperLearner ({CV_FOLDS} folds geográficos)...")
    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=grupos)):
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X[tr_idx])
        X_val_sc = scaler.transform(X[val_idx])
        y_tr, y_val = y[tr_idx], y[val_idx]

        Z_tr  = generar_oof(modelos, X_tr_sc, y_tr,
                             cv_folds=CV_FOLDS, seed=SEED)
        Z_val = np.zeros((len(val_idx), len(modelos)))
        for k, (nombre, modelo) in enumerate(modelos.items()):
            m = clone(modelo)
            m.fit(X_tr_sc, y_tr)
            Z_val[:, k] = m.predict(X_val_sc)

        meta = LinearRegression(positive=False)
        meta.fit(Z_tr, y_tr)
        maes.append(mean_absolute_error(y_val, meta.predict(Z_val)))
        print(f"    Fold espacial {fold_idx + 1}/{CV_FOLDS} — "
              f"MAE_log={maes[-1]:.5f}")

    return float(np.mean(maes)), float(np.std(maes))


def entrenar_modelo_final(modelos: dict,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray) -> tuple:
    """
    Entrena el SuperLearner final sobre todo el train:
      1. Genera OOF con CV 5-fold sobre X_train escalado.
      2. Entrena el meta-aprendiz sobre las OOF.
      3. Re-entrena cada modelo base en todo X_train.
      4. Predice en X_test → Z_test.
      5. Meta-aprendiz predice sobre Z_test.

    Retorna: (meta, Z_train_oof, y_pred_test, bases_finales, scaler)
    """
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    print("\n  Generando OOF para el meta-aprendiz final...")
    Z_train = generar_oof(modelos, X_tr_sc, y_train)

    meta = LinearRegression(positive=False)
    meta.fit(Z_train, y_train)

    print("  Entrenando modelos base finales en todo el train...")
    Z_test = np.zeros((X_te_sc.shape[0], len(modelos)))
    for k, (nombre, modelo) in enumerate(modelos.items()):
        m = clone(modelo)
        m.fit(X_tr_sc, y_train)
        Z_test[:, k] = m.predict(X_te_sc)
        print(f"    {nombre} entrenado")

    return meta, Z_train, meta.predict(Z_test), {}, scaler


# =============================================================================
# SECCIÓN 9: DIAGNÓSTICOS Y VISUALIZACIONES
# =============================================================================

def plot_pesos_meta(meta: LinearRegression,
                    nombres: list,
                    best_params: dict) -> None:
    """
    """
    coefs   = meta.coef_
    origenes = [best_params.get(n, {}).get("model_id_origen", "?")
                for n in nombres]
    labels  = [f"{n}\n({o})" for n, o in zip(nombres, origenes)]
    colores = ["#2ecc71" if c > 0 else "#e74c3c" for c in coefs]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, coefs, color=colores, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coeficiente del meta-aprendiz (OLS)")
    ax.set_title(f"Pesos del SuperLearner — {MODEL_ID}\n"
                 f"(hiperparámetros óptimos del registry; intercepto="
                 f"{meta.intercept_:.4f})")
    for bar, val in zip(bars, coefs):
        ax.text(val + (0.005 if val >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "pesos_meta.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/pesos_meta.png")


def plot_correlacion_bases(Z: np.ndarray, nombres: list) -> None:
    """
    Mapa de calor de la correlación entre las predicciones OOF de los modelos base.

    Alta correlación es qie el SL gana poco al incluir ambos modelos.
    Baja correlación signifca que   el SL se beneficia de la diversidad de errores.
    """
    corr = np.corrcoef(Z.T)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Correlación de Pearson")
    ax.set_xticks(range(len(nombres)))
    ax.set_yticks(range(len(nombres)))
    ax.set_xticklabels(nombres)
    ax.set_yticklabels(nombres)
    ax.set_title(f"Correlación entre predicciones OOF — {MODEL_ID}\n"
                 "(bases con hiperparámetros óptimos del registry)")
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if abs(corr[i, j]) < 0.7 else "white")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "correlacion_bases.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/correlacion_bases.png")


def plot_cv_comparacion(mae_rand: float, std_rand: float,
                         mae_esp: float, std_esp: float) -> None:
    """
    Gráfico de barras comparando MAE del CV vs CV espacial.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    labels  = ["CV\n(5-fold)", "CV Espacial\n(5×5 grid)"]
    values  = [mae_rand, mae_esp]
    errors  = [std_rand, std_esp]
    colors  = ["#3498db", "#e74c3c"]

    bars = ax.bar(labels, values, yerr=errors, color=colors,
                  capsize=6, edgecolor="white")
    ax.set_ylabel("MAE log(price)")
    ax.set_title(f"CV vs CV Espacial — {MODEL_ID}\n"
                 f"Δ = {mae_esp - mae_rand:+.5f}")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "cv_comparacion.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/cv_comparacion.png")


def plot_mae_individuales(Z: np.ndarray,
                           y: np.ndarray,
                           nombres: list,
                           mae_sl: float) -> None:
    """

    """
    maes_base = [mean_absolute_error(y, Z[:, k]) for k in range(len(nombres))]

    fig, ax = plt.subplots(figsize=(9, 4))
    x_pos = np.arange(len(nombres) + 1)
    labels_plot = nombres + ["SuperLearner"]
    values_plot = maes_base + [mae_sl]
    colors_plot = ["#95a5a6"] * len(nombres) + ["#f39c12"]

    bars = ax.bar(x_pos, values_plot, color=colors_plot, edgecolor="white")
    ax.axhline(mae_sl, color="#f39c12", lw=1.5, linestyle="--", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_plot)
    ax.set_ylabel("MAE log(price) — OOF / CV")
    ax.set_title(f"Modelos base óptimos vs SuperLearner — {MODEL_ID}\n"
                 "(naranja = SL; gris = bases con mejores hiperparámetros)")
    for bar, val in zip(bars, values_plot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "mae_individuales_vs_sl.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/mae_individuales_vs_sl.png")


def plot_registry_best_params(best_params: dict) -> None:
    """
    Tabla visual con los hiperparámetros clave de cada modelo base y su
    model_id de origen en el registry
    """
    nombres = list(best_params.keys())

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    filas = []
    for nombre in nombres:
        p       = best_params[nombre]["params"]
        origen  = best_params[nombre].get("model_id_origen", "?")
        mae_esp = best_params[nombre].get("esp_mae_log", "?")
        mae_str = f"{mae_esp:.5f}" if isinstance(mae_esp, float) else str(mae_esp)
        resumen = ", ".join(f"{k}={v}" for k, v in list(p.items())[:4])
        if len(p) > 4:
            resumen += f", ... (+{len(p)-4} más)"
        filas.append([nombre, origen, mae_str, resumen])

    tabla = ax.table(
        cellText=filas,
        colLabels=["Modelo Base", "Registry ID", "MAE CV Espacial", "Hiperparámetros Clave"],
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1.2, 1.6)

    for j in range(4):
        tabla[(0, j)].set_facecolor("#2c3e50")
        tabla[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title(f"Hiperparámetros óptimos por modelo base — {MODEL_ID}\n"
                 "(fuente: model_registry.xlsx, ordenado por MAE CV espacial)",
                 fontsize=10, pad=20)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "registry_best_params.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/registry_best_params.png")


# =============================================================================
# SECCIÓN 10: SUBMISSION
# =============================================================================

def generar_submission(test: pd.DataFrame,
                        y_pred_log: np.ndarray) -> str:
    """
    Genera el CSV de predicciones en formato para Kaggle (property_id, price).
    Revierte la transformación log → COP antes de guardar.
    """
    sub = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),
    })
    fecha    = date.today().strftime("%Y%m%d")
    sub_name = f"submission_{MODEL_ID}_{fecha}.csv"
    sub.to_csv(SUBMISSIONS / sub_name, index=False)
    print(f"  Submission: 03_submissions/{sub_name}  ({len(sub):,} filas)")
    return sub_name


# =============================================================================
# SECCIÓN 11: REGISTRO EN model_registry.xlsx
# =============================================================================

def registrar(mae_rand: float, std_rand: float,
               mae_esp:  float, std_esp:  float,
               mae_train: float,
               n_features: int,
               nombres_bases: list,
               pesos_meta: list,
               origenes: list,
               sub_name: str) -> None:
    """
    Registro del modelo SL_002 en el registry. Incluye los model_id de origen de cada base.
    """
    pesos_str    = " | ".join(f"{n}:{w:.3f}"
                               for n, w in zip(nombres_bases, pesos_meta))
    origenes_str = " | ".join(f"{n}:{o}"
                               for n, o in zip(nombres_bases, origenes))
    nueva = {
        "model_id":          MODEL_ID,
        "autor":             AUTOR,
        "fecha":             date.today().isoformat(),
        "algoritmo":         "SuperLearner",
        "n_features":        n_features,
        "mae_cv_rand_log":   round(mae_rand,  6),
        "std_cv_rand_log":   round(std_rand,  6),
        "mae_cv_esp_log":    round(mae_esp,   6),
        "std_cv_esp_log":    round(std_esp,   6),
        "mae_train_log":     round(mae_train, 6),
        "delta_sesgo":       round(mae_esp - mae_rand, 6),
        "kaggle_public_MAE": None,
        "submission_file":   sub_name,
        "notas": (
            f"SL hiperparámetros óptimos del registry. "
            f"Bases: {origenes_str}. "
            f"Meta: OLS. CV {CV_FOLDS}-fold + CV espacial {SPATIAL_GRID}×{SPATIAL_GRID}. "
            f"Pesos meta: [{pesos_str}]. Δ={mae_esp - mae_rand:+.5f}."
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

    print(f"  Registry actualizado: 02_outputs/model_registry.xlsx "
          f"({len(df_reg)} modelos)")


# =============================================================================
# SECCIÓN 12: MAIN llamar las funciones 
# =============================================================================

def main() -> None:
    print(f"{'='*65}")
    print(f"  SUPERLEARNER — {MODEL_ID}  (mejores hiperparámetros del registry)")
    print(f"{'='*65}")

    # ── [1/8] Carga ───────────────────────────────────────────────────────────
    print("\n[1/8] Cargando datos...")
    train, test = cargar_datos()
    y_train = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    # ── [2/8] Features ────────────────────────────────────────────────────────
    print("\n[2/8] Construyendo features...")
    X_train_df   = construir_features(train)
    feature_cols = list(X_train_df.columns)
    X_test_df    = construir_features(test, fit_cols=feature_cols)
    X_train      = X_train_df.values.astype(float)
    X_test       = X_test_df.values.astype(float)
    print(f"  Features: {len(feature_cols)}")

    # ── [3/8] Grupos espaciales ───────────────────────────────────────────────
    print(f"\n[3/8] Asignando grupos espaciales ({SPATIAL_GRID}×{SPATIAL_GRID})...")
    grupos = asignar_grupos_espaciales(train)
    print(f"  Bloques únicos: {len(np.unique(grupos))}")

    # ── [4/8] Modelos base con mejores hiperparámetros ────────────────────────
    print("\n[4/8] Construyendo modelos base óptimos (BEST_PARAMS)...")
    modelos = construir_modelos_base_optimos(BEST_PARAMS)
    for nombre in modelos.keys():
        origen  = BEST_PARAMS[nombre].get("model_id_origen", "?")
        mae_esp = BEST_PARAMS[nombre].get("esp_mae_log", "?")
        print(f"  {nombre:6s} ← {origen}  (MAE CV espacial individual: {mae_esp})")

    # ── [5/8] CV del SL ───────────────────────────────────────────────────────
    print("\n[5/8] CV del SuperLearner (5-fold)...")
    scaler_global = StandardScaler()
    X_tr_sc = scaler_global.fit_transform(X_train)
    mae_rand, std_rand, Z_oof = cv_aleatorio_sl(modelos, X_tr_sc, y_train)
    print(f"  MAE_log CV: {mae_rand:.5f} ± {std_rand:.5f}")

    # ── [6/8] CV espacial del SL ──────────────────────────────────────────────
    print("\n[6/8] CV espacial del SuperLearner...")
    mae_esp, std_esp = cv_espacial_sl(modelos, X_train, y_train, grupos)
    sesgo = mae_esp - mae_rand
    print(f"  MAE_log CV espacial: {mae_esp:.5f} ± {std_esp:.5f}")
    print(f"  Sesgo Δ = {sesgo:+.5f}")

    # ── [7/8] Modelo final ────────────────────────────────────────────────────
    print("\n[7/8] Entrenando SuperLearner final...")
    meta, Z_train, y_pred_test, _, scaler_final = entrenar_modelo_final(
        modelos, X_train, y_train, X_test
    )
    mae_train = float(mean_absolute_error(y_train, meta.predict(Z_train)))
    print(f"  MAE_log in-sample: {mae_train:.5f}")
    print(f"  Pesos meta: "
          + " | ".join(f"{n}={w:.3f}"
                       for n, w in zip(modelos.keys(), meta.coef_)))

    # ── [8/8] Diagnósticos, submission y registro ─────────────────────────────
    print("\n[8/8] Diagnósticos, submission y registro...")
    origenes = [BEST_PARAMS[n].get("model_id_origen", "?")
                for n in modelos.keys()]

    plot_pesos_meta(meta, list(modelos.keys()), BEST_PARAMS)
    plot_correlacion_bases(Z_oof, list(modelos.keys()))
    plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp)
    plot_mae_individuales(Z_oof, y_train, list(modelos.keys()), mae_rand)
    plot_registry_best_params(BEST_PARAMS)

    sub_name = generar_submission(test, y_pred_test)
    registrar(
        mae_rand=mae_rand, std_rand=std_rand,
        mae_esp=mae_esp,   std_esp=std_esp,
        mae_train=mae_train,
        n_features=len(feature_cols),
        nombres_bases=list(modelos.keys()),
        pesos_meta=list(meta.coef_),
        origenes=origenes,
        sub_name=sub_name,
    )

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"{'='*65}")
    print(f"  Modelos base (óptimos del registry):")
    for nombre in modelos.keys():
        origen  = BEST_PARAMS[nombre].get("model_id_origen", "?")
        mae_ind = BEST_PARAMS[nombre].get("esp_mae_log", "?")
        print(f"    {nombre:6s} ← {origen}  (MAE CV espacial: {mae_ind})")
    print(f"  MAE_log CV          : {mae_rand:.5f} ± {std_rand:.5f}")
    print(f"  MAE_log CV espacial : {mae_esp:.5f} ± {std_esp:.5f}")
    print(f"  Sesgo Δ             : {sesgo:+.5f}")
    print(f"  MAE_log in-sample   : {mae_train:.5f}")
    print(f"  Submission          : 03_submissions/{sub_name}")
    print(f"  Registry            : 02_outputs/model_registry.xlsx")
    print(f"  Gráficos            : 02_outputs/Models/SuperLearner/{MODEL_ID}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()