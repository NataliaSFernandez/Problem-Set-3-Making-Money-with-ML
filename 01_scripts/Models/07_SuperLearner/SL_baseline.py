#!/usr/bin/env python3
"""
SuperLearner — SL_001  (Baseline con hiperparámetros defaults)
============================================================

  Pasos:
    1. Dividir el train en V folds.
    2. Por cada fold v:
         - Entrenar cada modelo base en los folds diferentes a v
         - Predecir en el fold v → predicciones out-of-fold (OOF).
    3. Las predicciones OOF de todos los modelos forman la matriz Z.
    4. Entrenar el meta-aprendiz con Z como features y y como target.
    5. Para predecir en test:
         - Entrenar cada modelo base en Todo el train.
         - Generar predicciones para test  o sea columnas de Z_test.
         - El meta-aprendiz predice sobre Z_test.

SL_001
  Usa los hiperparámetros más simples para los 6 modelos base:
  Para que corra rapido y sirva como baseline inicial. Luego se pueden ajustar los hiperparámetros para mejorar el rendimiento, 
  pero este modelo ya debería superar al mejor modelo individual (LPM) gracias a la combinación de modelos diversos.
    1. LPM  — Linear Probability / OLS (LinearRegression, sin hiperparámetros)
    2. EN   — Elastic Net (alpha=1.0, l1_ratio=0.5 — punto medio Ridge+LASSO)
    3. CART — Decision Tree (defaults: sin límite de profundidad)
    4. RF   — Random Forest (100 árboles, defaults sklearn)
    5. NN   — Red Neuronal MLP (1 capa oculta 100 neuronas, defaults sklearn)
    6. Boost— Gradient Boosting (100 estimadores, defaults sklearn)

  Meta-aprendiz: OLS sin intercepto (pesos no negativos opcionalmente).
  El OLS como meta-aprendiz es la elección canónica del SuperLearner
  original — aprende los pesos óptimos de cada modelo base.

Pipeline
--------
1.  Carga  train_final.csv / test_final.csv
3.  StandardScaler ajustado SOLO sobre train
4.  Generación de predicciones OOF con CV  5-fold (KFold)
5.  CV  del SL: MAE_rand 
6.  CV espacial del SL: MAE_esp 
7.  Modelo final: entrena todos los bases + meta-aprendiz sobre todo el train
8.  Diagnósticos: pesos del meta-aprendiz, correlación de predicciones base,
    comparación CV  vs espacial
9.  Submission → 03_submissions/
10. Registro → 02_outputs/model_registry.xlsx

Outputs
-------
  02_outputs/Models/SuperLearner/SL_001/pesos_meta.png
  02_outputs/Models/SuperLearner/SL_001/correlacion_bases.png
  02_outputs/Models/SuperLearner/SL_001/cv_comparacion.png
  03_submissions/submission_SL_001_YYYYMMDD.csv
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
import openpyxl          # noqa: F401 — requerido por pd.ExcelWriter
import pandas as pd
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

AUTOR    = "Natalia"   
MODEL_ID = "SL_001"
SEED     = 42

CV_FOLDS     = 5   # folds para CV y para generación de OOF
SPATIAL_GRID = 5   # cuadrícula 5×5 = 25 bloques geográficos


BASE        = Path(__file__).parent.parent.parent.parent  
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "SuperLearner" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

for d in [SUBMISSIONS, DIR_MODEL, BASE / "02_outputs"]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECCIÓN 2: FEATURES 
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
    # Variables extraídas del título y descripción del anuncio
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
    # Variables geoespaciales de OpenStreetMap
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
# SECCIÓN 3: CARGA DE DATOS
# =============================================================================

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga train_final.csv y test_final.csv desde la carpeta de datos procesados.
    """
    train = pd.read_csv(PROCESSED / "train_final.csv")
    test  = pd.read_csv(PROCESSED / "test_final.csv")
    return train, test


# =============================================================================
# SECCIÓN 4: INGENIERÍA DE FEATURES
# =============================================================================

def construir_features(df: pd.DataFrame,
                       fit_cols: list | None = None) -> pd.DataFrame:
    """
    Construye la matriz de features X a partir del DataFrame df

    Parámetros
    ----------
    df       : DataFrame con columnas raw (train o test)
    fit_cols : lista de columnas del train; si se pasa, alinea test al mismo
               orden y rellena con 0 las columnas faltantes en test.

    Proceso
    -------
    1. Toma las columnas STRUCTURAL, TEXT y OSM que existan en df.
    2. Agrega is_apartment como dummy (1 si property_type == 'Apartamento').
    3. Crea dummies para month y year (captura estacionalidad no lineal).
    4. Imputa NaN con la mediana de la columna (columnas numéricas).
    5. Alinea con fit_cols si se proporciona.
    """
    out = pd.DataFrame(index=df.index)

    # Variables que realmente existen en el CSV (las OSM/texto pueden faltar)
    available_struct = [c for c in STRUCTURAL if c in df.columns]
    available_text   = [c for c in TEXT        if c in df.columns]
    available_osm    = [c for c in OSM         if c in df.columns]

    for col in available_struct + available_text + available_osm:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    # Dummy de tipo de propiedad
    if "property_type" in df.columns:
        out["es_apartamento"] = (df["property_type"] == "Apartamento").astype(int)

    # Dummies temporales (capturan ciclos estacionales no lineales)
    if "month" in df.columns:
        month_dummies = pd.get_dummies(df["month"], prefix="mes", drop_first=True)
        out = pd.concat([out, month_dummies], axis=1)
        out.drop(columns=["month"], errors="ignore", inplace=True)

    if "year" in df.columns:
        year_dummies = pd.get_dummies(df["year"], prefix="anio", drop_first=True)
        out = pd.concat([out, year_dummies], axis=1)
        out.drop(columns=["year"], errors="ignore", inplace=True)

    # Imputación de NaN con la mediana de la columna
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    # Alinear con las columnas del train (para no tener columnas extra en test)
    if fit_cols is not None:
        for col in fit_cols:
            if col not in out.columns:
                out[col] = 0
        out = out[fit_cols]

    return out


# =============================================================================
# SECCIÓN 5: DEFINICIÓN DE LOS MODELOS BASE  
# =============================================================================

def get_modelos_base() -> dict:
    """
    Retorna un diccionario {nombre: estimador} con los 6 modelos base.

    LPM  — LinearRegression: OLS clásico

    EN   — ElasticNet(alpha=1.0, l1_ratio=0.5): penalización intermedia
           alpha=1.0 es el
           default de sklearn.

    CART — DecisionTreeRegressor(random_state=SEED): árbol sin podar.
           Con defaults crece hasta hoja pura, muy propenso a sobreajuste.
           Es el caso extremo de la familia de árboles.

    RF   — RandomForestRegressor(n_estimators=100, random_state=SEED):
           100 árboles es el default de sklearn. sqrt features por split.

    NN   — MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, ...):
           Una capa oculta de 100 neuronas, ReLU (default sklearn).
           max_iter=500 para asegurar convergencia con datos grandes.

    Boost— GradientBoostingRegressor(n_estimators=100, random_state=SEED):
           100 estimadores, learning_rate=0.1, max_depth=3 (todos defaults).
           El boosting canónico de Friedman (2001).

    Nota: todos los modelos trabajan en log(price). El escalado de X
    se hace fuera, en el pipeline de CV, para evitar data leakage.
    """
    return {
        "LPM":   LinearRegression(),
        "EN":    ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000,
                            random_state=SEED),
        "CART":  DecisionTreeRegressor(random_state=SEED),
        "RF":    RandomForestRegressor(n_estimators=100, random_state=SEED,
                                       n_jobs=-1),
        "NN":    MLPRegressor(hidden_layer_sizes=(100,), activation="relu",
                              max_iter=500, random_state=SEED,
                              early_stopping=True, validation_fraction=0.1),
        "Boost": GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    }


# =============================================================================
# SECCIÓN 6: GRUPOS ESPACIALES (cuadrícula 5×5)
# =============================================================================

def asignar_grupos_espaciales(df: pd.DataFrame,
                               n: int = SPATIAL_GRID) -> np.ndarray:
    """
    Asigna cada propiedad a uno de los n² bloques de una cuadrícula geográfica.

    Se divide el rango [lat_min, lat_max] en n partes iguales y el rango
    [lon_min, lon_max] en n partes iguales. Cada propiedad pertenece al
    bloque (i, j) según su latitud y longitud.

    Con n=5 se generan 25 bloques → GroupKFold con 5 folds agrupa ~5 bloques
    por fold. Esto garantiza que en cada fold se excluyen zonas geográficas
    completas del entrenamiento, simulando la extrapolación a Chapinero.
    """

    lat_col = next((c for c in ["lat"] if c in df.columns), None)
    lon_col = next((c for c in ["lon"] if c in df.columns), None)


    lat = df[lat_col].values
    lon = df[lon_col].values

    lat_bins = np.linspace(lat.min(), lat.max() + 1e-9, n + 1)
    lon_bins = np.linspace(lon.min(), lon.max() + 1e-9, n + 1)

    lat_idx = np.digitize(lat, lat_bins) - 1
    lon_idx = np.digitize(lon, lon_bins) - 1

    return lat_idx * n + lon_idx


# =============================================================================
# SECCIÓN 7: GENERACIÓN DE PREDICCIONES OOF  
# =============================================================================

def generar_oof(modelos: dict,
                X: np.ndarray,
                y: np.ndarray,
                cv_folds: int = CV_FOLDS,
                seed: int = SEED) -> np.ndarray:
    """
    Genera la meta-matriz Z de predicciones out-of-fold (OOF) usando KFold.

    Para cada fold v (v=1..CV_FOLDS):
      - Entrena cada modelo en los folds diferente a v 
      - Predice en el fold v → columna correspondiente de Z.

    Retorna Z: array de forma (n_train, n_modelos).
    Cada columna Z[:,k] son las predicciones OOF del modelo k.

    El scaling se hace DENTRO del loop para evitar data leakage:
    el StandardScaler se ajusta solo sobre el fold de entrenamiento.

    Modelos NN y EN requieren datos escalados. LPM también se beneficia.
    RF, CART y Boost son invariantes a la escala, pero escalarlos no los
    perjudica — se escalan igual para consistencia del pipeline.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    nombres = list(modelos.keys())
    Z = np.zeros((len(y), len(nombres)))

    print(f"  Generando predicciones OOF ({cv_folds} folds)...")
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr        = y[train_idx]

        # Scaling dentro del fold: evita filtrar información del val al train
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        for k, (nombre, modelo) in enumerate(modelos.items()):
            # Clonar el modelo para no reusar pesos entre folds
            from sklearn.base import clone
            m = clone(modelo)
            m.fit(X_tr_sc, y_tr)
            Z[val_idx, k] = m.predict(X_val_sc)

        print(f"    Fold {fold_idx + 1}/{cv_folds} completado")

    return Z


# =============================================================================
# SECCIÓN 8: CV DEL SUPERLEARNER
# =============================================================================

def cv_aleatorio_sl(modelos: dict,
                    X: np.ndarray,
                    y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Estima el MAE del SuperLearner con CV (KFold).

    Procedimiento anidado:
      - Outer loop: CV_FOLDS folds (evalúa el SL completo).
      - Inner loop: dentro de cada fold outer, genera OOF con CV_FOLDS-1 folds
        para entrenar el meta-aprendiz.

    Simplificación práctica: usamos las OOF globales ) y
    entrenamos el meta-aprendiz en cada fold outer con los índices del train.
    Esto evita el CV doblemente anidado y es el enfoque estándar en la práctica.

    Retorna: (mae_medio, std_mae, Z_oof)
    """
    from sklearn.base import clone

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    nombres = list(modelos.keys())
    maes = []

    # Generamos OOF una sola vez (eficiencia)
    Z = generar_oof(modelos, X, y)

    print(f"\n  CV  del meta-aprendiz ({CV_FOLDS} folds)...")
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(Z)):
        Z_tr, Z_val = Z[tr_idx], Z[val_idx]
        y_tr, y_val = y[tr_idx],  y[val_idx]

        meta = LinearRegression(positive=False)  # OLS como meta-aprendiz
        meta.fit(Z_tr, y_tr)
        pred = meta.predict(Z_val)
        maes.append(mean_absolute_error(y_val, pred))

    return float(np.mean(maes)), float(np.std(maes)), Z


# =============================================================================
# SECCIÓN 9: CV ESPACIAL DEL SUPERLEARNER
# =============================================================================

def cv_espacial_sl(modelos: dict,
                   X: np.ndarray,
                   y: np.ndarray,
                   grupos: np.ndarray) -> tuple[float, float]:
    """
    Estima el MAE del SuperLearner con CV espacial (GroupKFold).

    A diferencia del CV, en cada fold outer se excluyen bloques
    geográficos completos.

    Para la generación de OOF dentro de cada fold outer, se usa KFold
    sobre el sub-train (sin los bloques del val), para no contaminar.

    Retorna: (mae_medio, std_mae)
    """
    from sklearn.base import clone

    gkf  = GroupKFold(n_splits=CV_FOLDS)
    maes = []

    print(f"\n  CV espacial del SuperLearner ({CV_FOLDS} folds geográficos)...")
    for fold_idx, (tr_idx, val_idx) in enumerate(
            gkf.split(X, y, groups=grupos)):

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        grupos_tr   = grupos[tr_idx]

        # Scaling del fold exterior: ajustado solo sobre sub-train
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        # Generar OOF del sub-train para entrenar el meta-aprendiz
        Z_tr  = generar_oof(modelos, X_tr_sc, y_tr,
                             cv_folds=CV_FOLDS, seed=SEED)

        # Entrenar bases en todo el sub-train para predecir en val
        Z_val = np.zeros((len(val_idx), len(modelos)))
        for k, (nombre, modelo) in enumerate(modelos.items()):
            m = clone(modelo)
            m.fit(X_tr_sc, y_tr)
            Z_val[:, k] = m.predict(X_val_sc)

        # Meta-aprendiz sobre OOF del sub-train
        meta = LinearRegression(positive=False)
        meta.fit(Z_tr, y_tr)
        pred = meta.predict(Z_val)
        maes.append(mean_absolute_error(y_val, pred))

        print(f"    Fold espacial {fold_idx + 1}/{CV_FOLDS} — "
              f"MAE_log={maes[-1]:.5f}")

    return float(np.mean(maes)), float(np.std(maes))


# =============================================================================
# SECCIÓN 10: MODELO FINAL
# =============================================================================

def entrenar_modelo_final(modelos: dict,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray) -> tuple:
    """
    Entrena el SuperLearner final sobre todo el train:
      1. Genera OOF con CV 5-fold sobre X_train.
      2. Entrena el meta-aprendiz sobre las OOF.
      3. Re-entrena cada modelo base en todo X_train.
      4. Predice en X_test con los bases → Z_test.
      5. Meta-aprendiz predice sobre Z_test.

    Retorna: (meta_aprendiz, Z_train_oof, y_pred_test, bases_entrenados, scaler)
    """
    from sklearn.base import clone

    # Escalar con TODOS los datos de train (modelo de producción)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    # OOF para meta-aprendiz
    print("\n  Generando OOF para el meta-aprendiz final...")
    Z_train = generar_oof(modelos, X_tr_sc, y_train)

    # Meta-aprendiz
    meta = LinearRegression(positive=False)
    meta.fit(Z_train, y_train)

    # Bases finales (entrenados en TODO el train escalado)
    print("  Entrenando modelos base finales en todo el train...")
    bases_finales = {}
    Z_test = np.zeros((X_te_sc.shape[0], len(modelos)))
    for k, (nombre, modelo) in enumerate(modelos.items()):
        m = clone(modelo)
        m.fit(X_tr_sc, y_train)
        Z_test[:, k] = m.predict(X_te_sc)
        bases_finales[nombre] = m
        print(f"    {nombre} entrenado")

    y_pred_test = meta.predict(Z_test)
    return meta, Z_train, y_pred_test, bases_finales, scaler


# =============================================================================
# SECCIÓN 11: DIAGNÓSTICOS Y VISUALIZACIONES
# =============================================================================

def plot_pesos_meta(meta: LinearRegression,
                    nombres: list) -> None:
    """
    Gráfico de barras horizontales con los coeficientes del meta-aprendiz.

    Un coeficiente grande indica que el meta-aprendiz confía mucho en ese
    modelo base. Coeficientes negativos son posibles en OLS sin restricción
    y se interpretan como "corrección a la baja" de modelos que sesgan.

    En el SuperLearner, los pesos son no negativos
    y suman 1 . Aquí usamos OLS sin restricción porque
    es más flexible y suele funcionar igual de bien en la práctica.
    """
    coefs = meta.coef_
    colores = ["#2ecc71" if c > 0 else "#e74c3c" for c in coefs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(nombres, coefs, color=colores, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coeficiente del meta-aprendiz (OLS)")
    ax.set_title(f"Pesos del SuperLearner — {MODEL_ID}\n"
                 f"(Intercepto = {meta.intercept_:.4f})")
    # Anotar valor exacto en cada barra
    for bar, val in zip(bars, coefs):
        ax.text(val + (0.005 if val >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "pesos_meta.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/pesos_meta.png")


def plot_correlacion_bases(Z: np.ndarray,
                            nombres: list) -> None:
    """
    Mapa de calor de la correlación entre las predicciones OOF de los modelos base.

    Alta correlación entre dos modelos → el SL gana poco al incluir ambos
    (la diversidad del ensamble disminuye).
    Baja correlación → el SL se beneficia al combinarlos (errores complementarios).

    La diversidad de predicciones es la razón por la que el SL supera al
    mejor modelo individual: si los errores están correlacionados, el promedio
    no reduce la varianza.
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
                 "(baja correlación = más diversidad = mejor ensamble)")
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
                         mae_esp: float,  std_esp: float) -> None:
    """
    Gráfico de barras comparando MAE del CV vs CV espacial.

    La brecha (Δ = MAE_esp − MAE_rand):
    cuánto sobreestima el CV  el rendimiento real en Chapinero.
    Una Δ pequeña indica que el SL generaliza bien geográficamente.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["CV \n", "CV Espacial\n"]
    values = [mae_rand, mae_esp]
    errors = [std_rand, std_esp]
    colors = ["#3498db", "#e74c3c"]

    bars = ax.bar(labels, values, yerr=errors, color=colors,
                  capsize=6, edgecolor="white")
    ax.set_ylabel("MAE log(price)")
    ax.set_title(f"CV  vs CV Espacial — {MODEL_ID}\n"
                 f"Δ = {mae_esp - mae_rand:+.5f}  "
                 f"({'↑ SL sobreestima en CV' if mae_esp > mae_rand else '↓ CV espacial igual o mejor'})")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "cv_comparacion.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/cv_comparacion.png")


def plot_mae_individuales(modelos: dict,
                           X: np.ndarray,
                           y: np.ndarray,
                           Z: np.ndarray,
                           mae_sl: float) -> None:
    """
    Compara el MAE de cada modelo base (evaluado sobre sus predicciones OOF)
    vs el MAE del SuperLearner ensamblado.

    Permite visualizar si el SL supera al mejor modelo individual —
    que es la principal justificación para usarlo.
    """
    from sklearn.base import clone

    nombres = list(modelos.keys())
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
    ax.set_title(f"Modelos base vs SuperLearner — {MODEL_ID}\n"
                 "(naranja = SL; gris = bases individuales)")
    for bar, val in zip(bars, values_plot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "mae_individuales_vs_sl.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/mae_individuales_vs_sl.png")


# =============================================================================
# SECCIÓN 12: SUBMISSION
# =============================================================================

def generar_submission(test: pd.DataFrame,
                        y_pred_log: np.ndarray) -> str:
    """
    Genera el CSV de predicciones en formato Kaggle (property_id, price).
    Revierte la transformación log → COP antes de guardar.
    """
    sub = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),
    })
    fecha    = date.today().strftime("%Y%m%d")
    sub_name = f"{MODEL_ID}_baseline.csv"
    sub.to_csv(SUBMISSIONS / sub_name, index=False)
    print(f"  Submission: 03_submissions/{sub_name}  ({len(sub):,} filas)")
    return sub_name


# =============================================================================
# SECCIÓN 13: REGISTRO EN model_registry.xlsx
# =============================================================================

def registrar(mae_rand: float, std_rand: float,
               mae_esp:  float, std_esp:  float,
               mae_train: float,
               n_features: int,
               nombres_bases: list,
               pesos_meta: list,
               sub_name: str) -> None:
    """
    Agrega o actualiza la fila de MODEL_ID en el registro central.
    Si MODEL_ID ya existe, reemplaza la fila (no duplica).
    """
    pesos_str = " | ".join(f"{n}:{w:.3f}"
                            for n, w in zip(nombres_bases, pesos_meta))
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
        "kaggle_public_MAE": None,   # llenar manualmente después de subir
        "submission_file":   sub_name,
        "notas": (
            f"SL defaults. Bases: LPM, EN(α=1,l1=0.5), CART, "
            f"RF(100), NN(100), Boost(100). "
            f"Meta: OLS. CV {CV_FOLDS}-fold + espacial {SPATIAL_GRID}×{SPATIAL_GRID}. "
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
# SECCIÓN 14: MAIN
# =============================================================================

def main() -> None:
    print(f"{'='*65}")
    print(f"  SUPERLEARNER — {MODEL_ID}  (hiperparámetros defaults)")
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
    n_grupos = len(np.unique(grupos))
    print(f"  Bloques únicos: {n_grupos}")

    # ── [4/8] Modelos base ────────────────────────────────────────────────────
    print("\n[4/8] Definiendo modelos base (defaults)...")
    modelos = get_modelos_base()
    print(f"  Bases: {list(modelos.keys())}")

    # ── [5/8] CV del SL ─────────────────────────────────────────────
    print("\n[5/8] CV  del SuperLearner...")
    # Escalamos X una sola vez fuera del OOF para el CV 
    scaler_global = StandardScaler()
    X_tr_sc = scaler_global.fit_transform(X_train)

    mae_rand, std_rand, Z_oof = cv_aleatorio_sl(modelos, X_tr_sc, y_train)
    print(f"  MAE_log CV : {mae_rand:.5f} ± {std_rand:.5f}  ")

    # ── [6/8] CV espacial del SL ──────────────────────────────────────────────
    print("\n[6/8] CV espacial del SuperLearner...")
    mae_esp, std_esp = cv_espacial_sl(modelos, X_train, y_train, grupos)
    sesgo = mae_esp - mae_rand
    print(f"  MAE_log CV espacial:  {mae_esp:.5f} ± {std_esp:.5f}  ")
    print(f"  Sesgo Δ = {sesgo:+.5f}  "
          f"({'SL sobreestimado en CV ' if sesgo > 0 else 'sin sesgo o subestimado'})")

    # ── [7/8] Modelo final ────────────────────────────────────────────────────
    print("\n[7/8] Entrenando SuperLearner final (todos los datos)...")
    meta, Z_train, y_pred_test, bases_finales, scaler_final = entrenar_modelo_final(
        modelos, X_train, y_train, X_test
    )
    y_pred_train = meta.predict(Z_train)
    mae_train    = float(mean_absolute_error(y_train, y_pred_train))
    print(f"  MAE_log train (in-sample): {mae_train:.5f}")
    print(f"  Pesos meta-aprendiz: "
          + " | ".join(f"{n}={w:.3f}"
                       for n, w in zip(modelos.keys(), meta.coef_)))

    # ── [8/8] Diagnósticos, submission y registro ─────────────────────────────
    print("\n[8/8] Diagnósticos, submission y registro...")
    plot_pesos_meta(meta, list(modelos.keys()))
    plot_correlacion_bases(Z_oof, list(modelos.keys()))
    plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp)
    plot_mae_individuales(modelos, X_tr_sc, y_train, Z_oof, mae_rand)

    sub_name = generar_submission(test, y_pred_test)
    registrar(
        mae_rand=mae_rand, std_rand=std_rand,
        mae_esp=mae_esp,   std_esp=std_esp,
        mae_train=mae_train,
        n_features=len(feature_cols),
        nombres_bases=list(modelos.keys()),
        pesos_meta=list(meta.coef_),
        sub_name=sub_name,
    )

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"{'='*65}")
    print(f"  Bases: {list(modelos.keys())}")
    print(f"  MAE_log CV  : {mae_rand:.5f} ± {std_rand:.5f} ")
    print(f"  MAE_log CV espacial  : {mae_esp:.5f} ± {std_esp:.5f} ")
    print(f"  Sesgo Δ              : {sesgo:+.5f}")
    print(f"  MAE_log in-sample    : {mae_train:.5f}")
    print(f"  Submission           : 03_submissions/{sub_name}")
    print(f"  Registry             : 02_outputs/model_registry.xlsx")
    print(f"  Gráficos             : 02_outputs/Models/SuperLearner/{MODEL_ID}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()