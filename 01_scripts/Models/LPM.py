"""
Regresión Lineal (OLS) para predicción de log(price) de vivienda en Bogotá
===========================================================================
Problem Set 3 — MECA 4107 (Big Data & ML para Economía Aplicada)
 
Modelo
------
Regresión Lineal por Mínimos Cuadrados Ordinarios (OLS):
 
  min  (1/2n) ||y - Xβ||²
 
Por qué predecir log(price) en lugar de price
----------------------------------------------
  Los precios tienen distribución sesgada a la derecha (cola larga de
  propiedades muy caras). Al usar log(price):
    1. La distribución se vuelve más simétrica y el MAE en log-escala
       penaliza errores relativos en vez de absolutos.
    2. Los coeficientes se interpretan como semi-elasticidades:
       β_k ≈ cambio porcentual en precio ante un cambio unitario en x_k.
 
Pipeline
--------
1. Carga  train_final.csv / test_final.csv
2. Ingeniería de features: dummies + interacciones hedónicas
3. StandardScaler ajustado SOLO sobre train (evita data leakage)
4. CV aleatorio 5-fold  (KFold)                   → MAE_random
5. CV espacial leave-one-block-out (GroupKFold)    → MAE_spatial
6. Comparación  Δ = MAE_spatial − MAE_random       (sesgo de optimismo)
7. Modelo final entrenado sobre TODO el train
8. Diagnósticos: coeficientes, residuos, comparación CV
9. Submission → 03_submissions/  (formato exacto del template Kaggle)
10. Registro automático → 02_outputs/model_registry.xlsx
 
Outputs
-------
  02_outputs/Models/LinearRegression/LR_001/coeficientes.png
  02_outputs/Models/LinearRegression/LR_001/residuos.png
  02_outputs/Models/LinearRegression/LR_001/cv_comparacion.png
  03_submissions/submission_LR_001_<fecha>.csv
  02_outputs/model_registry.xlsx
"""

# =============================================================================
# SECCIÓN 0: IMPORTACIONES
# =============================================================================
 
import warnings
from datetime import date
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")   # backend sin pantalla: necesario para scripts/servidores
import matplotlib.pyplot as plt
import numpy as np
import openpyxl          # noqa: F401 — requerido internamente por pd.ExcelWriter
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================

AUTOR    = "Natalia"   
MODEL_ID = "LR_001"
SEED     = 42         
 
# Cross-validation
CV_FOLDS     = 5     
SPATIAL_GRID = 5      # divide Bogotá en una cuadrícula 5×5 = 25 bloques espaciales
 
# Rutas relativas a la raíz del proyecto (3 niveles arriba de este script)
BASE        = Path(__file__).parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "LinearRegression" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

# =============================================================================
# SECCIÓN 2: DEFINICIÓN DE GRUPOS DE FEATURES
# =============================================================================
 
STRUCTURAL = [
    # Atributos físicos del inmueble — las variables "clásicas" del modelo hedónico
    "surface_total",    # superficie total en m²
    "surface_covered",  # superficie cubierta (sin terrazas/jardines)
    "rooms",            # número de habitaciones
    "bedrooms",         # número de alcobas
    "bathrooms",        # número de baños
    "month",            # mes de publicación (captura estacionalidad)
    "year",             # año de publicación (captura tendencia temporal)
]
 
TEXT = [
    # Variables extraídas de la  descripción de los anuncios.
    "remodelado",       # 1 si el texto menciona remodelación
    "vista_panoramica", # 1 si menciona vista panorámica
    "deposito",         # 1 si menciona depósito/bodega
    "conjunto_cerrado", # 1 si menciona conjunto cerrado o seguridad
    "balcon_terraza",   # 1 si menciona balcón o terraza
    "tfidf_premium",    # score TF-IDF de términos asociados a propiedades de lujo
    "parqueaderos_txt", # número de parqueaderos mencionados en el texto
    "piso_txt",         # número de piso extraído del texto
    "gimnasio",         # 1 si menciona gimnasio o zona fitness
    "amenidades",       # score agregado de amenidades mencionadas
    "num_amenidades",   # conteo de amenidades distintas mencionadas
]
 
OSM = [
    # Variables construidas con OpenStreetMap — capturan la localización.
    "dist_cbd_km",          # distancia al CBD (centro financiero) en km
    "dist_transmilenio_m",  # distancia a la estación de TransMilenio más cercana
    "dist_via_arterial_m",  # distancia a vía arterial principal
    "dist_hospital_m",      # distancia al hospital más cercano
    "dist_centro_com_m",    # distancia al centro comercial más cercano
    "dist_parque_m",        # distancia al parque más cercano
    "n_restaurantes_500m",  # número de restaurantes en radio de 500 m
    "n_bancos_500m",        # número de bancos en radio de 500 m
    "walkability_score",    # índice de caminabilidad (densidad peatonal)
    "densidad_vial",        # metros de vía por km² en el entorno del inmueble
]
 
# =============================================================================
# SECCIÓN 3: CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los datasets procesados con features OSM y texto ya integradas."""
    train = pd.read_csv(PROCESSED / "train_final.csv")
    test  = pd.read_csv(PROCESSED / "test_final.csv")
    return train, test
 
 
def construir_features(df: pd.DataFrame,
                       fit_cols: list | None = None) -> pd.DataFrame:
    """
    Construye la matriz de features X 
    Pasos:
      1. Codifica property_type como variable binaria.
         es_apartamento = 1 si Apartamento, 0 si Casa.
      2. Selecciona las variables de los tres grupos (STRUCTURAL, TEXT, OSM)
         más la variable es_apartamento.
      3. Si se pasa fit_cols, cuadra el orden de columnas de test con train.
         Esto evita errores si test tiene columnas en distinto orden que train.
    """
    d = df.copy()
    d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)
 
    all_cols = STRUCTURAL + TEXT + OSM + ["es_apartamento"]
    all_cols = [c for c in all_cols if c in d.columns]
 
    X = d[all_cols]
    if fit_cols is not None:
        X = X[fit_cols]
    return X

# =============================================================================
# SECCIÓN 4: GRUPOS ESPACIALES
# =============================================================================
 
def construir_grupos_espaciales(df: pd.DataFrame, grid_size: int = SPATIAL_GRID) -> np.ndarray:
    """
    Asigna cada propiedad a un bloque de una cuadrícula geográfica grid_size × grid_size.
 
    Lógica:
      1. Normaliza latitud y longitud al rango [0, grid_size).
      2. El bloque = fila_grid * grid_size + columna_grid.
      Propiedades vecinas caen en el mismo bloque → GroupKFold las mantiene
        siempre juntas en el mismo fold, simulando la extrapolación a Chapinero.
    """
    lat = df["lat"].values
    lon = df["lon"].values
 
    # Normalizar coordenadas al rango [0, grid_size)
    lat_norm = (lat - lat.min()) / (lat.max() - lat.min() + 1e-9) * grid_size
    lon_norm = (lon - lon.min()) / (lon.max() - lon.min() + 1e-9) * grid_size
 
    # Convertir a índice de bloque entero
    fila = np.floor(lat_norm).astype(int).clip(0, grid_size - 1)
    col  = np.floor(lon_norm).astype(int).clip(0, grid_size - 1)
 
    grupos = fila * grid_size + col
    return grupos
 
# =============================================================================
# SECCIÓN 5: CROSS-VALIDATION
# =============================================================================
 
def cv_aleatorio(model: LinearRegression,
                 X: np.ndarray,
                 y: np.ndarray) -> tuple[float, float]:
    """
    CV aleatorio con KFold estándar.
 
    Retorna: (mae_log_medio, std)
      mae_log: MAE en log-escala, por la transformación log(price).
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
 
    maes = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
 
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, pred))
 
    return float(np.mean(maes)), float(np.std(maes))
 
 
def cv_espacial(model: LinearRegression,
                X: np.ndarray,
                y: np.ndarray,
                grupos: np.ndarray) -> tuple[float, float]:
    """
    CV espacial con GroupKFold.
 
    Cada fold excluye un grupo de bloques geográficos completos del
    entrenamiento.

    Retorna: (mae_log_medio, std)
    """
    gkf = GroupKFold(n_splits=CV_FOLDS)
 
    maes = []
    for train_idx, val_idx in gkf.split(X, y, groups=grupos):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
 
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, pred))
 
    return float(np.mean(maes)), float(np.std(maes))
 
# =============================================================================
# SECCIÓN 6: DIAGNÓSTICOS Y VISUALIZACIONES
# =============================================================================
 
def plot_coeficientes(model: LinearRegression,
                      feature_names: list,
                      scaler: StandardScaler) -> None:
    """
    Gráfica de barras horizontales con los coeficientes estandarizados.
 
    Los coeficientes están en escala estandarizada (X fue escalado con
    StandardScaler antes del ajuste),  son comparables entre sí:
    un coeficiente más grande en valor absoluto indica mayor influencia
    sobre log(price) por desviación estándar de la feature.
 
    Verde = efecto positivo sobre el precio. Rojo = efecto negativo.
    """
    coefs = pd.Series(model.coef_, index=feature_names).sort_values()
 
    # Mostrar los n/2 más negativos y los n/2 más positivos
    n   = min(30, len(coefs))
    top = pd.concat([coefs.head(n // 2), coefs.tail(n // 2)])
 
    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.35)))
    colors  = ["#e74c3c" if v < 0 else "#2ecc71" for v in top.values]
    ax.barh(list(top.index), list(top.values), color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coeficiente OLS (escala estandarizada)")
    ax.set_title(f"Top {n} coeficientes — {MODEL_ID}\n"
                 f"(Intercepto = {model.intercept_:.4f})")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "coeficientes.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/coeficientes.png")
 
 
def plot_residuos(y_true: np.ndarray,
                  y_pred: np.ndarray) -> None:
    """
    Panel de diagnóstico de residuos:
      - Izquierda: residuos vs. valores predichos (detecta heterocedasticidad)
      - Derecha:   distribución de residuos (detecta sesgo sistemático)
 
    Para OLS los residuos deberían distribuirse como una normal =~N(0, σ²).
    Desviaciones indican falta de linealidad o heterocedasticidad, lo que
    motiva el uso de modelos más flexibles (Random Forest, Boosting).
    """
    residuos = y_true - y_pred
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    # Panel izquierdo: residuos vs. predichos
    axes[0].scatter(y_pred, residuos, alpha=0.3, s=5, color="#3498db")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Predicción log(price)")
    axes[0].set_ylabel("Residuo (log)")
    axes[0].set_title(f"Residuos vs. Predichos — {MODEL_ID}")
 
    # Panel derecho: histograma de residuos
    axes[1].hist(residuos, bins=60, color="#3498db", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("Residuo (log)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title(f"Distribución de residuos — {MODEL_ID}\n"
                      f"media={residuos.mean():.4f}, std={residuos.std():.4f}")
 
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "residuos.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/residuos.png")
 
 
def plot_cv_comparacion(mae_rand: float, std_rand: float,
                        mae_esp: float,  std_esp: float) -> None:
    """
    Compara el MAE del CV aleatorio vs. el CV espacial.
    cuánto subestima el CV aleatorio el error real al predecir en Chapinero.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
 
    labels = ["CV Aleatorio\n(optimista)", "CV Espacial\n(honesto)"]
    means  = [mae_rand, mae_esp]
    stds   = [std_rand, std_esp]
    colors = ["#3498db", "#e74c3c"]
 
    bars = ax.bar(labels, means, yerr=stds, color=colors,
                  capsize=6, edgecolor="white", width=0.5)
 
    # Etiqueta de valor sobre cada barra
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.1,
                f"{val:.5f}", ha="center", va="bottom", fontsize=10)
 
    sesgo = mae_esp - mae_rand
    ax.set_ylabel("MAE log(price)")
    ax.set_title(f"CV Aleatorio vs. Espacial — {MODEL_ID}\n"
                 f"Sesgo Δ = {sesgo:+.5f}")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "cv_comparacion.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/cv_comparacion.png")
 
 
# =============================================================================
# SECCIÓN 7: SUBMISSION
# =============================================================================
 
def generar_submission(test: pd.DataFrame,
                       y_pred_log: np.ndarray) -> str:
    """
    Exporta el CSV de submission en el formato exacto que pide Kaggle.
 
    Pasos:
      1. Exponencia las predicciones: log(price) → price (en COP).
      2. Genera el nombre de archivo con fecha para rastrear versiones.
      3. Guarda en 03_submissions/.
    """
    sub = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),   # revertir log → COP
    })
    fecha    = date.today().strftime("%Y%m%d")
    sub_name = f"submission_{MODEL_ID}_{fecha}.csv"
    sub.to_csv(SUBMISSIONS / sub_name, index=False)
    print(f"  Submission: 03_submissions/{sub_name}  ({len(sub):,} filas)")
    return sub_name
 
 
# =============================================================================
# SECCIÓN 8: REGISTRO EN MODEL REGISTRY
# =============================================================================
 
def registrar_modelo(mae_rand_log: float, std_rand_log: float,
                     mae_esp_log:  float, std_esp_log:  float,
                     mae_train_log: float,
                     n_features: int,
                     feature_cols: list,
                     sub_name: str) -> None:
    sesgo = mae_esp_log - mae_rand_log
 
    nueva = {
        "model_id":          MODEL_ID,
        "fecha":             str(date.today()),
        "autor":             AUTOR,
        "algoritmo":         "LinearRegression",
        "n_features":        n_features,
        "n_params":          n_features + 1,  # coeficientes + intercepto
        "cv_folds":          CV_FOLDS,
        # --- MAE de CV aleatorio (optimista) ---
        "cv_mae_log":        round(mae_rand_log, 5),
        "cv_std_log":        round(std_rand_log, 5),
        # --- MAE de CV espacial (honesto, proxy Chapinero) ---
        "esp_mae_log":       round(mae_esp_log,  5),
        "esp_std_log":       round(std_esp_log,  5),
        # --- MAE sobre train (mide sobreajuste) ---
        "train_mae_log":     round(mae_train_log, 5),
        "kaggle_public_MAE": None,   # ← llenar manualmente tras subir a Kaggle
        # --- no hay hiperparámetros que reportar en OLS ---
        "l1_ratio":          None,
        "alpha":             None,
        # --- contexto de features ---
        "features_grupos":  (f"structural={len([c for c in STRUCTURAL if c in feature_cols])}, "
                              f"text={len([c for c in TEXT if c in feature_cols])}, "
                              f"osm={len([c for c in OSM if c in feature_cols])}, "
                              f"es_apartamento"),
        "spatial_grid":      f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "submission_file":   sub_name,
        "notas": (
            f"OLS sin regularización. Baseline interpretable. "
            f"CV aleatorio MAE_log={mae_rand_log:.5f} (optimista). "
            f"CV espacial  MAE_log={mae_esp_log:.5f} (honesto). "
            f"Sesgo Δ={sesgo:+.5f}. "
            f"{n_features} features (structural + text + osm + es_apartamento)."
        ),
    }
    df_new = pd.DataFrame([nueva])
 
    if REGISTRY.exists():
        df_old = pd.read_excel(REGISTRY, engine="openpyxl")
        # Eliminar fila previa del mismo MODEL_ID para no duplicar
        df_old = df_old[df_old["model_id"] != MODEL_ID]
        df_reg = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_reg = df_new
 
    with pd.ExcelWriter(REGISTRY, engine="openpyxl") as writer:
        df_reg.to_excel(writer, index=False, sheet_name="registry")
        # Ajustar el ancho de cada columna al contenido más largo
        ws = writer.sheets["registry"]
        for col in ws.columns:
            max_len = max(
                len(str(col[0].value)) if col[0].value else 0,
                *(len(str(c.value)) if c.value else 0 for c in col[1:]),
            )
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)
 
    print(f"  Registro: 02_outputs/model_registry.xlsx  ({len(df_reg)} modelos)")
 
 
# =============================================================================
# SECCIÓN 9: LLAMAR LAS FUNCIOENS ARRIBA EN ORDEN PARA EJECUTAR EL PIPELINE COMPLETO
# =============================================================================
 
def main() -> None:
    print(f"{'='*60}")
    print(f"  REGRESIÓN LINEAL (OLS) — {MODEL_ID}")
    print(f"{'='*60}")
 
    # ── [1/8] Carga ───────────────────────────────────────────────────────────
    print("\n[1/8] Cargando datos...")
    train, test = cargar_datos()
    # Aplicar log al precio: estabiliza la varianza y hace la distribución más
    # simétrica. Los coeficientes OLS se interpretan como semi-elasticidades.
    y_train = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")
 
    # ── [2/8] Features ────────────────────────────────────────────────────────
    print("\n[2/8] Construyendo features...")
    X_train_df   = construir_features(train)
    feature_cols = list(X_train_df.columns)   # guardar orden para alinear test
    X_test_df    = construir_features(test, fit_cols=feature_cols)
    X_train_raw  = X_train_df.values
    X_test_raw   = X_test_df.values
    print(f"  Features: {len(feature_cols)}  "
          f"({len([c for c in STRUCTURAL if c in feature_cols])} estructurales + "
          f"{len([c for c in TEXT if c in feature_cols])} texto + "
          f"{len([c for c in OSM if c in feature_cols])} OSM + es_apartamento)")
 
    # ── [3/8] Grupos espaciales ───────────────────────────────────────────────
    # Creamos los grupos una sola vez y los reutilizamos en el CV espacial.
    print("\n[3/8] Construyendo grupos espaciales...")
    grupos = construir_grupos_espaciales(train)
    n_grupos = len(np.unique(grupos))
    print(f"  Cuadrícula {SPATIAL_GRID}×{SPATIAL_GRID} → {n_grupos} bloques con datos")
 
    # ── [4/8] Estandarización ─────────────────────────────────────────────────
    # OLS es invariante a la escala, pero StandardScaler hace que los
    # coeficientes sean comparables entre features (betas estandarizados).
    # CRÍTICO: el scaler se ajusta SOLO sobre train para evitar data leakage.
    print("\n[4/8] Estandarizando features...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
    print(f"  StandardScaler ajustado sobre train ({X_train.shape[0]:,} obs)")
 
    # ── [5/8] CV aleatorio ────────────────────────────────────────────────────
    print(f"\n[5/8] CV aleatorio ({CV_FOLDS}-fold KFold)...")
    # Instanciamos el modelo; fit_intercept=True añade la constante automáticamente.
    model = LinearRegression(fit_intercept=True)
    mae_rand, std_rand = cv_aleatorio(model, X_train, y_train)
    print(f"  MAE_log aleatorio = {mae_rand:.5f} ± {std_rand:.5f}  (optimista)")
 
    # ── [6/8] CV espacial ─────────────────────────────────────────────────────
    print(f"\n[6/8] CV espacial ({CV_FOLDS}-fold GroupKFold por bloque geográfico)...")
    mae_esp, std_esp = cv_espacial(model, X_train, y_train, grupos)
    sesgo = mae_esp - mae_rand
    print(f"  MAE_log espacial  = {mae_esp:.5f} ± {std_esp:.5f}  (honesto)")
    print(f"  Sesgo Δ           = {sesgo:+.5f}  "
          f"({'CV aleatorio es optimista' if sesgo > 0 else 'sin sesgo detectado'})")
 
    # ── [7/8] Modelo final ────────────────────────────────────────────────────
    # Se entrena sobre TODO el train con los hiperparámetros (ninguno en OLS).
    print("\n[7/8] Ajustando modelo final sobre todo el train...")
    model_final = LinearRegression(fit_intercept=True)
    model_final.fit(X_train, y_train)
 
    # MAE sobre train para diagnosticar sobreajuste
    y_pred_train = model_final.predict(X_train)
    mae_train    = mean_absolute_error(y_train, y_pred_train)
    print(f"  MAE_log train     = {mae_train:.5f}  "
          f"(comparar con CV para detectar sobreajuste)")
 
    n_nonzero = np.sum(model_final.coef_ != 0)
    print(f"  Coeficientes      = {len(model_final.coef_)} "
          f"(todos ≠ 0 en OLS, sin selección de variables)")
    print(f"  Intercepto        = {model_final.intercept_:.4f}")
    print(f"  R²  sobre train   = {model_final.score(X_train, y_train):.4f}")
 
    # ── [8/8] Diagnósticos, submission y registro ─────────────────────────────
    print("\n[8/8] Diagnósticos, submission y registro...")
 
    # Predicción sobre test
    y_pred_test = model_final.predict(X_test)
 
    # Gráficas
    plot_coeficientes(model_final, feature_cols, scaler)
    plot_residuos(y_train, y_pred_train)
    plot_cv_comparacion(mae_rand, std_rand, mae_esp, std_esp)
 
    # Submission
    sub_name = generar_submission(test, y_pred_test)
 
    # Registro
    registrar_modelo(
        mae_rand_log  = mae_rand,
        std_rand_log  = std_rand,
        mae_esp_log   = mae_esp,
        std_esp_log   = std_esp,
        mae_train_log = mae_train,
        n_features    = len(feature_cols),
        feature_cols  = feature_cols,
        sub_name      = sub_name,
    )
 
    print(f"\n{'='*60}")
    print(f"  LISTO — {MODEL_ID}")
    print(f"  MAE_log aleatorio = {mae_rand:.5f} ")
    print(f"  MAE_log espacial  = {mae_esp:.5f} ")
    print(f"  Sesgo Δ           = {sesgo:+.5f}")
    print(f"  Submission:  03_submissions/{sub_name}")
    print(f"{'='*60}")
 
 
if __name__ == "__main__":
    main()
 