"""
Elastic Net para predicción de log(price) de vivienda en Bogotá
===============================================================
Problem Set 3 — MECA 4107 (Big Data & ML para Economía Aplicada)

Modelo
------
Elastic Net combina LASSO y Ridge en una sola penalización:

  min  (1/2n) ||y - Xβ||²  +  α [ ρ ||β||₁  +  (1-ρ)/2 ||β||² ]

  α  = intensidad de la regularización total  (lambda en R/glmnet)
  ρ  = l1_ratio: fracción de la penalización que corresponde a LASSO
       ρ=1 → LASSO puro  (lleva coeficientes exactamente a cero → selección)
       ρ=0 → Ridge puro  (encoge todos los coeficientes → no selecciona)
       0<ρ<1 → Elastic Net (encoge Y selecciona)

  En este problema tenemos p ~ 30-40 features y n ~ 38 000 observaciones
  (p << n), así que OLS está identificado. Pero la regularización ayuda a
  reducir la varianza cuando algunas features son colineales (superficie_total
  y superficie_cubierta, por ejemplo) y mejora la generalización a Chapinero.

Motivación para CV espacial (01_SpatialData.ipynb)
---------------------------------------------------
  El train cubre todo Bogotá; el test (Kaggle) es solo Chapinero.
  Los precios de inmuebles vecinos están correlados espacialmente:
    Cov(ε(sᵢ), ε(sⱼ)) > 0  si sᵢ y sⱼ están cerca.
  Con CV aleatorio (KFold), el punto de validación sᵢ suele tener vecinos
  sⱼ en el conjunto de entrenamiento. El modelo "aprende" el ruido local de
  sⱼ y lo reutiliza en sᵢ → el error de validación parece más bajo de lo que
  realmente será en Chapinero (sesgo de optimismo).
  Con CV espacial (GroupKFold por bloque geográfico), los vecinos de sᵢ
  SIEMPRE están en entrenamiento → el error de validación es honesto.
  Δ = MAE_espacial − MAE_aleatorio mide ese sesgo.

Pipeline
--------
1. Carga  train_final.csv / test_final.csv
2. Ingeniería de features: dummies + interacciones hedónicas
3. StandardScaler ajustado SOLO sobre train (evita data leakage)
4. CV aleatorio 5-fold  (KFold)                   → MAE_random
5. CV espacial leave-one-block-out (GroupKFold)    → MAE_spatial
6. Comparación  Δ = MAE_spatial − MAE_random       (sesgo de optimismo)
7. Modelo final con hiperparámetros del CV espacial (más conservador)
8. Submission → 03_submissions/  (formato exacto del template Kaggle)
9. Registro automático → 02_outputs/model_registry.xlsx

Outputs
-------
  02_outputs/Models/ElasticNet/EN_001/coeficientes.png
  02_outputs/Models/ElasticNet/EN_001/cv_aleatorio_vs_espacial.png
  02_outputs/Models/ElasticNet/EN_001/residuos.png
  03_submissions/submission_EN_001_*.csv
  02_outputs/model_registry.xlsx
"""

# =============================================================================
# SECCIÓN 0: IMPORTACIONES
# =============================================================================

import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend sin pantalla: necesario para servidores/scripts
import matplotlib.pyplot as plt
import numpy as np
import openpyxl          # noqa: F401 — requerido internamente por pd.ExcelWriter
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
# Centralizar los hiperparámetros aquí hace el script reproducible y fácil
# de iterar: cambiar MODEL_ID a "EN_002" y ajustar parámetros genera una
# nueva entrada en el registry sin tocar el resto del código.

AUTOR    = "Jonathan"
MODEL_ID = "EN_001"
SEED     = 42   # semilla para reproducibilidad en KFold y ElasticNet

# Grilla de l1_ratio que se explorar en la búsqueda de hiperparámetros.
# Incluye los extremos (0.1=casi Ridge, 1.0=LASSO puro) y puntos intermedios.
L1_RATIOS    = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
N_ALPHAS     = 100    # número de valores de α que ElasticNetCV prueba en su grilla log-espaciada
CV_FOLDS     = 5      # número de folds para ambos CV (aleatorio y espacial)
SPATIAL_GRID = 5      # divide Bogotá en una cuadrícula 5×5 = 25 bloques espaciales

# Rutas relativas a la raíz del proyecto (4 niveles arriba de este script)
BASE        = Path(__file__).parent.parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "ElasticNet" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

# Crear todas las carpetas necesarias al arrancar el script.
# parents=True crea directorios intermedios; exist_ok=True no falla si ya existen.
for d in [SUBMISSIONS, DIR_MODEL, BASE / "02_outputs"]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECCIÓN 2: DEFINICIÓN DE GRUPOS DE FEATURES
# =============================================================================
# Se definen tres grupos separados para poder reportar la contribución de cada
# fuente de información (Properati, texto libre, OSM) en el análisis final.

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
    # Variables extraídas del título y descripción de los anuncios.
    # Capturan atributos que el propietario menciona pero que no están en
    # los campos estructurados de Properati.
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
    # Rosen (1974): el precio hedónico refleja también los servicios del entorno.
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
    Construye la matriz de features X con las variables del dataset procesado.

    Pasos:
      1. Codifica property_type como variable binaria: es_apartamento=1 si Casa, 0 si Apartamento.
      2. Selecciona las variables de los tres grupos (STRUCTURAL, TEXT, OSM)
         más la variable es_apartamento.
      3. Si se pasa fit_cols, alinea el orden de columnas de test con train.
    """
    d = df.copy()
    d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)
    all_cols = STRUCTURAL + TEXT + OSM + ["es_apartamento"]
    all_cols = [c for c in all_cols if c in d.columns]
    X = d[all_cols]
    if fit_cols is not None:
        X = X[fit_cols]
    return X


def crear_grupos_espaciales(lat: np.ndarray,
                             lon: np.ndarray) -> np.ndarray:
    """
    Asigna a cada observación un ID de bloque geográfico (0..24 con grid 5×5).
    Se usa como argumento 'groups' en GroupKFold: todos los inmuebles del
    mismo bloque caen en el mismo fold, garantizando separación geográfica.

    Implementación:
      - pd.qcut divide cada coordenada en G cuantiles de igual tamaño.
      - El ID final es lat_bin * G + lon_bin → entero único por celda.
      - duplicates="drop" maneja casos donde muchos puntos tienen la misma
        coordenada (evita que qcut falle por bins con 0 observaciones).
    """
    G        = SPATIAL_GRID
    lat_bins = pd.qcut(lat, G, labels=False, duplicates="drop")
    lon_bins = pd.qcut(lon, G, labels=False, duplicates="drop")
    return (lat_bins * G + lon_bins).astype(int)


# =============================================================================
# SECCIÓN 4: CV ALEATORIO (KFold) — EN_tp_random
# =============================================================================

def cv_aleatorio(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Busca el mejor (l1_ratio, α) con CV estándar de 5 pliegues aleatorios.

    Por qué es 'optimista' (cuaderno 01_SpatialData.ipynb):
      KFold mezcla aleatoriamente las observaciones. Con datos espaciales,
      el punto de validación sᵢ tiene alta probabilidad de tener vecinos
      en el conjunto de entrenamiento. Los errores de vecinos están
      correlados (autocorrelación espacial), así que el modelo parece
      generalizar mejor de lo que en realidad hará en Chapinero.

    Implementación:
      1. Para cada l1_ratio, ElasticNetCV hace su propia búsqueda de α
         en una grilla log-espaciada de N_ALPHAS puntos, usando los mismos
         5 folds aleatorios. Esto es equivalente a expand.grid(alpha, lambda)
         + trainControl(method="cv", number=5) en R/caret.
      2. Con el α óptimo encontrado, calculamos el MAE out-of-fold real
         usando cross_val_score + Pipeline (StandardScaler dentro del fold
         para no filtrar información del conjunto de validación).
      3. El modelo con menor MAE_log define los hiperparámetros 'random'.
    """
    print(f"\n{'='*60}")
    print(f"  CV ALEATORIO ({CV_FOLDS}-fold, KFold) — EN_tp_random")
    print(f"{'='*60}")

    # Escalamos aquí solo para que ElasticNetCV encuentre el α correcto.
    # El MAE real se calcula dentro del Pipeline (ver abajo) para evitar
    # que la escala del fold de validación contamine la búsqueda.
    scaler:     StandardScaler = StandardScaler()
    X_sc        = scaler.fit_transform(X)
    best_mae:   float = np.inf
    best_ratio: float = L1_RATIOS[0]
    best_alpha: float = 1e-3
    resultados  = []

    for ratio in L1_RATIOS:
        # ElasticNetCV: busca α internamente con CV de 5 pliegues.
        # n_jobs=-1 usa todos los núcleos disponibles en paralelo.
        enc = ElasticNetCV(
            l1_ratio=ratio, n_alphas=N_ALPHAS, cv=CV_FOLDS,
            max_iter=10_000, random_state=SEED, n_jobs=-1,
        )
        enc.fit(X_sc, y)

        # Calculamos el MAE out-of-fold con el α elegido.
        # Pipeline garantiza que el StandardScaler se ajusta solo sobre
        # el fold de entrenamiento en cada iteración → sin data leakage.
        kf    = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        model = ElasticNet(alpha=enc.alpha_, l1_ratio=ratio,
                           max_iter=10_000, random_state=SEED)
        scores = -cross_val_score(
            Pipeline([("sc", StandardScaler()), ("en", model)]),
            X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        mae_log = float(scores.mean())
        resultados.append({"l1_ratio": ratio, "alpha": enc.alpha_,
                            "mae_log": mae_log})
        print(f"  l1_ratio={ratio:.2f} → α={enc.alpha_:.6f}, "
              f"MAE_log={mae_log:.5f}")

        if mae_log < best_mae:
            best_mae   = mae_log
            best_ratio = ratio
            best_alpha = enc.alpha_

    print(f"\n  → Mejor: l1_ratio={best_ratio}, α={best_alpha:.6f}, "
          f"MAE_log={best_mae:.5f}  ← optimista (puede subestimar error real)")
    return best_ratio, best_alpha, pd.DataFrame(resultados), best_mae


# =============================================================================
# SECCIÓN 5: CV ESPACIAL (GroupKFold — leave-one-block-out) — EN_tp_spatial
# =============================================================================

def cv_espacial(X: np.ndarray, y: np.ndarray,
                grupos: np.ndarray) -> tuple:
    """
    Busca el mejor (l1_ratio, α) con CV espacial leave-one-block-out.

    Equivalente R (cuaderno 01_SpatialData.ipynb, celdas 46-52):
      location_folds <- spatial_leave_location_out_cv(train, group=Neighborhood)
      folds_train    <- lapply(splits, function(s) s$in_id)
      fitControl     <- trainControl(method="cv", index=folds_train)
      EN_tp_spatial  <- train(..., trControl=fitControl)

    Cómo funciona GroupKFold:
      - Recibe un vector 'groups' con el ID del bloque de cada observación.
      - En cada fold, TODOS los inmuebles de uno o más bloques van a
        validación; el resto va a entrenamiento.
      - Ningún inmueble del fold de validación tiene vecinos en entrenamiento
        → la autocorrelación espacial no infla artificialmente el desempeño.

    Por qué usar los hiperparámetros del CV espacial en el modelo final:
      El CV espacial simula mejor lo que pasará en Chapinero: un bloque
      geográfico completo no visto durante el entrenamiento. Los hiperparámetros
      que minimizan el MAE espacial son más conservadores (mayor α, menor
      varianza) y probablemente generalizan mejor a la zona de prueba.
    """
    print(f"\n{'='*60}")
    print(f"  CV ESPACIAL leave-one-block-out (GroupKFold) — EN_tp_spatial")
    print(f"{'='*60}")

    n_groups = len(np.unique(grupos))
    # n_splits no puede superar el número de grupos distintos
    n_splits = min(CV_FOLDS, n_groups)

    scaler:     StandardScaler = StandardScaler()
    X_sc        = scaler.fit_transform(X)
    best_mae:   float = np.inf
    best_ratio: float = L1_RATIOS[0]
    best_alpha: float = 1e-3
    resultados  = []

    gkf = GroupKFold(n_splits=n_splits)

    for ratio in L1_RATIOS:
        # Pasamos los folds espaciales directamente a ElasticNetCV.
        # list(gkf.split(...)) convierte el generador en una lista de
        # tuplas (train_idx, val_idx) que ElasticNetCV puede consumir.
        enc = ElasticNetCV(
            l1_ratio=ratio, n_alphas=N_ALPHAS,
            cv=list(gkf.split(X_sc, y, groups=grupos)),
            max_iter=10_000, random_state=SEED, n_jobs=-1,
        )
        enc.fit(X_sc, y)

        # MAE out-of-fold con separación geográfica estricta.
        # oof_pred acumula las predicciones de validación de cada fold
        # sin que el StandardScaler vea los datos de validación.
        model    = ElasticNet(alpha=enc.alpha_, l1_ratio=ratio,
                              max_iter=10_000, random_state=SEED)
        oof_pred = np.zeros_like(y)
        for tr_idx, val_idx in gkf.split(X, y, groups=grupos):
            sc_fold = StandardScaler()
            X_tr_sc = sc_fold.fit_transform(X[tr_idx])   # fit solo en train
            X_va_sc = sc_fold.transform(X[val_idx])      # transform en val
            model.fit(X_tr_sc, y[tr_idx])
            oof_pred[val_idx] = model.predict(X_va_sc)

        mae_log = float(mean_absolute_error(y, oof_pred))
        resultados.append({"l1_ratio": ratio, "alpha": enc.alpha_,
                            "mae_log": mae_log})
        print(f"  l1_ratio={ratio:.2f} → α={enc.alpha_:.6f}, "
              f"MAE_log={mae_log:.5f}")

        if mae_log < best_mae:
            best_mae   = mae_log
            best_ratio = ratio
            best_alpha = enc.alpha_

    print(f"\n  → Mejor: l1_ratio={best_ratio}, α={best_alpha:.6f}, "
          f"MAE_log={best_mae:.5f}  ← honesto (separación geográfica)")
    return best_ratio, best_alpha, pd.DataFrame(resultados), best_mae


# =============================================================================
# SECCIÓN 6: MODELO FINAL Y MÉTRICAS OOF
# =============================================================================

def ajustar_modelo_final(X: np.ndarray, y: np.ndarray,
                         l1_ratio: float, alpha: float):
    """
    Re-entrena el modelo sobre TODOS los datos de train con los hiperparámetros
    elegidos por el CV espacial. Usar todos los datos maximiza la información
    disponible para las predicciones sobre Chapinero.
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)   # ajusta y transforma en un solo paso
    model  = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                        max_iter=10_000, random_state=SEED)
    model.fit(X_sc, y)
    return scaler, model   # devolvemos el scaler para transformar el test set


def mae_oof_random(X: np.ndarray, y: np.ndarray,
                   l1_ratio: float, alpha: float) -> tuple:
    """
    Calcula el MAE out-of-fold con CV aleatorio para los hiperparámetros dados.
    Sirve como línea de base 'optimista' con la que comparar el CV espacial.
    """
    kf  = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    pip = Pipeline([
        ("sc", StandardScaler()),
        ("en", ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                          max_iter=10_000, random_state=SEED)),
    ])
    oof_pred = np.zeros_like(y)
    for tr_idx, val_idx in kf.split(X):
        pip.fit(X[tr_idx], y[tr_idx])
        oof_pred[val_idx] = pip.predict(X[val_idx])

    mae_log = float(mean_absolute_error(y, oof_pred))
    # Deshacemos el log para reportar el MAE en COP:  MAE_COP = E[|price - price_hat|]
    mae_cop = float(mean_absolute_error(np.exp(y), np.exp(oof_pred)))
    return mae_log, mae_cop


def mae_oof_espacial(X: np.ndarray, y: np.ndarray,
                     grupos: np.ndarray,
                     l1_ratio: float, alpha: float) -> tuple:
    """
    Calcula el MAE out-of-fold con CV espacial para los hiperparámetros dados.
    Este es el mejor proxy del MAE que obtendremos en Kaggle (Chapinero).
    """
    gkf      = GroupKFold(n_splits=min(CV_FOLDS, len(np.unique(grupos))))
    oof_pred = np.zeros_like(y)
    for tr_idx, val_idx in gkf.split(X, y, groups=grupos):
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X[tr_idx])
        X_va_sc = sc.transform(X[val_idx])
        m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                       max_iter=10_000, random_state=SEED)
        m.fit(X_tr_sc, y[tr_idx])
        oof_pred[val_idx] = m.predict(X_va_sc)

    mae_log = float(mean_absolute_error(y, oof_pred))
    mae_cop = float(mean_absolute_error(np.exp(y), np.exp(oof_pred)))
    return mae_log, mae_cop


# =============================================================================
# SECCIÓN 7: GRÁFICOS DE DIAGNÓSTICO
# =============================================================================

def plot_coeficientes(model: ElasticNet, feature_names: list,
                      l1_ratio: float, alpha: float) -> None:
    """
    Muestra los coeficientes no nulos del modelo final ordenados por magnitud.
    Los coeficientes están en escala estandarizada → son comparables entre sí:
    un coeficiente más grande en valor absoluto indica mayor influencia sobre
    log(price) por desviación estándar de la feature.
    Verde = efecto positivo sobre el precio. Rojo = efecto negativo.
    """
    coefs    = pd.Series(model.coef_, index=feature_names)
    coefs_nz = coefs[coefs != 0].sort_values()     # solo los seleccionados por LASSO
    n        = min(30, len(coefs_nz))               # mostrar como máximo 30
    # Mostrar los n/2 más negativos (abajo) y los n/2 más positivos (arriba)
    top      = pd.concat([coefs_nz.head(n // 2), coefs_nz.tail(n // 2)])

    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.35)))
    colors  = ["#e74c3c" if v < 0 else "#2ecc71" for v in top.values]
    ax.barh(list(top.index), list(top.values), color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coeficiente Elastic Net (escala estandarizada)")
    ax.set_title(f"Top {n} coeficientes — {MODEL_ID}\n"
                 f"(l1_ratio={l1_ratio:.2f}, α={alpha:.2e})")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "coeficientes.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/coeficientes.png")


def plot_cv_comparacion(df_rand: pd.DataFrame, df_esp: pd.DataFrame) -> None:
    """
    Compara el MAE del CV aleatorio vs el CV espacial para cada l1_ratio.

    La brecha sombreada entre las dos curvas es el sesgo de optimismo:
    cuanto mayor es la zona roja, más el CV aleatorio subestima el error real.
    Esta comparación es central en 01_SpatialData.ipynb — la idea de que
    EN_tp_random < EN_tp_spatial en validación, pero la diferencia se revela
    al predecir en la zona geográfica desconocida (Chapinero).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_rand["l1_ratio"], df_rand["mae_log"],
            "o-", color="#3498db", lw=2, label="CV aleatorio (optimista)")
    ax.plot(df_esp["l1_ratio"],  df_esp["mae_log"],
            "s--", color="#e74c3c", lw=2, label="CV espacial (honesto)")
    # fill_between necesita arrays numpy; los stubs de matplotlib tienen un
    # tipo incorrecto para y1/y2, de ahí el # type: ignore.
    x_fb  = df_rand["l1_ratio"].to_numpy()
    y1_fb = df_rand["mae_log"].to_numpy()
    y2_fb = df_esp["mae_log"].to_numpy()
    ax.fill_between(x_fb, y1_fb, y2_fb, alpha=0.12, color="#e74c3c", label="Sesgo de optimismo Δ")  # type: ignore[arg-type]
    ax.set_xlabel("l1_ratio  (0=Ridge … 1=LASSO)")
    ax.set_ylabel("MAE log(price)")
    ax.set_title(f"CV aleatorio vs CV espacial — {MODEL_ID}\n"
                 "La brecha es el sesgo de optimismo del CV aleatorio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(DIR_MODEL / "cv_aleatorio_vs_espacial.png", dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/cv_aleatorio_vs_espacial.png")


def plot_residuos(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Dos gráficos de diagnóstico de residuos sobre el conjunto de entrenamiento:
      - Izquierda: residuo vs. valor ajustado. Si hay un patrón → sesgo sistemático.
        Una nube sin forma alrededor de 0 indica que el modelo no está sesgado.
      - Derecha: histograma de residuos. Si es aproximadamente normal → los errores
        son simétricos, lo que implica que el modelo no sobreestima ni subestima
        sistemáticamente (relevante para la discusión de negocio del PS).
    """
    res = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, res, alpha=0.2, s=5, color="#3498db")
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("log(price) predicho")
    axes[0].set_ylabel("Residuo  (log)")
    axes[0].set_title(f"Residuos vs. Ajustados — {MODEL_ID}")
    axes[1].hist(res, bins=60, color="#3498db", edgecolor="white")
    axes[1].set_xlabel("Residuo  (log)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title(f"Distribución residuos — media={res.mean():.4f}")
    plt.tight_layout()
    fig.savefig(DIR_MODEL / "residuos.png", dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/residuos.png")


# =============================================================================
# SECCIÓN 8: SUBMISSION PARA KAGGLE
# =============================================================================

# Ruta al template de Kaggle con los property_id en el orden oficial
TEMPLATE_PATH = Path(
    "/Users/macbook/Downloads/uniandes-bdml-202610-ps-3/submission_template.csv"
)


def generar_submission(test: pd.DataFrame, y_pred_log: np.ndarray,
                       l1_ratio: float, alpha: float) -> str:
    """
    Genera el CSV de predicciones en el formato exacto del template de Kaggle.

    Puntos importantes:
      - Se hace exp(y_pred_log) para volver a la escala original en COP.
        El modelo predice log(price); Kaggle evalúa en COP (MAE en pesos).
      - El merge con el template garantiza que el orden de property_id sea
        el mismo que espera Kaggle, sin importar el orden en nuestro test.csv.
      - El nombre del archivo incluye los hiperparámetros clave para que sea
        fácil identificar qué submission corresponde a qué modelo en el log.
    """
    pred = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),   # volver de log(COP) a COP
    })
    template   = pd.read_csv(TEMPLATE_PATH)
    # Left join: el template define el orden; pred aporta los precios
    submission = template[["property_id"]].merge(pred, on="property_id",
                                                  how="left")

    sub_name = (
        f"submission_{MODEL_ID}_l1r{int(l1_ratio*100):03d}"
        f"_a{alpha:.0e}.csv"
    ).replace("-", "n")   # reemplazar '-' para evitar confusión en nombres de archivo
    submission.to_csv(SUBMISSIONS / sub_name, index=False)

    print(f"\n  Submission: 03_submissions/{sub_name}")
    print(f"  Filas: {len(submission):,} | "
          f"price media: {submission['price'].mean()/1e6:.1f}M COP")
    return sub_name


# =============================================================================
# SECCIÓN 9: REGISTRO EN model_registry.xlsx
# =============================================================================

def registrar(sub_name: str, feature_cols: list,
              l1_ratio_rand: float, alpha_rand: float,
              l1_ratio_esp: float,  alpha_esp: float,
              n_nonzero: int,
              mae_rand_log: float, mae_rand_cop: float,
              mae_esp_log:  float, mae_esp_cop: float,
              mae_train_log: float) -> None:
    """
    Añade (o actualiza) una fila en el registro central de modelos.

    El registro permite comparar todos los modelos del equipo en un solo lugar:
    EN_001, NN_001, RF_001, etc. La columna kaggle_public_MAE se llena
    manualmente después de subir el CSV a Kaggle.

    Si el MODEL_ID ya existe en el Excel, la fila anterior se reemplaza
    en lugar de duplicarse → útil al re-correr el script con ajustes.
    """
    sesgo = mae_esp_log - mae_rand_log   # Δ: sesgo de optimismo del CV aleatorio
    nueva = {
        "model_id":           MODEL_ID,
        "fecha":              str(date.today()),
        "autor":              AUTOR,
        "algoritmo":          "ElasticNet",
        "n_features":         len(feature_cols),
        "n_coefs_nonzero":    n_nonzero,    # features seleccionadas (coef ≠ 0)
        "cv_folds":           CV_FOLDS,
        # --- métricas de CV aleatorio (optimista) ---
        "rand_cv_mae_log":    round(mae_rand_log, 5),
        "rand_cv_mae_cop_M":  round(mae_rand_cop / 1e6, 2),
        # --- métricas de CV espacial (honesto, proxy Chapinero) ---
        "esp_cv_mae_log":     round(mae_esp_log, 5),
        "esp_cv_mae_cop_M":   round(mae_esp_cop / 1e6, 2),
        # --- sesgo de optimismo ---
        "sesgo_delta_log":    round(sesgo, 5),
        "train_mae_log":      round(mae_train_log, 5),
        "kaggle_public_MAE":  None,   # ← llenar manualmente tras subir a Kaggle
        # --- hiperparámetros de ambos CV (para análisis comparativo) ---
        "l1_ratio_spatial":   l1_ratio_esp,
        "alpha_spatial":      round(alpha_esp, 8),
        "l1_ratio_random":    l1_ratio_rand,
        "alpha_random":       round(alpha_rand, 8),
        # --- contexto ---
        "spatial_grid":       f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "submission_file":    sub_name,
        "notas": (
            f"CV aleatorio MAE_log={mae_rand_log:.5f} (optimista). "
            f"CV espacial  MAE_log={mae_esp_log:.5f} (honesto). "
            f"Sesgo Δ={sesgo:+.5f}. "
            f"Modelo final: l1_ratio={l1_ratio_esp}, α={alpha_esp:.2e}. "
            f"{n_nonzero}/{len(feature_cols)} coefs≠0."
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
# SECCIÓN 10: ENTRY POINT
# =============================================================================

def main() -> None:
    print(f"{'='*60}")
    print(f"  ELASTIC NET — {MODEL_ID}")
    print(f"{'='*60}")

    # ── [1/8] Carga ───────────────────────────────────────────────────────────
    print("\n[1/8] Cargando datos...")
    train, test = cargar_datos()
    # Aplicar log al precio: estabiliza la varianza y hace la distribución más
    # simétrica. También es lo que pide el PS ("predecir log(price)").
    y_train     = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    # ── [2/8] Features ────────────────────────────────────────────────────────
    print("\n[2/8] Construyendo features...")
    X_train_df   = construir_features(train)
    feature_cols = list(X_train_df.columns)   # guardamos el orden para alinear test
    X_test_df    = construir_features(test, fit_cols=feature_cols)
    X_train      = X_train_df.values
    X_test       = X_test_df.values
    print(f"  Features: {len(feature_cols)}  "
          f"({len(STRUCTURAL)} estructurales + {len(TEXT)} texto + "
          f"{len(OSM)} OSM + es_apartamento)")

    # ── [3/8] Grupos espaciales ───────────────────────────────────────────────
    # Creamos los grupos una sola vez y los reutilizamos en ambos CV.
    grupos = crear_grupos_espaciales(
        train["lat"].values, train["lon"].values
    )
    n_bloques = len(np.unique(grupos))
    print(f"\n[3/8] Bloques espaciales: {n_bloques} "
          f"(cuadrícula {SPATIAL_GRID}×{SPATIAL_GRID})")

    # ── [4/8] CV aleatorio ────────────────────────────────────────────────────
    print("\n[4/8] CV aleatorio (KFold, EN_tp_random)...")
    ratio_rand, alpha_rand, df_rand, _ = cv_aleatorio(X_train, y_train)
    mae_rand_log, mae_rand_cop = mae_oof_random(
        X_train, y_train, ratio_rand, alpha_rand
    )
    print(f"  MAE aleatorio (log): {mae_rand_log:.5f} ← optimista")

    # ── [5/8] CV espacial ─────────────────────────────────────────────────────
    print("\n[5/8] CV espacial (GroupKFold, EN_tp_spatial)...")
    ratio_esp, alpha_esp, df_esp, _ = cv_espacial(X_train, y_train, grupos)
    mae_esp_log, mae_esp_cop = mae_oof_espacial(
        X_train, y_train, grupos, ratio_esp, alpha_esp
    )
    print(f"  MAE espacial (log): {mae_esp_log:.5f} ← honesto")

    # Sesgo de optimismo: cuánto subestima el CV aleatorio el error real
    sesgo = mae_esp_log - mae_rand_log
    print(f"\n  ── Sesgo de optimismo del CV aleatorio ──────────────────")
    print(f"  Δ = MAE_espacial − MAE_aleatorio = {sesgo:+.5f}")
    print(f"  El CV aleatorio subestima el error real en {abs(sesgo):.5f} "
          f"unidades log-precio.")
    print(f"  El CV espacial es el mejor proxy del rendimiento en Chapinero.")

    # ── [6/8] Modelo final ────────────────────────────────────────────────────
    # Usamos los hiperparámetros del CV espacial: son más conservadores y
    # están optimizados para generalizar a zonas geográficas no vistas.
    print("\n[6/8] Ajustando modelo final (hiperparámetros del CV espacial)...")
    scaler, model = ajustar_modelo_final(X_train, y_train, ratio_esp, alpha_esp)
    y_pred_train  = model.predict(scaler.transform(X_train))
    mae_train_log = float(mean_absolute_error(y_train, y_pred_train))
    mae_train_cop = float(
        mean_absolute_error(np.exp(y_train), np.exp(y_pred_train))
    )
    # Contar coeficientes no nulos: refleja cuántas features seleccionó el EN
    n_nonzero = int(np.sum(model.coef_ != 0))
    print(f"  MAE train (log): {mae_train_log:.5f} | "
          f"MAE train (COP): {mae_train_cop/1e6:.1f}M")
    print(f"  Coefs ≠ 0: {n_nonzero}/{len(feature_cols)}")

    # ── [7/8] Gráficos ────────────────────────────────────────────────────────
    print("\n[7/8] Generando gráficos...")
    plot_coeficientes(model, feature_cols, ratio_esp, alpha_esp)
    plot_cv_comparacion(df_rand, df_esp)
    plot_residuos(y_train, y_pred_train)

    # ── [8/8] Submission + Registro ───────────────────────────────────────────
    print("\n[8/8] Submission y registro...")
    y_pred_test = model.predict(scaler.transform(X_test))
    sub_name    = generar_submission(test, y_pred_test, ratio_esp, alpha_esp)
    registrar(
        sub_name=sub_name,
        feature_cols=feature_cols,
        l1_ratio_rand=ratio_rand, alpha_rand=alpha_rand,
        l1_ratio_esp=ratio_esp,   alpha_esp=alpha_esp,
        n_nonzero=n_nonzero,
        mae_rand_log=mae_rand_log, mae_rand_cop=mae_rand_cop,
        mae_esp_log=mae_esp_log,   mae_esp_cop=mae_esp_cop,
        mae_train_log=mae_train_log,
    )

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESUMEN FINAL — {MODEL_ID}")
    print(f"{'='*60}")
    print(f"  Hiperparámetros (CV espacial): "
          f"l1_ratio={ratio_esp}, α={alpha_esp:.2e}")
    print(f"  Coefs ≠ 0 : {n_nonzero}/{len(feature_cols)}")
    print(f"  CV aleatorio MAE_log: {mae_rand_log:.5f}  "
          f"({mae_rand_cop/1e6:.1f}M COP)  ← optimista")
    print(f"  CV espacial  MAE_log: {mae_esp_log:.5f}  "
          f"({mae_esp_cop/1e6:.1f}M COP)  ← honesto / proxy Chapinero")
    print(f"  Sesgo Δ             : {sesgo:+.5f}  "
          f"({'↑ CV aleatorio subestima' if sesgo > 0 else '↓ similar'})")
    print(f"  Submission : 03_submissions/{sub_name}")
    print(f"  Registry   : 02_outputs/model_registry.xlsx")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    main()
