"""
Red Neuronal para predicción de log(price) de vivienda en Bogotá
================================================================
Problem Set 3 — MECA 4107 (Big Data & ML para Economía Aplicada)

Arquitectura — MLP feedforward para regresión hedónica
-------------------------------------------------------
  Entrada  : p features estandarizadas (estructurales + OSM + texto)
  Capa 1   : Dense(256, ReLU) + Dropout(0.3) + kernel_regularizer ElasticNet
  Capa 2   : Dense(128, ReLU) + Dropout(0.2) + kernel_regularizer ElasticNet
  Capa 3   : Dense(64,  ReLU)
  Salida   : Dense(1, lineal)   ← regresión continua

  Pérdida : MSE  — diferenciable, guía el backprop (algoritmo de optimización)
  Métrica : MAE  — interpretable en COP, es el criterio de Kaggle
  Optimizer: Adam — adapta la tasa de aprendizaje de cada peso por separado

Regularización (tres mecanismos combinados)
-------------------------------------------
  1. Elastic Net sobre pesos de cada capa: λ₁||W||₁ + λ₂||W||₂²
     Añade una penalización a la pérdida por el tamaño de los pesos.
     Equivale a shrinkage en regresión lineal, pero aplicado a la red.
  2. Dropout: en cada mini-batch, apaga aleatoriamente una fracción φ de
     neuronas. Evita que las neuronas se "especialicen" en memorizar el
     ruido del conjunto de entrenamiento (co-adaptación).
  3. Early Stopping: detiene el entrenamiento cuando la pérdida en
     validación deja de mejorar. Restaura los pesos del mejor epoch.
     Evita sobreajuste sin necesidad de fijar el número de épocas a priori.

Por qué predecir log(price) en lugar de price
----------------------------------------------
  Los precios tienen distribución sesgada a la derecha (cola larga de
  propiedades muy caras). Al usar log(price) la distribución se vuelve
  más simétrica y normal, lo que facilita el entrenamiento. Además, el
  MAE en log-escala penaliza errores relativos en lugar de absolutos,
  que es más apropiado para precios que varían en órdenes de magnitud.

CV espacial (01_SpatialData.ipynb)
-----------------------------------
  El train cubre toda Bogotá; el test (Kaggle) es solo Chapinero.
  La red tiene mucha capacidad (>200k parámetros) y puede memorizar
  patrones locales del ruido espacial. Con validation_split=0.2 aleatorio,
  los puntos de validación tienen vecinos en el entrenamiento → el error
  de validación es optimista. El CV espacial por cuadrícula geográfica
  evalúa si la red generaliza a zonas no vistas, que es el verdadero reto.

Pipeline
--------
1. Carga  train_final.csv / test_final.csv
2. Misma ingeniería de features que 01_elastic_net.py (comparabilidad directa)
3. StandardScaler ajustado SOLO sobre train (evita data leakage)
4. Entrenamiento con validation_split=0.2 + Early Stopping
5. CV espacial leave-one-block-out por cuadrícula geográfica 5×5
6. Diagnósticos: curvas de entrenamiento, residuos, CV espacial
7. Submission → 03_submissions/  (formato exacto del template Kaggle)
8. Registro automático → 02_outputs/model_registry.xlsx

Outputs
-------
  02_outputs/Models/NeuralNetwork/NN_001/historia_entrenamiento.png
  02_outputs/Models/NeuralNetwork/NN_001/residuos.png
  02_outputs/Models/NeuralNetwork/NN_001/cv_espacial.png
  03_submissions/submission_NN_001_*.csv
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
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
# Desactiva el modo interactivo de Keras para que no intente detectar
# si está dentro de un notebook (evita llamadas a sys.stdout en threads).
tf.keras.utils.disable_interactive_logging()

# Semilla global para reproducibilidad. Se fija tanto en numpy como en TF
# porque las redes neuronales tienen múltiples fuentes de aleatoriedad:
# inicialización de pesos, orden de mini-batches, Dropout.
SEED = 101010
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
# Centralizar hiperparámetros aquí hace el script fácil de iterar:
# cambiar MODEL_ID a "NN_002" y ajustar valores genera una nueva entrada
# en el registry sin modificar el resto del código.

AUTOR    = "Jonathan"
MODEL_ID = "NN_001"

# Hiperparámetros del entrenamiento
EPOCHS     = 200    # número máximo de épocas; Early Stopping detiene antes si corresponde
BATCH_SIZE = 512    # tamaño del mini-batch (potencia de 2 para eficiencia en GPU/CPU)
VAL_SPLIT  = 0.2    # fracción del train reservada para monitoreo durante el entrenamiento
PATIENCE   = 15     # épocas sin mejora en val_loss antes de activar Early Stopping

# Hiperparámetros de regularización
L1_REG    = 1e-4    # peso de la penalización L1 (LASSO) sobre los pesos de la red
L2_REG    = 1e-4    # peso de la penalización L2 (Ridge) sobre los pesos de la red
DROPOUT_1 = 0.3     # fracción de neuronas apagadas en la primera capa oculta
DROPOUT_2 = 0.2     # fracción de neuronas apagadas en la segunda capa oculta

SPATIAL_GRID = 5    # cuadrícula 5×5 = 25 bloques geográficos para CV espacial

# Rutas relativas a la raíz del proyecto (3 niveles arriba de este script)
BASE        = Path(__file__).parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "NeuralNetwork" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

# Crear carpetas antes de que el Tee abra el log
for d in [SUBMISSIONS, DIR_MODEL, BASE / "02_outputs"]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECCIÓN 2: DEFINICIÓN DE GRUPOS DE FEATURES
# =============================================================================
# Idénticas a 01_elastic_net.py: esto permite comparar los dos modelos de forma
# limpia — cualquier diferencia en desempeño se debe al algoritmo, no a las features.

STRUCTURAL = [
    # Atributos físicos del inmueble — variables "clásicas" del modelo hedónico
    "surface_total",    # superficie total en m²
    "surface_covered",  # superficie cubierta (sin terrazas/jardines)
    "rooms",            # número de habitaciones
    "bedrooms",         # número de alcobas
    "bathrooms",        # número de baños
    "month",            # mes de publicación (estacionalidad)
    "year",             # año de publicación (tendencia temporal)
]

TEXT = [
    # Variables extraídas del título y descripción del anuncio.
    # Capturan atributos que el propietario menciona pero que no están
    # en los campos estructurados de Properati.
    "remodelado",       # 1 si el texto menciona remodelación
    "vista_panoramica", # 1 si menciona vista panorámica
    "deposito",         # 1 si menciona depósito/bodega
    "conjunto_cerrado", # 1 si menciona conjunto cerrado o seguridad
    "balcon_terraza",   # 1 si menciona balcón o terraza
    "tfidf_premium",    # score TF-IDF de términos de lujo
    "parqueaderos_txt", # número de parqueaderos en el texto
    "piso_txt",         # número de piso extraído del texto
    "gimnasio",         # 1 si menciona gimnasio
    "amenidades",       # score agregado de amenidades mencionadas
    "num_amenidades",   # conteo de amenidades distintas
]

OSM = [
    # Variables de localización construidas con OpenStreetMap.
    # Rosen (1974): el precio hedónico incluye los servicios del entorno.
    "dist_cbd_km",          # distancia al CBD en km
    "dist_transmilenio_m",  # distancia a estación de TransMilenio más cercana
    "dist_via_arterial_m",  # distancia a vía arterial principal
    "dist_hospital_m",      # distancia al hospital más cercano
    "dist_centro_com_m",    # distancia al centro comercial más cercano
    "dist_parque_m",        # distancia al parque más cercano
    "n_restaurantes_500m",  # restaurantes en 500 m de radio
    "n_bancos_500m",        # bancos en 500 m de radio
    "walkability_score",    # índice de caminabilidad
    "densidad_vial",        # metros de vía por km² en el entorno
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
    Idéntica a 01_elastic_net.py para garantizar comparabilidad directa.

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


# =============================================================================
# SECCIÓN 4: ARQUITECTURA DE LA RED
# =============================================================================

def construir_modelo(n_features: int) -> keras.Model:
    """
    Define y compila el MLP para regresión de log(price).

    Decisiones de diseño:
      - 3 capas ocultas (256→128→64): progresión decreciente que fuerza a la
        red a comprimir la información en representaciones cada vez más abstractas.
        Más capas = mayor capacidad de capturar no-linealidades hedónicas (p.ej.
        el efecto del piso interactuando con la zona y el tamaño).
      - ReLU: función de activación estándar para capas ocultas. Introduce
        no-linealidad sin saturarse para valores positivos (problema de gradientes
        desvanecientes que tiene tanh/sigmoid).
      - Salida lineal (sin activación): produce un número real sin restricción,
        que es lo que necesitamos para predecir log(price).
      - Elastic Net sobre pesos (kernel_regularizer): penaliza pesos grandes,
        reduciendo el sobreajuste a los patrones locales del ruido espacial.
      - Dropout: apaga aleatoriamente neuronas durante el entrenamiento,
        forzando a la red a aprender representaciones redundantes y robustas.
      - MSE como pérdida: diferenciable → el optimizador puede calcular gradientes.
        MAE como métrica: directamente interpretable en COP, coherente con Kaggle.
    """
    # Regularizador Elastic Net compartido por las capas 1 y 2
    reg = regularizers.l1_l2(l1=L1_REG, l2=L2_REG)

    model = keras.Sequential([
        # Capa 1: 256 neuronas con ReLU y regularización Elastic Net
        layers.Dense(256, activation="relu", kernel_regularizer=reg,
                     input_shape=(n_features,), name="capa_1"),
        # Dropout 30%: en cada mini-batch se apagan 0.3*256 ≈ 77 neuronas al azar
        layers.Dropout(DROPOUT_1, seed=SEED, name="dropout_1"),

        # Capa 2: 128 neuronas (mitad de la anterior → compresión de representación)
        layers.Dense(128, activation="relu", kernel_regularizer=reg,
                     name="capa_2"),
        layers.Dropout(DROPOUT_2, seed=SEED, name="dropout_2"),

        # Capa 3: 64 neuronas sin regularización explícita (ya viene comprimida)
        layers.Dense(64, activation="relu", name="capa_3"),

        # Capa de salida: 1 neurona, activación lineal → predice log(price)
        layers.Dense(1, name="salida"),
    ], name=f"MLP_{MODEL_ID}")

    model.compile(
        optimizer=keras.optimizers.Adam(),   # Adam: ajusta la tasa de aprendizaje por parámetro
        loss="mse",       # función de pérdida: MSE, diferenciable → guía el backprop
        metrics=["mae"],  # métrica de monitoreo: MAE, interpretable en COP
    )
    return model


# =============================================================================
# SECCIÓN 5: CALLBACKS — EARLY STOPPING Y REDUCCIÓN DE LR
# =============================================================================

def make_callbacks() -> list:
    """
    Define los callbacks que controlan el entrenamiento:

    EarlyStopping:
      Monitorea val_loss (MSE en el conjunto de validación). Si no mejora
      durante PATIENCE épocas consecutivas, detiene el entrenamiento y
      restaura los pesos del epoch con menor val_loss.
      → Evita sobreajuste sin necesidad de fijar épocas a priori.
      → El 'patience' actúa como un trade-off: muy bajo = para prematuramente,
        muy alto = paga el costo del sobreajuste antes de parar.

    ReduceLROnPlateau:
      Si val_loss no mejora en 7 épocas, reduce la tasa de aprendizaje a la
      mitad. Permite escapar de mínimos planos que detienen el progreso antes
      de que Early Stopping se active.
      → min_lr=1e-6 evita que la tasa de aprendizaje caiga a cero.
    """
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",          # observar la pérdida en validación
            patience=PATIENCE,           # épocas de tolerancia sin mejora
            restore_best_weights=True,   # al terminar, usar los pesos del mejor epoch
            verbose=0,                   # sin output desde hilos de TF (evita deadlock con _Tee)
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,      # nueva LR = LR × 0.5
            patience=7,      # esperar 7 épocas sin mejora antes de reducir
            min_lr=1e-6,     # cota inferior para la tasa de aprendizaje
            verbose=0,
        ),
    ]


# =============================================================================
# SECCIÓN 6: ENTRENAMIENTO PRINCIPAL
# =============================================================================

def entrenar(X_sc: np.ndarray,
             y: np.ndarray) -> tuple[keras.Model, keras.callbacks.History]:
    """
    Entrena la red con validation_split=0.2 y Early Stopping.

    Por qué validation_split y no un fold del CV:
      Con ~38 000 observaciones, reservar el 20% (≈7 600) para validación
      es estadísticamente suficiente para estimar el error de generalización.
      Por la Ley de los Grandes Números, el MAE en esa muestra converge al
      MAE real con ruido del orden O(1/√7600) ≈ 1%, que es aceptable.
      El CV completo sería más preciso pero mucho más costoso (5 redes en
      lugar de 1). Esta es la misma decisión que toma el cuaderno del curso.

    Importante: el X_sc ya viene estandarizado con el scaler ajustado solo
    sobre train. Keras hace el split interno de validación ANTES de usar esos
    datos para actualizar pesos → no hay data leakage.
    """
    model = construir_modelo(X_sc.shape[1])
    model.summary()   # imprime la arquitectura con el número de parámetros

    print(f"\n  Entrenando (máx {EPOCHS} épocas, "
          f"early stopping patience={PATIENCE})...")
    history = model.fit(
        X_sc, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,   # reserva el 20% final del array para validación
        callbacks=make_callbacks(),
        verbose=0,   # silencioso época a época; Early Stopping imprime su propio mensaje
    )
    epochs_run = len(history.history["loss"])
    best_val   = min(history.history["val_loss"])
    # Este print lo hace el hilo principal (no TF) → sin riesgo de deadlock
    print(f"  Early Stopping: detenido en época {epochs_run}/{EPOCHS} | "
          f"mejor val_loss (MSE): {best_val:.6f} | "
          f"mejor val_MAE: {min(history.history['val_mae']):.6f}")
    return model, history


# =============================================================================
# SECCIÓN 7: CV ESPACIAL
# =============================================================================

def cv_espacial(X_raw: np.ndarray, y: np.ndarray,
                lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Evalúa la generalización geográfica de la red con CV leave-one-block-out.

    Por qué es importante para las redes (01_SpatialData.ipynb):
      Las redes neuronales tienen mayor capacidad que el Elastic Net
      (>200k parámetros vs. ~40 coeficientes). Esta capacidad extra les
      permite memorizar patrones locales del ruido espacial más agresivamente.
      Como resultado, el sesgo de optimismo del CV aleatorio es MAYOR para
      redes que para modelos lineales: la brecha entre CV aleatorio y CV
      espacial es más grande, y predecir en Chapinero es más difícil.

    Implementación:
      - Bogotá se divide en SPATIAL_GRID×SPATIAL_GRID bloques con qcut.
      - Los bloques se agrupan en 5 folds espaciales.
      - En cada fold: entrenamos una red desde cero (mismos hiperparámetros)
        sobre los bloques de entrenamiento y evaluamos en el bloque retenido.
      - Cada red re-entrena su propio StandardScaler solo sobre train
        (anti data-leakage estricto).
      - El MAE medio sobre los 5 folds es el proxy del MAE en Chapinero.
    """
    print(f"\n{'='*60}")
    print(f"  CV ESPACIAL — cuadrícula {SPATIAL_GRID}×{SPATIAL_GRID}")
    print(f"{'='*60}")

    G = SPATIAL_GRID
    # qcut divide cada coordenada en G cuantiles de igual tamaño (misma cantidad
    # de observaciones en cada bin, no el mismo ancho geográfico)
    lat_bins = pd.qcut(lat, G, labels=False, duplicates="drop")
    lon_bins = pd.qcut(lon, G, labels=False, duplicates="drop")
    # ID de celda: entero único por combinación (lat_bin, lon_bin)
    celda  = lat_bins * G + lon_bins
    # Agrupar las celdas en 5 folds espaciales
    grupos = np.array_split(np.unique(celda), 5)

    maes_log: list[float] = []
    maes_cop: list[float] = []

    for i, grp in enumerate(grupos):
        mask_val = np.isin(celda, grp)   # observaciones en el bloque de validación
        mask_tr  = ~mask_val              # resto va a entrenamiento
        if mask_val.sum() == 0 or mask_tr.sum() == 0:
            continue

        # Estandarizar: el scaler se ajusta solo sobre los datos de entrenamiento
        sc    = StandardScaler()
        X_tr  = sc.fit_transform(X_raw[mask_tr])
        X_val = sc.transform(X_raw[mask_val])   # aplicar la misma escala al val
        y_tr, y_val = y[mask_tr], y[mask_val]

        # Entrenar una red nueva para este fold (misma arquitectura e hiperparámetros)
        m = construir_modelo(X_tr.shape[1])
        m.fit(
            X_tr, y_tr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.15,   # 15% interno para que Early Stopping funcione
            callbacks=make_callbacks(),
            verbose=0,
        )
        y_hat = m.predict(X_val, verbose=0).flatten()

        mae_log = float(mean_absolute_error(y_val, y_hat))
        # Para el MAE en COP: deshacemos el log en ambas series antes de comparar
        mae_cop = float(mean_absolute_error(np.exp(y_val), np.exp(y_hat)))
        maes_log.append(mae_log)
        maes_cop.append(mae_cop)
        print(f"  Fold {i+1}/5: MAE_log={mae_log:.5f}, "
              f"MAE_COP={mae_cop/1e6:.1f}M")

    arr_log = np.array(maes_log)
    arr_cop = np.array(maes_cop)
    print(f"\n  MAE_log medio: {arr_log.mean():.5f} ± {arr_log.std():.5f}")
    print(f"  MAE_COP medio: {arr_cop.mean()/1e6:.1f}M "
          f"± {arr_cop.std()/1e6:.1f}M")
    return arr_log, arr_cop


# =============================================================================
# SECCIÓN 8: GRÁFICOS DE DIAGNÓSTICO
# =============================================================================

def plot_historia(history: keras.callbacks.History, epochs_run: int) -> None:
    """
    Grafica las curvas de pérdida (MSE) y métrica (MAE) durante el entrenamiento.

    Cómo interpretar:
      - Si Train ≈ Validación → el modelo generaliza bien (no hay overfitting).
      - Si Train << Validación y la brecha crece con las épocas → overfitting.
        En ese caso, Dropout o mayor regularización L1/L2 ayudarían.
      - El epoch donde Early Stopping restaura los pesos es el mínimo de la
        curva de validación.
    """
    ep = range(1, epochs_run + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel izquierdo: pérdida (MSE) — lo que el optimizador minimiza
    axes[0].plot(ep, history.history["loss"],     label="Train",      color="#3498db")
    axes[0].plot(ep, history.history["val_loss"], label="Validación", color="#e74c3c")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("MSE — log(price)")
    axes[0].set_title(f"Pérdida (MSE) — {MODEL_ID}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel derecho: MAE — lo que reportamos e interpretamos
    axes[1].plot(ep, history.history["mae"],     label="Train",      color="#3498db")
    axes[1].plot(ep, history.history["val_mae"], label="Validación", color="#e74c3c")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MAE — log(price)")
    axes[1].set_title(f"Métrica MAE — {MODEL_ID}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Curvas de entrenamiento — {MODEL_ID}\n"
                 f"(Early Stopping en época {epochs_run})", y=1.02)
    plt.tight_layout()
    fig.savefig(DIR_MODEL / "historia_entrenamiento.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/historia_entrenamiento.png")


def plot_residuos(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Diagnóstico de residuos sobre el conjunto de entrenamiento completo.

    Panel izquierdo — Residuos vs. Ajustados:
      Una nube sin patrón alrededor de 0 indica ausencia de sesgo sistemático.
      Un patrón en forma de abanico → heterocedasticidad (errores más grandes
      para precios altos). Una curva → no-linealidad no capturada.

    Panel derecho — Histograma de residuos:
      Distribución aproximadamente simétrica alrededor de 0 → el modelo no
      sobreestima ni subestima sistemáticamente. Relevante para la discusión
      de negocio: sobreestimar = la empresa paga más de lo que vale el inmueble.
    """
    res = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, res, alpha=0.15, s=4, color="#9b59b6")
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("log(price) predicho")
    axes[0].set_ylabel("Residuo  (log)")
    axes[0].set_title(f"Residuos vs. Ajustados — {MODEL_ID}")

    axes[1].hist(res, bins=60, color="#9b59b6", edgecolor="white")
    axes[1].set_xlabel("Residuo  (log)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title(f"Distribución residuos — media={res.mean():.4f}")

    plt.tight_layout()
    fig.savefig(DIR_MODEL / "residuos.png", dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/residuos.png")


def plot_cv_espacial(maes_log: np.ndarray, maes_cop: np.ndarray,
                     mae_val_log: float) -> None:
    """
    Compara el MAE de cada fold espacial contra el MAE de validación interna.

    La línea azul punteada es el MAE de validation_split=0.2 (optimista).
    Las barras son el MAE real de cada bloque geográfico no visto.
    Si las barras son consistentemente más altas que la línea → sesgo de
    optimismo confirmado: la red generaliza peor de lo que parece.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    folds = np.arange(1, len(maes_log) + 1)

    # Panel izquierdo: MAE en log-escala (escala del modelo)
    axes[0].bar(folds, maes_log, color="#9b59b6", edgecolor="white")
    axes[0].axhline(mae_val_log, color="blue", lw=1.5, linestyle="--",
                    label=f"Val. interno ({mae_val_log:.4f})")
    axes[0].set_xlabel("Fold espacial")
    axes[0].set_ylabel("MAE log(price)")
    axes[0].set_title(f"CV espacial — {MODEL_ID} (log)")
    axes[0].legend(fontsize=8)

    # Panel derecho: MAE en millones de COP (escala de negocio)
    axes[1].bar(folds, maes_cop / 1e6, color="#9b59b6", edgecolor="white")
    axes[1].axhline(maes_cop.mean() / 1e6, color="black", lw=1,
                    linestyle="--",
                    label=f"Media={maes_cop.mean()/1e6:.0f}M")
    axes[1].set_xlabel("Fold espacial")
    axes[1].set_ylabel("MAE (millones COP)")
    axes[1].set_title(f"CV espacial — {MODEL_ID} (COP)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(DIR_MODEL / "cv_espacial.png", dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/cv_espacial.png")


# =============================================================================
# SECCIÓN 9: SUBMISSION PARA KAGGLE
# =============================================================================

def generar_submission(test: pd.DataFrame, y_pred_log: np.ndarray,
                       epochs_run: int, total_params: int) -> str:
    """
    Genera el CSV de predicciones en el formato exacto del template de Kaggle.

    Puntos importantes:
      - exp(y_pred_log) convierte de log(COP) a COP.
        Kaggle evalúa el MAE en precios originales (pesos colombianos).
      - El merge con el template garantiza el orden correcto de property_id,
        independiente del orden en nuestro test.csv.
      - El nombre incluye épocas y parámetros para identificar el submission.
    """
    submission = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),   # revertir log → precios en COP
    })
    template = pd.read_csv(
        Path("/Users/macbook/Downloads/uniandes-bdml-202610-ps-3") /
        "submission_template.csv"
    )
    # Left join: el template define el orden; submission aporta los precios
    submission = (template[["property_id"]]
                  .merge(submission, on="property_id", how="left"))

    sub_name = f"submission_{MODEL_ID}_ep{epochs_run}_p{total_params//1000}k.csv"
    submission.to_csv(SUBMISSIONS / sub_name, index=False)

    print(f"\n  Submission guardado: 03_submissions/{sub_name}")
    print(f"  Filas: {len(submission):,} | "
          f"price media: {submission['price'].mean()/1e6:.1f}M COP")
    return sub_name


# =============================================================================
# SECCIÓN 10: REGISTRO EN model_registry.xlsx
# =============================================================================

def registrar(sub_name: str, n_features: int, total_params: int,
              epochs_run: int,
              mae_train_log: float, mae_train_cop: float,
              mae_val_log: float,
              mae_esp_log: float, mae_esp_cop: float) -> None:
    """
    Añade (o actualiza) la fila de MODEL_ID en el registro central de modelos.

    El registro permite comparar todos los modelos del equipo en un solo lugar.
    La columna kaggle_public_MAE se llena manualmente después de subir el CSV.
    Si MODEL_ID ya existe, la fila anterior se reemplaza (no se duplica).

    Columnas clave para el análisis comparativo:
      - cv_mae_log:  MAE del validation_split=0.2 (optimista, como referencia)
      - esp_mae_log: MAE del CV espacial (honesto, proxy de Chapinero)
      - El Δ entre ambos (notas) mide el sesgo de optimismo de la NN
    """
    delta_espacial = mae_esp_log - mae_val_log   # sesgo de optimismo

    nueva = {
        "model_id":          MODEL_ID,
        "fecha":             str(date.today()),
        "autor":             AUTOR,
        "algoritmo":         "NeuralNetwork",
        "n_features":        n_features,
        "n_params":          total_params,    # parámetros totales de la red
        "epochs_run":        epochs_run,      # épocas reales (Early Stopping)
        "cv_folds":          f"val_split={VAL_SPLIT}",
        # --- MAE de validación interna (optimista) ---
        "cv_mae_log":        round(mae_val_log,   5),
        "cv_mae_cop_M":      None,   # no calculable directamente de history
        # --- MAE de CV espacial (honesto, proxy Chapinero) ---
        "esp_mae_log":       round(mae_esp_log,   5),
        "esp_mae_cop_M":     round(mae_esp_cop / 1e6, 2),
        # --- MAE sobre train (mide sobreajuste si está muy por debajo del CV) ---
        "train_mae_log":     round(mae_train_log, 5),
        "kaggle_public_MAE": None,   # ← llenar manualmente tras subir a Kaggle
        # --- hiperparámetros (para comparar con EN_001) ---
        "l1_ratio":          None,   # no aplica (Elastic Net es el regularizador de pesos)
        "alpha":             None,
        # --- contexto de features y regularización ---
        "features_grupos":  (f"structural={len(STRUCTURAL)}, "
                              f"text={len(TEXT)}, osm={len(OSM)}, "
                              f"es_apartamento"),
        "arquitectura":      f"p→256(Drop{DROPOUT_1})→128(Drop{DROPOUT_2})→64→1",
        "reg_l1_l2":         f"l1={L1_REG}, l2={L2_REG}",
        "early_stopping":    f"patience={PATIENCE}",
        "spatial_grid":      f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "submission_file":   sub_name,
        "notas": (
            f"MLP 256→128→64→1 con ReLU y Dropout. "
            f"Regularización Elastic Net en capas 1-2 (l1={L1_REG},l2={L2_REG}). "
            f"EarlyStopping patience={PATIENCE}. "
            f"Épocas efectivas: {epochs_run}/{EPOCHS}. "
            f"MAE_log val.interno={mae_val_log:.5f} (optimista). "
            f"MAE_log CV espacial={mae_esp_log:.5f} (honesto). "
            f"Sesgo Δ={delta_espacial:+.5f}."
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
        # Ajustar ancho de columnas al contenido más largo de cada una
        ws = writer.sheets["registry"]
        for col in ws.columns:
            max_len = max(
                len(str(col[0].value)) if col[0].value else 0,
                *(len(str(c.value)) if c.value else 0 for c in col[1:]),
            )
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)

    print(f"\n  Registro actualizado: 02_outputs/model_registry.xlsx")
    print(f"  Modelos en el registry: {len(df_reg)}")


# =============================================================================
# SECCIÓN 11: ENTRY POINT
# =============================================================================

def main() -> None:
    print(f"{'='*60}")
    print(f"  RED NEURONAL — {MODEL_ID}")
    print(f"{'='*60}")

    # ── [1/7] Carga ───────────────────────────────────────────────────────────
    print("\n[1/7] Cargando datos...")
    train, test = cargar_datos()
    # Transformar el precio a log(price): normaliza la distribución y hace el
    # MAE en log-escala equivalente al error porcentual relativo.
    y_train_log = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    # ── [2/7] Features ────────────────────────────────────────────────────────
    print("\n[2/7] Construyendo features...")
    X_train_df   = construir_features(train)
    feature_cols = list(X_train_df.columns)   # guardamos el orden para alinear test
    X_test_df    = construir_features(test, fit_cols=feature_cols)
    print(f"  Features: {len(feature_cols)} "
          f"({len(STRUCTURAL)} estructurales + {len(TEXT)} texto + "
          f"{len(OSM)} OSM + es_apartamento)")

    # ── [3/7] Estandarización ─────────────────────────────────────────────────
    # Las redes neuronales son muy sensibles a la escala de las features:
    # features con rangos muy distintos hacen que el gradiente favorezca unas
    # dimensiones sobre otras. StandardScaler: x_scaled = (x - μ) / σ.
    # CRÍTICO: el scaler se ajusta SOLO sobre train y se aplica a test.
    # Si se ajustara sobre test, estaríamos usando información del futuro
    # para transformar los datos de entrenamiento (data leakage).
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_df.values)   # ajustar Y transformar
    X_test_sc  = scaler.transform(X_test_df.values)        # solo transformar

    # ── [4/7] Entrenamiento principal ─────────────────────────────────────────
    print("\n[3/7] Entrenando red neuronal (validation_split=0.2)...")
    model, history = entrenar(X_train_sc, y_train_log)

    epochs_run    = len(history.history["loss"])
    # Mejor MAE de validación durante el entrenamiento (en el epoch restaurado)
    mae_val_log   = float(min(history.history["val_mae"]))
    total_params  = int(model.count_params())

    # MAE sobre TODO el train (modo inferencia: Dropout desactivado)
    y_pred_train  = model.predict(X_train_sc, verbose=0).flatten()
    mae_train_log = float(mean_absolute_error(y_train_log, y_pred_train))
    mae_train_cop = float(
        mean_absolute_error(np.exp(y_train_log), np.exp(y_pred_train))
    )
    print(f"  Parámetros totales: {total_params:,}")
    print(f"  MAE train (log): {mae_train_log:.5f} | "
          f"MAE train (COP): {mae_train_cop/1e6:.1f}M")
    # Si MAE_train << MAE_val → sobreajuste → aumentar Dropout o regularización

    # ── [5/7] CV espacial ─────────────────────────────────────────────────────
    # Usamos X_train_df.values (sin escalar) porque cv_espacial escala
    # internamente en cada fold para evitar data leakage entre folds.
    print("\n[4/7] CV espacial...")
    maes_esp_log, maes_esp_cop = cv_espacial(
        X_train_df.values, y_train_log,
        train["lat"].values, train["lon"].values,
    )

    # ── [6/7] Gráficos ────────────────────────────────────────────────────────
    print("\n[5/7] Generando gráficos de diagnóstico...")
    plot_historia(history, epochs_run)
    plot_residuos(y_train_log, y_pred_train)
    plot_cv_espacial(maes_esp_log, maes_esp_cop, mae_val_log)

    # ── [7/7] Submission + Registro ───────────────────────────────────────────
    print("\n[6/7] Generando submission...")
    y_pred_test = model.predict(X_test_sc, verbose=0).flatten()
    sub_name    = generar_submission(test, y_pred_test, epochs_run, total_params)

    print("\n[7/7] Actualizando registro...")
    registrar(
        sub_name=sub_name,
        n_features=len(feature_cols),
        total_params=total_params,
        epochs_run=epochs_run,
        mae_train_log=mae_train_log,
        mae_train_cop=mae_train_cop,
        mae_val_log=mae_val_log,
        mae_esp_log=float(maes_esp_log.mean()),
        mae_esp_cop=float(maes_esp_cop.mean()),
    )

    # ── Resumen final ─────────────────────────────────────────────────────────
    delta = float(maes_esp_log.mean()) - mae_val_log
    print(f"\n{'='*60}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"{'='*60}")
    print(f"  Arquitectura: p → 256(Drop{DROPOUT_1}) → "
          f"128(Drop{DROPOUT_2}) → 64 → 1")
    print(f"  Parámetros  : {total_params:,}")
    print(f"  Épocas      : {epochs_run}/{EPOCHS}")
    print(f"  MAE val int (log): {mae_val_log:.5f}   ← optimista")
    print(f"  MAE CV esp  (log): {maes_esp_log.mean():.5f}   "
          f"Δ={delta:+.5f}  ← proxy Chapinero")
    print(f"  MAE train   (log): {mae_train_log:.5f}")
    print(f"  Submission  : 03_submissions/{sub_name}")
    print(f"  Registry    : 02_outputs/model_registry.xlsx")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    main()
