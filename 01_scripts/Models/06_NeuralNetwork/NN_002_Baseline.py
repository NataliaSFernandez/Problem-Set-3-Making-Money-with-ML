"""
Red Neuronal (PyTorch) para predicción de log(price) de vivienda en Bogotá
===========================================================================
Problem Set 3 — MECA 4107 (Big Data & ML para Economía Aplicada)

Por qué PyTorch en lugar de TensorFlow
---------------------------------------
  TF 2.21 + Keras 3 generan un deadlock en macOS ARM64 (Apple Silicon):
  los callbacks de Python (EarlyStopping, ReduceLROnPlateau) son objetos
  Python que TF intenta invocar desde hilos internos de C++. En macOS ARM,
  ese cruce de hilos intenta adquirir el GIL de Python mientras el hilo
  principal ya lo retiene esperando a TF → interbloqueo permanente.
  PyTorch usa MPS (Metal Performance Shaders) como backend nativo en Apple
  Silicon y el bucle de entrenamiento corre completamente en Python,
  sin callbacks de C++, eliminando el problema.

Arquitectura — MLP feedforward para regresión hedónica
-------------------------------------------------------
  Entrada  : p features estandarizadas (estructurales + OSM + texto)
  Capa 1   : Linear(p→256) + ReLU + Dropout(0.3)
  Capa 2   : Linear(256→128) + ReLU + Dropout(0.2)
  Capa 3   : Linear(128→64)  + ReLU
  Salida   : Linear(64→1)   ← regresión continua, sin activación

  Pérdida  : MSE  — diferenciable en todos sus puntos → backprop puede
             calcular gradientes exactos en cada iteración.
  Métrica  : MAE  — interpretable en COP, es el criterio de Kaggle.
  Optimizer: Adam — adapta la tasa de aprendizaje por parámetro usando
             estimaciones del primer y segundo momento del gradiente.

Regularización (cuatro mecanismos combinados)
---------------------------------------------
  1. L2 via weight_decay en Adam: λ₂·Σ W²
     Adam ya tiene este término integrado: en cada paso reduce cada peso
     en proporción a su magnitud. Equivale a Ridge en regresión lineal.
     Efecto: encoge los pesos hacia 0 uniformemente → reduce varianza.

  2. L1 sobre pesos (manual en el loss): λ₁·Σ|W|
     Suma del valor absoluto de cada peso, añadida a la pérdida MSE.
     El gradiente de |W| es sign(W) = ±1 por peso, lo que empuja cada
     peso hacia 0 independientemente de su magnitud → produce sparsity
     (algunos pesos exactamente 0). Equivale a LASSO en regresión lineal.
     Juntos L1+L2 forman Elastic Net sobre los pesos de la red.

  3. Dropout: en cada mini-batch, apaga aleatoriamente una fracción φ de
     neuronas (pone su activación a 0). Esto obliga a la red a aprender
     representaciones redundantes → ninguna neurona puede especializarse
     en memorizar una muestra específica del ruido del train.
     φ=0.3 en capa 1 (más agresivo, más parámetros que encadenar),
     φ=0.2 en capa 2 (más suave, representación ya comprimida).

  4. Early Stopping: monitorea val_loss al final de cada época. Si no
     mejora durante PATIENCE épocas consecutivas, detiene el entrenamiento
     y restaura los pesos del epoch con menor val_loss (restore_best_weights).
     Evita sobreajuste sin fijar épocas a priori: la red para sola cuando
     empieza a memorizar el train en lugar de generalizar.

Por qué predecir log(price) en lugar de price
----------------------------------------------
  Los precios tienen distribución sesgada a la derecha (cola larga de
  propiedades muy caras). Al usar log(price) la distribución se vuelve
  más simétrica y casi normal, lo que facilita el entrenamiento: el MSE
  en log-escala trata errores proporcionales por igual sin importar si la
  propiedad vale 200M o 2B COP. Además, el MAE en log-escala penaliza
  errores relativos, que es más apropiado que errores absolutos cuando
  los precios varían en órdenes de magnitud.

CV espacial (01_SpatialData.ipynb)
-----------------------------------
  El train cubre toda Bogotá; el test (Kaggle) es solo Chapinero.
  Los precios de inmuebles vecinos están correlados espacialmente:
    Cov(ε(sᵢ), ε(sⱼ)) > 0  si sᵢ y sⱼ están cerca.
  Con CV aleatorio estándar, el punto de validación sᵢ suele tener vecinos
  en el train → el modelo "aprende" el ruido local y lo transfiere → el
  error de validación es artificialmente bajo (sesgo de optimismo).
  Con CV espacial leave-one-block-out, todos los vecinos de sᵢ están en
  el train → error de validación honesto, proxy real de Chapinero.
  Δ = MAE_espacial − MAE_interno cuantifica el sesgo de optimismo.

Pipeline
--------
1. Carga  train_final.csv / test_final.csv
2. Misma ingeniería de features que 01_elastic_net.py (comparabilidad directa)
3. StandardScaler ajustado SOLO sobre train (evita data leakage)
4. Entrenamiento con val split manual + Early Stopping en bucle Python
5. CV espacial leave-one-block-out por cuadrícula geográfica 5×5
6. Diagnósticos: curvas de entrenamiento, residuos, CV espacial
7. Submission → 03_submissions/
8. Registro automático → 02_outputs/model_registry.xlsx

Outputs
-------
  02_outputs/Models/NeuralNetwork/NN_002/historia_entrenamiento.png
  02_outputs/Models/NeuralNetwork/NN_002/residuos.png
  02_outputs/Models/NeuralNetwork/NN_002/cv_espacial.png
  03_submissions/submission_NN_002_*.csv
  02_outputs/model_registry.xlsx
"""

# =============================================================================
# SECCIÓN 0: IMPORTACIONES
# =============================================================================

import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend sin pantalla: evita intentar abrir una ventana
                        # gráfica en un script (falla en servidores sin display)
import matplotlib.pyplot as plt
import numpy as np
import openpyxl          # noqa: F401 — requerido internamente por pd.ExcelWriter
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Semillas para reproducibilidad. Las redes neuronales tienen múltiples fuentes
# de aleatoriedad: inicialización de pesos, orden de mini-batches, Dropout.
# Fijamos tanto numpy (para los splits y operaciones fuera de PyTorch) como
# torch (para inicialización de pesos y Dropout interno).
SEED = 101010
np.random.seed(SEED)
torch.manual_seed(SEED)

# Selección automática del backend de cómputo:
#   - MPS  (Metal Performance Shaders): GPU de Apple Silicon → más rápido
#   - CUDA : GPU de NVIDIA → más rápido en servidores Linux
#   - CPU  : fallback universal → más lento pero siempre disponible
# En macOS con M1/M2/M3, MPS está disponible y es significativamente más
# rápido que CPU para operaciones de álgebra lineal densa (forward/backward).
DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))


# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================
# Centralizar hiperparámetros aquí hace el script fácil de iterar:
# cambiar MODEL_ID a "NN_003" y ajustar valores genera una nueva entrada
# en el registry sin modificar el resto del código.

AUTOR    = "Jonathan"
MODEL_ID = "06_NN"

# ── Hiperparámetros del entrenamiento ─────────────────────────────────────────
EPOCHS     = 200   # máximo de épocas; Early Stopping detiene antes si converge
BATCH_SIZE = 512   # tamaño del mini-batch. Potencia de 2 para eficiencia en
                   # MPS/CUDA. Más grande = gradiente menos ruidoso pero más
                   # memoria; 512 es un buen balance para ~38k muestras.
VAL_SPLIT  = 0.2   # fracción del train reservada para monitoreo interno.
                   # Con ~38k obs., 20% = ~7600 muestras de validación,
                   # suficiente para estimar el error de generalización con
                   # ruido del orden O(1/√7600) ≈ 1% (Ley de Grandes Números).
PATIENCE   = 15    # épocas sin mejora en val_loss antes de activar Early
                   # Stopping. Trade-off: muy bajo = para prematuramente antes
                   # de que la red haya explorado el espacio; muy alto = paga
                   # el costo del sobreajuste esperando demasiado.
LR         = 1e-3  # tasa de aprendizaje inicial de Adam. 1e-3 es el default
                   # recomendado por los autores del algoritmo (Kingma & Ba 2015)
                   # y funciona bien para la mayoría de redes feedforward.

# ── Regularización Elastic Net sobre los pesos ────────────────────────────────
L1_REG = 1e-4   # λ₁: intensidad de la penalización L1.
                # Empuja cada peso hacia exactamente 0 (sparsity).
                # Gradiente: ∂(λ₁·Σ|W|)/∂W = λ₁·sign(W) → ±λ₁ por peso.
L2_REG = 1e-4   # λ₂: intensidad de la penalización L2 (weight_decay en Adam).
                # Encoge los pesos proporcionalmente a su magnitud.
                # Gradiente: ∂(λ₂·Σ W²)/∂W = 2λ₂·W → mayor penalización
                # para pesos grandes.

# ── Dropout ───────────────────────────────────────────────────────────────────
DROPOUT_1 = 0.3   # fracción de neuronas apagadas en la capa 1 (256 neuronas).
                  # En cada mini-batch, 0.3×256 ≈ 77 neuronas se ponen a 0.
                  # Más agresivo en la primera capa porque tiene más parámetros
                  # y por tanto más capacidad de memorizar ruido.
DROPOUT_2 = 0.2   # fracción de neuronas apagadas en la capa 2 (128 neuronas).
                  # Más suave porque la representación ya viene comprimida
                  # desde la capa anterior.

SPATIAL_GRID = 5  # lado de la cuadrícula geográfica: 5×5 = 25 bloques.
                  # Cada bloque agrupa inmuebles en una zona geográfica similar.
                  # 5 folds espaciales = 5 combinaciones de bloques para CV.

# ── Rutas ─────────────────────────────────────────────────────────────────────
# Path(__file__) da la ruta absoluta de este script.
# .parent.parent.parent.parent sube cuatro niveles: 06_NeuralNetwork/ → Models/ → 01_scripts/ → raíz.
BASE        = Path(__file__).parent.parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "NeuralNetwork" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

# Crear carpetas si no existen (parents=True crea intermedias, exist_ok=True
# no falla si ya existen).
for d in [SUBMISSIONS, DIR_MODEL, BASE / "02_outputs"]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECCIÓN 2: DEFINICIÓN DE GRUPOS DE FEATURES
# =============================================================================
# Las mismas features que 01_elastic_net.py. Esto es fundamental para que la
# comparación entre ElasticNet y Red Neuronal sea limpia: si ambos modelos usan
# exactamente las mismas variables, cualquier diferencia en desempeño se debe
# al algoritmo, no a la ingeniería de features.

STRUCTURAL = [
    # Atributos físicos del inmueble — variables "clásicas" del modelo hedónico
    # (Rosen 1974): el precio es la suma de los precios implícitos de cada
    # característica del bien.
    "surface_total",    # superficie total en m²: variable de mayor poder
                        # explicativo en modelos hedónicos de vivienda
    "surface_covered",  # superficie cubierta (excluye terrazas y jardines)
    "rooms",            # número total de habitaciones
    "bedrooms",         # número de alcobas (subconjunto de rooms)
    "bathrooms",        # número de baños
    "month",            # mes de publicación → captura estacionalidad del
                        # mercado (ej. más ventas en enero/julio)
    "year",             # año de publicación → captura tendencia temporal
                        # (inflación, ciclos del mercado inmobiliario)
]

TEXT = [
    # Variables extraídas del texto libre del anuncio (título + descripción).
    # Capturan atributos que el vendedor menciona pero que no están en los
    # campos estructurados de Properati. La extracción usa regex y TF-IDF.
    "remodelado",       # 1 si el texto menciona remodelación → prima de precio
    "vista_panoramica", # 1 si menciona vista panorámica → amenidad de lujo
    "deposito",         # 1 si menciona depósito/bodega → espacio adicional
    "conjunto_cerrado", # 1 si menciona conjunto cerrado o seguridad privada
    "balcon_terraza",   # 1 si menciona balcón o terraza → espacio exterior
    "tfidf_premium",    # score TF-IDF de términos de lujo (penthouse, jacuzzi,
                        # mármol, etc.) → proxy continuo de calidad del inmueble
    "parqueaderos_txt", # número de parqueaderos mencionados en el texto
    "piso_txt",         # número de piso extraído del texto (influye en precio:
                        # pisos altos suelen ser más caros en apartamentos)
    "gimnasio",         # 1 si menciona gimnasio o zona fitness
    "amenidades",       # score agregado de amenidades (piscina, sauna, etc.)
    "num_amenidades",   # conteo de tipos distintos de amenidades mencionadas
]

OSM = [
    # Variables de localización construidas con OpenStreetMap.
    # Rosen (1974): el precio hedónico incluye también los servicios y
    # externalidades del entorno (accesibilidad, amenidades urbanas, etc.).
    "dist_cbd_km",          # distancia al CBD (centro financiero de Bogotá)
                            # en km: mayor distancia → menor precio esperado
    "dist_transmilenio_m",  # distancia a la estación de TransMilenio más
                            # cercana en metros: accesibilidad al transporte
    "dist_via_arterial_m",  # distancia a vía arterial principal: accesibilidad
                            # y también proxy de ruido/contaminación
    "dist_hospital_m",      # distancia al hospital más cercano: accesibilidad
                            # a servicios de salud
    "dist_centro_com_m",    # distancia al centro comercial más cercano:
                            # proxy de oferta de comercio y entretenimiento
    "dist_parque_m",        # distancia al parque más cercano: amenidad verde,
                            # positivamente valorada en el mercado
    "n_restaurantes_500m",  # número de restaurantes en radio de 500 m:
                            # indicador de densidad urbana y vitalidad del barrio
    "n_bancos_500m",        # número de bancos en radio de 500 m:
                            # accesibilidad a servicios financieros
    "walkability_score",    # índice de caminabilidad (Walk Score): combina
                            # distancias a servicios cotidianos en un solo número
    "densidad_vial",        # metros de vía por km² en el entorno del inmueble:
                            # proxy de conectividad y densidad urbana
]


# =============================================================================
# SECCIÓN 3: CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los datasets procesados con features OSM y de texto ya integradas."""
    train = pd.read_csv(PROCESSED / "train_final.csv")
    test  = pd.read_csv(PROCESSED / "test_final.csv")
    return train, test


def construir_features(df: pd.DataFrame,
                       fit_cols: list | None = None) -> pd.DataFrame:
    """
    Construye la matriz de features X con las variables del dataset procesado.
    Idéntica a 01_elastic_net.py para garantizar comparabilidad directa entre
    modelos: cualquier diferencia en MAE se debe al algoritmo, no a las features.

    property_type es la única variable categórica. Como solo hay dos categorías
    (Apartamento y Casa), se codifica directamente como una variable binaria:
      es_apartamento = 1 si Apartamento, 0 si Casa.
    Esto es equivalente a get_dummies(drop_first=True) pero sin depender de
    que Pandas genere el nombre de columna dinámicamente.

    El parámetro fit_cols garantiza que el test tenga exactamente las mismas
    columnas en el mismo orden que el train, necesario para que el modelo
    (entrenado con las columnas de train) pueda procesar el test correctamente.
    """
    d = df.copy()

    # Codificación binaria de property_type: 1=Apartamento, 0=Casa.
    # Los datos no tienen nulos en esta columna, así que no hace falta imputar.
    d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)

    # Construir la lista de columnas a usar, filtrando las que no existan
    # en el DataFrame (ej. si en el futuro se elimina una feature del CSV).
    all_cols = STRUCTURAL + TEXT + OSM + ["es_apartamento"]
    all_cols = [c for c in all_cols if c in d.columns]
    X = d[all_cols]

    # Alinear test con train: garantiza el mismo orden de columnas.
    # Si fit_cols ya está definido (llamada desde el test), reordenamos.
    if fit_cols is not None:
        X = X[fit_cols]
    return X


# =============================================================================
# SECCIÓN 4: ARQUITECTURA DE LA RED (nn.Module)
# =============================================================================

class MLP(nn.Module):
    """
    Perceptrón multicapa (MLP) para regresión de log(price).

    Hereda de nn.Module, la clase base de todos los modelos en PyTorch.
    Al heredar, obtenemos automáticamente: registro de parámetros, manejo
    de device (CPU/MPS/CUDA), serialización (save/load), y el método
    .parameters() que el optimizador necesita para actualizar los pesos.

    Arquitectura: p → 256 → 128 → 64 → 1
    ----------------------------------------
    La progresión decreciente (256→128→64) fuerza a la red a comprimir la
    información en representaciones cada vez más abstractas. Esto es análogo
    a un autoencoder: la capa 1 detecta patrones de primer orden (relaciones
    lineales entre features), la capa 2 combina esos patrones, y la capa 3
    produce una representación compacta que la salida convierte en log(price).

    ReLU (Rectified Linear Unit): f(x) = max(0, x)
    -----------------------------------------------
    Es la activación estándar para capas ocultas. Ventajas sobre tanh/sigmoid:
      - No se satura para valores positivos → gradientes no desaparecen
        (problema de vanishing gradients en redes profundas)
      - Computacionalmente simple (una comparación)
      - Produce sparsity: aproximadamente 50% de neuronas en 0 en cada capa

    Salida lineal (sin activación)
    ------------------------------
    La capa de salida produce un número real sin restricción, que es exactamente
    lo que necesitamos para predecir log(price) ∈ ℝ. Si usáramos sigmoid o
    tanh, la salida estaría acotada y no podría representar valores arbitrarios.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        # nn.Sequential aplica las capas en orden, pasando la salida de cada
        # una como entrada de la siguiente. Es equivalente a definir el forward
        # manualmente pero más conciso para arquitecturas lineales.
        self.red = nn.Sequential(
            # Capa 1: transforma las p features en 256 representaciones.
            # Cada neurona aprende una combinación lineal de las p features
            # y luego ReLU introduce no-linealidad.
            nn.Linear(n_features, 256),
            nn.ReLU(),
            # Dropout al 30%: en cada mini-batch apaga ~77 de las 256 neuronas.
            # Durante la inferencia (.eval()) Dropout se desactiva automáticamente
            # y PyTorch escala las activaciones para compensar.
            nn.Dropout(DROPOUT_1),

            # Capa 2: comprime de 256 a 128 dimensiones.
            # La reducción a la mitad fuerza a la red a seleccionar las
            # representaciones más informativas de la capa anterior.
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_2),   # 20%: más suave, representación ya comprimida

            # Capa 3: comprime a 64 dimensiones. Sin Dropout: en este punto
            # la representación es suficientemente abstracta y comprimida
            # que el Dropout no aportaría regularización adicional significativa.
            nn.Linear(128, 64),
            nn.ReLU(),

            # Capa de salida: proyecta las 64 dimensiones a un único número
            # real → predicción de log(price). Sin activación: la salida es
            # una combinación lineal de las 64 neuronas previas.
            nn.Linear(64, 1),
        )
        # Inicializar pesos con Kaiming (He) normal antes de entrenar.
        self._init_pesos()

    def _init_pesos(self) -> None:
        """
        Inicialización de Kaiming (He) para capas con ReLU.

        Con inicialización aleatoria estándar (ej. N(0,1)), la varianza de las
        activaciones crece o decrece exponencialmente con la profundidad, haciendo
        el entrenamiento inestable. Kaiming calcula la varianza óptima por capa:

          W ~ N(0, sqrt(2 / n_in))

        donde n_in es el número de entradas de esa capa. El factor √2 compensa
        el hecho de que ReLU apaga aproximadamente la mitad de las neuronas,
        manteniendo la varianza constante a lo largo de las capas.

        El bias se inicializa en 0 (excepto la capa de salida, que se ajustará
        a la media del target en entrenar() para acelerar la convergencia).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso forward: calcula la predicción dado un batch de features.

        x: tensor de forma (batch_size, n_features)
        salida: tensor de forma (batch_size,) — una predicción por muestra.

        self.red(x) produce forma (batch_size, 1) porque la última capa es
        Linear(64→1). El squeeze(-1) elimina esa última dimensión para que
        la salida sea (batch_size,), compatible con el vector de targets y_t.
        """
        return self.red(x).squeeze(-1)


def penalizacion_l1(model: MLP) -> torch.Tensor:
    """
    Calcula la penalización L1: suma de |W| sobre todos los pesos de la red.

    Solo se penalizan los pesos (weight), NO los bias. El bias es un término
    de intercepción que no debería ser penalizado: penalizarlo lo forzaría
    hacia 0, lo que introduciría sesgo en las predicciones. Esta es la
    convención estándar en regularización de redes neuronales (equivalente
    a kernel_regularizer en Keras, que también excluye los bias por defecto).

    Se usa torch.zeros en lugar de Python sum() para que el acumulador sea
    un Tensor desde el inicio, lo que:
      a) Evita confusión de tipos para el type checker (sum empieza en int 0)
      b) Mantiene el grafo computacional conectado para backpropagation
      c) Asegura que el tensor esté en el device correcto (MPS/CUDA/CPU)
    """
    device = next(model.parameters()).device
    total  = torch.zeros(1, device=device)
    for name, p in model.named_parameters():
        if "weight" in name:   # excluir bias explícitamente
            total = total + p.abs().sum()
    return total


# =============================================================================
# SECCIÓN 5: ENTRENAMIENTO PRINCIPAL
# =============================================================================

def entrenar(X_sc: np.ndarray,
             y: np.ndarray,
             device: torch.device = DEVICE) -> tuple[MLP, dict]:
    """
    Entrena el MLP con un bucle Python puro (sin callbacks de C++).

    Por qué bucle Python en lugar de model.fit() de Keras
    -------------------------------------------------------
    En TF/Keras, model.fit() delega el bucle de entrenamiento a C++, que luego
    llama de vuelta a Python para ejecutar los callbacks (EarlyStopping, etc.).
    En macOS ARM64 ese patrón produce deadlocks. Con PyTorch controlamos el
    bucle directamente en Python: cada epoch, cada batch, cada callback es
    código Python puro que corre en el hilo principal, sin riesgo de deadlock.

    Por qué val split y no fold del CV espacial
    -------------------------------------------
    Con ~38k observaciones, el 20% de validación (≈7600 muestras) es suficiente
    para estimar el error de generalización con error O(1/√7600) ≈ 1%.
    El CV completo sería más preciso pero requiere entrenar 5 modelos en lugar
    de 1. La validación interna sirve para early stopping, no para reportar
    el error final: ese rol lo cumple el CV espacial (sección 6).

    Parámetros
    ----------
    X_sc : matriz de features ya estandarizada (StandardScaler ajustado en train)
    y    : vector de log(price) del conjunto de entrenamiento
    device: backend de cómputo (MPS, CUDA o CPU)

    Retorna
    -------
    model  : red entrenada con los pesos del mejor epoch (restore_best_weights)
    history: dict con listas de loss/val_loss/mae/val_mae por época
    """
    # ── Split train / val ──────────────────────────────────────────────────────
    # Tomamos los ÚLTIMOS n_val elementos como validación, igual que hace Keras
    # con validation_split. Importante: el scaler ya fue ajustado sobre todo el
    # train ANTES de llamar a esta función, así que no hay data leakage.
    n     = len(X_sc)
    n_val = int(n * VAL_SPLIT)
    X_tr,  X_val  = X_sc[:n - n_val],  X_sc[n - n_val:]
    y_tr,  y_val  = y[:n - n_val],     y[n - n_val:]

    # ── Convertir a tensores PyTorch en el device correcto ─────────────────────
    # float32: precisión suficiente para redes neuronales y más eficiente en
    # GPU que float64. Los datos de pandas/numpy vienen en float64 por defecto.
    # .to(device) copia el tensor a MPS/CUDA si está disponible.
    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).to(device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    # ── DataLoader: gestiona el mini-batch sampling ────────────────────────────
    # TensorDataset agrupa (X, y) para que DataLoader los muestree juntos.
    # shuffle=True: reordena el train aleatoriamente al inicio de cada época.
    # Esto es crucial: si el orden es siempre el mismo, el modelo puede aprender
    # artefactos del orden de los datos (ej. si los datos están ordenados por
    # precio, los últimos batches siempre ven los más caros).
    # El Generator fija la semilla para reproducibilidad del shuffling.
    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    # ── Modelo ─────────────────────────────────────────────────────────────────
    model = MLP(X_sc.shape[1]).to(device)

    # Inicializar el bias de la capa de salida en mean(y_train) ≈ log(price) ≈ 20.
    # Con bias=0 (inicialización por defecto de Kaiming), las predicciones iniciales
    # son ~0, pero los targets son ~20. Adam necesita ~333 épocas para mover el bias
    # hasta 20 con LR=1e-3 (cálculo: 20 unidades / 1e-3 por paso = 20000 pasos;
    # con 60 batches/época → 333 épocas). Inicializando en mean(y_train), la red
    # empieza cerca del valor correcto y solo necesita aprender las desviaciones.
    with torch.no_grad():
        model.red[-1].bias.fill_(float(y_tr.mean()))

    # ── Optimizador: Adam con L2 integrada (weight_decay) ─────────────────────
    # Adam (Adaptive Moment Estimation, Kingma & Ba 2015) mantiene estimaciones
    # del primer momento (media del gradiente) y segundo momento (varianza del
    # gradiente) para cada parámetro, adaptando la tasa de aprendizaje:
    #   θ_{t+1} = θ_t - lr * m̂_t / (√v̂_t + ε)
    # weight_decay implementa L2: en cada paso multiplica cada peso por (1 - lr*λ₂)
    # antes de la actualización del gradiente (AdamW-style decoupled regularization).
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=L2_REG)

    # ── Scheduler: ReduceLROnPlateau ───────────────────────────────────────────
    # Si val_loss no mejora durante 7 épocas consecutivas, reduce LR a la mitad.
    # Esto permite escapar de mínimos planos (plateaus) donde el gradiente es
    # pequeño pero la loss todavía puede disminuir con un paso más pequeño.
    # min_lr evita que la tasa de aprendizaje caiga a cero por reducciones
    # sucesivas (0.001 → 0.0005 → 0.00025 → ... → 0.000001).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=7, min_lr=1e-6
    )

    # MSELoss: calcula (1/N) Σ(ŷ - y)². Es diferenciable en todos sus puntos,
    # lo que permite calcular gradientes exactos mediante backpropagation.
    # MAE no es diferenciable en 0, lo que lo haría problemático como loss.
    criterion = nn.MSELoss()

    # ── Variables de control del entrenamiento ─────────────────────────────────
    best_val_loss  = float("inf")   # mejor val_loss visto hasta ahora
    best_state     = None           # state_dict del mejor epoch (para restore)
    patience_count = 0              # contador de épocas sin mejora
    history: dict  = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

    print(f"\n  Entrenando en {device} (máx {EPOCHS} épocas, "
          f"early stopping patience={PATIENCE})...")

    for epoch in range(1, EPOCHS + 1):

        # ── Fase de entrenamiento ──────────────────────────────────────────────
        # model.train() activa Dropout: en cada forward pass, las neuronas
        # se apagan aleatoriamente según las probabilidades definidas.
        model.train()
        batch_losses: list[float] = []

        for xb, yb in loader:
            # optimizer.zero_grad(): limpia los gradientes acumulados del batch
            # anterior. PyTorch acumula gradientes por defecto (útil para
            # gradient accumulation), así que hay que limpiarlos manualmente.
            optimizer.zero_grad()

            # Forward pass: calcula las predicciones del batch actual.
            pred = model(xb)

            # Pérdida = MSE + L1 regularización.
            # MSE mide qué tan bien predice el modelo en este batch.
            # L1 penaliza pesos grandes para reducir el sobreajuste.
            # La L2 ya está integrada en el optimizer (weight_decay), así que
            # no hace falta añadirla aquí manualmente.
            loss = criterion(pred, yb)
            if L1_REG > 0:
                loss = loss + L1_REG * penalizacion_l1(model)

            # Backward pass: calcula ∂loss/∂W para cada parámetro W de la red
            # usando la regla de la cadena (backpropagation).
            loss.backward()

            # Gradient clipping: si la norma total del gradiente supera max_norm,
            # escala todos los gradientes proporcionalmente para que la norma
            # total sea exactamente max_norm. Previene que un batch con error
            # muy grande produzca una actualización que destruya los pesos
            # aprendidos hasta ese momento (explosión de gradientes).
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Actualización de pesos: θ ← θ - lr * ∇loss (con la adaptación de Adam).
            optimizer.step()

            # Registrar la loss de este batch (ya es float, fuera del grafo).
            batch_losses.append(loss.item())

        # ── Fase de validación ─────────────────────────────────────────────────
        # model.eval() desactiva Dropout: todas las neuronas están activas.
        # PyTorch también escala las activaciones para compensar el Dropout
        # que se aplicó durante el entrenamiento.
        model.eval()
        with torch.no_grad():
            # torch.no_grad(): desactiva el cálculo del grafo computacional.
            # En validación no necesitamos gradientes → ahorra memoria y tiempo.
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()  # MSE en validación
            val_mae  = (val_pred - y_val_t).abs().mean().item()  # MAE en val
            tr_pred  = model(X_tr_t)
            tr_mae   = (tr_pred - y_tr_t).abs().mean().item()    # MAE en train

        # Pérdida de entrenamiento = promedio sobre todos los batches de la época.
        train_loss = float(np.mean(batch_losses))
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mae"].append(tr_mae)
        history["val_mae"].append(val_mae)

        # ReduceLROnPlateau: si val_loss no mejoró en 7 épocas, reduce LR × 0.5.
        scheduler.step(val_loss)

        # ── Early Stopping con restore_best_weights ────────────────────────────
        if val_loss < best_val_loss:
            # Nuevo mínimo: guardar los pesos actuales como los mejores.
            # .cpu().clone() copia los tensores a CPU antes de guardarlos,
            # para no ocupar memoria del device (MPS/GPU) con los pesos guardados.
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            # Sin mejora: incrementar el contador.
            patience_count += 1
            if patience_count >= PATIENCE:
                # Se agotó la paciencia: detener el entrenamiento.
                print(f"  Early Stopping en época {epoch}/{EPOCHS}")
                break

    # Restaurar los pesos del mejor epoch encontrado durante el entrenamiento.
    # Equivalente a restore_best_weights=True en el callback de Keras.
    # Esto garantiza que el modelo retorna con los mejores pesos vistos,
    # no con los pesos del último epoch (que podrían ser peores si hubo
    # sobreajuste en las últimas épocas antes de activarse el Early Stopping).
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    epochs_run = len(history["loss"])
    print(f"  Épocas efectivas: {epochs_run} | "
          f"mejor val_loss (MSE): {best_val_loss:.6f} | "
          f"mejor val_MAE: {min(history['val_mae']):.6f}")
    return model, history


def predecir(model: MLP, X_sc: np.ndarray,
             device: torch.device = DEVICE) -> np.ndarray:
    """
    Genera predicciones en modo inferencia (Dropout desactivado).

    model.eval() es imprescindible: sin él, Dropout seguiría activo y las
    predicciones serían ruidosas e irreproducibles entre llamadas. En modo
    eval, PyTorch escala las activaciones por (1 - dropout_prob) para que
    la esperanza matemática de la salida sea la misma que en entrenamiento.

    .cpu().numpy() mueve el tensor de vuelta a CPU y lo convierte a numpy,
    necesario porque sklearn (mean_absolute_error) no acepta tensores PyTorch.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_sc, dtype=torch.float32).to(device)
        return model(X_t).cpu().numpy()


# =============================================================================
# SECCIÓN 6: CV ESPACIAL
# =============================================================================

def cv_espacial(X_raw: np.ndarray, y: np.ndarray,
                lat: np.ndarray, lon: np.ndarray,
                device: torch.device = DEVICE) -> tuple[np.ndarray, np.ndarray]:
    """
    Evalúa la generalización geográfica de la red con CV leave-one-block-out.

    Por qué el CV espacial es más honesto que la validación interna
    ---------------------------------------------------------------
    La validación interna (VAL_SPLIT=0.2) es optimista porque los puntos de
    validación tienen vecinos espaciales en el train. La autocorrelación espacial
    (Cov(ε(sᵢ), ε(sⱼ)) > 0 para sᵢ ≈ sⱼ) hace que el modelo "transfiera"
    información del ruido local del train al fold de validación.
    Con CV espacial, todos los vecinos del fold de validación están en el train,
    simulando el escenario real de predecir en Chapinero (zona no vista).

    Implementación leave-one-block-out
    -----------------------------------
    1. Se divide Bogotá en una cuadrícula SPATIAL_GRID × SPATIAL_GRID = 25 bloques.
       qcut crea bloques de igual tamaño (mismo número de observaciones),
       no de igual área geográfica.
    2. Los 25 bloques se agrupan en 5 folds (5 grupos de ~5 bloques cada uno).
    3. En cada fold: los inmuebles de ese grupo de bloques van a validación,
       el resto a entrenamiento. Se entrena un modelo nuevo desde cero con
       los mismos hiperparámetros y se evalúa en el bloque retenido.
    4. Cada modelo entrena su propio StandardScaler solo sobre su train,
       para evitar data leakage entre folds (el scaler nunca ve el fold de val).
    5. El MAE medio sobre los 5 folds es el proxy del MAE en Chapinero.

    X_raw : features sin escalar (el scaler se ajusta dentro de cada fold)
    """
    print(f"\n{'='*60}")
    print(f"  CV ESPACIAL — cuadrícula {SPATIAL_GRID}×{SPATIAL_GRID}")
    print(f"{'='*60}")

    G = SPATIAL_GRID

    # qcut divide la latitud y longitud en G cuantiles de igual tamaño.
    # duplicates="drop" evita errores cuando muchos puntos comparten la misma
    # coordenada exacta (lo que haría que qcut intentara crear bins vacíos).
    lat_bins = pd.qcut(lat, G, labels=False, duplicates="drop")
    lon_bins = pd.qcut(lon, G, labels=False, duplicates="drop")

    # ID de celda: entero único por par (lat_bin, lon_bin).
    # lat_bin * G + lon_bin asigna un ID distinto a cada una de las G² celdas.
    celda = lat_bins * G + lon_bins

    # Dividir las celdas únicas en 5 grupos para los 5 folds.
    # array_split reparte las celdas lo más equitativamente posible.
    grupos = np.array_split(np.unique(celda), 5)

    maes_log: list[float] = []
    maes_cop: list[float] = []

    for i, grp in enumerate(grupos):
        # mask_val: True para los inmuebles cuya celda está en el fold de validación
        # mask_tr:  True para el resto (van a entrenamiento)
        mask_val = np.isin(celda, grp)
        mask_tr  = ~mask_val

        # Si el fold queda vacío (puede pasar con pocos datos o cuadrícula grande),
        # saltarlo para no entrenar con datos insuficientes.
        if mask_val.sum() == 0 or mask_tr.sum() == 0:
            continue

        # Estandarizar dentro del fold: el scaler se ajusta SOLO sobre el train
        # del fold. Si usáramos el scaler global (ajustado sobre todo el train),
        # el fold de validación habría influido en la media/desviación usada
        # para transformar el fold de validación → data leakage.
        sc    = StandardScaler()
        X_tr  = sc.fit_transform(X_raw[mask_tr])    # ajustar Y transformar
        X_val = sc.transform(X_raw[mask_val])        # solo transformar (sin ver val)
        y_tr, y_val = y[mask_tr], y[mask_val]

        # Entrenar un modelo nuevo desde cero para este fold.
        # El "_" descarta el history porque en CV espacial solo nos interesa
        # el MAE final, no las curvas de entrenamiento por fold.
        m, _ = entrenar(X_tr, y_tr, device=device)
        y_hat = predecir(m, X_val, device=device)

        # MAE en escala log(price): métrica del modelo.
        mae_log = float(mean_absolute_error(y_val, y_hat))

        # MAE en COP: exp revierte el log para volver a precios originales.
        # np.clip evita overflow en np.exp si el modelo predice valores extremos
        # (log(price) > 709 causaría inf en float64).
        mae_cop = float(mean_absolute_error(
            np.exp(y_val), np.exp(np.clip(y_hat, 10, 30))
        ))
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
# SECCIÓN 7: GRÁFICOS DE DIAGNÓSTICO
# =============================================================================

def plot_historia(history: dict, epochs_run: int) -> None:
    """
    Grafica las curvas de pérdida (MSE) y métrica (MAE) para train y validación.

    A diferencia de sklearn MLPRegressor (que solo expone la curva de train),
    PyTorch lleva el historial completo por época gracias al bucle manual.
    Esto permite analizar:
      - Si train ≈ validación → el modelo generaliza bien (sin overfitting).
      - Si train << validación y la brecha crece → overfitting: el modelo
        memoriza el train. Solución: más Dropout, mayor L1/L2, menos épocas.
      - El punto de Early Stopping es donde la curva de validación empieza
        a subir (o deja de bajar): los pesos restaurados corresponden a ese
        mínimo de val_loss.
    """
    ep = range(1, epochs_run + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel izquierdo: pérdida MSE — lo que el optimizador minimiza.
    # La brecha entre train y val en el eje vertical indica el overfitting.
    axes[0].plot(ep, history["loss"],     label="Train",      color="#3498db")
    axes[0].plot(ep, history["val_loss"], label="Validación", color="#e74c3c")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("MSE — log(price)")
    axes[0].set_title(f"Pérdida (MSE) — {MODEL_ID}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel derecho: MAE — lo que reportamos e interpretamos.
    # MAE en log-escala ≈ error porcentual relativo (para valores pequeños).
    axes[1].plot(ep, history["mae"],     label="Train",      color="#3498db")
    axes[1].plot(ep, history["val_mae"], label="Validación", color="#e74c3c")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MAE — log(price)")
    axes[1].set_title(f"Métrica MAE — {MODEL_ID}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Curvas de entrenamiento — {MODEL_ID}\n"
                 f"(Early Stopping en época {epochs_run}, device={DEVICE})",
                 y=1.02)
    plt.tight_layout()
    fig.savefig(DIR_MODEL / "historia_entrenamiento.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/historia_entrenamiento.png")


def plot_residuos(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Diagnóstico de residuos del modelo final sobre TODO el train.

    Panel izquierdo — Residuos vs. Ajustados:
      Grafica e_i = y_i - ŷ_i contra ŷ_i. En un modelo bien especificado:
      - La nube debe estar centrada en 0 sin patrón: ausencia de sesgo.
      - Una forma de abanico (varianza crece con ŷ) indica heterocedasticidad:
        errores más grandes para propiedades más caras. En log-escala este
        problema es menos severo que en escala original.
      - Una curva sistemática indica no-linealidad no capturada por el modelo.

    Panel derecho — Histograma de residuos:
      Distribución simétrica alrededor de 0 → el modelo no sobreestima ni
      subestima sistemáticamente. Asimetría → sesgo sistemático (ej. el modelo
      siempre subestima precios altos), relevante para la discusión de negocio.
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

    La línea punteada azul es el MAE del val_split=0.2 (optimista).
    Las barras son el MAE real en cada bloque geográfico no visto durante
    el entrenamiento de ese fold.

    Si las barras son consistentemente más altas que la línea → sesgo de
    optimismo confirmado: la red generaliza peor a zonas no vistas de lo
    que sugiere la validación interna.
    Δ = media(barras) − línea cuantifica ese sesgo.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    folds = np.arange(1, len(maes_log) + 1)

    # Panel izquierdo: en escala log(price) — escala del modelo.
    axes[0].bar(folds, maes_log, color="#9b59b6", edgecolor="white")
    axes[0].axhline(mae_val_log, color="blue", lw=1.5, linestyle="--",
                    label=f"Val. interno ({mae_val_log:.4f})")
    axes[0].set_xlabel("Fold espacial")
    axes[0].set_ylabel("MAE log(price)")
    axes[0].set_title(f"CV espacial — {MODEL_ID} (log)")
    axes[0].legend(fontsize=8)

    # Panel derecho: en millones de COP — escala de negocio.
    # Permite interpretar el error en términos económicos concretos.
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
# SECCIÓN 8: SUBMISSION PARA KAGGLE
# =============================================================================

def generar_submission(test: pd.DataFrame, y_pred_log: np.ndarray,
                       epochs_run: int, total_params: int) -> str:
    """
    Genera el CSV de predicciones en el formato exacto del template de Kaggle.

    Los pasos son:
      1. exp(y_pred_log) revierte la transformación log → vuelve a precios en COP.
         Kaggle evalúa el MAE en precios originales (pesos colombianos), no en log.
      2. El merge con el template garantiza el orden exacto de property_id que
         espera Kaggle, independientemente del orden de nuestro test.csv.
      3. El nombre del archivo incluye épocas y parámetros para identificar
         fácilmente qué submission corresponde a qué modelo.
    """
    submission = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),  # revertir log → precios en COP
    })
    template = pd.read_csv(
        Path("/Users/macbook/Downloads/uniandes-bdml-202610-ps-3") /
        "submission_template.csv"
    )
    # Left join: el template define el orden de filas; submission aporta los precios.
    # Si algún property_id del template no está en submission (no debería pasar),
    # quedaría con NaN en price → detectaría un bug en la construcción del test.
    submission = (template[["property_id"]]
                  .merge(submission, on="property_id", how="left"))

    sub_name = f"submission_{MODEL_ID}_ep{epochs_run}_p{total_params//1000}k.csv"
    submission.to_csv(SUBMISSIONS / sub_name, index=False)

    print(f"\n  Submission guardado: 03_submissions/{sub_name}")
    print(f"  Filas: {len(submission):,} | "
          f"price media: {submission['price'].mean()/1e6:.1f}M COP")
    return sub_name


# =============================================================================
# SECCIÓN 9: REGISTRO EN model_registry.xlsx
# =============================================================================

def registrar(sub_name: str, n_features: int, total_params: int,
              epochs_run: int,
              mae_train_log: float, mae_train_cop: float,
              mae_val_log: float,
              mae_esp_log: float, mae_esp_cop: float) -> None:
    """
    Añade (o actualiza) la fila de MODEL_ID en el registro central de modelos.

    El registro permite comparar EN_001, NN_002 y futuros modelos en un solo
    lugar. Si MODEL_ID ya existe, la fila anterior se reemplaza (no se duplica),
    lo que facilita re-correr el script con ajustes de hiperparámetros.

    Columnas clave para el análisis comparativo:
      - cv_mae_log:  MAE del val_split=0.2 → optimista (referencia interna)
      - esp_mae_log: MAE del CV espacial → honesto (proxy de Chapinero)
      - Δ = esp_mae_log - cv_mae_log → sesgo de optimismo de la val interna
      - kaggle_public_MAE: se llena manualmente tras subir el CSV a Kaggle
    """
    # Sesgo de optimismo: cuánto subestima la validación interna el error real.
    # Si Δ > 0: la val interna es más optimista que el CV espacial (lo esperado).
    # Si Δ ≈ 0: los datos no tienen autocorrelación espacial fuerte (inusual).
    delta_espacial = mae_esp_log - mae_val_log

    nueva = {
        "model_id":          MODEL_ID,
        "fecha":             str(date.today()),
        "autor":             AUTOR,
        "algoritmo":         "NeuralNetwork_PyTorch",
        "n_features":        n_features,
        "n_params":          total_params,   # suma de todos los parámetros entrenables
        "epochs_run":        epochs_run,     # épocas efectivas (Early Stopping)
        "cv_folds":          f"val_split={VAL_SPLIT}",
        # MAE de validación interna (optimista: vecinos en el train)
        "cv_mae_log":        round(mae_val_log,   5),
        "cv_mae_cop_M":      None,   # no calculable directamente sin el split guardado
        # MAE de CV espacial (honesto: separación geográfica estricta)
        "esp_mae_log":       round(mae_esp_log,   5),
        "esp_mae_cop_M":     round(mae_esp_cop / 1e6, 2),
        # MAE en train (si muy por debajo del CV → sobreajuste)
        "train_mae_log":     round(mae_train_log, 5),
        "kaggle_public_MAE": None,   # ← llenar manualmente tras subir a Kaggle
        "l1_ratio":          None,   # no aplica (ElasticNet es regularizador de pesos)
        "alpha":             None,
        "features_grupos":  (f"structural={len(STRUCTURAL)}, "
                              f"text={len(TEXT)}, osm={len(OSM)}, "
                              f"es_apartamento"),
        "arquitectura":      (f"p→256(Drop{DROPOUT_1})→"
                              f"128(Drop{DROPOUT_2})→64→1 (ReLU, MPS)"),
        "reg_l1_l2":         f"l1={L1_REG}, l2={L2_REG}",
        "early_stopping":    f"patience={PATIENCE}",
        "spatial_grid":      f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "device":            str(DEVICE),
        "submission_file":   sub_name,
        "notas": (
            f"MLP PyTorch 256→128→64→1 con ReLU, Dropout y Elastic Net. "
            f"Backend: {DEVICE}. "
            f"l1={L1_REG}, l2={L2_REG} (weight_decay). "
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
        # Eliminar fila previa del mismo MODEL_ID para no duplicar registros.
        df_old = df_old[df_old["model_id"] != MODEL_ID]
        df_reg = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_reg = df_new

    with pd.ExcelWriter(REGISTRY, engine="openpyxl") as writer:
        df_reg.to_excel(writer, index=False, sheet_name="registry")
        # Ajustar el ancho de cada columna al contenido más largo.
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
# SECCIÓN 10: ENTRY POINT
# =============================================================================

def main() -> None:
    print(f"{'='*60}")
    print(f"  RED NEURONAL PyTorch — {MODEL_ID}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    # ── [1/7] Carga ───────────────────────────────────────────────────────────
    print("\n[1/7] Cargando datos...")
    train, test = cargar_datos()
    # Transformar price a log(price): estabiliza la varianza y hace la
    # distribución más simétrica → entrenamiento más estable y MAE en
    # log-escala penaliza errores relativos en lugar de absolutos.
    y_train_log = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    # ── [2/7] Features ────────────────────────────────────────────────────────
    print("\n[2/7] Construyendo features...")
    X_train_df   = construir_features(train)
    feature_cols = list(X_train_df.columns)   # orden de columnas del train
    # Pasamos feature_cols al test para garantizar el mismo orden de columnas.
    X_test_df    = construir_features(test, fit_cols=feature_cols)
    print(f"  Features: {len(feature_cols)} "
          f"({len(STRUCTURAL)} estructurales + {len(TEXT)} texto + "
          f"{len(OSM)} OSM + es_apartamento)")

    # ── [3/7] Estandarización ─────────────────────────────────────────────────
    # Las redes neuronales son muy sensibles a la escala de las features:
    # si una feature tiene rango [0, 1000000] y otra [0, 1], el gradiente
    # favorece la primera → convergencia lenta o inestable.
    # StandardScaler: x_scaled = (x - μ) / σ → media 0, desviación 1.
    # CRÍTICO: ajustar (fit) SOLO sobre train y transformar (transform) test.
    # Si fitteamos sobre test, usaríamos información del futuro para escalar
    # los datos de entrenamiento → data leakage.
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_df.values)   # ajustar Y transformar
    X_test_sc  = scaler.transform(X_test_df.values)        # solo transformar

    # ── [4/7] Entrenamiento principal ─────────────────────────────────────────
    print("\n[3/7] Entrenando red neuronal (PyTorch, MPS)...")
    model, history = entrenar(X_train_sc, y_train_log)

    epochs_run  = len(history["loss"])
    # Mejor MAE de validación durante el entrenamiento: corresponde al epoch
    # cuyos pesos fueron restaurados por Early Stopping.
    mae_val_log = float(min(history["val_mae"]))
    # Número total de parámetros entrenables: suma de elementos en pesos y bias
    # de todas las capas. p.numel() da el total de elementos del tensor p.
    total_params = sum(p.numel() for p in model.parameters())

    # Predicciones sobre TODO el train en modo eval (Dropout desactivado).
    # Sirve para calcular el MAE de entrenamiento y diagnosticar sobreajuste:
    # si MAE_train << MAE_val → la red memorizó el train sin generalizar.
    y_pred_train      = predecir(model, X_train_sc)
    # np.clip evita overflow en np.exp: log(price) > 709 → exp → inf en float64.
    # El rango [10, 30] cubre precios razonables: exp(10)≈22k COP, exp(30)≈1e13 COP.
    y_pred_train_clip = np.clip(y_pred_train, 10, 30)
    mae_train_log = float(mean_absolute_error(y_train_log, y_pred_train))
    mae_train_cop = float(
        mean_absolute_error(np.exp(y_train_log), np.exp(y_pred_train_clip))
    )
    print(f"  Parámetros totales: {total_params:,}")
    print(f"  MAE train (log): {mae_train_log:.5f} | "
          f"MAE train (COP): {mae_train_cop/1e6:.1f}M")

    # ── [5/7] CV espacial ─────────────────────────────────────────────────────
    # Pasamos X_train_df.values (sin escalar) porque cv_espacial escala
    # internamente dentro de cada fold → evita leakage entre folds.
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
    # Clip antes de exp para evitar overflow (mismo razonamiento que arriba).
    y_pred_test = np.clip(predecir(model, X_test_sc), 10, 30)
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
    # Δ > 0 confirma el sesgo de optimismo de la validación interna:
    # la red generaliza peor a zonas no vistas de lo que sugiere val_split.
    delta = float(maes_esp_log.mean()) - mae_val_log
    print(f"\n{'='*60}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"{'='*60}")
    print(f"  Device      : {DEVICE}")
    print(f"  Arquitectura: p → 256(Drop{DROPOUT_1}) → "
          f"128(Drop{DROPOUT_2}) → 64 → 1")
    print(f"  Regulariz.  : L1={L1_REG}, L2={L2_REG} (Elastic Net)")
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
