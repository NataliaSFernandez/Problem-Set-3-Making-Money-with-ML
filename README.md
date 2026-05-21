# Problem Set 3 — Making Money with ML?
### *"It's all about location, location, location!"*

**MECA 4107 · Big Data and Machine Learning para Economía Aplicada · 2026-10**  
**Universidad de los Andes — Facultad de Economía**

**Equipo 04** — Daniela Solano R. · Jonathan Melo Sarta · Natalia Suescún F.

---

## Objetivo

Desarrollar un modelo predictivo de precios de vivienda en el barrio **Chapinero**, Bogotá, usando datos de Properati de toda la ciudad como entrenamiento. El reto central es la **transferencia espacial**: el modelo se entrena fuera de Chapinero y debe predecir dentro. La pregunta de negocio es si el modelo es confiable para comprar propiedades sin repetir el error de Zillow (sobrepredicción sistemática que destruye capital).

La métrica de evaluación en Kaggle es el **Mean Absolute Error (MAE)** sobre el precio en pesos colombianos.

---

## Mejor modelo

| Modelo | Algoritmo | Kaggle MAE público | CV espacial MAE log | Sesgo Δ |
|---|---|---|---|---|
| **SL_003** | SuperLearner NNLS (XGB_009 + NN_003 [+ RF_005]) | **$199,851,632** | 0.23244 | +0.06700 |
| XGB_009 | XGBoost + KNN mediana K=30 + early stopping espacial | $207,886,154 | 0.21079 | +0.03480 |
| XGB_010 | XGBoost + KNN ponderado por distancia K=30 | $215,636,744 | 0.17260 | +0.02231 |

> SL_003 es el mejor modelo del repositorio con MAE $199,851,632. Compara internamente SL-2 (XGB_009 + NN_003) vs SL-3 (XGB_009 + NN_003 + RF_005) y selecciona el ganador por CV espacial, con parsimonia hacia SL-2 si la diferencia es menor a 1 std.

---

## Cómo reproducir los resultados

### Requisitos

- Python 3.11
- Cuenta de Google Cloud con acceso a BigQuery (solo para `02_add_osm_features.py`, primera ejecución ~69 GB ≈ $0.43 USD, o gratis dentro del free tier mensual de 1 TB)

### 1. Clonar el repositorio y crear entorno virtual

```bash
git clone https://github.com/NataliaSFernandez/Problem-Set-3-Making-Money-with-ML.git
cd Problem-Set-3-Making-Money-with-ML
python3.11 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Configurar credenciales de Google Cloud

```bash
cp .env.example .env
# Editar .env: poner la ruta completa al JSON del service account
export GOOGLE_APPLICATION_CREDENTIALS="/ruta/completa/a/credentials.json"
```

> El archivo `.env` y el JSON de credenciales están en `.gitignore` y nunca deben subirse al repositorio.

### 3. Preparar los datos (orden obligatorio)

```bash
python 01_scripts/DataPreparation/01_extract_text_features.py
python 01_scripts/DataPreparation/02_add_osm_features.py
```

Esto genera en `00_data/processed/` los archivos `train_final.csv` y `test_final.csv`, que son el input de todos los modelos. La carpeta `00_data/raw/osm/` se crea automáticamente como caché de la descarga OSM — a partir de la segunda ejecución el costo BigQuery es $0.

### 4. Correr cualquier modelo

Cada script en `01_scripts/Models/` es autocontenido: crea las carpetas de output necesarias, guarda los diagnósticos en `02_outputs/Models/<familia>/<modelo_id>/`, genera el CSV de submission en `03_submissions/`, y actualiza `02_outputs/model_registry.xlsx`. Las dependencias (xgboost, openpyxl, scipy) se instalan automáticamente si no están presentes.

```bash
# Ejemplo: mejor modelo Kaggle
python 01_scripts/Models/04_Boosting/XGB_009_KNN.py

# Ejemplo: mejor CV espacial
python 01_scripts/Models/04_Boosting/XGB_010_KNNWeighted.py

# SuperLearner (requiere que los modelos base ya hayan corrido)
python 01_scripts/Models/07_SuperLearner/SL_003.py
```

---

## Estructura del repositorio

```
Problem-Set-3-Making-Money-with-ML/
│
├── 00_data/
│   ├── raw/
│   │   ├── train.csv                              # Datos Properati — toda Bogotá (train)
│   │   ├── test.csv                               # Datos Properati — solo Chapinero (test)
│   │   └── osm/                                   # Caché POIs y red vial OSM (auto-generado)
│   └── processed/
│       ├── ngrams_seleccionados.csv               # N-gramas seleccionados para features de texto
│       ├── train_texto.csv                        # Train + variables hedónicas de texto
│       ├── test_texto.csv
│       ├── train_osm.csv                          # Train + variables de localización OSM
│       ├── test_osm.csv
│       ├── train_final.csv                        # Dataset final para modelado
│       └── test_final.csv
│
├── 01_scripts/
│   ├── DataPreparation/
│   │   ├── 01_extract_text_features.py            # Variables hedónicas desde título/descripción
│   │   ├── 02_add_osm_features.py                 # Variables de localización desde BigQuery OSM
│   │   ├── 02_extract_text_features_lasso.py      # Variante: selección de n-gramas por Lasso
│   │   └── 03_clean_explore.py                    # Limpieza y análisis exploratorio
│   │
│   └── Models/
│       ├── 00_LinearProbabilityModel.py           # LR_001: OLS baseline, 29 features
│       ├── 01_ElasticNet/                         # EN_001 (base) · EN_002 (interacciones + poly)
│       ├── 02_CART/                               # CART_001 (baseline) · CART_002 (GridSearch esp.)
│       ├── 03_RandomForest/                       # RF_001 … RF_005 (grid depth, tuning espacial)
│       ├── 04_Boosting/
│       │   ├── XGB_001_baseline.py                # Solo estructurales, defaults
│       │   ├── XGB_002_allfeats.py                # Features completas, sin tuning
│       │   ├── XGB_003_depth_tuning.py            # GridSearch depth + lr
│       │   ├── XGB_004_randomsearch50.py          # RandomizedSearch 50 iter, CV espacial
│       │   ├── XGB_005_target_encoding.py         # Target encoding espacial LOO
│       │   ├── XGB_006_randomsearch150.py         # RandomizedSearch 150 iter
│       │   ├── XGB_007_latlon.py                  # lat/lon + early stopping espacial
│       │   ├── XGB_008_filtro_chapinero.py        # Filtro IQR precio/m² Chapinero
│       │   ├── XGB_009_KNN.py                     # KNN mediana K=30 LOO haversine
│       │   └── XGB_010_KNNWeighted.py             # KNN ponderado 1/dist K=30 LOO
│       ├── 06_NeuralNetwork/                      # NN_002 (MLP 3 capas) · NN_003 (BN + 4 capas)
│       └── 07_SuperLearner/                       # SL_001 (baseline) · SL_002 (bases óptimas) · SL_003 (NNLS, mejor Kaggle ⭐)
│
├── 02_outputs/
│   ├── model_registry.xlsx                        # Registro centralizado de todos los modelos
│   └── Models/
│       ├── 01_ElasticNet/EN_001/ EN_002/
│       ├── 02_CART/CART_001/ CART_002/
│       ├── 03_RandomForest/RF_001/ … RF_005/
│       ├── 04_Boosting/
│       │   └── XGB_007/ … XGB_010/                # feature_importance.png · curva_aprendizaje.png
│       │                                          # residuos.png · cv_comparacion.png
│       ├── 06_NeuralNetwork/NN_002/ NN_003/
│       └── 07_SuperLearner/SL_001/ SL_002/ SL_003/
│
├── 03_submissions/                                # CSVs de predicciones enviadas a Kaggle (20)
│   ├── submission_LR_001_20260517.csv
│   ├── submission_EN_001_l1r100_a2en02.csv
│   ├── submission_EN_002_l1r100_a2en02.csv
│   ├── CART_dNone_leaf1_ccp00_cv5_CART_001.csv
│   ├── CART_d10_leaf20_ccp00001_cv5_CART_002.csv
│   ├── RF_ntrees100_dNone_leaf1_mfsqrt_cv5_featRF_001.csv
│   ├── RF_ntrees100_dNone_leaf1_mfsqrt_cv5_featRF_002.csv
│   ├── RF_ntrees500_d10_leaf5_mfsqrt_cv5_featRF_004.csv
│   ├── RF_ntrees500_d15_leaf5_mfsqrt_cv5_gridESP_RF_005.csv
│   ├── submission_NN_002_ep75_p48k.csv
│   ├── XGB_nest100_d6_lr03_allfeats_cv5_XGB_002.csv
│   ├── XGB_nest356_d6_lr0078_cv5_XGB_004.csv
│   ├── XGB_iter150_nest564_d8_lr0075_cv5_XGB_006.csv
│   ├── XGB_latlon_earlystop_XGB_007.csv
│   ├── XGB_filtro_chapinero_earlystop_XGB_008.csv
│   ├── XGB_knn30_latlon_earlystop_XGB_009.csv
│   ├── XGB_knn30_weighted_earlystop_XGB_010.csv
│   ├── SL_001_baseline.csv
│   ├── submission_SL_002_20260519.csv
│   └── submission_SL_003_YYYYMMDD.csv            # ⭐ MEJOR MAE KAGGLE: $199,851,632
│
├── .env.example                                   # Plantilla de credenciales Google Cloud
├── .gitignore
└── README.md
```

---

## Datos y variables

### Fuente base

| Fuente | Uso |
|---|---|
| Properati Colombia | Precios, características estructurales y texto de anuncios |
| BigQuery — `bigquery-public-data.geo_openstreetmap` | POIs y red vial de Bogotá |

### Variables construidas

**Desde texto del anuncio (`title` + `description`) — `01_extract_text_features.py`:**

| Variable | Tipo | Descripción |
|---|---|---|
| `remodelado` | 1/NaN | Menciona remodelación o inmueble nuevo |
| `vista_panoramica` | 1/NaN | Menciona vistas, cerros o panorámica |
| `deposito` | 1/NaN | Menciona depósito, bodega o cuarto útil |
| `conjunto_cerrado` | 1/NaN | Menciona portería, vigilancia o áreas comunes |
| `balcon_terraza` | 1/NaN | Menciona balcón, terraza, BBQ o patio |
| `tfidf_premium` | float | Score TF-IDF de términos de lujo |
| `parqueaderos_txt` | int/NaN | Número de garajes mencionados en el texto |
| `piso_txt` | int/NaN | Piso del apartamento mencionado en el texto |
| `gimnasio` | 1/NaN | Menciona gimnasio, gym o fitness |
| `amenidades` | 1/NaN | Menciona al menos una amenidad |
| `num_amenidades` | float/NaN | Cantidad de tipos de amenidad mencionados |

> Convención de missings: `1` = patrón presente en el anuncio; `NaN` = no mencionado. Un `NaN` no equivale a ausencia del atributo físico.

**Desde OpenStreetMap — `02_add_osm_features.py`:**

| Variable | Unidad | Descripción |
|---|---|---|
| `dist_cbd_km` | km | Distancia al CBD (Plaza de Bolívar) |
| `dist_transmilenio_m` | m | Distancia al portal/estación TM más cercana |
| `dist_via_arterial_m` | m | Distancia a autopista o avenida principal |
| `dist_hospital_m` | m | Distancia al hospital o clínica más cercana |
| `dist_centro_com_m` | m | Distancia al centro comercial más cercano |
| `dist_parque_m` | m | Distancia al parque o área verde más cercana |
| `n_restaurantes_500m` | conteo | Restaurantes y cafés en radio 500 m |
| `n_bancos_500m` | conteo | Bancos y cajeros en radio 500 m |
| `walkability_score` | 0–100 | Categorías de servicio accesibles en 800 m |
| `densidad_vial` | seg/km² | Segmentos viales por km² en radio 500 m |

---

## Registro de modelos

Todos los modelos están documentados en `02_outputs/model_registry.xlsx`. La tabla resume el mejor modelo por familia de algoritmo.

| ID | Algoritmo | Features | CV rand MAE log | CV esp MAE log | Sesgo Δ | Kaggle MAE |
|---|---|---|---|---|---|---|
| SL_003 | SuperLearner NNLS | 32+37 | 0.16545 | 0.23244 | +0.06700 | **$199,851,632** ⭐ |
| XGB_009 | XGBoost | 32 | 0.17598 | 0.21079 | +0.03480 | $207,886,154 |
| XGB_010 | XGBoost | 32 | 0.15029 | 0.17260 | +0.02231 | $215,636,744 |
| SL_002 | SuperLearner | 21 | — | — | +0.04118 | $289,712,137 |
| RF_005 | RandomForest | 29 | 0.19172 | 0.26364 | +0.07192 | $289,569,353 |
| NN_002 | NeuralNetwork | 29 | 0.21823 | 0.24954 | +0.03131 | $294,510,969 |
| CART_002 | CART | 29 | 0.23260 | 0.27405 | +0.04144 | $289,662,028 |
| EN_002 | ElasticNet | 37 | 0.26348 | 0.29778 | +0.03430 | $313,972,117 |
| LR_001 | LinearRegression | 29 | 0.27905 | 0.31016 | +0.03111 | $318,727,924 |

**Submissions en Kaggle: 21** (Daniela Solano R., Jonathan Melo Sarta, Natalia Suescún F.)

---

## Estrategia de validación

Se implementaron dos estrategias en paralelo para todos los modelos:

**CV aleatorio** (KFold, 5 pliegues): estimación optimista. Asume que cualquier propiedad puede aparecer en cualquier pliegue — no replica el desafío real de predecir en Chapinero.

**CV espacial** (GroupKFold, cuadrícula 5×5): divide el territorio en 25 bloques geográficos y garantiza que ningún bloque del test esté representado en el train. Es el estimador honesto de generalización a Chapinero y el que debe guiar la selección del modelo.

El **sesgo Δ** = MAE_espacial − MAE_aleatorio mide la degradación al pasar de validación optimista a honesta. XGB_010 tiene el Δ más bajo del repositorio (+0.02231), lo que indica que su ventaja sobre otros modelos es robusta a la transferencia geográfica.

---

## Mapeo scripts → figuras de las slides

| Figura / tabla | Script que la genera | Ruta del output |
|---|---|---|
| Tabla resumen todos los modelos | — | `02_outputs/model_registry.xlsx` |
| Feature importance — XGB_009 | `XGB_009_KNN.py` | `02_outputs/Models/04_Boosting/XGB_009/feature_importance.png` |
| Curva de aprendizaje — XGB_009 | `XGB_009_KNN.py` | `02_outputs/Models/04_Boosting/XGB_009/curva_aprendizaje.png` |
| Residuos vs. predichos — XGB_009 | `XGB_009_KNN.py` | `02_outputs/Models/04_Boosting/XGB_009/residuos.png` |
| CV aleatorio vs. espacial — XGB_009 | `XGB_009_KNN.py` | `02_outputs/Models/04_Boosting/XGB_009/cv_comparacion.png` |
| Feature importance — XGB_010 | `XGB_010_KNNWeighted.py` | `02_outputs/Models/04_Boosting/XGB_010/feature_importance.png` |
| Diagnósticos RF_005 | `RF_005_gridsearch_esp.py` | `02_outputs/Models/03_RandomForest/RF_005/` |
| Diagnósticos NN_003 | `NN_003_bn_4layers.py` | `02_outputs/Models/06_NeuralNetwork/NN_003/` |
| Comparación SL-2 vs SL-3 | `SL_003.py` | `02_outputs/Models/SuperLearner/SL_003/comparacion_sl2_sl3.png` |
| Pesos NNLS ganador | `SL_003.py` | `02_outputs/Models/SuperLearner/SL_003/pesos_nnls_ganador.png` |
| CV vs CV espacial SL_003 | `SL_003.py` | `02_outputs/Models/SuperLearner/SL_003/cv_comparacion_ganador.png` |
| Correlación bases SL_003 | `SL_003.py` | `02_outputs/Models/SuperLearner/SL_003/correlacion_bases.png` |
| MAE individuales vs SL | `SL_003.py` | `02_outputs/Models/SuperLearner/SL_003/mae_individuales_vs_sl.png` |
| EDA y distribución de precios | `03_clean_explore.py` | `02_outputs/figures/` |

---

## Entregables

- `best_equipo_04.pdf` — Best Model Deep Dive (SL_003, Kaggle MAE $199,851,632)
- `compare_equipo_04.pdf` — Algorithm Comparison
- Repositorio público en GitHub https://github.com/NataliaSFernandez/Problem-Set-3-Making-Money-with-ML
- 20 submissions en Kaggle ✓

---

## Fuentes

| Fuente | Uso |
|---|---|
| [Properati Colombia](https://www.properati.com.co) | Listados de vivienda con precio, características y texto |
| [BigQuery — geo_openstreetmap](https://console.cloud.google.com/bigquery) | POIs y red vial de Bogotá |
| Rosen, S. (1974). *Hedonic Prices and Implicit Markets*. JPE, 82(1), 34–55. | Marco teórico de precios hedónicos |
