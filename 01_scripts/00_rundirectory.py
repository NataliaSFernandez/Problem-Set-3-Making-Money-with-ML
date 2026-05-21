"""
00_rundirectory.py
==================
Script maestro del pipeline de reproducibilidad.

Ejecutar desde la raiz del repositorio:
    python 01_scripts/00_rundirectory.py

Esto corre en orden los scripts de preparacion de datos y los
siete modelos requeridos, genera todos los outputs en 02_outputs/
y todas las submissions en 03_submissions/.

Tiempo estimado total: 45-90 minutos segun hardware.
(El paso mas lento es 02_add_osm_features.py si no hay cache OSM local.)

Equipo 04
- Natalia Suescun
- Daniela Solano
- Jonathan Melo
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# DEPENDENCIAS
# =============================================================================

DEPENDENCIAS = [
    'google-cloud-bigquery',
    'google-cloud-bigquery-storage',
    'pyarrow',
    'xgboost',
    'scikit-learn',
    'pandas',
    'numpy',
    'openpyxl',
    'scipy',
    'torch',
    'matplotlib',
    'seaborn',
    'tqdm',
]


def instalar_dependencias(paquetes: list) -> None:
    """Instala los paquetes faltantes antes de correr el pipeline."""
    import importlib
    import subprocess
    import sys

    # Mapeo nombre pip -> nombre de import (cuando difieren)
    import_names = {
        'google-cloud-bigquery': 'google.cloud.bigquery',
        'google-cloud-bigquery-storage': 'google.cloud.bigquery_storage',
        'scikit-learn': 'sklearn',
        'pyarrow': 'pyarrow',
        'torch': 'torch',
    }

    faltantes = []
    for paquete in paquetes:
        import_name = import_names.get(paquete, paquete)
        try:
            importlib.import_module(import_name)
        except ImportError:
            faltantes.append(paquete)

    if not faltantes:
        print('Todas las dependencias estan instaladas.')
        return

    print(f'Instalando {len(faltantes)} paquete(s) faltante(s): {faltantes}')
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '--quiet'] + faltantes
    )
    print('Instalacion completada.')


# =============================================================================
# CONFIGURACION
# =============================================================================

# Detectar raiz del repo (este script vive en 01_scripts/)
BASE = Path(__file__).parent.parent

SEPARADOR = "=" * 72


def run_script(script_path: Path, descripcion: str) -> bool:
    """
    Ejecuta un script Python como subproceso.
    Retorna True si tuvo exito, False si fallo.
    """
    print(f"\n{SEPARADOR}")
    print(f"EJECUTANDO: {descripcion}")
    print(f"Script:     {script_path.relative_to(BASE)}")
    print(SEPARADOR)

    inicio = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE),
    )

    elapsed = time.time() - inicio
    minutos = elapsed / 60

    if result.returncode == 0:
        print(f"\nCompletado en {minutos:.1f} minutos")
        return True
    else:
        print(f"\nERROR en {script_path.name} (codigo {result.returncode})")
        print("El pipeline se detiene aqui.")
        return False


# =============================================================================
# INICIO
# =============================================================================

instalar_dependencias(DEPENDENCIAS)

inicio_total = time.time()
inicio_dt = datetime.now()

print(f"""
{SEPARADOR}
PROBLEM SET 3: MAKING MONEY WITH ML?
Pipeline de Reproducibilidad Completo
{SEPARADOR}
Directorio de trabajo : {BASE}
Inicio                : {inicio_dt.strftime('%Y-%m-%d %H:%M:%S')}
Tiempo estimado       : 45-90 minutos
{SEPARADOR}
""")

# =============================================================================
# PASO 1: PREPARACION DE DATOS
# =============================================================================
# Genera train_texto.csv / test_texto.csv con variables hedónicas
# extraídas del campo title y description de cada listado.
# Output: 00_data/processed/train_texto.csv, test_texto.csv

ok = run_script(
    BASE / "01_scripts" / "DataPreparation" / "01_extract_text_features.py",
    "PASO 1a: Extraccion de variables de texto (hedónicas)"
)
if not ok:
    sys.exit(1)

# =============================================================================
# PASO 2: VARIABLES OSM
# =============================================================================
# Descarga POIs y red vial de Bogotá desde BigQuery y calcula
# variables de localización para cada inmueble (haversine).
# Primera ejecucion: ~69 GB BigQuery (~$0.43 USD o gratis en free tier).
# Ejecuciones siguientes: $0 (lee desde cache en 00_data/raw/osm/).
# Output: 00_data/processed/train_osm.csv, test_osm.csv
#         00_data/processed/train_final.csv, test_final.csv

ok = run_script(
    BASE / "01_scripts" / "DataPreparation" / "02_add_osm_features.py",
    "PASO 1b: Variables de localización OSM (BigQuery + haversine)"
)
if not ok:
    sys.exit(1)

# =============================================================================
# PASO 3: MODELOS
# =============================================================================
# Cada script carga train_final.csv / test_final.csv desde
# 00_data/processed/, entrena con CV espacial 5x5 + CV aleatorio
# 5-fold, genera diagnosticos en 02_outputs/Models/<familia>/<id>/,
# guarda la submission en 03_submissions/, y actualiza
# 02_outputs/model_registry.xlsx.

# -- 3.1 Linear Probability Model -------------------------------------------
# OLS sin regularizacion. Baseline interpretable.
# 29 features: structural + text + osm + es_apartamento.
# CV espacial MAE_log = 0.31016. Kaggle MAE = $318,727,924.

ok = run_script(
    BASE / "01_scripts" / "Models" / "00_LinearProbabilityModel.py",
    "PASO 3.1: Linear Probability Model — LR_001 (baseline OLS)"
)
if not ok:
    sys.exit(1)

# -- 3.2 Elastic Net ---------------------------------------------------------
# ElasticNet con log-transforms, interacciones y poly espacial.
# l1_ratio=1.0, alpha=1.94e-02. 17/37 coefs != 0.
# CV espacial MAE_log = 0.29778. Kaggle MAE = $313,972,117.

ok = run_script(
    BASE / "01_scripts" / "Models" / "01_ElasticNet" / "EN_002_interactions.py",
    "PASO 3.2: Elastic Net — EN_002 (log-transforms + poly espacial)"
)
if not ok:
    sys.exit(1)

# -- 3.3 CART ----------------------------------------------------------------
# Arbol de decision con GridSearchCV espacial.
# Params optimos: depth=10, min_samples_leaf=20, ccp_alpha=0.0001.
# CV espacial MAE_log = 0.27405. Kaggle MAE = $289,662,028.

ok = run_script(
    BASE / "01_scripts" / "Models" / "02_CART" / "CART_002_tuned.py",
    "PASO 3.3: CART — CART_002 (GridSearch espacial, depth=10)"
)
if not ok:
    sys.exit(1)

# -- 3.4 Random Forest -------------------------------------------------------
# 500 arboles, min_samples_leaf=5, max_depth=15.
# Grid search espacial sobre depth=[8, 10, 15].
# CV espacial MAE_log = 0.26364. Kaggle MAE = $289,569,353.

ok = run_script(
    BASE / "01_scripts" / "Models" / "03_RandomForest" / "RF_005_gridsearch_esp.py",
    "PASO 3.4: Random Forest — RF_005 (grid depth, CV espacial, n=500)"
)
if not ok:
    sys.exit(1)

# -- 3.5 XGBoost (mejor Kaggle individual) ----------------------------------
# XGB_009: lat/lon + precio mediano K=30 vecinos (haversine, LOO).
# Early stopping espacial. RandomizedSearch 50 iter.
# CV espacial MAE_log = 0.21079. Kaggle MAE = $207,886,154.

ok = run_script(
    BASE / "01_scripts" / "Models" / "04_Boosting" / "XGB_009_KNN.py",
    "PASO 3.5: XGBoost — XGB_009 (KNN mediana K=30 LOO)"
)
if not ok:
    sys.exit(1)

# -- 3.6 Neural Network ------------------------------------------------------
# MLP PyTorch 4 capas con Batch Normalization.
# 256->256->128->64->1 (ReLU). Dropout 0.2/0.1. patience=30.
# 37 features (log-transforms + poly espacial).
# CV espacial MAE_log = 0.24331. Kaggle MAE = $216,987,903.

ok = run_script(
    BASE / "01_scripts" / "Models" / "06_NeuralNetwork" / "NN_003_bn_4layers.py",
    "PASO 3.6: Neural Network — NN_003 (BN + 4 capas, PyTorch)"
)
if not ok:
    sys.exit(1)

# -- 3.7 SuperLearner --------------------------------------------------------
# Compara SL-2 (XGB_009 + NN_003) vs SL-3 (XGB_009 + NN_003 + RF_005).
# Meta-aprendiz: NNLS (LinearRegression positive=True, pesos >= 0).
# Ganador seleccionado por MAE_esp (CV espacial 5x5).
# Si la diferencia entre SL-2 y SL-3 es < 1 std, gana SL-2 por parsimonia.
# Kaggle MAE = $199,851,632.  <-- MEJOR DEL REPO

ok = run_script(
    BASE / "01_scripts" / "Models" / "07_SuperLearner" / "SL_003.py",
    "PASO 3.7: SuperLearner — SL_003 (SL-2 vs SL-3, meta NNLS) *** MEJOR KAGGLE ***"
)
if not ok:
    sys.exit(1)

# =============================================================================
# RESUMEN FINAL
# =============================================================================

fin_total = time.time()
fin_dt = datetime.now()
total_min = (fin_total - inicio_total) / 60

print(f"""
{SEPARADOR}
PIPELINE COMPLETADO
{SEPARADOR}
Iniciado   : {inicio_dt.strftime('%Y-%m-%d %H:%M:%S')}
Finalizado : {fin_dt.strftime('%Y-%m-%d %H:%M:%S')}
Duracion   : {total_min:.1f} minutos
{SEPARADOR}

Todos los resultados han sido generados en:

  00_data/processed/
      train_final.csv, test_final.csv (y versiones intermedias)

  02_outputs/
      model_registry.xlsx                    -- tabla resumen de todos los modelos
      Models/00_LPM/                         -- diagnosticos LR_001
      Models/01_ElasticNet/EN_002/           -- diagnosticos EN_002
      Models/02_CART/CART_002/               -- diagnosticos CART_002
      Models/03_RandomForest/RF_005/         -- diagnosticos RF_005
      Models/04_Boosting/XGB_009/            -- diagnosticos XGB_009
      Models/06_NeuralNetwork/NN_003/        -- diagnosticos NN_003
      Models/07_SuperLearner/SL_003/         -- comparacion_sl2_sl3.png
                                             -- pesos_nnls_ganador.png
                                             -- cv_comparacion_ganador.png
                                             -- correlacion_bases.png
                                             -- mae_individuales_vs_sl.png

  03_submissions/
      submission_LR_001_20260517.csv
      submission_EN_002_l1r100_a2en02.csv
      CART_d10_leaf20_ccp00001_cv5_CART_002.csv
      RF_ntrees500_d15_leaf5_mfsqrt_cv5_gridESP_RF_005.csv
      XGB_knn30_latlon_earlystop_XGB_009.csv
      submission_NN_003_ep182_p118k.csv
      submission_SL_003_YYYYMMDD.csv         <-- mejor MAE Kaggle: $199,851,632

{SEPARADOR}
""")
