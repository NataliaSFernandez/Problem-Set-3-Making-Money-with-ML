"""
XGBoost - XGB_010 ENSEMBLE
============================
Ensemble simple de XGB_007 y XGB_009 — promedio de predicciones.

Justificacion (SuperLearners, cuaderno del curso):
  El SuperLearner es un ensemble donde los pesos de cada modelo se
  aprenden de los datos. El ensemble simple es la version base del
  mismo principio — todos los modelos tienen el mismo peso (0.5 cada uno).

  Cada modelo comete errores diferentes:
  - XGB_007 (lat/lon + early stopping): MAE Kaggle = 225M
  - XGB_009 (lat/lon + KNN vecinos):    MAE Kaggle = 207M

  Al promediar, los errores individuales se compensan parcialmente
  y el error promedio tiende a ser menor que el de cualquier modelo solo.

Pipeline
--------
1. Leer submission XGB_007
2. Leer submission XGB_009
3. Promediar precio por property_id
4. Generar nueva submission
5. Registro -> 02_outputs/model_registry.xlsx

Estructura de carpetas generada automaticamente al correr el script:
  <repo>/
  ├── 03_submissions/  <- submissions de XGB_007 y XGB_009 (input)
  └── 03_submissions/  <- nueva submission ensemble (output)
"""

# =============================================================================
# SECCION -1: INSTALACION AUTOMATICA DE DEPENDENCIAS
# =============================================================================

import subprocess
import sys

def instalar(paquete):
    subprocess.check_call([sys.executable, "-m", "pip", "install", paquete, "-q"])

try:
    import openpyxl
except ImportError:
    print("  Instalando openpyxl...")
    instalar("openpyxl")


# =============================================================================
# SECCION 0: IMPORTACIONES
# =============================================================================

import warnings
from datetime import date
from pathlib import Path

import numpy as np
import openpyxl  # noqa: F401
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# SECCION 1: CONFIGURACION
# =============================================================================

MODEL_ID = "XGB_010"

BASE        = Path(__file__).parent.parent.parent.parent
SUBMISSIONS = BASE / "03_submissions"
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

# Submissions base del ensemble
SUB_007 = SUBMISSIONS / "XGB_latlon_earlystop_XGB_007.csv"
SUB_009 = SUBMISSIONS / "XGB_knn30_latlon_earlystop_XGB_009.csv"

# Pesos del ensemble — iguales por ahora
# XGB_009 es el mejor individualmente pero XGB_007 aporta diversidad
PESO_007 = 0.5
PESO_009 = 0.5


# =============================================================================
# SECCION 2: ENSEMBLE
# =============================================================================

def generar_ensemble():
    """
    Promedio ponderado de las predicciones de XGB_007 y XGB_009.

    Se hace en escala de precios (COP), no en log-precio, porque
    el MAE de Kaggle se calcula en COP. Promediar en log y luego
    aplicar exp da resultados diferentes a promediar directamente
    en COP — el promedio en COP es lo correcto para minimizar MAE.
    """
    print(f"  Leyendo XGB_007: {SUB_007.name}")
    df_007 = pd.read_csv(SUB_007)

    print(f"  Leyendo XGB_009: {SUB_009.name}")
    df_009 = pd.read_csv(SUB_009)

    # Verificar que tienen los mismos property_id en el mismo orden
    assert list(df_007["property_id"]) == list(df_009["property_id"]), \
        "Los property_id no coinciden entre las dos submissions"

    # Promedio ponderado en escala COP
    precio_ensemble = PESO_007 * df_007["price"] + PESO_009 * df_009["price"]

    sub = pd.DataFrame({
        "property_id": df_007["property_id"],
        "price":       precio_ensemble,
    })

    print(f"  Precio promedio ensemble: ${precio_ensemble.mean():,.0f} COP")
    print(f"  Precio XGB_007:           ${df_007['price'].mean():,.0f} COP")
    print(f"  Precio XGB_009:           ${df_009['price'].mean():,.0f} COP")

    sub_name = f"XGB_ensemble_007_009_{MODEL_ID}.csv"
    sub.to_csv(SUBMISSIONS / sub_name, index=False)
    print(f"  Submission: 03_submissions/{sub_name}  ({len(sub):,} filas)")
    return sub_name


# =============================================================================
# SECCION 3: REGISTRO
# =============================================================================

def registrar(sub_name):
    nueva = {
        "model_id":          MODEL_ID,
        "fecha":             str(date.today()),
        "algoritmo":         "Ensemble",
        "n_features":        "XGB_007 + XGB_009",
        "notas": (
            f"Ensemble simple: promedio ponderado XGB_007 (w={PESO_007}) "
            f"y XGB_009 (w={PESO_009}). "
            f"XGB_007 MAE Kaggle=225M, XGB_009 MAE Kaggle=207M. "
            f"Promedio en escala COP para minimizar MAE directamente."
        ),
        "kaggle_public_MAE": None,
        "submission_file":   sub_name,
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
# SECCION 4: MAIN
# =============================================================================

def main():
    print(f"{'='*60}")
    print(f"  ENSEMBLE — {MODEL_ID}  (XGB_007 x{PESO_007} + XGB_009 x{PESO_009})")
    print(f"{'='*60}")

    print("\n[1/2] Generando ensemble...")
    sub_name = generar_ensemble()

    print("\n[2/2] Registrando...")
    registrar(sub_name)

    print(f"\n{'='*60}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"  Modelos: XGB_007 (w={PESO_007}) + XGB_009 (w={PESO_009})")
    print(f"  Submission: 03_submissions/{sub_name}")
    print(f"  Subir a Kaggle y anotar MAE publico en registry")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
