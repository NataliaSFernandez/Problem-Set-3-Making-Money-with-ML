"""
SuperLearner — SL_003
=====================
Problem Set 3 — MECA 4107 (Big Data & ML para Economía Aplicada)

Compara dos configuraciones de SuperLearner con meta-aprendiz NNLS:
  · SL-2: bases = [XGB_009, NN_003]
  · SL-3: bases = [XGB_009, NN_003, RF_005]

Al finalizar guarda submission, gráficos y registro SOLO del ganador
(el de menor MAE_esp en CV espacial).

Meta-aprendiz: NNLS — LinearRegression(positive=True)
------------------------------------------------------
NNLS implementa la combinación convexa recomendada por van der Laan et al.
(2007): estima pesos w_i ≥ 0 minimizando el error OOF, a diferencia del
promedio simple que fija w_i = 1/k. Si SL-3 aporta diversidad útil con
RF_005, NNLS aprenderá w_RF > 0; si no, lo descartará (w_RF → 0).

Selección entre SL-2 y SL-3
----------------------------
Criterio: MAE_esp (CV espacial GroupKFold 5×5). El CV espacial simula
predecir en zonas geográficas no vistas (como Chapinero). El ganador es
el que obtiene menor MAE_esp. En caso de empate estadístico (diferencia
< 1 std), se elige SL-2 por parsimonia.

Estrategia computacional
------------------------
OOF y CV espacial se corren UNA sola vez para los 3 modelos en paralelo
(mismo fold, misma muestra). Luego se evalúan NNLS sobre dos sub-matrices:
  Z2 = [XGB_OOF | NN_OOF]         → SL-2
  Z3 = [XGB_OOF | NN_OOF | RF_OOF] → SL-3
Los pesos NNLS finales se aprenden sobre el OOF aleatorio completo y
se aplican en el CV espacial (aproximación estándar en práctica: evita
25 re-entrenamientos de NN en el CV espacial anidado).

Pipeline
--------
1.  Carga  train_final.csv / test_final.csv
2.  Tuning XGB_009 (n_iter=20, GroupKFold espacial)
3.  precio_vecinos (K=30, LOO haversine) sobre train completo
4.  OOF 5-fold para XGB_009 + NN_003 + RF_005 (un solo pase)
5.  Evaluación NNLS random: SL-2 vs SL-3
6.  Evaluación NNLS espacial: SL-2 vs SL-3
7.  Selección del ganador por MAE_esp
8.  Modelo final del ganador en todo el train
9.  Submission + gráficos + registro (solo del ganador)

Outputs
-------
  02_outputs/Models/SuperLearner/SL_003/comparacion_sl2_sl3.png    ← nuevo: SL-2 vs SL-3
  02_outputs/Models/SuperLearner/SL_003/pesos_nnls_ganador.png     ← equiv. pesos_meta (SL_002)
  02_outputs/Models/SuperLearner/SL_003/cv_comparacion_ganador.png ← equiv. cv_comparacion (SL_002)
  02_outputs/Models/SuperLearner/SL_003/correlacion_bases.png      ← equiv. correlacion_bases (SL_002)
  02_outputs/Models/SuperLearner/SL_003/mae_individuales_vs_sl.png ← equiv. mae_individuales (SL_002)
  03_submissions/submission_SL_003_YYYYMMDD.csv
  02_outputs/model_registry.xlsx
"""

# =============================================================================
# SECCIÓN -1: INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS
# =============================================================================

import os, subprocess, sys

# Evita conflictos OpenMP en macOS cuando XGBoost + sklearn (RF) usan
# múltiples threads en el mismo proceso → segfault.
# Debe setearse ANTES de importar numpy/scipy/sklearn.
os.environ.setdefault("OMP_NUM_THREADS",     "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

def _instalar(paquete: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", paquete, "-q"])

for _pkg in ["openpyxl", "scipy"]:
    try:
        __import__(_pkg)
    except ImportError:
        print(f"  Instalando {_pkg}...")
        _instalar(_pkg)

# xgboost: instalar el paquete Python, pero el import puede fallar igualmente
# si falta la librería de sistema libomp (macOS). Se captura más abajo.
try:
    _instalar("xgboost")
except Exception:
    pass


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
import torch
import torch.nn as nn
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# XGBoost puede fallar en macOS si falta libomp (brew install libomp).
# Fallback automático a GradientBoostingRegressor de sklearn con parámetros
# equivalentes aproximados — mismo comportamiento, sin dependencia de sistema.
try:
    from xgboost import XGBRegressor
    _USE_XGB = True
except Exception:
    _USE_XGB = False
    print("  ⚠  xgboost no disponible (¿falta libomp?). "
          "Usando GradientBoostingRegressor (sklearn) como fallback.")

warnings.filterwarnings("ignore")


# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================

AUTOR    = "Equipo"
MODEL_ID = "SL_003"
SEED     = 42

CV_FOLDS     = 5
SPATIAL_GRID = 5
K_VECINOS    = 30
N_ITER_TUNE  = 20

# ── RF_005: hiperparámetros óptimos del model_registry ────────────────────────
# Fuente: BEST_PARAMS de SL_002 → RF_005 (MAE_esp = 0.16414).
# Se usan las mismas features que XGB_009 (incluido precio_vecinos) para
# que la diversidad del ensemble provenga del algoritmo (bagging vs boosting)
# y no de las features. Esto maximiza la señal de la comparación SL-2 vs SL-3.
RF_PARAMS = {
    "n_estimators":     500,
    "max_depth":         15,
    "min_samples_leaf":   5,
    "max_features":    "sqrt",
    "random_state":     SEED,
    "n_jobs":              1,   # n_jobs=-1 junto con XGBoost causa segfault en macOS
}

# ── NN_003: hiperparámetros fijos (idénticos al script original) ─────────────
NN_EPOCHS     = 300
NN_BATCH_SIZE = 512
NN_VAL_SPLIT  = 0.2
NN_PATIENCE   = 30
NN_LR         = 1e-3
NN_L1_REG     = 5e-5
NN_L2_REG     = 5e-5
NN_DROPOUT_1  = 0.2
NN_DROPOUT_2  = 0.1
NN_SEED       = 101010

# MPS (Apple Silicon) tiene problemas de estabilidad con BatchNorm1d y
# DataLoader dentro de loops repetidos (segfault en algunos builds de PyTorch).
# Se fuerza CPU para garantizar reproducibilidad en cualquier máquina.
# Si quieres MPS en tu entorno y sabes que es estable, cambia a:
#   DEVICE = torch.device("mps")
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))

BASE        = Path(__file__).parent.parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "SuperLearner" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

for d in [SUBMISSIONS, DIR_MODEL, BASE / "02_outputs"]:
    d.mkdir(parents=True, exist_ok=True)

# Espacio de búsqueda para XGBoost; si no está disponible se usa el fallback GBR.
XGB_PARAM_DIST = {
    "max_depth":        randint(3, 10),
    "learning_rate":    uniform(0.01, 0.19),
    "subsample":        uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 10),
    "gamma":            uniform(0, 0.5),
    "reg_lambda":       uniform(0.5, 4.5),
    "reg_alpha":        uniform(0, 1.0),
}

# GradientBoostingRegressor no acepta parámetros XGB-específicos.
# Se usa este espacio reducido cuando xgboost no está disponible.
GBR_PARAM_DIST = {
    "max_depth":        randint(3, 10),
    "learning_rate":    uniform(0.01, 0.19),
    "subsample":        uniform(0.6, 0.4),
    "min_samples_leaf": randint(1, 10),
}


def _make_xgb(params: dict):
    """
    Instancia XGBRegressor o GradientBoostingRegressor según disponibilidad.

    Cuando xgboost falla por falta de libomp (macOS sin brew install libomp)
    o cualquier otro error de sistema, se cae automáticamente a sklearn GBR.
    La traducción de parámetros es aproximada:
      min_child_weight → min_samples_leaf  (control de complejidad del árbol)
      colsample_bytree, gamma, reg_lambda, reg_alpha → descartados (sin equiv.)
    El rendimiento del fallback es ligeramente inferior al de XGBoost nativo,
    pero garantiza que el script corre en cualquier entorno sin configuración.
    """
    if _USE_XGB:
        xgb_keys = {"max_depth", "learning_rate", "subsample", "colsample_bytree",
                    "min_child_weight", "gamma", "reg_lambda", "reg_alpha",
                    "n_estimators", "random_state", "n_jobs", "verbosity"}
        clean = {k: v for k, v in params.items() if k in xgb_keys}
        # n_jobs=1 evita conflicto OpenMP con RF cuando ambos corren en el mismo proceso
        return XGBRegressor(**clean, random_state=SEED, n_jobs=1, verbosity=0)
    else:
        gbr_params = {
            "n_estimators":     params.get("n_estimators", 300),
            "max_depth":        params.get("max_depth", 6),
            "learning_rate":    params.get("learning_rate", 0.1),
            "subsample":        params.get("subsample", 0.8),
            "min_samples_leaf": params.get("min_child_weight",
                                           params.get("min_samples_leaf", 1)),
            "random_state":     SEED,
        }
        return GradientBoostingRegressor(**gbr_params)


def _param_dist_xgb() -> dict:
    """Devuelve el espacio de búsqueda correcto según el backend disponible."""
    return XGB_PARAM_DIST if _USE_XGB else GBR_PARAM_DIST


def _base_model_xgb():
    """Modelo base para RandomizedSearchCV (sin parámetros tuneados aún)."""
    if _USE_XGB:
        return XGBRegressor(n_estimators=300, random_state=SEED,
                            n_jobs=-1, verbosity=0)
    else:
        return GradientBoostingRegressor(n_estimators=300, random_state=SEED)


# =============================================================================
# SECCIÓN 2: FEATURES XGB_009 / RF_005 (compartidas)
# =============================================================================
# XGB y RF comparten el mismo pipeline de features: árboles son invariantes
# a monotransformaciones → no se aplican log-transforms. RF usa las mismas
# features que XGB (incluyendo precio_vecinos) para que la diversidad del
# ensemble venga del algoritmo (bagging vs boosting), no de las features.

XGB_STRUCTURAL = [
    "surface_total", "surface_covered", "rooms",
    "bedrooms", "bathrooms", "month", "year",
]
XGB_OSM = [
    "dist_cbd_km", "dist_transmilenio_m", "dist_via_arterial_m",
    "dist_hospital_m", "dist_centro_com_m", "dist_parque_m",
    "n_restaurantes_500m", "n_bancos_500m", "walkability_score", "densidad_vial",
]
XGB_TEXT = [
    "remodelado", "vista_panoramica", "deposito", "conjunto_cerrado",
    "balcon_terraza", "tfidf_premium", "parqueaderos_txt", "piso_txt",
    "gimnasio", "amenidades", "num_amenidades",
]
XGB_GEO = ["lat", "lon"]


def construir_features_xgb(df: pd.DataFrame,
                            fit_cols: list | None = None) -> pd.DataFrame:
    d = df.copy()
    d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)
    all_cols = (XGB_STRUCTURAL + XGB_OSM + XGB_TEXT + XGB_GEO
                + ["precio_vecinos", "es_apartamento"])
    cols = [c for c in all_cols if c in d.columns]
    X = d[cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())
    if fit_cols is not None:
        X = X.reindex(columns=fit_cols, fill_value=0)
    return X


# =============================================================================
# SECCIÓN 3: FEATURES NN_003
# =============================================================================
# NN requiere log-transforms y polinomio espacial porque el gradiente
# descendente es sensible a la escala y distribución de los inputs.
# Diferente a XGB/RF: la diversidad algorítmica XGB↔NN es mayor que XGB↔RF.

NN_STRUCTURAL = [
    "log_surface_total", "log_surface_covered", "surface_ratio",
    "rooms", "rooms_sq", "bedrooms", "bathrooms", "bath_surface",
    "month", "year",
]
NN_TEXT = [
    "remodelado", "vista_panoramica", "deposito", "conjunto_cerrado",
    "balcon_terraza", "tfidf_premium", "parqueaderos_txt", "piso_txt",
    "gimnasio", "amenidades", "num_amenidades",
]
NN_OSM = [
    "log_dist_cbd_km", "log_dist_transmilenio_m", "log_dist_via_arterial_m",
    "log_dist_hospital_m", "log_dist_centro_com_m", "log_dist_parque_m",
    "n_restaurantes_500m", "n_bancos_500m", "walkability_score", "densidad_vial",
]
NN_SPATIAL_POLY = ["lat", "lon", "lat_sq", "lon_sq", "lat_lon"]


def _derivar_features_nn(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_surface_total"]   = np.log1p(d["surface_total"])
    d["log_surface_covered"] = np.log1p(d["surface_covered"])
    d["surface_ratio"]       = d["surface_covered"] / (d["surface_total"] + 1)
    d["rooms_sq"]            = d["rooms"] ** 2
    d["bath_surface"]        = d["bathrooms"] * np.log1p(d["surface_covered"])
    for col in ["dist_cbd_km", "dist_transmilenio_m", "dist_via_arterial_m",
                "dist_hospital_m", "dist_centro_com_m", "dist_parque_m"]:
        d[f"log_{col}"] = np.log1p(d[col]) if col in d.columns else 0.0
    d["lat_sq"]  = d["lat"] ** 2
    d["lon_sq"]  = d["lon"] ** 2
    d["lat_lon"] = d["lat"] * d["lon"]
    d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)
    return d


def construir_features_nn(df: pd.DataFrame,
                          fit_cols: list | None = None) -> pd.DataFrame:
    d    = _derivar_features_nn(df)
    cols = [c for c in (NN_STRUCTURAL + NN_TEXT + NN_OSM
                        + NN_SPATIAL_POLY + ["es_apartamento"])
            if c in d.columns]
    X = d[cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())
    if fit_cols is not None:
        X = X.reindex(columns=fit_cols, fill_value=0)
    return X


# =============================================================================
# SECCIÓN 4: ARQUITECTURA NN_003 (PyTorch)
# =============================================================================

class MLP(nn.Module):
    """p → 256 → 256 → 128 → 64 → 1  (BN + ReLU + Dropout, igual que NN_003)."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        d1, d2 = NN_DROPOUT_1, NN_DROPOUT_2
        self.red = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(256, 256),        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(d2),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_pesos()

    def _init_pesos(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x).squeeze(-1)


def _l1(model: MLP) -> torch.Tensor:
    device = next(model.parameters()).device
    total  = torch.zeros(1, device=device)
    for name, p in model.named_parameters():
        if "weight" in name:
            total = total + p.abs().sum()
    return total


def entrenar_nn(X_sc: np.ndarray, y: np.ndarray) -> MLP:
    """Entrena MLP con early stopping sobre split interno (VAL_SPLIT=0.2)."""
    torch.manual_seed(NN_SEED)
    n     = len(X_sc)
    n_val = int(n * NN_VAL_SPLIT)
    X_tr, X_val = X_sc[:n - n_val], X_sc[n - n_val:]
    y_tr, y_val = y[:n - n_val],    y[n - n_val:]

    def _t(a): return torch.tensor(a, dtype=torch.float32).to(DEVICE)

    loader = DataLoader(
        TensorDataset(_t(X_tr), _t(y_tr)),
        batch_size=NN_BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(NN_SEED),
    )

    model = MLP(X_sc.shape[1]).to(DEVICE)
    with torch.no_grad():
        model.red[-1].bias.fill_(float(y_tr.mean()))

    optimizer  = torch.optim.Adam(model.parameters(), lr=NN_LR, weight_decay=NN_L2_REG)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-6)
    criterion  = nn.MSELoss()
    best_loss  = float("inf")
    best_state = None
    no_improve = 0
    X_val_t, y_val_t = _t(X_val), _t(y_val)

    for _ in range(1, NN_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if NN_L1_REG > 0:
                loss = loss + NN_L1_REG * _l1(model)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= NN_PATIENCE:
                break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


def predecir_nn(model: MLP, X_sc: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_sc, dtype=torch.float32).to(DEVICE)).cpu().numpy()


# =============================================================================
# SECCIÓN 5: PRECIO VECINOS (K=30, LOO haversine)
# =============================================================================

def _knn_precio(ref_df: pd.DataFrame, query_df: pd.DataFrame,
                k: int, loo: bool) -> np.ndarray:
    """
    Precio mediano de K vecinos más cercanos (haversine, BallTree).
    loo=True: excluye la propiedad misma para evitar leakage en train.
    loo=False: para test o validación (no tienen precio propio que filtrar).
    """
    coords_ref   = np.radians(ref_df[["lat", "lon"]].values)
    coords_query = np.radians(query_df[["lat", "lon"]].values)
    log_price    = np.log(ref_df["price"].values)
    tree         = BallTree(coords_ref, metric="haversine")
    if loo:
        _, idx = tree.query(coords_query, k=k + 1)
        return np.array([np.median(log_price[i[1:]]) for i in idx])
    else:
        _, idx = tree.query(coords_query, k=k)
        return np.array([np.median(log_price[i]) for i in idx])


def agregar_precio_vecinos(train_df: pd.DataFrame,
                           test_df: pd.DataFrame) -> tuple:
    """Precio vecinos sobre el train completo (para tuning y modelo final)."""
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["precio_vecinos"] = _knn_precio(train_df, train_df, K_VECINOS, loo=True)
    test_df["precio_vecinos"]  = _knn_precio(train_df, test_df,  K_VECINOS, loo=False)
    return train_df, test_df


def _agregar_pv_fold(tr_df: pd.DataFrame, va_df: pd.DataFrame) -> tuple:
    """
    Precio vecinos dentro de un fold de CV (sin leakage de validación).
    XGB y RF comparten este cálculo en cada fold → se llama UNA sola vez
    y los resultados se usan para construir las features de ambos modelos.
    """
    tr_df = tr_df.copy()
    va_df = va_df.copy()
    tr_df["precio_vecinos"] = _knn_precio(tr_df, tr_df, K_VECINOS, loo=True)
    va_df["precio_vecinos"] = _knn_precio(tr_df, va_df, K_VECINOS, loo=False)
    return tr_df, va_df


# =============================================================================
# SECCIÓN 6: GRUPOS ESPACIALES
# =============================================================================

def asignar_grupos(df: pd.DataFrame, n: int = SPATIAL_GRID) -> np.ndarray:
    lat = df["lat"].values
    lon = df["lon"].values
    lat_n = (lat - lat.min()) / (lat.max() - lat.min() + 1e-9) * n
    lon_n = (lon - lon.min()) / (lon.max() - lon.min() + 1e-9) * n
    fila  = np.floor(lat_n).astype(int).clip(0, n - 1)
    col   = np.floor(lon_n).astype(int).clip(0, n - 1)
    return fila * n + col


# =============================================================================
# SECCIÓN 7: TUNING XGB_009
# =============================================================================

def tuning_xgb(train_xgb: pd.DataFrame,
               y: np.ndarray,
               grupos: np.ndarray) -> dict:
    """
    RandomizedSearchCV con GroupKFold espacial para XGB_009.
    Usa train_xgb (con precio_vecinos del train completo) porque el tuning
    solo determina hiperparámetros, no predicciones de CV → el leakage leve
    de precio_vecinos es aceptable aquí.
    """
    print(f"  Tuning XGB_009 (n_iter={N_ITER_TUNE}, CV espacial)...")
    X   = construir_features_xgb(train_xgb)
    gkf = GroupKFold(n_splits=CV_FOLDS)
    rs  = RandomizedSearchCV(
        _base_model_xgb(),
        param_distributions=_param_dist_xgb(),
        n_iter=N_ITER_TUNE, cv=gkf,
        scoring="neg_mean_absolute_error",
        random_state=SEED, n_jobs=-1, refit=False, verbose=0,
    )
    rs.fit(X, y, groups=grupos)
    best = rs.best_params_.copy()
    best["n_estimators"] = 300
    print(f"  Mejores params XGB: {best}")
    print(f"  MAE_esp tuning XGB: {-rs.best_score_:.5f}")
    return best


# =============================================================================
# SECCIÓN 8: OOF PARA LOS 3 MODELOS (un solo pase de CV)
# =============================================================================

def generar_oof_todos(train_df: pd.DataFrame,
                      y: np.ndarray,
                      kf: KFold,
                      xgb_params: dict) -> tuple:
    """
    Genera OOF para XGB_009, NN_003 y RF_005 en UN solo pase de CV.

    Eficiencia: precio_vecinos se calcula UNA sola vez por fold y se
    comparte entre XGB y RF (ambos usan el mismo feature set). Esto evita
    duplicar la búsqueda BallTree (O(n log n)) que es el paso más costoso
    del pipeline de XGB_009.

    Garantías anti-leakage por fold:
      · precio_vecinos: recomputado en _agregar_pv_fold() → val no contamina train.
      · StandardScaler (NN): fit solo sobre el fold de entrenamiento.
      · RF y XGB no escalan, pero usan las mismas features sin leakage.

    Retorna: (oof_xgb, oof_nn, oof_rf)
    """
    n      = len(y)
    oof_xgb = np.zeros(n)
    oof_nn  = np.zeros(n)
    oof_rf  = np.zeros(n)

    print(f"  Generando OOF ({CV_FOLDS} folds, 3 modelos)...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[val_idx].reset_index(drop=True)
        y_tr  = y[tr_idx]
        y_va  = y[val_idx]

        # ── precio_vecinos compartido para XGB y RF ───────────────────────────
        tr_xf, va_xf = _agregar_pv_fold(tr_df, va_df)
        X_tr_xf = construir_features_xgb(tr_xf)
        X_va_xf = construir_features_xgb(va_xf, fit_cols=X_tr_xf.columns.tolist())

        # ── XGB_009 ───────────────────────────────────────────────────────────
        xgb_m = _make_xgb(xgb_params)
        xgb_m.fit(X_tr_xf, y_tr)
        oof_xgb[val_idx] = xgb_m.predict(X_va_xf)

        # ── RF_005 ────────────────────────────────────────────────────────────
        rf_m = RandomForestRegressor(**RF_PARAMS)
        rf_m.fit(X_tr_xf, y_tr)
        oof_rf[val_idx] = rf_m.predict(X_va_xf)

        # ── NN_003 ────────────────────────────────────────────────────────────
        X_tr_nn = construir_features_nn(tr_df)
        X_va_nn = construir_features_nn(va_df, fit_cols=X_tr_nn.columns.tolist())
        scaler  = StandardScaler()
        nn_m    = entrenar_nn(scaler.fit_transform(X_tr_nn), y_tr)
        oof_nn[val_idx] = predecir_nn(nn_m, scaler.transform(X_va_nn))

        mae_xgb = mean_absolute_error(y_va, oof_xgb[val_idx])
        mae_nn  = mean_absolute_error(y_va, oof_nn[val_idx])
        mae_rf  = mean_absolute_error(y_va, oof_rf[val_idx])
        print(f"    Fold {fold+1}/{CV_FOLDS}: "
              f"XGB={mae_xgb:.5f} | NN={mae_nn:.5f} | RF={mae_rf:.5f}")

    return oof_xgb, oof_nn, oof_rf


# =============================================================================
# SECCIÓN 9: EVALUACIÓN NNLS (CV aleatorio sobre OOF)
# =============================================================================

def cv_nnls_meta(Z_oof: np.ndarray,
                 y: np.ndarray,
                 kf: KFold) -> tuple:
    """
    Evalúa el meta-aprendiz NNLS con KFold CV sobre la matriz OOF Z.

    Cada observación es predicha por un NNLS entrenado en los otros folds
    de Z — garantía de evaluación sin sesgo (van der Laan, 2007 §3).
    Z contiene predicciones OOF válidas (cada fila fue generada por un modelo
    que nunca vio esa observación), por lo que esta segunda capa de CV sobre
    Z es correcta.

    Retorna: (mae_rand, std_rand, coefs_medios)
    """
    maes  = []
    coefs = []
    for tr_idx, val_idx in kf.split(Z_oof):
        meta = LinearRegression(positive=True)
        meta.fit(Z_oof[tr_idx], y[tr_idx])
        maes.append(mean_absolute_error(y[val_idx], meta.predict(Z_oof[val_idx])))
        coefs.append(meta.coef_)
    return float(np.mean(maes)), float(np.std(maes)), np.mean(coefs, axis=0)


# =============================================================================
# SECCIÓN 10: CV ESPACIAL (un solo pase para los 3 modelos)
# =============================================================================

def cv_espacial_todos(train_df: pd.DataFrame,
                      y: np.ndarray,
                      grupos: np.ndarray,
                      xgb_params: dict) -> tuple:
    """
    CV espacial (GroupKFold) para los 3 modelos en un solo pase.

    Retorna las predicciones espaciales concatenadas de los 3 modelos y
    el target espacial. Con estos arrays se pueden evaluar AMBAS configuraciones
    (SL-2 y SL-3) sin re-entrenar ningún modelo.

    Aproximación en los pesos NNLS
    --------------------------------
    Los pesos NNLS se aprenden sobre el OOF ALEATORIO (Sección 9), no sobre
    el OOF espacial. Aplicarlos aquí es una aproximación: los pesos óptimos
    para CV aleatorio pueden diferir ligeramente de los óptimos para CV
    espacial. Esta aproximación evita el costo de 25 entrenamientos de NN
    adicionales (5 folds externos × 5 folds internos) que requeriría el
    enfoque anidado completo.

    Garantías anti-leakage
    ----------------------
    precio_vecinos se recomputa por fold espacial: las observaciones del
    bloque geográfico de validación NO contribuyen al cálculo de precio_vecinos
    del bloque de entrenamiento.
    """
    gkf          = GroupKFold(n_splits=CV_FOLDS)
    y_esp_list   = []
    p_xgb_list   = []
    p_nn_list    = []
    p_rf_list    = []

    print(f"\n  CV espacial ({CV_FOLDS} folds geográficos, 3 modelos)...")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_df, y, groups=grupos)):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[val_idx].reset_index(drop=True)
        y_tr, y_val = y[tr_idx], y[val_idx]

        # ── precio_vecinos compartido XGB + RF ────────────────────────────────
        tr_xf, va_xf = _agregar_pv_fold(tr_df, va_df)
        X_tr_xf = construir_features_xgb(tr_xf)
        X_va_xf = construir_features_xgb(va_xf, fit_cols=X_tr_xf.columns.tolist())

        xgb_m = _make_xgb(xgb_params)
        xgb_m.fit(X_tr_xf, y_tr)
        p_xgb = xgb_m.predict(X_va_xf)

        rf_m = RandomForestRegressor(**RF_PARAMS)
        rf_m.fit(X_tr_xf, y_tr)
        p_rf = rf_m.predict(X_va_xf)

        # ── NN_003 ────────────────────────────────────────────────────────────
        X_tr_nn = construir_features_nn(tr_df)
        X_va_nn = construir_features_nn(va_df, fit_cols=X_tr_nn.columns.tolist())
        scaler  = StandardScaler()
        nn_m    = entrenar_nn(scaler.fit_transform(X_tr_nn), y_tr)
        p_nn    = predecir_nn(nn_m, scaler.transform(X_va_nn))

        mae2 = mean_absolute_error(y_val, (p_xgb + p_nn) / 2)
        mae3 = mean_absolute_error(y_val, (p_xgb + p_nn + p_rf) / 3)
        print(f"    Fold {fold+1}/{CV_FOLDS}: "
              f"XGB={mean_absolute_error(y_val, p_xgb):.5f} | "
              f"NN={mean_absolute_error(y_val, p_nn):.5f} | "
              f"RF={mean_absolute_error(y_val, p_rf):.5f} | "
              f"avg2={mae2:.5f} | avg3={mae3:.5f}")

        y_esp_list.append(y_val)
        p_xgb_list.append(p_xgb)
        p_nn_list.append(p_nn)
        p_rf_list.append(p_rf)

    return (np.concatenate(y_esp_list),
            np.concatenate(p_xgb_list),
            np.concatenate(p_nn_list),
            np.concatenate(p_rf_list))


# =============================================================================
# SECCIÓN 11: SELECCIÓN DEL GANADOR
# =============================================================================

def seleccionar_ganador(mae_esp_2: float, std_esp_2: float,
                        mae_esp_3: float, std_esp_3: float) -> str:
    """
    Selecciona el ganador por MAE_esp.

    Criterio de parsimonia: si la diferencia entre SL-2 y SL-3 es menor
    que 1 std del ganador en CV espacial (diferencia no significativa),
    se prefiere SL-2 por tener menos parámetros en el meta-aprendiz
    (2 pesos vs 3 → menor varianza de estimación).
    """
    diff = abs(mae_esp_2 - mae_esp_3)
    std_ganador = std_esp_2 if mae_esp_2 <= mae_esp_3 else std_esp_3
    if diff < std_ganador:
        # Diferencia no significativa → parsimonia → SL-2
        print(f"  Diferencia |Δ|={diff:.5f} < std={std_ganador:.5f} → empate estadístico.")
        print(f"  Se elige SL-2 por parsimonia (menos parámetros en meta-aprendiz).")
        return "SL-2"
    ganador = "SL-2" if mae_esp_2 < mae_esp_3 else "SL-3"
    print(f"  Ganador: {ganador}  (Δ={mae_esp_3 - mae_esp_2:+.5f})")
    return ganador


# =============================================================================
# SECCIÓN 12: MODELO FINAL DEL GANADOR
# =============================================================================

def entrenar_modelo_final(train_xgb: pd.DataFrame,
                          train_df: pd.DataFrame,
                          y: np.ndarray,
                          test_xgb: pd.DataFrame,
                          test_df: pd.DataFrame,
                          xgb_params: dict,
                          ganador: str,
                          meta_ganador: LinearRegression) -> tuple:
    """
    Entrena los modelos base del ganador sobre todo el train.
    Los pesos del meta-aprendiz (meta_ganador) ya fueron aprendidos sobre
    el OOF completo: no se re-aprenden aquí para evitar sobreajuste in-sample.

    Retorna: (y_pred_test, feat_xgb, feat_nn)
    """
    print(f"  Entrenando XGB_009 final...")
    X_tr_xgb = construir_features_xgb(train_xgb)
    X_te_xgb = construir_features_xgb(test_xgb, fit_cols=X_tr_xgb.columns.tolist())
    xgb_f    = XGBRegressor(**xgb_params, random_state=SEED, n_jobs=-1, verbosity=0)
    xgb_f.fit(X_tr_xgb, y)
    pred_xgb_test = xgb_f.predict(X_te_xgb)
    print(f"    XGB_009 listo ({X_tr_xgb.shape[1]} features)")

    print(f"  Entrenando NN_003 final...")
    X_tr_nn  = construir_features_nn(train_df)
    X_te_nn  = construir_features_nn(test_df, fit_cols=X_tr_nn.columns.tolist())
    scaler   = StandardScaler()
    nn_f     = entrenar_nn(scaler.fit_transform(X_tr_nn), y)
    pred_nn_test = predecir_nn(nn_f, scaler.transform(X_te_nn))
    print(f"    NN_003 listo ({X_tr_nn.shape[1]} features)")

    if ganador == "SL-3":
        print(f"  Entrenando RF_005 final...")
        rf_f = RandomForestRegressor(**RF_PARAMS)
        rf_f.fit(X_tr_xgb, y)
        pred_rf_test = rf_f.predict(X_te_xgb)
        print(f"    RF_005 listo")
        Z_test = np.column_stack([pred_xgb_test, pred_nn_test, pred_rf_test])
    else:
        Z_test = np.column_stack([pred_xgb_test, pred_nn_test])

    y_pred_test = meta_ganador.predict(Z_test)

    # MAE in-sample del ensemble final
    if ganador == "SL-3":
        Z_tr = np.column_stack([
            xgb_f.predict(X_tr_xgb),
            predecir_nn(nn_f, scaler.transform(X_tr_nn)),
            rf_f.predict(X_tr_xgb),
        ])
    else:
        Z_tr = np.column_stack([
            xgb_f.predict(X_tr_xgb),
            predecir_nn(nn_f, scaler.transform(X_tr_nn)),
        ])
    mae_train = float(mean_absolute_error(y, meta_ganador.predict(Z_tr)))

    return y_pred_test, X_tr_xgb.columns.tolist(), X_tr_nn.columns.tolist(), mae_train


# =============================================================================
# SECCIÓN 13: DIAGNÓSTICOS
# =============================================================================

def plot_comparacion_configs(mae_rand_2: float, std_rand_2: float,
                             mae_esp_2:  float, std_esp_2:  float,
                             mae_rand_3: float, std_rand_3: float,
                             mae_esp_3:  float, std_esp_3:  float,
                             ganador: str) -> None:
    """
    Gráfico de barras comparando SL-2 vs SL-3 en CV aleatorio y CV espacial.
    Permite visualizar si RF_005 aporta ganancia real en generalización espacial.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    configs = ["SL-2\n(XGB+NN)", "SL-3\n(XGB+NN+RF)"]
    colors  = ["#3498db", "#27ae60"]

    for ax, (mae2, std2, mae3, std3, titulo) in zip(
        axes,
        [(mae_rand_2, std_rand_2, mae_rand_3, std_rand_3, "CV Aleatorio (5-fold)"),
         (mae_esp_2,  std_esp_2,  mae_esp_3,  std_esp_3,  "CV Espacial (5×5 grid)")]
    ):
        bars = ax.bar(configs, [mae2, mae3], yerr=[std2, std3],
                      color=colors, capsize=6, edgecolor="white", alpha=0.85)
        ax.set_ylabel("MAE log(price)")
        ax.set_title(titulo)
        delta = mae3 - mae2
        ax.set_xlabel(f"Δ(SL3−SL2) = {delta:+.5f}  |  Ganador: {ganador}",
                      fontsize=9)
        for bar, val in zip(bars, [mae2, mae3]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.5f}", ha="center", va="bottom", fontsize=9)
        # Resalta al ganador
        idx_g = 0 if ganador == "SL-2" else 1
        bars[idx_g].set_edgecolor("black")
        bars[idx_g].set_linewidth(2)

    fig.suptitle(f"SL_003 — Comparación SL-2 vs SL-3\nGanador: {ganador}",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "comparacion_sl2_sl3.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/comparacion_sl2_sl3.png")


def plot_pesos_nnls(coefs: np.ndarray, nombres: list, ganador: str) -> None:
    """
    Pesos medios del NNLS ganador (promediados sobre los 5 folds de CV).
    Permite verificar que no hay pesos negativos y visualizar cuánto pesa
    cada base en la combinación óptima.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    colores = ["#3498db", "#e74c3c", "#27ae60"][:len(nombres)]
    bars    = ax.bar(nombres, coefs, color=colores, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Peso NNLS (medio sobre 5 folds)")
    ax.set_title(f"Pesos meta-aprendiz NNLS — {ganador}\n"
                 f"(LinearRegression positive=True, pesos ≥ 0 garantizados)")
    for bar, val in zip(bars, coefs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "pesos_nnls_ganador.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/pesos_nnls_ganador.png")


def plot_cv_comparacion(mae_rand: float, std_rand: float,
                        mae_esp: float,  std_esp: float,
                        ganador: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels  = ["CV\n(5-fold)", "CV Espacial\n(5×5 grid)"]
    values  = [mae_rand, mae_esp]
    errors  = [std_rand, std_esp]
    colors  = ["#3498db", "#e74c3c"]
    bars    = ax.bar(labels, values, yerr=errors, color=colors,
                     capsize=6, edgecolor="white")
    ax.set_ylabel("MAE log(price)")
    ax.set_title(f"CV vs CV Espacial — {MODEL_ID} ({ganador})\n"
                 f"Δ = {mae_esp - mae_rand:+.5f}")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "cv_comparacion_ganador.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/cv_comparacion_ganador.png")


def plot_correlacion_bases(Z_oof: np.ndarray,
                           nombres: list,
                           ganador: str) -> None:
    """
    Mapa de calor de correlación de Pearson entre las predicciones OOF
    de los modelos base del ganador.

    Interpretación:
      · Correlación alta (→ 1): los modelos fallan en las mismas observaciones
        → el ensemble gana poco combinándolos.
      · Correlación baja (→ 0): errores independientes → mayor ganancia potencial.

    Justifica empíricamente la selección de bases (XGB + NN tienen distinto
    sesgo inductivo → esperamos correlación baja; XGB + RF, al ser ambos
    tree-based, tendrán correlación más alta).
    """
    corr = np.corrcoef(Z_oof.T)
    fig, ax = plt.subplots(figsize=(max(5, len(nombres) * 1.8),
                                   max(4, len(nombres) * 1.5)))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Correlación de Pearson")
    ax.set_xticks(range(len(nombres)))
    ax.set_yticks(range(len(nombres)))
    ax.set_xticklabels(nombres)
    ax.set_yticklabels(nombres)
    ax.set_title(f"Correlación entre predicciones OOF — {MODEL_ID} ({ganador})\n"
                 "Baja correlación valida la diversidad del ensemble")
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center",
                    fontsize=10,
                    color="black" if abs(corr[i, j]) < 0.75 else "white")
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "correlacion_bases.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/correlacion_bases.png")


def plot_mae_individuales_vs_sl(oof_dict: dict,
                                y: np.ndarray,
                                mae_sl_rand: float,
                                mae_sl_esp: float,
                                ganador: str) -> None:
    """
    Barras comparando el MAE OOF de cada modelo base contra el MAE
    del SuperLearner ganador (CV aleatorio y CV espacial).

    Valida que el ensemble mejora (o al menos no empeora) respecto a las
    mejores bases individuales. Si el SL es peor que alguna base, indica
    un problema en la configuración (pesos patológicos, leakage, etc.).
    """
    nombres  = list(oof_dict.keys())
    maes_ind = [mean_absolute_error(y, oof_dict[n]) for n in nombres]

    fig, axes = plt.subplots(1, 2, figsize=(max(9, (len(nombres) + 1) * 2), 5))

    for ax, (mae_sl, titulo, color_sl) in zip(
        axes,
        [(mae_sl_rand, "CV Aleatorio (5-fold)",    "#f39c12"),
         (mae_sl_esp,  "CV Espacial (5×5 grid)",   "#e67e22")]
    ):
        x_pos      = np.arange(len(nombres) + 1)
        labels_all = nombres + [f"SuperLearner\n({ganador})"]
        values_all = maes_ind + [mae_sl]
        colors_all = ["#95a5a6"] * len(nombres) + [color_sl]

        bars = ax.bar(x_pos, values_all, color=colors_all, edgecolor="white")
        ax.axhline(mae_sl, color=color_sl, lw=1.5, linestyle="--", alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_all, fontsize=8)
        ax.set_ylabel("MAE log(price)")
        ax.set_title(f"Bases vs SuperLearner — {titulo}")
        for bar, val in zip(bars, values_all):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Modelos base vs {ganador} — {MODEL_ID}", fontsize=11)
    plt.tight_layout()
    fig.savefig(str(DIR_MODEL / "mae_individuales_vs_sl.png"), dpi=150)
    plt.close(fig)
    print(f"  Guardado: {MODEL_ID}/mae_individuales_vs_sl.png")


# =============================================================================
# SECCIÓN 14: SUBMISSION Y REGISTRO
# =============================================================================

def generar_submission(test_df: pd.DataFrame, y_pred_log: np.ndarray) -> str:
    sub = pd.DataFrame({
        "property_id": test_df["property_id"],
        "price":       np.exp(y_pred_log),
    })
    fecha    = date.today().strftime("%Y%m%d")
    sub_name = f"submission_{MODEL_ID}_{fecha}.csv"
    sub.to_csv(SUBMISSIONS / sub_name, index=False)
    print(f"  Submission: 03_submissions/{sub_name}  ({len(sub):,} filas)")
    return sub_name


def registrar(ganador: str, coefs_ganador: np.ndarray, nombres_ganador: list,
              mae_rand: float, std_rand: float,
              mae_esp: float,  std_esp: float,
              mae_train: float,
              n_feat_xgb: int, n_feat_nn: int,
              mae_esp_2: float, mae_esp_3: float,
              sub_name: str) -> None:
    pesos_str = " | ".join(f"{n}:{w:.3f}"
                           for n, w in zip(nombres_ganador, coefs_ganador))
    # Usa los mismos nombres de columna que el resto de modelos del registry
    # para que el ranking sea directo sin necesidad de unificar columnas.
    nueva = {
        "model_id":          MODEL_ID,
        "autor":             AUTOR,
        "fecha":             date.today().isoformat(),
        "algoritmo":         "SuperLearner",
        "n_features":        n_feat_xgb + n_feat_nn,
        "cv_mae_log":        round(mae_rand,  6),   # mismo nombre que modelos individuales
        "cv_std_log":        round(std_rand,  6),
        "esp_mae_log":       round(mae_esp,   6),   # mismo nombre que modelos individuales
        "esp_std_log":       round(std_esp,   6),
        "train_mae_log":     round(mae_train, 6),
        "sesgo_delta_log":   round(mae_esp - mae_rand, 6),
        "kaggle_public_MAE": None,
        "submission_file":   sub_name,
        "notas": (
            f"Ganador: {ganador} (SL-2 esp={mae_esp_2:.5f} vs "
            f"SL-3 esp={mae_esp_3:.5f}). "
            f"Meta: NNLS (positive=True). Pesos: [{pesos_str}]. "
            f"Bases: XGB_009 (KNN K=30 LOO) + NN_003 (PyTorch 256x256x128x64)"
            + (" + RF_005 (500 árboles depth=15)" if ganador == "SL-3" else "") +
            f". CV {CV_FOLDS}-fold + espacial {SPATIAL_GRID}x{SPATIAL_GRID}. "
            f"Δ={mae_esp - mae_rand:+.5f}."
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

    print(f"  Registry actualizado ({len(df_reg)} modelos)")


# =============================================================================
# SECCIÓN 15: MAIN
# =============================================================================

def main() -> None:
    print(f"{'='*65}")
    print(f"  SUPERLEARNER — {MODEL_ID}  (SL-2 vs SL-3 con NNLS)")
    print(f"{'='*65}")

    # ── [1/10] Carga ──────────────────────────────────────────────────────────
    print("\n[1/10] Cargando datos...")
    train = pd.read_csv(PROCESSED / "train_final.csv")
    test  = pd.read_csv(PROCESSED / "test_final.csv")
    y_train = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    # ── [2/10] Grupos espaciales ──────────────────────────────────────────────
    print(f"\n[2/10] Grupos espaciales ({SPATIAL_GRID}×{SPATIAL_GRID})...")
    grupos = asignar_grupos(train)
    print(f"  Bloques únicos: {len(np.unique(grupos))}")

    # ── [3/10] precio_vecinos sobre el train completo ─────────────────────────
    print(f"\n[3/10] precio_vecinos (K={K_VECINOS}, LOO haversine)...")
    train_xgb, test_xgb = agregar_precio_vecinos(train, test)

    # ── [4/10] Tuning XGB_009 ─────────────────────────────────────────────────
    print("\n[4/10] Tuning XGB_009...")
    xgb_params = tuning_xgb(train_xgb, y_train, grupos)

    # ── [5/10] OOF para los 3 modelos ────────────────────────────────────────
    print("\n[5/10] OOF 5-fold para XGB_009, NN_003 y RF_005...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    oof_xgb, oof_nn, oof_rf = generar_oof_todos(train, y_train, kf, xgb_params)

    mae_ind = {
        "XGB": mean_absolute_error(y_train, oof_xgb),
        "NN":  mean_absolute_error(y_train, oof_nn),
        "RF":  mean_absolute_error(y_train, oof_rf),
    }
    print(f"\n  MAE OOF individual: XGB={mae_ind['XGB']:.5f} | "
          f"NN={mae_ind['NN']:.5f} | RF={mae_ind['RF']:.5f}")

    # ── [6/10] Evaluación NNLS random: SL-2 y SL-3 ───────────────────────────
    print("\n[6/10] CV NNLS aleatorio (SL-2 y SL-3)...")
    Z2 = np.column_stack([oof_xgb, oof_nn])
    Z3 = np.column_stack([oof_xgb, oof_nn, oof_rf])

    mae_rand_2, std_rand_2, coefs_rand_2 = cv_nnls_meta(Z2, y_train, kf)
    mae_rand_3, std_rand_3, coefs_rand_3 = cv_nnls_meta(Z3, y_train, kf)
    print(f"  SL-2 (XGB+NN)      MAE_rand: {mae_rand_2:.5f} ± {std_rand_2:.5f}  "
          f"pesos≈{coefs_rand_2.round(3)}")
    print(f"  SL-3 (XGB+NN+RF)   MAE_rand: {mae_rand_3:.5f} ± {std_rand_3:.5f}  "
          f"pesos≈{coefs_rand_3.round(3)}")

    # Ajustar NNLS sobre el OOF completo para usar en spatial CV y modelo final
    meta2 = LinearRegression(positive=True).fit(Z2, y_train)
    meta3 = LinearRegression(positive=True).fit(Z3, y_train)

    # ── [7/10] CV espacial: SL-2 y SL-3 ──────────────────────────────────────
    print("\n[7/10] CV espacial para los 3 modelos...")
    y_esp, p_xgb_esp, p_nn_esp, p_rf_esp = cv_espacial_todos(
        train, y_train, grupos, xgb_params
    )

    # Aplicar pesos NNLS (aprendidos del OOF aleatorio) a predicciones espaciales
    Z_esp_2 = np.column_stack([p_xgb_esp, p_nn_esp])
    Z_esp_3 = np.column_stack([p_xgb_esp, p_nn_esp, p_rf_esp])

    mae_esp_2 = float(mean_absolute_error(y_esp, meta2.predict(Z_esp_2)))
    mae_esp_3 = float(mean_absolute_error(y_esp, meta3.predict(Z_esp_3)))

    # std_esp: variabilidad entre folds espaciales
    # Se reconstruye fold a fold para calcular std
    gkf = GroupKFold(n_splits=CV_FOLDS)
    folds_mae_2, folds_mae_3 = [], []
    start = 0
    for _, val_idx in gkf.split(train, y_train, groups=grupos):
        n_val = len(val_idx)
        y_v    = y_esp[start:start + n_val]
        z2_v   = Z_esp_2[start:start + n_val]
        z3_v   = Z_esp_3[start:start + n_val]
        folds_mae_2.append(mean_absolute_error(y_v, meta2.predict(z2_v)))
        folds_mae_3.append(mean_absolute_error(y_v, meta3.predict(z3_v)))
        start += n_val
    std_esp_2 = float(np.std(folds_mae_2))
    std_esp_3 = float(np.std(folds_mae_3))

    print(f"\n  SL-2 MAE_esp: {mae_esp_2:.5f} ± {std_esp_2:.5f}")
    print(f"  SL-3 MAE_esp: {mae_esp_3:.5f} ± {std_esp_3:.5f}")

    # ── [8/10] Selección del ganador ──────────────────────────────────────────
    print("\n[8/10] Seleccionando ganador...")
    ganador = seleccionar_ganador(mae_esp_2, std_esp_2, mae_esp_3, std_esp_3)

    if ganador == "SL-2":
        mae_rand_g, std_rand_g = mae_rand_2, std_rand_2
        mae_esp_g,  std_esp_g  = mae_esp_2,  std_esp_2
        coefs_g  = meta2.coef_
        nombres_g = ["XGB_009", "NN_003"]
        meta_g   = meta2
    else:
        mae_rand_g, std_rand_g = mae_rand_3, std_rand_3
        mae_esp_g,  std_esp_g  = mae_esp_3,  std_esp_3
        coefs_g   = meta3.coef_
        nombres_g = ["XGB_009", "NN_003", "RF_005"]
        meta_g    = meta3

    print(f"  {ganador} seleccionado | MAE_esp={mae_esp_g:.5f} | "
          f"pesos={dict(zip(nombres_g, coefs_g.round(3)))}")

    # ── [9/10] Modelo final del ganador ───────────────────────────────────────
    print(f"\n[9/10] Entrenando modelo final ({ganador})...")
    y_pred_test, feat_xgb, feat_nn, mae_train = entrenar_modelo_final(
        train_xgb, train, y_train, test_xgb, test,
        xgb_params, ganador, meta_g
    )
    print(f"  MAE_log in-sample: {mae_train:.5f}")

    # ── [10/10] Diagnósticos, submission y registro ───────────────────────────
    print("\n[10/10] Diagnósticos, submission y registro...")
    plot_comparacion_configs(mae_rand_2, std_rand_2, mae_esp_2, std_esp_2,
                             mae_rand_3, std_rand_3, mae_esp_3, std_esp_3,
                             ganador)
    plot_pesos_nnls(coefs_g, nombres_g, ganador)
    plot_cv_comparacion(mae_rand_g, std_rand_g, mae_esp_g, std_esp_g, ganador)

    # Gráficos equivalentes a SL_002, solo para el ganador
    Z_oof_g = (Z2 if ganador == "SL-2"
               else np.column_stack([oof_xgb, oof_nn, oof_rf]))
    plot_correlacion_bases(Z_oof_g, nombres_g, ganador)

    oof_dict_g = {"XGB_009": oof_xgb, "NN_003": oof_nn}
    if ganador == "SL-3":
        oof_dict_g["RF_005"] = oof_rf
    plot_mae_individuales_vs_sl(oof_dict_g, y_train,
                                mae_rand_g, mae_esp_g, ganador)

    sub_name = generar_submission(test, y_pred_test)
    registrar(ganador, coefs_g, nombres_g,
              mae_rand_g, std_rand_g, mae_esp_g, std_esp_g,
              mae_train, len(feat_xgb), len(feat_nn),
              mae_esp_2, mae_esp_3, sub_name)

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"{'='*65}")
    print(f"  Comparación:")
    print(f"    SL-2 (XGB+NN)     MAE_rand={mae_rand_2:.5f} | MAE_esp={mae_esp_2:.5f}")
    print(f"    SL-3 (XGB+NN+RF)  MAE_rand={mae_rand_3:.5f} | MAE_esp={mae_esp_3:.5f}")
    print(f"  Ganador       : {ganador}")
    print(f"  Pesos NNLS    : {dict(zip(nombres_g, coefs_g.round(4)))}")
    print(f"  MAE_rand      : {mae_rand_g:.5f} ± {std_rand_g:.5f}")
    print(f"  MAE_esp       : {mae_esp_g:.5f} ± {std_esp_g:.5f}")
    print(f"  Δ sesgo       : {mae_esp_g - mae_rand_g:+.5f}")
    print(f"  MAE in-sample : {mae_train:.5f}")
    print(f"  Device NN     : {DEVICE}")
    print(f"  Submission    : 03_submissions/{sub_name}")
    print(f"  Gráficos      : 02_outputs/Models/SuperLearner/{MODEL_ID}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
