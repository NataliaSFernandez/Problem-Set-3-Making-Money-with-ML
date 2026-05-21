"""
Análisis de Errores para Slide Deck — MECA 4107 

Mejor modelo: SL_003 (SuperLearner SL-2: XGB_009 + NN_003, meta-aprendiz NNLS)
  · Ganador: SL-2 (sin RF_005) — SL-2 MAE_esp=0.23244 vs SL-3 MAE_esp=0.23123
    Diferencia < 1 std → parsimonia → SL-2 seleccionado.
  · Pesos NNLS finales: [XGB_009=0.580, NN_003=0.454]
  · XGB_009: n_estimators=265, max_depth=6, lr=0.026, subsample=0.749,
             colsample_bytree=0.89, min_child_weight=6, gamma=0.321,
             reg_lambda=3.1131, reg_alpha=0.669
  · NN_003:  p→256(BN,Drop0.2)→256(BN,Drop0.2)→128(BN,Drop0.1)→64(BN)→1
             l1=5e-5, l2=5e-5, patience=30, 182 épocas
  · Kaggle MAE público: 199,851,631.7 COP

Estrategia de residuos
----------------------
Las predicciones OOF (out-of-fold) se re-generan corriendo el mismo pipeline
que el script original pero SOLO el pase de OOF (sin tuning, sin CV espacial,
sin submission). Los residuos son válidos: cada observación fue predicha por
modelos que nunca la vieron.

Gráficas generadas (10 figuras)
---------------------------------
01  Distribución de residuos — ¿simétrica o sesgada?
02  Donut sesgo: sobrepredicción vs subpredicción (casos + impacto COP)
03  MAE y % sobrepredicción por cuartil de precio (Q1–Q4)
04  Mapa espacial de errores (lat/lon, magnitud del error)
05  Pérdida financiera acumulada (curva tipo Zillow)
06  Scatter precio real vs predicho con línea 45°
07  Error relativo (%) por cuartil — intuitivo para audiencias no técnicas
08  Comparación MAE Kaggle: SL_003, XGB_009, NN_003, RF_002, CART_002, EN_002, LR_001
09  Heatmap espacial de MAE y sesgo por bloque geográfico (5×5)
10  Tabla métricas de negocio por cuartil

Convenciones heredadas (idénticas a todos los scripts del proyecto)
--------------------------------------------------------------------
- Target: log(price); predicciones en COP con np.exp()
- StandardScaler ajustado SOLO en fold de train (antidata leakage)
- precio_vecinos: K=30, LOO haversine, recomputado por fold
- Grupos espaciales: grid 5×5 de cuantiles lat × lon

Outputs
-------
  02_outputs/ErrorAnalysis_SL003/01_distribucion_residuos.png
  02_outputs/ErrorAnalysis_SL003/02_sesgo_donut.png
  02_outputs/ErrorAnalysis_SL003/03_mae_por_segmento.png
  02_outputs/ErrorAnalysis_SL003/04_mapa_errores.png
  02_outputs/ErrorAnalysis_SL003/05_perdida_financiera.png
  02_outputs/ErrorAnalysis_SL003/06_scatter_real_vs_pred.png
  02_outputs/ErrorAnalysis_SL003/07_error_relativo_segmento.png
  02_outputs/ErrorAnalysis_SL003/08_comparacion_modelos_kaggle.png
  02_outputs/ErrorAnalysis_SL003/09_heatmap_espacial.png
  02_outputs/ErrorAnalysis_SL003/10_tabla_metricas_negocio.png
"""

# =============================================================================
# SECCIÓN -1: VARIABLES DE ENTORNO (antes de cualquier import)
# =============================================================================
# Evita conflictos OpenMP entre XGBoost + RF en el mismo proceso (segfault macOS)
import os
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

# =============================================================================
# SECCIÓN 0: IMPORTACIONES
# =============================================================================

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# XGBoost con fallback automático a GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    _USE_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    _USE_XGB = False
    print("  xgboost no disponible. Usando GradientBoostingRegressor como fallback.")

warnings.filterwarnings("ignore")

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================

SEED         = 42
CV_FOLDS     = 5
SPATIAL_GRID = 5
K_VECINOS    = 30
NN_SEED      = 101010

# Rutas del proyecto
BASE        = Path(__file__).parent.parent
PROCESSED   = BASE / "00_data" / "processed"
OUT_DIR     = BASE / "02_outputs" / "ErrorAnalysis_SL003"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paleta de colores del proyecto ────────────────────────────────────────────
C_MAIN    = "#E07B54"   # naranja — sobrepredicción / error
C_GOOD    = "#4C9BE8"   # azul    — subpredicción / oportunidad
C_NEUTRAL = "#6C757D"   # gris
C_DARK    = "#2C3E50"   # casi negro
C_LIGHT   = "#F8F4EF"   # fondo claro

# ── Hiperparámetros XGB_009 — extraídos directamente del model_registry ──────
# Fuente: fila XGB_009 del registry (columnas de hiperparámetros tuneados).
# Tuning: RandomizedSearchCV, n_iter=20, GroupKFold espacial 5×5.
# Resultado: MAE_rand=0.17598 | MAE_esp=0.21079 | Δ=+0.03480
# Features: structural=7, osm=10, text=11, geo=2 (lat+lon),
#            precio_vecinos=1 (K=30 LOO), es_apartamento → 32 total
XGB_BEST_PARAMS = {
    "n_estimators":     265,    # early stopping espacial
    "max_depth":          6,
    "learning_rate":   0.026,
    "subsample":       0.749,
    "colsample_bytree": 0.89,   # solo XGBoost nativo (ignorado en fallback GBR)
    "min_child_weight":    6,   # solo XGBoost nativo
    "gamma":            0.321,  # solo XGBoost nativo
    "reg_lambda":      3.1131,  # solo XGBoost nativo
    "reg_alpha":        0.669,  # solo XGBoost nativo
}

# ── Pesos NNLS del SL_003 (ganador: SL-2 = XGB_009 + NN_003) ─────────────────
# Fuente: notas del registry — "Pesos: [XGB_009:0.580 | NN_003:0.454]"
# Estos pesos se usan como punto de partida para el meta-aprendiz OOF;
# el NNLS de validación los re-estima fold a fold.
NNLS_FIXED_WEIGHTS = np.array([0.580, 0.454])  # [XGB, NN]

# Nota: SL_003 ganó con SL-2 (XGB+NN) — SL-2 MAE_esp=0.23244 vs SL-3 MAE_esp=0.23123
# La diferencia fue < 1 std → parsimonia → SL-2 seleccionado (sin RF_005).

# ── Hiperparámetros RF_005 (idénticos a SL_003) ───────────────────────────────
RF_PARAMS = {
    "n_estimators":     500,
    "max_depth":         15,
    "min_samples_leaf":   5,
    "max_features":    "sqrt",
    "random_state":     SEED,
    "n_jobs":              1,
}

# ── NN_003: hiperparámetros (idénticos a SL_003) ─────────────────────────────
NN_EPOCHS     = 300
NN_BATCH_SIZE = 512
NN_VAL_SPLIT  = 0.2
NN_PATIENCE   = 30
NN_LR         = 1e-3
NN_L1_REG     = 5e-5
NN_L2_REG     = 5e-5
NN_DROPOUT_1  = 0.2
NN_DROPOUT_2  = 0.1

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))

# ── MAE Kaggle de los 7 modelos resaltados (para la gráfica 08) ──────────────
# Fuente: model_registry.xlsx (columna kaggle_public_MAE).
# SL_003 es la submission final del equipo (mejor score).
# XGB_009 es la base más fuerte del ensemble (fila XGB_009 del registry).
MODELOS_COMPARACION = {
    "SL_003":  199_851_631.7,   # ← mejor submission del equipo
    "XGB_009": 207_886_154.44,  # base del ensemble (fila XGB_009 del registry)
    "NN_003":  216_987_903.1,
    "RF_002":  281_835_336.12,
    "CART_002":289_662_027.55,
    "EN_002":  313_972_117.03,
    "LR_001":  318_727_924.35,
}

# =============================================================================
# SECCIÓN 2: VARIABLES DE FEATURES (idénticas a SL_003)
# =============================================================================

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

# =============================================================================
# SECCIÓN 3: FUNCIONES DE FEATURES (idénticas a SL_003)
# =============================================================================

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
# SECCIÓN 4: ARQUITECTURA NN_003 (PyTorch — idéntica a SL_003)
# =============================================================================

class MLP(nn.Module):
    """p → 256 → 256 → 128 → 64 → 1  (BN + ReLU + Dropout)."""

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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x).squeeze(-1)


def _l1_loss(model: MLP) -> torch.Tensor:
    device = next(model.parameters()).device
    total  = torch.zeros(1, device=device)
    for name, p in model.named_parameters():
        if "weight" in name:
            total = total + p.abs().sum()
    return total


def entrenar_nn(X_sc: np.ndarray, y: np.ndarray) -> MLP:
    """Entrena MLP con early stopping (val_split=0.2). Idéntico a SL_003."""
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
    model      = MLP(X_sc.shape[1]).to(DEVICE)
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
                loss = loss + NN_L1_REG * _l1_loss(model)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_state = val_loss, {k: v.cpu().clone()
                                               for k, v in model.state_dict().items()}
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
# SECCIÓN 5: PRECIO VECINOS (idéntico a SL_003)
# =============================================================================

def _knn_precio(ref_df: pd.DataFrame, query_df: pd.DataFrame,
                k: int, loo: bool) -> np.ndarray:
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


def _agregar_pv_fold(tr_df: pd.DataFrame,
                     va_df: pd.DataFrame) -> tuple:
    """precio_vecinos dentro de un fold — sin leakage de validación."""
    tr_df = tr_df.copy()
    va_df = va_df.copy()
    tr_df["precio_vecinos"] = _knn_precio(tr_df, tr_df, K_VECINOS, loo=True)
    va_df["precio_vecinos"] = _knn_precio(tr_df, va_df, K_VECINOS, loo=False)
    return tr_df, va_df

# =============================================================================
# SECCIÓN 6: HELPER XGB (fallback automático)
# =============================================================================

def _make_xgb(params: dict):
    if _USE_XGB:
        xgb_keys = {"max_depth", "learning_rate", "subsample", "colsample_bytree",
                    "min_child_weight", "gamma", "reg_lambda", "reg_alpha",
                    "n_estimators", "random_state", "n_jobs", "verbosity"}
        clean = {k: v for k, v in params.items() if k in xgb_keys}
        return XGBRegressor(**clean, random_state=SEED, n_jobs=1, verbosity=0)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
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

# =============================================================================
# SECCIÓN 7: OOF DEL SL_003 (XGB_009 + NN_003 con NNLS)
# =============================================================================

def generar_oof_sl003(train_df: pd.DataFrame,
                      y_log: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera predicciones OOF del SL_003 (SL-2: XGB_009 + NN_003) con KFold 5.

    Reproduce EXACTAMENTE el pipeline de SL_003 (05_sl003.py):

    PASO 1 — OOF de bases (idéntico a generar_oof_todos en SL_003):
      Por cada fold:
        · precio_vecinos recomputado solo con datos del fold de train (LOO).
        · XGB_009 entrena sobre X_train del fold → predice X_val.
        · NN_003 entrena sobre X_train del fold con scaler ajustado en fold train
          → predice X_val.
      Resultado: oof_xgb y oof_nn completos (n observaciones cada uno).

    PASO 2 — Meta NNLS (idéntico a cv_nnls_meta + meta2 = LinearRegression(...).fit en SL_003):
      · Se ajusta UN SOLO meta LinearRegression(positive=True) sobre el OOF
        completo [oof_xgb | oof_nn] vs y_log.
      · oof_sl = meta.predict([oof_xgb | oof_nn])
      · Este es exactamente el mismo objeto `meta2` que SL_003 usa para
        construir la submission final.

    Nota: los pesos NNLS_FIXED_WEIGHTS (del registry) se usan SOLO para
    comparar/verificar — no para generar predicciones.

    Retorna:
      oof_sl  — predicciones SL_003 en escala log(price)
      oof_xgb — predicciones XGB_009 individuales en escala log(price)
      oof_nn  — predicciones NN_003 individuales en escala log(price)
    """
    kf  = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    n   = len(y_log)
    oof_xgb = np.zeros(n)
    oof_nn  = np.zeros(n)

    # ── PASO 1: OOF de las dos bases (copia exacta de generar_oof_todos) ─────
    print(f"\n  [OOF paso 1/2] Bases XGB_009 + NN_003 — {CV_FOLDS} folds")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df), 1):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[val_idx].reset_index(drop=True)
        y_tr  = y_log[tr_idx]
        y_va  = y_log[val_idx]

        # precio_vecinos sin leakage: recomputado con solo datos del fold train
        tr_xf, va_xf = _agregar_pv_fold(tr_df, va_df)
        X_tr_xf = construir_features_xgb(tr_xf)
        X_va_xf = construir_features_xgb(va_xf, fit_cols=X_tr_xf.columns.tolist())

        # XGB_009
        xgb_m = _make_xgb(XGB_BEST_PARAMS)
        xgb_m.fit(X_tr_xf, y_tr)
        oof_xgb[val_idx] = xgb_m.predict(X_va_xf)

        # NN_003 — scaler ajustado SOLO en fold train (antidata leakage)
        X_tr_nn = construir_features_nn(tr_df)
        X_va_nn = construir_features_nn(va_df, fit_cols=X_tr_nn.columns.tolist())
        sc      = StandardScaler()
        nn_m    = entrenar_nn(sc.fit_transform(X_tr_nn), y_tr)
        oof_nn[val_idx] = predecir_nn(nn_m, sc.transform(X_va_nn))

        mae_xgb = mean_absolute_error(y_va, oof_xgb[val_idx])
        mae_nn  = mean_absolute_error(y_va, oof_nn[val_idx])
        print(f"    Fold {fold}: XGB={mae_xgb:.5f} | NN={mae_nn:.5f}")

    mae_total_xgb = mean_absolute_error(y_log, oof_xgb)
    mae_total_nn  = mean_absolute_error(y_log, oof_nn)
    print(f"  MAE_log OOF total → XGB={mae_total_xgb:.5f} | NN={mae_total_nn:.5f}")

    # ── PASO 2: Meta NNLS sobre el OOF completo (igual que meta2 en SL_003) ──
    # En SL_003: meta2 = LinearRegression(positive=True).fit(Z2, y_train)
    # donde Z2 = np.column_stack([oof_xgb, oof_nn])
    # Luego: oof_sl = meta2.predict(Z2)
    print(f"\n  [OOF paso 2/2] Meta NNLS sobre OOF completo...")
    Z2   = np.column_stack([oof_xgb, oof_nn])
    meta = LinearRegression(positive=True)
    meta.fit(Z2, y_log)
    oof_sl = meta.predict(Z2)

    coefs = meta.coef_
    intercept = meta.intercept_
    mae_total_sl = mean_absolute_error(y_log, oof_sl)

    print(f"  Pesos NNLS re-estimados : [XGB={coefs[0]:.4f}, NN={coefs[1]:.4f}]  "
          f"intercept={intercept:.4f}")
    print(f"  Pesos NNLS del registry  : [XGB={NNLS_FIXED_WEIGHTS[0]:.4f}, "
          f"NN={NNLS_FIXED_WEIGHTS[1]:.4f}]")
    print(f"  MAE_log OOF SL_003      : {mae_total_sl:.5f}")

    return oof_sl, oof_xgb, oof_nn

# =============================================================================
# SECCIÓN 8: MÉTRICAS DE NEGOCIO
# =============================================================================

def calcular_metricas(y_real: np.ndarray,
                      y_pred: np.ndarray) -> dict:
    """
    Residuo = precio_real − precio_predicho
      < 0 → sobrepredicción: modelo sobrevalora → empresa paga de más → pérdida de capital
      > 0 → subpredicción:  modelo subvalora → empresa ofrece poco → oportunidad perdida
    """
    residuos       = y_real - y_pred
    mask_sobre     = residuos < 0
    mask_sub       = residuos > 0

    return {
        "residuos":        residuos,
        "prop_sobrepred":  float(mask_sobre.mean()),
        "prop_subpred":    float(mask_sub.mean()),
        "perdida_cop":     float(np.abs(residuos[mask_sobre]).sum()),
        "n_sobre":         int(mask_sobre.sum()),
        "oportunidad_cop": float(residuos[mask_sub].sum()),
        "n_sub":           int(mask_sub.sum()),
        "mae_cop":         float(np.abs(residuos).mean()),
        "bias":            float(residuos.mean()),   # > 0 → subpredice; < 0 → sobrepredice
    }

# =============================================================================
# SECCIÓN 9: ESTILO GRÁFICO
# =============================================================================

def _estilo_publicacion() -> None:
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.6,
        "figure.dpi":        150,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

# =============================================================================
# SECCIÓN 10: GRÁFICAS
# =============================================================================

# ── 01: Distribución de residuos ─────────────────────────────────────────────

def plot_distribucion_residuos(m: dict, out: Path) -> None:
    res = m["residuos"] / 1e6   # millones de COP

    fig, ax = plt.subplots(figsize=(10, 5))
    n_bins  = 80
    counts, bins, patches = ax.hist(res, bins=n_bins,
                                    edgecolor="white", linewidth=0.3)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(C_MAIN if left < 0 else C_GOOD)
        patch.set_alpha(0.85)

    ax.axvline(0, color=C_DARK, lw=2.0, ls="-",  label="Sin error (residuo = 0)")
    ax.axvline(m["bias"] / 1e6, color="#E74C3C", lw=2.0, ls="--",
               label=f"Sesgo promedio: {m['bias']/1e6:+.1f} M COP")

    legend_patches = [
        mpatches.Patch(color=C_MAIN, alpha=0.85,
                       label=f"Sobrepredicción — {m['prop_sobrepred']:.1%} de propiedades"),
        mpatches.Patch(color=C_GOOD, alpha=0.85,
                       label=f"Subpredicción — {m['prop_subpred']:.1%} de propiedades"),
    ]
    ax.legend(handles=legend_patches + [ax.get_lines()[0], ax.get_lines()[1]],
              fontsize=10)

    ax.set_xlabel("Residuo: Precio Real − Predicho  (millones de COP)", fontsize=12)
    ax.set_ylabel("N° de propiedades", fontsize=12)
    ax.set_title("Distribución de Residuos — SL_003 (XGB_009 + NN_003, NNLS)",
                 fontsize=13, fontweight="bold")

    texto = (f"MAE = {m['mae_cop']/1e6:.1f} M COP\n"
             f"Sesgo = {m['bias']/1e6:+.1f} M COP\n"
             f"N = {len(m['residuos']):,} propiedades")
    ax.text(0.98, 0.97, texto, transform=ax.transAxes, va="top", ha="right",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.4",
                                   facecolor=C_LIGHT, alpha=0.9))
    plt.tight_layout()
    plt.savefig(out / "01_distribucion_residuos.png")
    plt.close()
    print("  [OK] 01_distribucion_residuos.png")


# ── 02: Donut sobrepredicción vs subpredicción ───────────────────────────────

def plot_sesgo_donut(m: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, sizes, labels, centro, titulo in [
        (
            axes[0],
            [m["prop_sobrepred"], m["prop_subpred"]],
            [f"Sobrepredicción\n{m['prop_sobrepred']:.1%} de propiedades\n"
             f"({m['n_sobre']:,} casos)",
             f"Subpredicción\n{m['prop_subpred']:.1%} de propiedades\n"
             f"({m['n_sub']:,} casos)"],
            f"{m['n_sobre'] + m['n_sub']:,}\npropiedades",
            "Proporción de casos",
        ),
        (
            axes[1],
            [m["perdida_cop"], m["oportunidad_cop"]],
            [f"Pérdida de capital\n{m['perdida_cop']/1e9:.1f} MM COP",
             f"Oportunidad perdida\n{m['oportunidad_cop']/1e9:.1f} MM COP"],
            f"{(m['perdida_cop']+m['oportunidad_cop'])/1e9:.0f} MM\nCOP total",
            "Impacto financiero acumulado",
        ),
    ]:
        wedges, _ = ax.pie(
            sizes, colors=[C_MAIN, C_GOOD], startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2.5)
        )
        ax.legend(wedges, labels, loc="lower center", fontsize=9,
                  bbox_to_anchor=(0.5, -0.22), ncol=1)
        ax.text(0, 0, centro, ha="center", va="center",
                fontsize=10, color=C_DARK, fontweight="bold")
        ax.set_title(titulo, fontsize=12, fontweight="bold")

    fig.suptitle("Asimetría de Errores: Sobrepredicción vs Subpredicción — SL_003",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out / "02_sesgo_donut.png")
    plt.close()
    print("  [OK] 02_sesgo_donut.png")


# ── 03: MAE y % sobrepredicción por cuartil de precio ────────────────────────

def plot_mae_por_segmento(train: pd.DataFrame, y_pred: np.ndarray,
                          m: dict, out: Path) -> None:
    df = train.copy()
    df["y_pred"]   = y_pred
    df["residuo"]  = df["price"] - y_pred
    df["segmento"] = pd.qcut(
        df["price"], q=4,
        labels=["Q1\n(Bajo)", "Q2\n(Medio-Bajo)", "Q3\n(Medio-Alto)", "Q4\n(Alto)"]
    )

    def _stats(g):
        res = g["residuo"]
        return pd.Series({
            "MAE_M":      res.abs().mean() / 1e6,
            "pct_sobre":  (res < 0).mean(),
            "precio_med": g["price"].median() / 1e6,
            "n":          len(g),
        })
    resumen = df.groupby("segmento", observed=True).apply(_stats).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel izquierdo: MAE por segmento ──
    ax = axes[0]
    colors_bar = [C_MAIN if v > resumen["MAE_M"].mean() else C_GOOD
                  for v in resumen["MAE_M"]]
    bars = ax.bar(resumen["segmento"], resumen["MAE_M"],
                  color=colors_bar, edgecolor="white", linewidth=1.5, alpha=0.88)
    ax.axhline(resumen["MAE_M"].mean(), color=C_DARK, lw=1.8, ls="--",
               label=f"MAE promedio: {resumen['MAE_M'].mean():.1f} M COP")
    for bar, val in zip(bars, resumen["MAE_M"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f} M", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("MAE (millones de COP)", fontsize=11)
    ax.set_xlabel("Cuartil de precio", fontsize=11)
    ax.set_title("MAE por Segmento de Precio", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    # ── Panel derecho: % sobrepredicción ──
    ax2 = axes[1]
    colors_pct = [C_MAIN if p > 0.5 else C_GOOD for p in resumen["pct_sobre"]]
    bars2 = ax2.bar(resumen["segmento"], resumen["pct_sobre"] * 100,
                    color=colors_pct, edgecolor="white", linewidth=1.5, alpha=0.88)
    ax2.axhline(50, color=C_DARK, lw=1.8, ls="--", label="50% (neutro)")
    for bar, val in zip(bars2, resumen["pct_sobre"]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{val:.1%}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("% de sobrepredicciones", fontsize=11)
    ax2.set_xlabel("Cuartil de precio", fontsize=11)
    ax2.set_title("Proporción de Sobrepredicción por Segmento", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)

    fig.suptitle("Análisis de Errores por Segmento de Mercado — SL_003",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "03_mae_por_segmento.png")
    plt.close()
    print("  [OK] 03_mae_por_segmento.png")


# ── 04: Mapa espacial de errores ─────────────────────────────────────────────

def plot_mapa_errores(train: pd.DataFrame, y_pred: np.ndarray,
                      out: Path) -> None:
    df = train.copy()
    df["residuo"]     = df["price"] - y_pred
    df["error_abs_M"] = df["residuo"].abs() / 1e6
    df["tipo"]        = df["residuo"].apply(
        lambda r: "Sobrepredicción" if r < 0 else "Subpredicción"
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, grupo_nombre, color, titulo in [
        (axes[0], "Sobrepredicción", C_MAIN, "Sobrepredicción (Real < Predicho)"),
        (axes[1], "Subpredicción",   C_GOOD, "Subpredicción   (Real > Predicho)"),
    ]:
        g = df[df["tipo"] == grupo_nombre]
        cmap_custom = mcolors.LinearSegmentedColormap.from_list(
            "err", ["#EEEEEE", color], N=256
        )
        sc = ax.scatter(
            g["lon"], g["lat"],
            c=g["error_abs_M"], cmap=cmap_custom,
            s=np.clip(g["error_abs_M"] * 0.25, 2, 55),
            alpha=0.55, edgecolors="none", rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.82)
        cbar.set_label("Error absoluto (M COP)", fontsize=9)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitud", fontsize=10)
        ax.set_ylabel("Latitud",  fontsize=10)
        ax.set_facecolor("#EBEBEB")
        # Anotar conteo y MAE parcial
        ax.text(0.02, 0.97,
                f"N = {len(g):,}\nMAE = {g['error_abs_M'].mean():.1f} M COP",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Mapa Espacial de Errores de Predicción — SL_003",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "04_mapa_errores.png")
    plt.close()
    print("  [OK] 04_mapa_errores.png")


# ── 05: Curva de pérdida financiera acumulada ─────────────────────────────────

def plot_perdida_financiera(train: pd.DataFrame, y_pred: np.ndarray,
                            m: dict, out: Path) -> None:
    df = train.copy()
    df["residuo"] = df["price"] - y_pred
    sobre = (df[df["residuo"] < 0]
             .assign(perdida=lambda d: d["residuo"].abs())
             .sort_values("perdida", ascending=False)
             .reset_index(drop=True))
    sobre["acum_MM"] = sobre["perdida"].cumsum() / 1e9

    pct = sobre.index / len(sobre) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(pct, sobre["acum_MM"], alpha=0.20, color=C_MAIN)
    ax.plot(pct, sobre["acum_MM"], color=C_MAIN, lw=2.5,
            label="Pérdida acumulada (sobrepredicción)")

    for pct_ref, ls in [(10, "--"), (20, ":")]:
        n_ref = int(len(sobre) * pct_ref / 100)
        if n_ref > 0:
            val = sobre.loc[n_ref - 1, "acum_MM"]
            ax.axvline(pct_ref, color=C_NEUTRAL, ls=ls, lw=1.5)
            ax.annotate(
                f"{pct_ref}% peores\n= {val:.0f} MM COP",
                xy=(pct_ref, val), xytext=(pct_ref + 2, val * 0.92),
                fontsize=9, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_NEUTRAL, lw=1.2),
            )

    ax.set_xlabel("% de propiedades con sobrepredicción (ordenadas por magnitud)",
                  fontsize=11)
    ax.set_ylabel("Pérdida acumulada (miles de millones de COP)", fontsize=11)
    ax.set_title(
        f"Curva de Pérdida Financiera por Sobrepredicción — SL_003\n"
        f"Total capital en riesgo: {m['perdida_cop']/1e9:.0f} MM COP",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    nota = (f"{m['n_sobre']:,} propiedades sobrepredichas ({m['prop_sobrepred']:.1%})\n"
            f"Pérdida promedio por caso: {m['perdida_cop']/m['n_sobre']/1e6:.1f} M COP")
    ax.text(0.98, 0.05, nota, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4",
                                   facecolor=C_LIGHT, alpha=0.9))
    plt.tight_layout()
    plt.savefig(out / "05_perdida_financiera.png")
    plt.close()
    print("  [OK] 05_perdida_financiera.png")


# ── 06: Scatter precio real vs predicho ──────────────────────────────────────

def plot_scatter_real_vs_pred(train: pd.DataFrame, y_pred: np.ndarray,
                               m: dict, out: Path) -> None:
    df = train.copy()
    df["pred_M"]  = y_pred / 1e6
    df["real_M"]  = df["price"] / 1e6
    df["residuo"] = df["price"] - y_pred
    df["tipo"]    = df["residuo"].apply(
        lambda r: "Sobrepredicción" if r < 0 else "Subpredicción"
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    for tipo, color, alpha in [("Sobrepredicción", C_MAIN, 0.45),
                                ("Subpredicción",   C_GOOD, 0.45)]:
        mask = df["tipo"] == tipo
        ax.scatter(df.loc[mask, "pred_M"], df.loc[mask, "real_M"],
                   color=color, alpha=alpha, s=5, label=tipo, rasterized=True)

    lim_max = df[["pred_M", "real_M"]].quantile(0.995).max() * 1.05
    lim_min = df[["pred_M", "real_M"]].quantile(0.005).min() * 0.95
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color=C_DARK, lw=2.0, ls="-", label="Predicción perfecta (45°)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Precio Predicho (millones de COP)", fontsize=11)
    ax.set_ylabel("Precio Real (millones de COP)", fontsize=11)
    ax.set_title("Precio Real vs Predicho — SL_003 (OOF 5-fold)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, markerscale=2.5)

    texto = (f"MAE = {m['mae_cop']/1e6:.1f} M COP\n"
             f"Sesgo = {m['bias']/1e6:+.1f} M COP\n"
             f"OOF: {CV_FOLDS}-fold CV")
    ax.text(0.03, 0.97, texto, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.4",
                                   facecolor=C_LIGHT, alpha=0.9))
    plt.tight_layout()
    plt.savefig(out / "06_scatter_real_vs_pred.png")
    plt.close()
    print("  [OK] 06_scatter_real_vs_pred.png")


# ── 07: Error relativo (%) por cuartil ───────────────────────────────────────

def plot_error_relativo_segmento(train: pd.DataFrame, y_pred: np.ndarray,
                                  out: Path) -> None:
    df = train.copy()
    df["residuo"]       = df["price"] - y_pred
    df["error_rel_pct"] = df["residuo"].abs() / df["price"] * 100
    df["sesgo_rel_pct"] = df["residuo"] / df["price"] * 100
    df["segmento"]      = pd.qcut(
        df["price"], q=4,
        labels=["Q1\n(Bajo)", "Q2\n(Medio-Bajo)", "Q3\n(Medio-Alto)", "Q4\n(Alto)"]
    )

    resumen = (df.groupby("segmento", observed=True)
               .agg(mean_pct=("error_rel_pct", "mean"),
                    med_pct =("error_rel_pct", "median"),
                    sesgo   =("sesgo_rel_pct", "mean"))
               .reset_index())

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(resumen))
    ancho = 0.35

    ax.bar(x - ancho / 2, resumen["mean_pct"], ancho,
           color=C_MAIN, alpha=0.85, label="Error relativo medio",
           edgecolor="white")
    ax.bar(x + ancho / 2, resumen["med_pct"],  ancho,
           color=C_GOOD, alpha=0.85, label="Error relativo mediano",
           edgecolor="white")

    for i, row in resumen.iterrows():
        tope = max(row["mean_pct"], row["med_pct"]) + 0.5
        ax.text(i, tope, f"Sesgo: {row['sesgo']:+.1f}%",
                ha="center", va="bottom", fontsize=9,
                color=C_DARK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(resumen["segmento"], fontsize=11)
    ax.set_ylabel("Error relativo (% del precio real)", fontsize=11)
    ax.set_xlabel("Cuartil de precio", fontsize=11)
    ax.set_title("Error Relativo por Segmento de Mercado — SL_003",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "07_error_relativo_segmento.png")
    plt.close()
    print("  [OK] 07_error_relativo_segmento.png")


# ── 08: Comparación de 7 modelos seleccionados ───────────────────────────────

def plot_comparacion_modelos_kaggle(out: Path) -> None:
    """
    Barchart horizontal con los 7 modelos resaltados en naranja en el registry.
    Fuente: model_registry.xlsx (columna kaggle_public_MAE).
    Nota: actualizar MODELOS_COMPARACION si SL_003 ya tiene MAE Kaggle definitivo.
    """
    FAMILIA = {
        "SL_003":  ("SuperLearner",      "#27AE60"),
        "XGB_009": ("XGBoost",           "#E07B54"),
        "NN_003":  ("Neural Network",    "#9B59B6"),
        "RF_002":  ("Random Forest",     "#4C9BE8"),
        "CART_002":("CART",              "#F39C12"),
        "EN_002":  ("Elastic Net",       "#1ABC9C"),
        "LR_001":  ("Regresión Lineal",  "#95A5A6"),
    }

    data = (pd.DataFrame(
                [{"model_id": k, "mae_M": v / 1e6, "familia": FAMILIA[k][0],
                  "color": FAMILIA[k][1]}
                 for k, v in MODELOS_COMPARACION.items()])
            .sort_values("mae_M")
            .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(data["model_id"], data["mae_M"],
                   color=data["color"], edgecolor="white",
                   linewidth=1.2, alpha=0.88)

    # Borde doble para el mejor modelo
    idx_sl = data.index[data["model_id"] == "SL_003"].tolist()
    if idx_sl:
        bars[idx_sl[0]].set_edgecolor(C_DARK)
        bars[idx_sl[0]].set_linewidth(3.0)
        bars[idx_sl[0]].set_alpha(1.0)

    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{row['mae_M']:.0f} M COP", va="center", ha="left", fontsize=9.5)

    # Línea de referencia: mejor modelo
    mejor = data["mae_M"].min()
    ax.axvline(mejor, color=C_DARK, lw=1.5, ls=":", alpha=0.5,
               label=f"Mejor: {mejor:.0f} M COP")

    # Leyenda de familias
    leyenda = [mpatches.Patch(color=FAMILIA[m][1], label=FAMILIA[m][0], alpha=0.88)
               for m in data["model_id"]]
    ax.legend(handles=leyenda, loc="lower right", fontsize=9, title="Algoritmo",
              title_fontsize=9)

    # Anotación de mejora relativa respecto a LR
    mae_lr  = data.loc[data["model_id"] == "LR_001", "mae_M"].values
    mae_sl  = data.loc[data["model_id"] == "SL_003", "mae_M"].values
    if len(mae_lr) and len(mae_sl):
        mejora = (mae_lr[0] - mae_sl[0]) / mae_lr[0] * 100
        ax.text(0.98, 0.02,
                f"SL_003 mejora {mejora:.1f}%\nrespecto a LR_001",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT, alpha=0.9))

    ax.set_xlabel("MAE Kaggle Público (millones de COP) — menor es mejor", fontsize=11)
    ax.set_title("Comparación de Modelos por MAE Kaggle\n"
                 "SL_003 · XGB_009 · NN_003 · RF_002 · CART_002 · EN_002 · LR_001",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "08_comparacion_modelos_kaggle.png")
    plt.close()
    print("  [OK] 08_comparacion_modelos_kaggle.png")


# ── 09: Heatmap espacial de MAE y sesgo (grid 5×5) ───────────────────────────

def plot_heatmap_espacial(train: pd.DataFrame, y_pred: np.ndarray,
                           out: Path) -> None:
    G  = SPATIAL_GRID
    df = train.copy()
    df["residuo"]   = df["price"] - y_pred
    df["error_abs"] = df["residuo"].abs() / 1e6

    df["lat_bin"] = pd.qcut(df["lat"], G, labels=False, duplicates="drop")
    df["lon_bin"] = pd.qcut(df["lon"], G, labels=False, duplicates="drop")

    grid_mae  = (df.groupby(["lat_bin", "lon_bin"])["error_abs"]
                 .mean().unstack(fill_value=np.nan))
    grid_bias = (df.groupby(["lat_bin", "lon_bin"])["residuo"]
                 .mean().unstack(fill_value=np.nan) / 1e6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, grid, titulo, fmt, cmap, label in [
        (axes[0], grid_mae,  "MAE promedio por Bloque Geográfico",
         ".0f", "YlOrRd",  "M COP"),
        (axes[1], grid_bias, "Sesgo promedio por Bloque Geográfico",
         "+.0f", "RdBu_r", "M COP"),
    ]:
        im = ax.imshow(grid.values, cmap=cmap, aspect="auto",
                       origin="lower", interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.85, label=label)
        vmax = np.nanmax(np.abs(grid.values))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid.values[i, j]
                if not np.isnan(val):
                    fc = "white" if abs(val) > vmax * 0.65 else "black"
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=9, color=fc)
        ax.set_xlabel("Bloque de Longitud (O → E)", fontsize=10)
        ax.set_ylabel("Bloque de Latitud (S → N)",  fontsize=10)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))

    fig.suptitle("Análisis Espacial de Errores — Grid 5×5 Bogotá | SL_003",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "09_heatmap_espacial.png")
    plt.close()
    print("  [OK] 09_heatmap_espacial.png")


# ── 10: Tabla métricas de negocio por cuartil ────────────────────────────────

def plot_tabla_metricas_negocio(train: pd.DataFrame, y_pred: np.ndarray,
                                 m: dict, out: Path) -> None:
    df = train.copy()
    df["y_pred"]   = y_pred
    df["residuo"]  = df["price"] - y_pred
    df["segmento"] = pd.qcut(
        df["price"], q=4,
        labels=["Q1 (Bajo)", "Q2 (Medio-Bajo)", "Q3 (Medio-Alto)", "Q4 (Alto)"]
    )

    def _fila(g: pd.DataFrame) -> pd.Series:
        res = g["residuo"]
        return pd.Series({
            "N prop.":                   f"{len(g):,}",
            "Precio mediano\n(M COP)":   f"{g['price'].median()/1e6:.0f}",
            "MAE\n(M COP)":              f"{res.abs().mean()/1e6:.1f}",
            "Sesgo\n(M COP)":            f"{res.mean()/1e6:+.1f}",
            "% Sobrepred.":              f"{(res < 0).mean():.1%}",
            "Pérdida capital\n(MM COP)": f"{res[res<0].abs().sum()/1e9:.1f}",
            "Oport. perdida\n(MM COP)":  f"{res[res>0].sum()/1e9:.1f}",
        })

    tabla = df.groupby("segmento", observed=True).apply(_fila).reset_index()
    tabla.columns = ["Segmento"] + list(tabla.columns[1:])
    fila_total = pd.Series({
        "Segmento":                  "TOTAL",
        "N prop.":                   f"{len(df):,}",
        "Precio mediano\n(M COP)":   f"{df['price'].median()/1e6:.0f}",
        "MAE\n(M COP)":              f"{m['mae_cop']/1e6:.1f}",
        "Sesgo\n(M COP)":            f"{m['bias']/1e6:+.1f}",
        "% Sobrepred.":              f"{m['prop_sobrepred']:.1%}",
        "Pérdida capital\n(MM COP)": f"{m['perdida_cop']/1e9:.1f}",
        "Oport. perdida\n(MM COP)":  f"{m['oportunidad_cop']/1e9:.1f}",
    })
    tabla = pd.concat([tabla, fila_total.to_frame().T], ignore_index=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    col_labels = list(tabla.columns)

    table = ax.table(
        cellText  = tabla.values.tolist(),
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row == 0:
            cell.set_facecolor(C_DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(tabla):   # fila TOTAL
            cell.set_facecolor("#FFF3CD")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8F9FA")
        else:
            cell.set_facecolor("white")

    ax.set_title("Métricas de Negocio por Segmento de Precio — SL_003",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out / "10_tabla_metricas_negocio.png")
    plt.close()
    print("  [OK] 10_tabla_metricas_negocio.png")


# =============================================================================
# SECCIÓN 11: MAIN
# =============================================================================

def main() -> None:
    print("=" * 65)
    print("  ANÁLISIS DE ERRORES — SL_003")
    print("  SL-2: XGB_009 (w=0.580) + NN_003 (w=0.454), meta NNLS")
    print("  Kaggle MAE público: 199,851,631 COP")
    print("  Stage 5: From Prediction to Business — MECA 4107")
    print("=" * 65)

    _estilo_publicacion()

    # ── [1/4] Carga ──────────────────────────────────────────────────────────
    print("\n[1/4] Cargando train_final.csv...")
    path_train = PROCESSED / "train_final.csv"
    if not path_train.exists():
        raise FileNotFoundError(
            f"No se encontró {path_train}.\n"
            "Corre primero los scripts de DataPreparation."
        )
    train   = pd.read_csv(path_train)
    y_log   = np.log(train["price"].values)
    y_real  = train["price"].values
    print(f"  train_final.csv: {train.shape[0]:,} filas × {train.shape[1]} columnas")
    print(f"  Device NN: {DEVICE}")

    # ── [2/4] OOF del SL_003 ─────────────────────────────────────────────────
    print("\n[2/4] Generando OOF del SL_003...")
    print("  (mismo pipeline que 05_sl003.py — sin tuning ni submission)")
    oof_sl_log, oof_xgb_log, oof_nn_log = generar_oof_sl003(train, y_log)

    # Convertir a COP para métricas de negocio
    y_pred_cop   = np.exp(oof_sl_log)

    # ── [3/4] Métricas ───────────────────────────────────────────────────────
    print("\n[3/4] Calculando métricas de negocio...")
    m = calcular_metricas(y_real, y_pred_cop)

    print(f"\n  {'Métrica':<38} {'Valor':>18}")
    print(f"  {'-'*56}")
    print(f"  {'MAE OOF (M COP)':<38} {m['mae_cop']/1e6:>17.1f}")
    print(f"  {'Sesgo promedio (M COP)':<38} {m['bias']/1e6:>+17.1f}")
    print(f"  {'% Sobrepredicción':<38} {m['prop_sobrepred']:>17.1%}")
    print(f"  {'% Subpredicción':<38} {m['prop_subpred']:>17.1%}")
    print(f"  {'N casos de sobrepredicción':<38} {m['n_sobre']:>17,}")
    print(f"  {'Pérdida de capital (MM COP)':<38} {m['perdida_cop']/1e9:>17.1f}")
    print(f"  {'Oportunidad perdida (MM COP)':<38} {m['oportunidad_cop']/1e9:>17.1f}")

    # ── [4/4] Gráficas ───────────────────────────────────────────────────────
    print(f"\n[4/4] Generando 10 figuras en {OUT_DIR}...")
    plot_distribucion_residuos(m, OUT_DIR)
    plot_sesgo_donut(m, OUT_DIR)
    plot_mae_por_segmento(train, y_pred_cop, m, OUT_DIR)
    plot_mapa_errores(train, y_pred_cop, OUT_DIR)
    plot_perdida_financiera(train, y_pred_cop, m, OUT_DIR)
    plot_scatter_real_vs_pred(train, y_pred_cop, m, OUT_DIR)
    plot_error_relativo_segmento(train, y_pred_cop, OUT_DIR)
    plot_comparacion_modelos_kaggle(OUT_DIR)          # sin datos OOF — usa registry
    plot_heatmap_espacial(train, y_pred_cop, OUT_DIR)
    plot_tabla_metricas_negocio(train, y_pred_cop, m, OUT_DIR)

    print(f"\n{'='*65}")
    print(f"  ✓  10 figuras guardadas en {OUT_DIR.name}/")
    print(f"  ✓  Figura 08 usa MAE Kaggle del registry (sin OOF)")
    print(f"  ✓  Actualiza MODELOS_COMPARACION['SL_003'] tras subir a Kaggle")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()