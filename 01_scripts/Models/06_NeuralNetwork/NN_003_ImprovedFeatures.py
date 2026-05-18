"""
Red Neuronal (PyTorch) para predicción de log(price) — NN_003
=============================================================
Problem Set 3 — MECA 4107 (Big Data & ML para Economía Aplicada)

Mejoras sobre NN_002
--------------------
1. Features log-transformadas + interacciones + polinomio espacial (igual que EN_002)
   - log(surface_total), log(surface_covered), surface_ratio
   - bath_surface = bathrooms × log(surface_covered)
   - rooms_sq = rooms²
   - log(distancias OSM) — corrige right-skew de dist_via_arterial (skew=1.6)
   - lat, lon, lat², lon², lat×lon — gradiente espacial de precios
   → 37 features vs 29 de NN_002. Aborda el patrón en cuña de residuos.

2. Batch Normalization tras cada capa oculta
   Re-normaliza activaciones en cada mini-batch, independientemente del
   StandardScaler inicial. Acelera convergencia y mejora generalización.
   Arquitectura: Linear → BN → ReLU → Dropout (orden estándar para tabular).

3. Regularización reducida (menos agresiva)
   NN_002: Dropout 0.3/0.2 + L1=L2=1e-4 → brecha train/val = 0.013 (subajuste)
   NN_003: Dropout 0.2/0.1 + L1=L2=5e-5 → permite más capacidad real

4. PATIENCE = 30 (vs 15 en NN_002)
   La curva de val MAE de NN_002 seguía bajando a época 75. Más paciencia
   permite al scheduler reducir el LR más veces antes de parar.

5. Arquitectura más ancha: p → 256 → 256 → 128 → 64 → 1
   Capa extra de 256 neuronas antes de la compresión. Más capacidad para
   capturar interacciones no-lineales entre las features de localización.

Diagnóstico de NN_002 que motiva estos cambios
----------------------------------------------
- Patrón en cuña en residuos (mismo que EN_001/002): residuos negativos
  sistemáticos para log(price) > 21 → features en escala cruda dificultan
  que la red aprenda el precio de propiedades de lujo.
- Val MAE seguía decreciendo a época 75 → early stopping prematuro.
- Brecha train/val = 0.013 → regularización excesiva para este dataset.
- CV espacial MAE_log = 0.2495; objetivo NN_003: bajar a < 0.22.
"""

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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 101010
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))


# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================

AUTOR    = "Jonathan"
MODEL_ID = "NN_003"

EPOCHS     = 300    # más épocas máximas — el early stopping decide cuándo parar
BATCH_SIZE = 512
VAL_SPLIT  = 0.2
PATIENCE   = 30     # NN_002 paró a época 75 con val MAE aún decreciendo
LR         = 1e-3

# Regularización más suave — NN_002 tenía brecha train/val de solo 0.013
L1_REG    = 5e-5
L2_REG    = 5e-5
DROPOUT_1 = 0.2    # reducido de 0.3
DROPOUT_2 = 0.1    # reducido de 0.2

SPATIAL_GRID = 5

BASE        = Path(__file__).parent.parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "NeuralNetwork" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

for d in [SUBMISSIONS, DIR_MODEL, BASE / "02_outputs"]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECCIÓN 2: GRUPOS DE FEATURES (igual que EN_002)
# =============================================================================

STRUCTURAL = [
    "log_surface_total",
    "log_surface_covered",
    "surface_ratio",
    "rooms",
    "rooms_sq",
    "bedrooms",
    "bathrooms",
    "bath_surface",
    "month",
    "year",
]

TEXT = [
    "remodelado",
    "vista_panoramica",
    "deposito",
    "conjunto_cerrado",
    "balcon_terraza",
    "tfidf_premium",
    "parqueaderos_txt",
    "piso_txt",
    "gimnasio",
    "amenidades",
    "num_amenidades",
]

OSM = [
    "log_dist_cbd_km",
    "log_dist_transmilenio_m",
    "log_dist_via_arterial_m",
    "log_dist_hospital_m",
    "log_dist_centro_com_m",
    "log_dist_parque_m",
    "n_restaurantes_500m",
    "n_bancos_500m",
    "walkability_score",
    "densidad_vial",
]

SPATIAL_POLY = [
    "lat",
    "lon",
    "lat_sq",
    "lon_sq",
    "lat_lon",
]


# =============================================================================
# SECCIÓN 3: CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED / "train_final.csv")
    test  = pd.read_csv(PROCESSED / "test_final.csv")
    return train, test


def _agregar_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["log_surface_total"]   = np.log1p(d["surface_total"])
    d["log_surface_covered"] = np.log1p(d["surface_covered"])
    d["surface_ratio"]       = d["surface_covered"] / (d["surface_total"] + 1)
    d["rooms_sq"]            = d["rooms"] ** 2
    d["bath_surface"]        = d["bathrooms"] * d["log_surface_covered"]

    dist_cols = [
        "dist_cbd_km", "dist_transmilenio_m", "dist_via_arterial_m",
        "dist_hospital_m", "dist_centro_com_m", "dist_parque_m",
    ]
    for col in dist_cols:
        d[f"log_{col}"] = np.log1p(d[col])

    d["lat_sq"]  = d["lat"] ** 2
    d["lon_sq"]  = d["lon"] ** 2
    d["lat_lon"] = d["lat"] * d["lon"]
    d["es_apartamento"] = (d["property_type"] == "Apartamento").astype(int)

    return d


def construir_features(df: pd.DataFrame,
                       fit_cols: list | None = None) -> pd.DataFrame:
    d        = _agregar_features_derivadas(df)
    all_cols = STRUCTURAL + TEXT + OSM + SPATIAL_POLY + ["es_apartamento"]
    all_cols = [c for c in all_cols if c in d.columns]
    X        = d[all_cols]
    if fit_cols is not None:
        X = X[fit_cols]
    return X


# =============================================================================
# SECCIÓN 4: ARQUITECTURA DE LA RED
# =============================================================================

class MLP(nn.Module):
    """
    MLP para regresión hedónica con Batch Normalization.

    Arquitectura: p → 256 → 256 → 128 → 64 → 1
    ---------------------------------------------
    Respecto a NN_002 (p → 256 → 128 → 64 → 1):
      - Capa extra de 256 en el frente: más capacidad para aprender interacciones
        de primer orden entre las 37 features antes de comprimir.
      - Batch Normalization (BN) tras cada capa lineal y antes de ReLU.

    Por qué BN antes de ReLU (no después)
    --------------------------------------
    BN normaliza las pre-activaciones (salida de Linear) a media≈0 y std≈1.
    ReLU luego corta los valores negativos. Si BN fuera después de ReLU,
    operaría sobre activaciones ya no-negativas y perdería parte de su efecto
    normalizador. El orden Linear→BN→ReLU→Dropout es el estándar empírico
    para redes feedforward en datos tabulares.

    BN y Dropout interacción
    ------------------------
    BN tiene su propio efecto regularizador (estimación ruidosa de media/std
    dentro del mini-batch). Con BN, el Dropout puede ser más suave — de ahí
    que reduzcamos a 0.2/0.1 en lugar de 0.3/0.2 de NN_002.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.red = nn.Sequential(
            # Capa 1: 37 → 256 con BN
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_1),

            # Capa 2: 256 → 256 (nueva respecto a NN_002) con BN
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_1),

            # Capa 3: 256 → 128 con BN
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_2),

            # Capa 4: 128 → 64 sin Dropout (representación ya comprimida)
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Salida: 64 → 1, sin activación
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


def penalizacion_l1(model: MLP) -> torch.Tensor:
    device = next(model.parameters()).device
    total  = torch.zeros(1, device=device)
    for name, p in model.named_parameters():
        if "weight" in name:
            total = total + p.abs().sum()
    return total


# =============================================================================
# SECCIÓN 5: ENTRENAMIENTO
# =============================================================================

def entrenar(X_sc: np.ndarray,
             y: np.ndarray,
             device: torch.device = DEVICE) -> tuple[MLP, dict]:
    n     = len(X_sc)
    n_val = int(n * VAL_SPLIT)
    X_tr,  X_val = X_sc[:n - n_val],  X_sc[n - n_val:]
    y_tr,  y_val = y[:n - n_val],     y[n - n_val:]

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).to(device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = MLP(X_sc.shape[1]).to(device)
    with torch.no_grad():
        model.red[-1].bias.fill_(float(y_tr.mean()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=L2_REG)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    history: dict  = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

    print(f"\n  Entrenando en {device} (máx {EPOCHS} épocas, "
          f"early stopping patience={PATIENCE})...")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        batch_losses: list[float] = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if L1_REG > 0:
                loss = loss + L1_REG * penalizacion_l1(model)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_mae  = (val_pred - y_val_t).abs().mean().item()
            tr_pred  = model(X_tr_t)
            tr_mae   = (tr_pred - y_tr_t).abs().mean().item()

        train_loss = float(np.mean(batch_losses))
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mae"].append(tr_mae)
        history["val_mae"].append(val_mae)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early Stopping en época {epoch}/{EPOCHS}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    epochs_run = len(history["loss"])
    print(f"  Épocas efectivas: {epochs_run} | "
          f"mejor val_loss (MSE): {best_val_loss:.6f} | "
          f"mejor val_MAE: {min(history['val_mae']):.6f}")
    return model, history


def predecir(model: MLP, X_sc: np.ndarray,
             device: torch.device = DEVICE) -> np.ndarray:
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
    print(f"\n{'='*60}")
    print(f"  CV ESPACIAL — cuadrícula {SPATIAL_GRID}×{SPATIAL_GRID}")
    print(f"{'='*60}")

    G        = SPATIAL_GRID
    lat_bins = pd.qcut(lat, G, labels=False, duplicates="drop")
    lon_bins = pd.qcut(lon, G, labels=False, duplicates="drop")
    celda    = lat_bins * G + lon_bins
    grupos   = np.array_split(np.unique(celda), 5)

    maes_log: list[float] = []
    maes_cop: list[float] = []

    for i, grp in enumerate(grupos):
        mask_val = np.isin(celda, grp)
        mask_tr  = ~mask_val

        if mask_val.sum() == 0 or mask_tr.sum() == 0:
            continue

        sc    = StandardScaler()
        X_tr  = sc.fit_transform(X_raw[mask_tr])
        X_val = sc.transform(X_raw[mask_val])
        y_tr, y_val = y[mask_tr], y[mask_val]

        m, _ = entrenar(X_tr, y_tr, device=device)
        y_hat = predecir(m, X_val, device=device)

        mae_log = float(mean_absolute_error(y_val, y_hat))
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
# SECCIÓN 7: GRÁFICOS
# =============================================================================

def plot_historia(history: dict, epochs_run: int) -> None:
    ep = range(1, epochs_run + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ep, history["loss"],     label="Train",      color="#3498db")
    axes[0].plot(ep, history["val_loss"], label="Validación", color="#e74c3c")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("MSE — log(price)")
    axes[0].set_title(f"Pérdida (MSE) — {MODEL_ID}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    folds = np.arange(1, len(maes_log) + 1)

    axes[0].bar(folds, maes_log, color="#9b59b6", edgecolor="white")
    axes[0].axhline(mae_val_log, color="blue", lw=1.5, linestyle="--",
                    label=f"Val. interno ({mae_val_log:.4f})")
    axes[0].axhline(0.24954, color="gray", lw=1, linestyle=":",
                    label="NN_002 ref (0.2495)")
    axes[0].set_xlabel("Fold espacial")
    axes[0].set_ylabel("MAE log(price)")
    axes[0].set_title(f"CV espacial — {MODEL_ID} (log)")
    axes[0].legend(fontsize=8)

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
# SECCIÓN 8: SUBMISSION
# =============================================================================

TEMPLATE_PATH = Path(
    "/Users/macbook/Downloads/uniandes-bdml-202610-ps-3/submission_template.csv"
)


def generar_submission(test: pd.DataFrame, y_pred_log: np.ndarray,
                       epochs_run: int, total_params: int) -> str:
    submission = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),
    })
    template = pd.read_csv(TEMPLATE_PATH)
    submission = (template[["property_id"]]
                  .merge(submission, on="property_id", how="left"))

    sub_name = f"submission_{MODEL_ID}_ep{epochs_run}_p{total_params//1000}k.csv"
    submission.to_csv(SUBMISSIONS / sub_name, index=False)

    print(f"\n  Submission: 03_submissions/{sub_name}")
    print(f"  Filas: {len(submission):,} | "
          f"price media: {submission['price'].mean()/1e6:.1f}M COP")
    return sub_name


# =============================================================================
# SECCIÓN 9: REGISTRO
# =============================================================================

def registrar(sub_name: str, n_features: int, total_params: int,
              epochs_run: int,
              mae_train_log: float, mae_train_cop: float,
              mae_val_log: float,
              mae_esp_log: float, mae_esp_cop: float) -> None:
    delta = mae_esp_log - mae_val_log
    nueva = {
        "model_id":          MODEL_ID,
        "fecha":             str(date.today()),
        "autor":             AUTOR,
        "algoritmo":         "NeuralNetwork_PyTorch",
        "n_features":        n_features,
        "n_params":          total_params,
        "epochs_run":        epochs_run,
        "cv_folds":          f"val_split={VAL_SPLIT}",
        "cv_mae_log":        round(mae_val_log,   5),
        "cv_mae_cop_M":      None,
        "esp_mae_log":       round(mae_esp_log,   5),
        "esp_mae_cop_M":     round(mae_esp_cop / 1e6, 2),
        "train_mae_log":     round(mae_train_log, 5),
        "kaggle_public_MAE": None,
        "features_grupos":  (f"structural={len(STRUCTURAL)}, "
                              f"text={len(TEXT)}, osm={len(OSM)}, "
                              f"spatial_poly={len(SPATIAL_POLY)}, es_apartamento"),
        "arquitectura":      (f"p→256(BN,Drop{DROPOUT_1})→"
                              f"256(BN,Drop{DROPOUT_1})→"
                              f"128(BN,Drop{DROPOUT_2})→64(BN)→1 (ReLU)"),
        "reg_l1_l2":         f"l1={L1_REG}, l2={L2_REG}",
        "early_stopping":    f"patience={PATIENCE}",
        "spatial_grid":      f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "device":            str(DEVICE),
        "submission_file":   sub_name,
        "notas": (
            f"NN_003: BN + 4 capas + features EN_002 + reg reducida + patience=30. "
            f"Backend: {DEVICE}. l1={L1_REG}, l2={L2_REG}. "
            f"Épocas: {epochs_run}/{EPOCHS}. "
            f"MAE_log val.interno={mae_val_log:.5f}. "
            f"MAE_log CV espacial={mae_esp_log:.5f}. "
            f"Sesgo Δ={delta:+.5f}."
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

    print(f"  Registro: 02_outputs/model_registry.xlsx  ({len(df_reg)} modelos)")


# =============================================================================
# SECCIÓN 10: ENTRY POINT
# =============================================================================

def main() -> None:
    print(f"{'='*60}")
    print(f"  RED NEURONAL PyTorch — {MODEL_ID}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")
    print("  Mejoras: BN + 4 capas + log-features + reg reducida + patience=30")

    print("\n[1/7] Cargando datos...")
    train, test = cargar_datos()
    y_train_log = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    print("\n[2/7] Construyendo features...")
    X_train_df   = construir_features(train)
    feature_cols = list(X_train_df.columns)
    X_test_df    = construir_features(test, fit_cols=feature_cols)
    print(f"  Features: {len(feature_cols)} "
          f"({len(STRUCTURAL)} estructurales + {len(TEXT)} texto + "
          f"{len(OSM)} OSM + {len(SPATIAL_POLY)} spatial + es_apartamento)")

    print("\n[3/7] Estandarizando...")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_df.values)
    X_test_sc  = scaler.transform(X_test_df.values)

    print("\n[4/7] Entrenando red neuronal...")
    model, history = entrenar(X_train_sc, y_train_log)

    epochs_run   = len(history["loss"])
    mae_val_log  = float(min(history["val_mae"]))
    total_params = sum(p.numel() for p in model.parameters())

    y_pred_train      = predecir(model, X_train_sc)
    y_pred_train_clip = np.clip(y_pred_train, 10, 30)
    mae_train_log = float(mean_absolute_error(y_train_log, y_pred_train))
    mae_train_cop = float(
        mean_absolute_error(np.exp(y_train_log), np.exp(y_pred_train_clip))
    )
    print(f"  Parámetros totales: {total_params:,}")
    print(f"  MAE train (log): {mae_train_log:.5f} | "
          f"MAE train (COP): {mae_train_cop/1e6:.1f}M")
    print(f"  MAE val   (log): {mae_val_log:.5f}  "
          f"(brecha train/val: {mae_val_log - mae_train_log:+.5f})")

    print("\n[5/7] CV espacial...")
    maes_esp_log, maes_esp_cop = cv_espacial(
        X_train_df.values, y_train_log,
        train["lat"].values, train["lon"].values,
    )

    print("\n[6/7] Generando gráficos...")
    plot_historia(history, epochs_run)
    plot_residuos(y_train_log, y_pred_train)
    plot_cv_espacial(maes_esp_log, maes_esp_cop, mae_val_log)

    print("\n[7/7] Submission y registro...")
    y_pred_test = np.clip(predecir(model, X_test_sc), 10, 30)
    sub_name    = generar_submission(test, y_pred_test, epochs_run, total_params)
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

    delta = float(maes_esp_log.mean()) - mae_val_log
    print(f"\n{'='*60}")
    print(f"  RESUMEN — {MODEL_ID}")
    print(f"{'='*60}")
    print(f"  Device      : {DEVICE}")
    print(f"  Arquitectura: p→256(BN)→256(BN)→128(BN)→64(BN)→1")
    print(f"  Regulariz.  : L1={L1_REG}, L2={L2_REG}, "
          f"Dropout {DROPOUT_1}/{DROPOUT_2}")
    print(f"  Parámetros  : {total_params:,}")
    print(f"  Épocas      : {epochs_run}/{EPOCHS}")
    print(f"  MAE val int (log): {mae_val_log:.5f}   ← optimista")
    print(f"  MAE CV esp  (log): {maes_esp_log.mean():.5f}   "
          f"Δ={delta:+.5f}  ← proxy Chapinero")
    print(f"  NN_002 ref  (log): 0.24954")
    print(f"  Mejora vs NN_002 : {0.24954 - maes_esp_log.mean():+.5f} MAE_log")
    print(f"  MAE train   (log): {mae_train_log:.5f}")
    print(f"  Submission  : 03_submissions/{sub_name}")
    print(f"  Registry    : 02_outputs/model_registry.xlsx")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
