"""
Elastic Net para predicción de log(price) de vivienda en Bogotá — EN_002
========================================================================
Problem Set 3 — MECA 4107 (Big Data & ML para Economía Aplicada)

Mejoras sobre EN_001
--------------------
1. Log-transformaciones de variables de distancia y superficie
   (skewness de surface_total = 89; distancias con skew > 1)
   → hace las relaciones linealmente capturables para Elastic Net.

2. Features de interacción hedónicas
   surface_ratio   = surface_covered / surface_total  (compacidad)
   bath_surface    = bathrooms × log(surface_covered) (lujo compacto)
   rooms_sq        = rooms²                           (no-linealidad)

3. Coordenadas espaciales polinomiales
   lat, lon, lat², lon², lat×lon
   Capturan el gradiente de precios sobre el mapa de Bogotá directamente,
   complementando los proxies OSM que EN_001 ya usaba.
   Crítico porque el test es Chapinero (zona geográfica específica).

Diagnóstico de EN_001 que motiva estos cambios
----------------------------------------------
- Patrón en cuña en residuos vs ajustados → no-linealidad no capturada.
- Histograma de residuos sesgado a la izquierda → sobrepredicción
  sistemática en propiedades caras.
- 17/29 features zeroed-out por LASSO → variables en escala cruda
  con alta skewness son penalizadas en exceso por la regularización.
- MAE_spatial = 0.2986 (200M COP); objetivo EN_002: bajar a <0.27.

Pipeline (igual que EN_001)
---------------------------
1. Carga train_final.csv / test_final.csv
2. Ingeniería de features: transformaciones + interacciones + spatial poly
3. StandardScaler ajustado SOLO sobre train
4. CV aleatorio 5-fold  (KFold)                   → MAE_random
5. CV espacial leave-one-block-out (GroupKFold)    → MAE_spatial
6. Comparación  Δ = MAE_spatial − MAE_random
7. Modelo final con hiperparámetros del CV espacial
8. Submission → 03_submissions/
9. Registro automático → 02_outputs/model_registry.xlsx
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
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL
# =============================================================================

AUTOR    = "Jonathan"
MODEL_ID = "EN_002"
SEED     = 42

L1_RATIOS    = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
N_ALPHAS     = 100
CV_FOLDS     = 5
SPATIAL_GRID = 5

BASE        = Path(__file__).parent.parent.parent.parent
PROCESSED   = BASE / "00_data" / "processed"
SUBMISSIONS = BASE / "03_submissions"
DIR_MODEL   = BASE / "02_outputs" / "Models" / "ElasticNet" / MODEL_ID
REGISTRY    = BASE / "02_outputs" / "model_registry.xlsx"

for d in [SUBMISSIONS, DIR_MODEL, BASE / "02_outputs"]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECCIÓN 2: GRUPOS DE FEATURES
# =============================================================================

STRUCTURAL = [
    "log_surface_total",    # log(surface_total+1) — corrige skew=89
    "log_surface_covered",  # log(surface_covered+1)
    "surface_ratio",        # surface_covered / surface_total (compacidad)
    "rooms",
    "rooms_sq",             # rooms² — captura no-linealidad
    "bedrooms",
    "bathrooms",
    "bath_surface",         # bathrooms × log(surface_covered) — lujo compacto
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
    # Distancias en log-escala (log1p): corrige right-skew
    "log_dist_cbd_km",
    "log_dist_transmilenio_m",
    "log_dist_via_arterial_m",
    "log_dist_hospital_m",
    "log_dist_centro_com_m",
    "log_dist_parque_m",
    # Counts y scores (escala original — ya menos sesgados)
    "n_restaurantes_500m",
    "n_bancos_500m",
    "walkability_score",
    "densidad_vial",
]

SPATIAL_POLY = [
    # Polinomio espacial de grado 2: captura gradiente de precios en el mapa
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
    """Calcula todas las variables transformadas e interacciones in-place."""
    d = df.copy()

    # ── Superficies en log-escala ─────────────────────────────────────────
    d["log_surface_total"]   = np.log1p(d["surface_total"])
    d["log_surface_covered"] = np.log1p(d["surface_covered"])
    # Ratio de compacidad: qué fracción de la superficie total está cubierta
    d["surface_ratio"]       = d["surface_covered"] / (d["surface_total"] + 1)

    # ── Interacciones hedónicas ───────────────────────────────────────────
    d["rooms_sq"]    = d["rooms"] ** 2
    d["bath_surface"] = d["bathrooms"] * d["log_surface_covered"]

    # ── Distancias en log-escala ──────────────────────────────────────────
    dist_cols = [
        "dist_cbd_km", "dist_transmilenio_m", "dist_via_arterial_m",
        "dist_hospital_m", "dist_centro_com_m", "dist_parque_m",
    ]
    for col in dist_cols:
        d[f"log_{col}"] = np.log1p(d[col])

    # ── Polinomio espacial ────────────────────────────────────────────────
    d["lat_sq"]  = d["lat"] ** 2
    d["lon_sq"]  = d["lon"] ** 2
    d["lat_lon"] = d["lat"] * d["lon"]

    # ── Dummy property_type ───────────────────────────────────────────────
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


def crear_grupos_espaciales(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    G        = SPATIAL_GRID
    lat_bins = pd.qcut(lat, G, labels=False, duplicates="drop")
    lon_bins = pd.qcut(lon, G, labels=False, duplicates="drop")
    return (lat_bins * G + lon_bins).astype(int)


# =============================================================================
# SECCIÓN 4: CV ALEATORIO
# =============================================================================

def cv_aleatorio(X: np.ndarray, y: np.ndarray) -> tuple:
    print(f"\n{'='*60}")
    print(f"  CV ALEATORIO ({CV_FOLDS}-fold, KFold) — EN_tp_random")
    print(f"{'='*60}")

    scaler      = StandardScaler()
    X_sc        = scaler.fit_transform(X)
    best_mae    = np.inf
    best_ratio  = L1_RATIOS[0]
    best_alpha  = 1e-3
    resultados  = []

    for ratio in L1_RATIOS:
        enc = ElasticNetCV(
            l1_ratio=ratio, n_alphas=N_ALPHAS, cv=CV_FOLDS,
            max_iter=10_000, random_state=SEED, n_jobs=-1,
        )
        enc.fit(X_sc, y)

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
          f"MAE_log={best_mae:.5f}  ← optimista")
    return best_ratio, best_alpha, pd.DataFrame(resultados), best_mae


# =============================================================================
# SECCIÓN 5: CV ESPACIAL
# =============================================================================

def cv_espacial(X: np.ndarray, y: np.ndarray,
                grupos: np.ndarray) -> tuple:
    print(f"\n{'='*60}")
    print(f"  CV ESPACIAL leave-one-block-out (GroupKFold) — EN_tp_spatial")
    print(f"{'='*60}")

    n_groups = len(np.unique(grupos))
    n_splits = min(CV_FOLDS, n_groups)

    scaler     = StandardScaler()
    X_sc       = scaler.fit_transform(X)
    best_mae   = np.inf
    best_ratio = L1_RATIOS[0]
    best_alpha = 1e-3
    resultados = []

    gkf = GroupKFold(n_splits=n_splits)

    for ratio in L1_RATIOS:
        enc = ElasticNetCV(
            l1_ratio=ratio, n_alphas=N_ALPHAS,
            cv=list(gkf.split(X_sc, y, groups=grupos)),
            max_iter=10_000, random_state=SEED, n_jobs=-1,
        )
        enc.fit(X_sc, y)

        model    = ElasticNet(alpha=enc.alpha_, l1_ratio=ratio,
                              max_iter=10_000, random_state=SEED)
        oof_pred = np.zeros_like(y)
        for tr_idx, val_idx in gkf.split(X, y, groups=grupos):
            sc_fold  = StandardScaler()
            X_tr_sc  = sc_fold.fit_transform(X[tr_idx])
            X_va_sc  = sc_fold.transform(X[val_idx])
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
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    model  = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                        max_iter=10_000, random_state=SEED)
    model.fit(X_sc, y)
    return scaler, model


def mae_oof_random(X: np.ndarray, y: np.ndarray,
                   l1_ratio: float, alpha: float) -> tuple:
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
    mae_cop = float(mean_absolute_error(np.exp(y), np.exp(oof_pred)))
    return mae_log, mae_cop


def mae_oof_espacial(X: np.ndarray, y: np.ndarray,
                     grupos: np.ndarray,
                     l1_ratio: float, alpha: float) -> tuple:
    gkf      = GroupKFold(n_splits=min(CV_FOLDS, len(np.unique(grupos))))
    oof_pred = np.zeros_like(y)
    for tr_idx, val_idx in gkf.split(X, y, groups=grupos):
        sc      = StandardScaler()
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
# SECCIÓN 7: GRÁFICOS
# =============================================================================

def plot_coeficientes(model: ElasticNet, feature_names: list,
                      l1_ratio: float, alpha: float) -> None:
    coefs    = pd.Series(model.coef_, index=feature_names)
    coefs_nz = coefs[coefs != 0].sort_values()
    n        = min(30, len(coefs_nz))
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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_rand["l1_ratio"], df_rand["mae_log"],
            "o-", color="#3498db", lw=2, label="CV aleatorio (optimista)")
    ax.plot(df_esp["l1_ratio"],  df_esp["mae_log"],
            "s--", color="#e74c3c", lw=2, label="CV espacial (honesto)")
    x_fb  = df_rand["l1_ratio"].to_numpy()
    y1_fb = df_rand["mae_log"].to_numpy()
    y2_fb = df_esp["mae_log"].to_numpy()
    ax.fill_between(x_fb, y1_fb, y2_fb, alpha=0.12, color="#e74c3c",
                    label="Sesgo de optimismo Δ")  # type: ignore[arg-type]
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
# SECCIÓN 8: SUBMISSION
# =============================================================================


def generar_submission(test: pd.DataFrame, y_pred_log: np.ndarray,
                       l1_ratio: float, alpha: float) -> str:
    submission = pd.DataFrame({
        "property_id": test["property_id"],
        "price":       np.exp(y_pred_log),
    })
    sub_name = (
        f"submission_{MODEL_ID}_l1r{int(l1_ratio*100):03d}"
        f"_a{alpha:.0e}.csv"
    ).replace("-", "n")
    submission.to_csv(SUBMISSIONS / sub_name, index=False)

    print(f"\n  Submission: 03_submissions/{sub_name}")
    print(f"  Filas: {len(submission):,} | "
          f"price media: {submission['price'].mean()/1e6:.1f}M COP")
    return sub_name



# =============================================================================
# SECCIÓN 9: REGISTRO
# =============================================================================

def registrar(sub_name: str, feature_cols: list,
              l1_ratio_rand: float, alpha_rand: float,
              l1_ratio_esp: float,  alpha_esp: float,
              n_nonzero: int,
              mae_rand_log: float, mae_rand_cop: float,
              mae_esp_log:  float, mae_esp_cop: float,
              mae_train_log: float) -> None:
    sesgo  = mae_esp_log - mae_rand_log
    nueva  = {
        "model_id":           MODEL_ID,
        "fecha":              str(date.today()),
        "autor":              AUTOR,
        "algoritmo":          "ElasticNet",
        "n_features":         len(feature_cols),
        "n_coefs_nonzero":    n_nonzero,
        "cv_folds":           CV_FOLDS,
        "rand_cv_mae_log":    round(mae_rand_log, 5),
        "rand_cv_mae_cop_M":  round(mae_rand_cop / 1e6, 2),
        "esp_cv_mae_log":     round(mae_esp_log, 5),
        "esp_cv_mae_cop_M":   round(mae_esp_cop / 1e6, 2),
        "sesgo_delta_log":    round(sesgo, 5),
        "train_mae_log":      round(mae_train_log, 5),
        "kaggle_public_MAE":  None,
        "l1_ratio_spatial":   l1_ratio_esp,
        "alpha_spatial":      round(alpha_esp, 8),
        "l1_ratio_random":    l1_ratio_rand,
        "alpha_random":       round(alpha_rand, 8),
        "spatial_grid":       f"{SPATIAL_GRID}x{SPATIAL_GRID}",
        "submission_file":    sub_name,
        "notas": (
            f"EN_002: log-transforms + interacciones + poly espacial. "
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
    print(f"  ELASTIC NET — {MODEL_ID}")
    print(f"{'='*60}")
    print("  Mejoras vs EN_001: log-transforms + interacciones + spatial poly")

    print("\n[1/8] Cargando datos...")
    train, test = cargar_datos()
    y_train     = np.log(train["price"].values)
    print(f"  TRAIN: {train.shape[0]:,} | TEST: {test.shape[0]:,}")

    print("\n[2/8] Construyendo features...")
    X_train_df   = construir_features(train)
    feature_cols = list(X_train_df.columns)
    X_test_df    = construir_features(test, fit_cols=feature_cols)
    X_train      = X_train_df.values
    X_test       = X_test_df.values
    print(f"  Features: {len(feature_cols)}  "
          f"({len(STRUCTURAL)} estructurales + {len(TEXT)} texto + "
          f"{len(OSM)} OSM + {len(SPATIAL_POLY)} spatial + es_apartamento)")

    grupos    = crear_grupos_espaciales(train["lat"].values, train["lon"].values)
    n_bloques = len(np.unique(grupos))
    print(f"\n[3/8] Bloques espaciales: {n_bloques} "
          f"(cuadrícula {SPATIAL_GRID}×{SPATIAL_GRID})")

    print("\n[4/8] CV aleatorio (KFold, EN_tp_random)...")
    ratio_rand, alpha_rand, df_rand, _ = cv_aleatorio(X_train, y_train)
    mae_rand_log, mae_rand_cop = mae_oof_random(
        X_train, y_train, ratio_rand, alpha_rand
    )
    print(f"  MAE aleatorio (log): {mae_rand_log:.5f} ← optimista")

    print("\n[5/8] CV espacial (GroupKFold, EN_tp_spatial)...")
    ratio_esp, alpha_esp, df_esp, _ = cv_espacial(X_train, y_train, grupos)
    mae_esp_log, mae_esp_cop = mae_oof_espacial(
        X_train, y_train, grupos, ratio_esp, alpha_esp
    )
    print(f"  MAE espacial (log): {mae_esp_log:.5f} ← honesto")

    sesgo = mae_esp_log - mae_rand_log
    print(f"\n  ── Sesgo de optimismo del CV aleatorio ──────────────────")
    print(f"  Δ = MAE_espacial − MAE_aleatorio = {sesgo:+.5f}")
    print(f"  El CV espacial es el mejor proxy del rendimiento en Chapinero.")

    print("\n[6/8] Ajustando modelo final (hiperparámetros del CV espacial)...")
    scaler, model = ajustar_modelo_final(X_train, y_train, ratio_esp, alpha_esp)
    y_pred_train  = model.predict(scaler.transform(X_train))
    mae_train_log = float(mean_absolute_error(y_train, y_pred_train))
    mae_train_cop = float(
        mean_absolute_error(np.exp(y_train), np.exp(y_pred_train))
    )
    n_nonzero = int(np.sum(model.coef_ != 0))
    print(f"  MAE train (log): {mae_train_log:.5f} | "
          f"MAE train (COP): {mae_train_cop/1e6:.1f}M")
    print(f"  Coefs ≠ 0: {n_nonzero}/{len(feature_cols)}")

    print("\n[7/8] Generando gráficos...")
    plot_coeficientes(model, feature_cols, ratio_esp, alpha_esp)
    plot_cv_comparacion(df_rand, df_esp)
    plot_residuos(y_train, y_pred_train)

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

    print(f"\n{'='*60}")
    print(f"  RESUMEN FINAL — {MODEL_ID}")
    print(f"{'='*60}")
    print(f"  Features          : {len(feature_cols)} "
          f"(vs 29 en EN_001 — +{len(feature_cols)-29} nuevas)")
    print(f"  Hiperparámetros   : l1_ratio={ratio_esp}, α={alpha_esp:.2e}")
    print(f"  Coefs ≠ 0         : {n_nonzero}/{len(feature_cols)}")
    print(f"  CV aleatorio MAE  : {mae_rand_log:.5f}  "
          f"({mae_rand_cop/1e6:.1f}M COP)  ← optimista")
    print(f"  CV espacial MAE   : {mae_esp_log:.5f}  "
          f"({mae_esp_cop/1e6:.1f}M COP)  ← honesto / proxy Chapinero")
    print(f"  EN_001 espacial   : 0.29860  (200.3M COP)  ← referencia")
    print(f"  Mejora vs EN_001  : {0.29860 - mae_esp_log:+.5f} MAE_log")
    print(f"  Sesgo Δ           : {sesgo:+.5f}")
    print(f"  Submission        : 03_submissions/{sub_name}")
    print(f"  Registry          : 02_outputs/model_registry.xlsx")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
