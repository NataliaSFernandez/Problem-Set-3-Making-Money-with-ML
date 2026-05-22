"""
Gráficas de Comparación 

Genera 5 figuras:
  01_comparacion_sl_xgb.png  — Barras horizontales SL vs XGB  
  02_comparacion_sl_nn.png   — Barras horizontales SL vs NN   (
  03_tradeoff_todos.png      — Trade-off scatter (todos los modelos)
  04_dumbbell_sl_xgb.png     — Dumbbell plot SL vs XGB 
  05_sesgo_sl_nn.png         — Panel de sesgo espacial SL vs NN 

"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
import numpy as np

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
OUT  = BASE / "02_outputs" / "ComparacionFiguras"
OUT.mkdir(parents=True, exist_ok=True)

# ── Paleta ────────────────────────────────────────────────────────────────────
C = {
    "SL":   "#2A7D6B",
    "XGB":  "#C0582A",
    "NN":   "#6B3A7D",
    "RF":   "#2B5F8E",
    "grid": "#E0E0E0",
    "text": "#1A1A1A",
    "muted":"#7A7A7A",
    "bg":   "#FAFAFA",
    "diag": "#BBBBBB",
}

# ── Datos ─────────────────────────────────────────────────────────────────────
MODELS = {
    "SL_003":  dict(label="SuperLearner", color=C["SL"],
                    rand=0.16545, esp=0.23244, delta=0.06700, kaggle=199.9),
    "XGB_009": dict(label="XGBoost",      color=C["XGB"],
                    rand=0.17598, esp=0.21079, delta=0.03480, kaggle=207.9),
    "NN_003":  dict(label="Neural Net",   color=C["NN"],
                    rand=0.13739, esp=0.24331, delta=0.10592, kaggle=217.0),
    "RF_002":  dict(label="Random Forest",color=C["RF"],
                    rand=0.17280, esp=0.17280, delta=0.00000, kaggle=281.8),
    "CART_002":dict(label="CART",         color="#8C7A3A",
                    rand=0.22825, esp=0.22825, delta=0.00000, kaggle=289.7),
    "EN_002":  dict(label="Elastic Net",  color="#5A7A5A",
                    rand=0.26300, esp=0.29800, delta=0.03500, kaggle=314.0),
    "LR_001":  dict(label="Reg. Lineal",  color="#8A8A8A",
                    rand=0.27873, esp=0.27873, delta=0.00000, kaggle=318.7),
}

# ── Estilo global ─────────────────────────────────────────────────────────────
def _rc():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.facecolor":    C["bg"],
        "figure.facecolor":  "white",
        "axes.edgecolor":    "#CCCCCC",
        "axes.linewidth":    0.7,
        "axes.grid":         True,
        "grid.color":        C["grid"],
        "grid.linewidth":    0.6,
        "grid.alpha":        1.0,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.color":       C["muted"],
        "ytick.color":       C["muted"],
        "xtick.major.size":  3,
        "figure.dpi":        150,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
        "savefig.pad_inches":0.18,
    })

# =============================================================================
# HELPER — barras horizontales con grid y ejes X visibles
# =============================================================================

def _comparacion_barras(m1_id, m2_id, metricas, titulo, fname):
    """
    Barras horizontales, un subpanel por métrica, escala X independiente.
    Ahora con:  grid visible · eje X con tick labels · fondo #FAFAFA por panel
    """
    m1 = MODELS[m1_id]
    m2 = MODELS[m2_id]
    n  = len(metricas)

    fig = plt.figure(figsize=(8.0, n * 1.72 + 1.4), facecolor="white")

    # Título
    fig.text(0.04, 0.975, titulo,
             fontsize=13, fontweight="bold", color=C["text"],
             va="top", ha="left")

    # Leyenda
    lx = 0.04
    for mid, m in [(m1_id, m1), (m2_id, m2)]:
        fig.add_artist(mpatches.FancyBboxPatch(
            (lx, 0.925), 0.026, 0.020,
            boxstyle="square,pad=0", facecolor=m["color"],
            transform=fig.transFigure, clip_on=False, alpha=0.88))
        fig.text(lx + 0.032, 0.935, mid,
                 fontsize=10, va="center", color=C["text"],
                 transform=fig.transFigure)
        lx += 0.19

    top    = 0.90
    bottom = 0.06
    total  = top - bottom
    h_each = total / n
    pad    = 0.022

    for i, met in enumerate(metricas):
        ybot = bottom + (n - 1 - i) * h_each + pad
        ytop = bottom + (n - i) * h_each - pad
        ax   = fig.add_axes([0.22, ybot, 0.62, ytop - ybot])

        v1  = m1[met["key"]] * met.get("scale", 1)
        v2  = m2[met["key"]] * met.get("scale", 1)
        fmt = met.get("fmt", ".5f")

        xmax = max(v1, v2) * 1.18
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1)

        # ── Barras ──────────────────────────────────────────────────────────
        ax.barh(0.66, v1, height=0.26, color=m1["color"], alpha=0.85,
                edgecolor="white", linewidth=0.6, align="center")
        ax.barh(0.30, v2, height=0.26, color=m2["color"], alpha=0.85,
                edgecolor="white", linewidth=0.6, align="center")

        # ── Valores al final de cada barra ──────────────────────────────────
        ax.text(v1 + xmax * 0.010, 0.66, f"{v1:{fmt}}",
                va="center", ha="left", fontsize=10.5,
                color=m1["color"], fontweight="bold")
        ax.text(v2 + xmax * 0.010, 0.30, f"{v2:{fmt}}",
                va="center", ha="left", fontsize=10.5,
                color=m2["color"], fontweight="bold")

        # ── Grid: líneas verticales ──────────────────────────────────────────
        ax.set_yticks([])
        ax.xaxis.set_tick_params(labelsize=8.5, colors=C["muted"])
        ax.tick_params(axis="x", which="both", length=3, color="#CCCCCC")

        # Spine inferior visible, resto no
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["bottom"].set_color("#CCCCCC")

        # Grid solo en X (líneas verticales sutiles)
        ax.grid(True, axis="x", color=C["grid"],
                linewidth=0.7, linestyle="-", alpha=1.0)
        ax.set_axisbelow(True)
        ax.set_facecolor(C["bg"])

        # Ticks X: 3 referencias limpias
        ax.set_xticks(np.linspace(0, xmax * 0.95, 4)[1:])

        # Etiqueta de la métrica a la izquierda (coordenadas figura)
        ax_pos = ax.get_position()
        mid_y  = (ax_pos.y0 + ax_pos.y1) / 2
        fig.text(0.20, mid_y, met["label"],
                 fontsize=10.5, va="center", ha="right",
                 color=C["text"], transform=fig.transFigure)

    plt.savefig(OUT / fname)
    plt.close()
    print(f"  [OK] {fname}")


# =============================================================================
# FIG 01 — SL vs XGB: barras horizontales
# =============================================================================

def fig_sl_xgb_barras():
    _comparacion_barras(
        "SL_003", "XGB_009",
        [
            dict(label="MAE CV\nAleatorio", key="rand",   fmt=".5f"),
            dict(label="MAE CV\nEspacial",  key="esp",    fmt=".5f"),
            dict(label="Kaggle\nMAE (M)",   key="kaggle", fmt=".1f"),
        ],
        "Comparación directa de métricas",
        "01_comparacion_sl_xgb.png",
    )


# =============================================================================
# FIG 02 — SL vs NN: barras horizontales
# =============================================================================

def fig_sl_nn_barras():
    _comparacion_barras(
        "SL_003", "NN_003",
        [
            dict(label="MAE CV\nAleatorio", key="rand",   fmt=".5f"),
            dict(label="MAE CV\nEspacial",  key="esp",    fmt=".5f"),
            dict(label="Kaggle\nMAE (M)",   key="kaggle", fmt=".1f"),
        ],
        "Comparación directa de métricas",
        "02_comparacion_sl_nn.png",
    )


# =============================================================================
# FIG 03 — Trade-off scatter (todos los modelos)
# =============================================================================

def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(C["bg"])

    lim = (0.12, 0.335)
    xs  = np.linspace(lim[0], lim[1], 300)

    ax.fill_between(xs, xs, lim[1], alpha=0.04, color="#CC4444", zorder=0)
    ax.text(0.288, 0.308, "spatial\noverfitting",
            fontsize=9, color="#CC4444", alpha=0.65,
            ha="center", va="bottom", style="italic")
    ax.plot(xs, xs, color=C["diag"], lw=1.0, ls="--", zorder=1)

    max_kag = max(m["kaggle"] for m in MODELS.values())
    min_kag = min(m["kaggle"] for m in MODELS.values())

    offsets = {
        "SL_003":   ( 0.004,  0.007), "XGB_009":  ( 0.004, -0.011),
        "NN_003":   (-0.003,  0.009), "RF_002":   ( 0.004,  0.006),
        "CART_002": ( 0.004,  0.006), "EN_002":   ( 0.004,  0.006),
        "LR_001":   ( 0.004, -0.011),
    }
    for mid, m in MODELS.items():
        sz = (max_kag - m["kaggle"]) / (max_kag - min_kag) * 280 + 55
        ax.scatter(m["rand"], m["esp"], s=sz, color=m["color"],
                   alpha=0.88, edgecolors="white", linewidths=1.2, zorder=4)
        ox, oy = offsets.get(mid, (0.004, 0.007))
        ax.text(m["rand"] + ox, m["esp"] + oy, m["label"],
                fontsize=10, color=m["color"], fontweight="semibold", va="bottom")

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("MAE — random CV  (in-distribution)", fontsize=11, color=C["text"], labelpad=8)
    ax.set_ylabel("MAE — spatial CV  (out-of-distribution)", fontsize=11, color=C["text"], labelpad=8)
    ax.set_title("Generalization trade-off: random vs spatial CV",
                 fontsize=12.5, color=C["text"], fontweight="medium", pad=12)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9.5, colors=C["muted"])

    leg_handles = [
        mlines.Line2D([], [], color="#888", marker="o", ls="none",
                      ms=9,  label=f"Kaggle ≤ {int(min_kag)} M COP"),
        mlines.Line2D([], [], color="#888", marker="o", ls="none",
                      ms=4,  label=f"Kaggle ≥ {int(max_kag)} M COP"),
        mlines.Line2D([], [], color=C["diag"], ls="--", lw=1.0, label="No spatial bias"),
    ]
    ax.legend(handles=leg_handles, loc="upper left", fontsize=9,
              framealpha=0.92, edgecolor="#DDDDDD", borderpad=0.8)

    plt.tight_layout()
    plt.savefig(OUT / "03_tradeoff_todos.png")
    plt.close()
    print("  [OK] 03_tradeoff_todos.png")


# =============================================================================
# FIG 04 — Dumbbell plot: SL vs XGB (adicional)
# ─────────────────────────────────────────────
# Muestra las 3 métricas clave como segmentos SL ↔ XGB.
# Permite ver de un vistazo en cuál métrica gana cada modelo y por cuánto.
# Ganador = punto más a la izquierda (menor MAE = mejor).
# =============================================================================

def fig_dumbbell_sl_xgb():
    sl  = MODELS["SL_003"]
    xgb = MODELS["XGB_009"]

    metricas = [
        ("MAE Kaggle (M COP)", "kaggle", 1,    ".1f", (185, 220)),
        ("MAE CV Espacial",    "esp",    1000, ".1f", (190, 250)),  # ×1000 para legibilidad
        ("MAE CV Aleatorio",   "rand",   1000, ".1f", (148, 200)),
    ]
    # Escala ×1000 en CV para que los números sean comparables visualmente
    labels_x = {
        "kaggle": "MAE (millones de COP)",
        "esp":    "MAE × 1000  (log price)",
        "rand":   "MAE × 1000  (log price)",
    }

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.set_facecolor(C["bg"])

    y_pos = np.arange(len(metricas))

    for i, (lab, key, scale, fmt, xlim) in enumerate(metricas):
        v_sl  = sl[key]  * scale
        v_xgb = xgb[key] * scale

        # Segmento conector
        ax.plot([v_sl, v_xgb], [i, i],
                color="#CCCCCC", lw=2.0, solid_capstyle="round", zorder=2)

        # Punto SL (círculo)
        ax.scatter(v_sl, i, s=130, color=C["SL"], zorder=5,
                   edgecolors="white", linewidths=1.2)
        # Punto XGB (cuadrado)
        ax.scatter(v_xgb, i, s=130, color=C["XGB"], zorder=5,
                   marker="s", edgecolors="white", linewidths=1.2)

        # Valores
        offset = (max(v_sl, v_xgb) - min(v_sl, v_xgb)) * 0.08
        for v, color, ha in [(v_sl,  C["SL"],  "center"), (v_xgb, C["XGB"], "center")]:
            ax.text(v, i + 0.22, f"{v:{fmt}}",
                    ha=ha, va="bottom", fontsize=10, color=color, fontweight="bold")

        # Ganador (pequeña etiqueta discreta)
        winner_color = C["SL"] if v_sl < v_xgb else C["XGB"]
        winner_label = "SL gana" if v_sl < v_xgb else "XGB gana"
        pct = abs(v_sl - v_xgb) / max(v_sl, v_xgb) * 100
        mid_x = (v_sl + v_xgb) / 2
        ax.text(mid_x, i - 0.30, f"{winner_label}  ({pct:.1f}%)",
                ha="center", va="top", fontsize=8.5,
                color=winner_color, style="italic", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[0] for m in metricas], fontsize=11, color=C["text"])
    ax.tick_params(axis="y", left=False)
    ax.tick_params(axis="x", labelsize=9, colors=C["muted"])
    ax.set_ylim(-0.75, len(metricas) - 0.4)

    # Spine limpio
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.grid(True, axis="x", color=C["grid"], linewidth=0.7, alpha=1.0)
    ax.set_axisbelow(True)

    ax.set_title("SL_003 vs XGB_009 — comparación por métrica",
                 fontsize=12.5, color=C["text"], fontweight="medium",
                 pad=14, loc="left")

    # Leyenda
    leg = [
        mlines.Line2D([], [], color=C["SL"],  marker="o", ls="none",
                      ms=7, label="SL_003  (SuperLearner)"),
        mlines.Line2D([], [], color=C["XGB"], marker="s", ls="none",
                      ms=7, label="XGB_009 (XGBoost)"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=9.5,
              framealpha=0.92, edgecolor="#DDDDDD")

    plt.tight_layout()
    plt.savefig(OUT / "04_dumbbell_sl_xgb.png")
    plt.close()
    print("  [OK] 04_dumbbell_sl_xgb.png")


# =============================================================================
# FIG 05 — Panel de sesgo espacial: SL vs NN  (adicional)
# ─────────────────────────────────────────────────────────
# Dos paneles:
#   Izquierda  — Scatter CV rand vs CV esp para SL y NN (+ diagonal)
#                Muestra el paradox: NN mejor in-sample, SL mejor out-of-sample
#   Derecha    — Bar chart del sesgo espacial Δ = CV_esp − CV_rand
#                Con línea de referencia y coloreado por magnitud del sesgo
# =============================================================================

def fig_sesgo_sl_nn():
    sl = MODELS["SL_003"]
    nn = MODELS["NN_003"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                             gridspec_kw={"wspace": 0.38})
    for ax in axes:
        ax.set_facecolor(C["bg"])

    # ── Panel A: scatter CV rand vs CV esp ────────────────────────────────────
    ax = axes[0]
    lim = (0.12, 0.27)
    xs  = np.linspace(lim[0], lim[1], 200)

    ax.fill_between(xs, xs, lim[1], alpha=0.06, color="#CC4444", zorder=0)
    ax.plot(xs, xs, color=C["diag"], lw=1.0, ls="--", zorder=1,
            label="Sin sesgo espacial")

    # Puntos SL y NN
    for mid, m, marker, sz in [
        ("SL_003", sl, "o", 160),
        ("NN_003", nn, "D", 160),
    ]:
        ax.scatter(m["rand"], m["esp"], s=sz, color=m["color"],
                   marker=marker, zorder=5,
                   edgecolors="white", linewidths=1.2)

    # Flecha de SL → NN (mostrando que NN se mueve izquierda pero sube)
    ax.annotate(
        "", xy=(nn["rand"], nn["esp"]), xytext=(sl["rand"], sl["esp"]),
        arrowprops=dict(arrowstyle="-|>", color="#AAAAAA",
                        lw=1.0, mutation_scale=9),
    )

    # Etiquetas
    ax.text(sl["rand"] + 0.003, sl["esp"] + 0.004,
            "SuperLearner", fontsize=10, color=C["SL"], fontweight="semibold")
    ax.text(nn["rand"] - 0.002, nn["esp"] + 0.004,
            "Neural Net",   fontsize=10, color=C["NN"], fontweight="semibold",
            ha="right")

    # Anotación del paradox
    ax.text(0.97, 0.04,
            "NN: mejor in-sample\nSL: mejor out-of-sample",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=C["muted"], style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("MAE — CV aleatorio", fontsize=10.5, color=C["text"])
    ax.set_ylabel("MAE — CV espacial",  fontsize=10.5, color=C["text"])
    ax.set_title("(A)  Ajuste general vs generalización espacial",
                 fontsize=11, color=C["text"], fontweight="medium", loc="left", pad=10)
    ax.legend(fontsize=9, framealpha=0.92, edgecolor="#DDDDDD")
    ax.tick_params(labelsize=9, colors=C["muted"])
    ax.text(0.22, 0.255, "sobreajuste\nespacial",
            fontsize=8, color="#CC4444", alpha=0.55, ha="center", style="italic")

    # ── Panel B: sesgo espacial Δ ─────────────────────────────────────────────
    ax2 = axes[1]

    modelos_b = [sl, nn]
    labels_b  = ["SL_003\n(SuperLearner)", "NN_003\n(Neural Net)"]
    deltas_b  = [sl["delta"], nn["delta"]]
    colors_b  = [C["SL"], C["NN"]]
    x_pos     = np.arange(2)

    bars = ax2.bar(x_pos, deltas_b, width=0.42,
                   color=colors_b, alpha=0.85,
                   edgecolor="white", linewidth=0.8)

    # Línea de referencia: XGB (el mejor en sesgo de los tree models)
    ax2.axhline(MODELS["XGB_009"]["delta"], color=C["XGB"],
                lw=1.4, ls="--", alpha=0.7,
                label=f"XGB_009 referencia  Δ={MODELS['XGB_009']['delta']:.3f}")

    # Valores encima de las barras
    for bar, val, color in zip(bars, deltas_b, colors_b):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f"Δ = {val:.3f}",
                 ha="center", va="bottom", fontsize=10.5,
                 color=color, fontweight="bold")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_b, fontsize=10.5, color=C["text"])
    ax2.set_ylabel("Sesgo espacial  Δ = MAE_esp − MAE_rand",
                   fontsize=10, color=C["text"])
    ax2.set_title("(B)  Sesgo espacial por modelo",
                  fontsize=11, color=C["text"], fontweight="medium", loc="left", pad=10)
    ax2.set_ylim(0, max(deltas_b) * 1.35)
    ax2.tick_params(labelsize=9, colors=C["muted"])
    ax2.legend(fontsize=9, framealpha=0.92, edgecolor="#DDDDDD")

    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    ax2.grid(True, axis="y", color=C["grid"], linewidth=0.7)
    ax2.set_axisbelow(True)

    fig.suptitle("SL_003 vs NN_003 — Análisis de generalización espacial",
                 fontsize=13, fontweight="bold", color=C["text"], y=1.02)

    plt.savefig(OUT / "05_sesgo_sl_nn.png")
    plt.close()
    print("  [OK] 05_sesgo_sl_nn.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    _rc()
    print("=" * 55)
    print("  GRÁFICAS COMPARACIÓN v2 — MECA 4107")
    print("=" * 55)

    print("\n[1/5] Barras SL vs XGB (con grid + ejes)")
    fig_sl_xgb_barras()

    print("[2/5] Barras SL vs NN  (con grid + ejes)")
    fig_sl_nn_barras()

    print("[3/5] Trade-off scatter (todos los modelos)")
    fig_tradeoff()

    print("[4/5] Dumbbell SL vs XGB  ← adicional")
    fig_dumbbell_sl_xgb()

    print("[5/5] Panel sesgo SL vs NN  ← adicional")
    fig_sesgo_sl_nn()

    print(f"\n  Figuras en: {OUT}")
    for f in ["01_comparacion_sl_xgb.png", "02_comparacion_sl_nn.png",
              "03_tradeoff_todos.png", "04_dumbbell_sl_xgb.png", "05_sesgo_sl_nn.png"]:
        print(f"  · {f}")


if __name__ == "__main__":
    main()