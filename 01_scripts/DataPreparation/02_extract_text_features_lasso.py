"""
Pipeline alternativo de extracción de variables de texto (LASSO sobre n-gramas)
===============================================================================
Motivación
----------
El script 01_extract_text_features.py usa patrones regex definidos manualmente.
Este pipeline sigue la lógica data-driven de Gentzkow & Shapiro (2010) /
Lecture 15 del curso: en lugar de decidir a priori qué frases importan, se
construye una matriz documento-término con todos los bi-gramas y tri-gramas del
corpus y se usa LASSO para seleccionar automáticamente cuáles predicen el precio.

Preprocesamiento del texto
--------------------------
1. Combinar `title` + `description` en un único texto por inmueble.
2. Pasar el texto por spaCy (modelo `es_core_news_md`) para:
   - Lematizar cada token (reduce formas flexionadas a su lema: "vistas" → "vista").
   - Eliminar stopwords y signos de puntuación.
3. Normalizar los lemas: minúsculas + eliminar tildes (unicodedata NFD).
   Equivalente al paso de lematización de ConTexto (LematizadorSpacy), seguido
   por la normalización estándar del pipeline 01.
4. Aplicar stemming con NLTK SnowballStemmer('spanish'):
   - Reduce aún más la variación morfológica ("apartamento" → "apartament").
   - Equivalente al stem_texto() de ConTexto (Stemmer).
   Aplicar stemming sobre los lemas (y no sobre el texto crudo) es la secuencia
   correcta: la lematización garantiza que el stemmer parte de formas canónicas.

Construcción de la matriz documento-término
-------------------------------------------
- n-gramas: bi-gramas y tri-gramas (ngram_range=(2, 3)).
- Vocabulario: los MAX_FEATURES n-gramas más frecuentes en el corpus de train.
- Vectorizador ajustado SOLO sobre train (anti data-leakage).
- Binarización: D_ij = 1{c_ij > 0} — idéntico al notebook del curso.

Selección de variables con LASSO
---------------------------------
- Outcome: log(price) calculado sobre train.
- Modelo: LassoCV con 5-fold cross-validation (selecciona lambda óptimo).
- Ajustado SOLO sobre train.
- Se conservan los n-gramas con coeficiente ≠ 0 → variables seleccionadas.

Outputs
-------
  00_data/processed/train_texto_lasso.csv  ← train original + columnas txt_*
  00_data/processed/test_texto_lasso.csv   ← test  original + columnas txt_*
  00_data/processed/ngrams_seleccionados.csv ← tabla de n-gramas y coeficientes LASSO

Nota
----
El nombre de cada columna de texto tiene el prefijo `txt_` y los espacios entre
palabras se reemplazan por `_` (e.g., "cocina integral" → "txt_cocin_integr").
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================

import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import spacy
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MaxAbsScaler

import nltk
nltk.download("stopwords", quiet=True)

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

BASE      = Path(__file__).parent.parent.parent
RAW       = BASE / "00_data" / "raw"
PROCESSED = BASE / "00_data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

TRAIN = RAW / "train.csv"
TEST  = RAW / "test.csv"
CACHE = BASE / "cache" / "text_lasso"
CACHE.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HIPERPARÁMETROS DEL PIPELINE
# =============================================================================

MAX_FEATURES = 5_000   # n-gramas más frecuentes a considerar en el vocabulario
CV_FOLDS     = 5       # folds para LassoCV
RANDOM_STATE = 42

# =============================================================================
# CARGA LAZY DEL MODELO DE SPACY Y RECURSOS NLTK
# (solo se inicializa si el paso de preprocesamiento no está cacheado)
# =============================================================================

_nlp     = None
_stemmer = None
_stopwords: set[str] = set()


def _iniciar_nlp() -> None:
    """Carga spaCy y NLTK la primera vez que se necesitan."""
    global _nlp, _stemmer, _stopwords
    if _nlp is not None:
        return
    print("  Cargando modelo de spaCy (es_core_news_md)...")
    _nlp     = spacy.load("es_core_news_md", disable=["ner", "parser"])
    _stemmer = SnowballStemmer("spanish")
    _stopwords_nltk = set(nltk_stopwords.words("spanish"))
    _stopwords_extra = {
        "inmueble", "propiedad", "apartamento", "apto", "casa", "bogota",
        "colombia", "oportunidad", "venta", "arriendo", "precio", "valor",
        "metros", "metro", "mt", "mts", "m2", "m³",
        "mas", "todo", "toda", "todos", "todas",
    }
    _stopwords = _stopwords_nltk | _stopwords_extra

# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# =============================================================================

def _quitar_tildes(texto: str) -> str:
    """Elimina diacríticos (tildes, diéresis, cedillas) de un texto UTF-8."""
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def preprocesar_texto(texto: str) -> str:
    """
    Devuelve una cadena de tokens stemizados lista para el vectorizador.

    Pasos:
      1. spaCy tokeniza y lematiza el texto original (con tildes, para que
         el modelo lingüístico funcione correctamente).
      2. Se descartan stopwords, signos de puntuación, espacios y tokens muy
         cortos (< 3 caracteres).
      3. Los lemas se normalizan: minúsculas + eliminación de tildes.
      4. Se aplica SnowballStemmer('spanish') sobre cada lema normalizado.

    Ejemplo:
        "Apartamentos remodelados con vistas panorámicas"
        → lemas spaCy : ["apartamento", "remodelado", "vista", "panorámico"]
        → normalizados: ["apartamento", "remodelado", "vista", "panoramico"]
        → stems        : ["apartament", "remodel", "vist", "panoram"]
        → salida       : "apartament remodel vist panoram"
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""

    assert _nlp is not None, "Llama _iniciar_nlp() antes de preprocesar_texto()"
    doc = _nlp(texto[:100_000])  # spaCy tiene límite de caracteres

    tokens = []
    for token in doc:
        # Descartar puntuación, espacios, stopwords spaCy y stopwords extra
        if token.is_punct or token.is_space:
            continue
        lemma = token.lemma_.lower()
        if lemma in _stopwords or token.is_stop:
            continue
        if len(lemma) < 3:
            continue

        # Normalizar lema (quitar tildes) y aplicar stemming
        lemma_norm = _quitar_tildes(lemma)
        stem       = _stemmer.stem(lemma_norm)  # type: ignore[union-attr]
        if len(stem) >= 3:
            tokens.append(stem)

    return " ".join(tokens)


def construir_corpus(df: pd.DataFrame) -> pd.Series:
    """Combina title + description, inicializa spaCy si hace falta y procesa fila a fila."""
    _iniciar_nlp()
    title = df["title"].fillna("").astype(str)
    desc  = df["description"].fillna("").astype(str)
    texto = (title + " " + desc)
    print(f"  Preprocesando {len(texto):,} documentos con spaCy + SnowballStemmer...")
    return texto.apply(preprocesar_texto)


# =============================================================================
# VECTORIZACIÓN Y BINARIZACIÓN
# =============================================================================

def ajustar_vectorizador(corpus_train: pd.Series) -> CountVectorizer:
    """
    Ajusta un CountVectorizer de bi-gramas y tri-gramas sobre el corpus de train.

    Parámetros del vectorizador:
      - ngram_range=(2, 3) : captura bi-gramas ("cocin integr") y tri-gramas
                             ("prim pis balcon").
      - max_features       : limita el vocabulario a los MAX_FEATURES n-gramas
                             más frecuentes para mantener la dimensionalidad
                             manejable antes de LASSO.
      - min_df=5           : descarta n-gramas que aparecen en menos de 5
                             documentos (muy raros = ruido).
      - analyzer='word'    : opera sobre palabras (tokens), no caracteres.

    Se ajusta SÓLO sobre train. Aplicarlo sobre test sería data leakage porque
    las frecuencias del vocabulario (max_features, min_df) se calcularían con
    información del set de evaluación.
    """
    vec = CountVectorizer(
        ngram_range=(2, 3),
        max_features=MAX_FEATURES,
        min_df=5,
        analyzer="word",
    )
    vec.fit(corpus_train)
    return vec


def binarizar(matrix) -> np.ndarray:
    """Convierte una matriz de conteos a indicadores 0/1 (D_ij = 1{c_ij > 0})."""
    return (matrix > 0).astype(np.float32)


# =============================================================================
# SELECCIÓN CON LASSO
# =============================================================================

def seleccionar_con_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Ajusta LassoCV sobre la matriz binarizada de n-gramas y retorna:
      - índices de las columnas con coeficiente ≠ 0.
      - DataFrame con los n-gramas seleccionados y sus coeficientes.

    Por qué LassoCV:
      - Con p >> n (muchos n-gramas, pocos inmuebles) OLS no está identificado.
      - LASSO penaliza con λ Σ|β_j| llevando a cero los n-gramas sin poder
        predictivo. Es el equivalente del cv.gamlr del cuaderno del curso.
      - La selección de λ se hace por validación cruzada (CV_FOLDS folds),
        eligiendo el λ que minimiza el MSE fuera de muestra.
    """
    print(f"  Ajustando LassoCV ({CV_FOLDS}-fold) sobre {X_train.shape[1]:,} n-gramas...")

    # Escalar la matriz (LASSO es sensible a la escala de las columnas)
    scaler  = MaxAbsScaler()
    X_sc    = scaler.fit_transform(X_train)

    lasso = LassoCV(
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        max_iter=5_000,
        n_jobs=-1,
    )
    lasso.fit(X_sc, y_train)

    coefs    = lasso.coef_
    seleccion = np.where(coefs != 0)[0]
    print(f"  λ óptimo: {lasso.alpha_:.6f} — n-gramas seleccionados: {len(seleccion):,}")

    tabla = pd.DataFrame({
        "ngrama":      feature_names[seleccion],
        "coeficiente": coefs[seleccion],
    }).sort_values("coeficiente", ascending=False).reset_index(drop=True)

    return seleccion, scaler, tabla


# =============================================================================
# CONSTRUCCIÓN DEL DATASET DE SALIDA
# =============================================================================

def agregar_columnas_texto(
    df: pd.DataFrame,
    X_bin: np.ndarray,
    feature_names: np.ndarray,
    seleccion: np.ndarray,
) -> pd.DataFrame:
    """
    Agrega al DataFrame original las columnas binarias de los n-gramas
    seleccionados por LASSO.

    Convención de nombres de columna:
      "cocin integr" → "txt_cocin_integr"   (espacios → guiones bajos, prefijo txt_)
    """
    X_sel = X_bin[:, seleccion]
    # toarray() convierte la sparse matrix a ndarray denso antes de pasar a pandas
    if scipy.sparse.issparse(X_sel):
        X_sel = X_sel.toarray()
    cols  = ["txt_" + name.replace(" ", "_") for name in feature_names[seleccion]]
    df_text = pd.DataFrame(X_sel, columns=cols, index=df.index)
    return pd.concat([df, df_text], axis=1)


# =============================================================================
# RESUMEN EN CONSOLA
# =============================================================================

def resumen(tabla: pd.DataFrame, top_n: int = 10) -> None:
    """Imprime los n-gramas con coeficientes más positivos y más negativos."""
    print(f"\n{'='*60}")
    print(f"  N-GRAMAS SELECCIONADOS POR LASSO  ({len(tabla):,} en total)")
    print(f"{'='*60}")

    print(f"\nTop {top_n} n-gramas asociados a MAYOR precio:")
    print(tabla.head(top_n).to_string(index=False))

    print(f"\nTop {top_n} n-gramas asociados a MENOR precio:")
    print(tabla.tail(top_n).iloc[::-1].to_string(index=False))


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("Cargando train.csv y test.csv...")
    train = pd.read_csv(TRAIN)
    test  = pd.read_csv(TEST)

    # ── 1. Preprocesamiento de texto (con caché) ──────────────────────────────
    # El procesamiento con spaCy es el paso más lento (~10 min para 50K docs).
    # Si el corpus ya fue procesado en una ejecución anterior, se carga del disco.
    _cache_train = CACHE / "corpus_train.parquet"
    _cache_test  = CACHE / "corpus_test.parquet"

    if _cache_train.exists() and _cache_test.exists():
        print("\n[1/4] Cargando corpus preprocesado desde caché...")
        corpus_train = pd.read_parquet(_cache_train)["texto"]
        corpus_test  = pd.read_parquet(_cache_test)["texto"]
        print(f"  → train: {len(corpus_train):,} docs  |  test: {len(corpus_test):,} docs")
    else:
        print("\n[1/4] Preprocesando texto (lematización + stemming)...")
        print("  → TRAIN")
        corpus_train = construir_corpus(train)
        print("  → TEST")
        corpus_test  = construir_corpus(test)
        corpus_train.to_frame("texto").to_parquet(_cache_train, index=False)
        corpus_test.to_frame("texto").to_parquet(_cache_test,   index=False)
        print(f"  Corpus guardado en caché: {CACHE}")

    # ── 2. Vectorización (con caché) ─────────────────────────────────────────
    _cache_Xtrain = CACHE / "X_train_bin.npz"
    _cache_Xtest  = CACHE / "X_test_bin.npz"
    _cache_names  = CACHE / "feature_names.npy"

    if _cache_Xtrain.exists() and _cache_Xtest.exists() and _cache_names.exists():
        print(f"\n[2/4] Cargando matrices binarizadas desde caché...")
        X_train_bin = scipy.sparse.load_npz(_cache_Xtrain)
        X_test_bin  = scipy.sparse.load_npz(_cache_Xtest)
        names       = np.load(_cache_names, allow_pickle=True)
    else:
        print(f"\n[2/4] Vectorizando (bi-gramas + tri-gramas, top {MAX_FEATURES:,})...")
        vec   = ajustar_vectorizador(corpus_train)
        names = np.array(vec.get_feature_names_out())

        X_train_bin = binarizar(vec.transform(corpus_train))
        X_test_bin  = binarizar(vec.transform(corpus_test))

        scipy.sparse.save_npz(_cache_Xtrain, X_train_bin)
        scipy.sparse.save_npz(_cache_Xtest,  X_test_bin)
        np.save(_cache_names, names)
        print(f"  Matrices guardadas en caché: {CACHE}")

    print(f"  Matriz train: {X_train_bin.shape[0]:,} × {X_train_bin.shape[1]:,}")
    print(f"  Matriz test : {X_test_bin.shape[0]:,}  × {X_test_bin.shape[1]:,}")

    # ── 3. LASSO sobre log(price) ────────────────────────────────────────────
    print("\n[3/4] Seleccionando n-gramas con LASSO sobre log(price)...")
    log_price_train = np.log(train["price"].values)

    seleccion, scaler, tabla = seleccionar_con_lasso(
        X_train_bin, log_price_train, names
    )

    # % de missings (ceros) por n-grama en train y test
    # Un 0 indica que el n-grama no aparece en ese documento.
    # mean() sobre la columna sparse da la fracción de 1s; 1 - eso = fracción de 0s.
    X_train_sel = X_train_bin[:, seleccion]
    X_test_sel  = X_test_bin[:, seleccion]
    # np.asarray(sparse.mean(axis=0)) → (1, k) → aplanar a (k,)
    freq_train = np.asarray(X_train_sel.mean(axis=0)).flatten()
    freq_test  = np.asarray(X_test_sel.mean(axis=0)).flatten()
    tabla["pct_missing_train"] = (1 - freq_train) * 100
    tabla["pct_missing_test"]  = (1 - freq_test)  * 100

    resumen(tabla)

    # Guardar tabla de n-gramas seleccionados
    tabla.to_csv(PROCESSED / "ngrams_seleccionados.csv", index=False)
    print(f"\n  Tabla guardada en 00_data/processed/ngrams_seleccionados.csv")

    # ── 4. Guardar datasets enriquecidos ─────────────────────────────────────
    print("\n[4/4] Guardando datasets enriquecidos...")
    train_out = agregar_columnas_texto(train, X_train_bin, names, seleccion)
    test_out  = agregar_columnas_texto(test,  X_test_bin,  names, seleccion)

    train_out.to_csv(PROCESSED / "train_texto_lasso.csv", index=False)
    test_out.to_csv( PROCESSED / "test_texto_lasso.csv",  index=False)

    n_cols = len(seleccion)
    print(f"\n  train_texto_lasso.csv : {len(train_out):,} filas, "
          f"{len(train_out.columns)} columnas ({n_cols} nuevas txt_*)")
    print(f"  test_texto_lasso.csv  : {len(test_out):,} filas, "
          f"{len(test_out.columns)} columnas ({n_cols} nuevas txt_*)")


if __name__ == "__main__":
    main()
