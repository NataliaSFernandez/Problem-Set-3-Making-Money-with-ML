"""
Extracción de variables hedónicas de texto (NLP)
================================================
Fuente de referencia : variables_hedónicas_Chapinero_v2.xlsx
                       → Categoría "Variables de Texto (NLP)" (filas 24-31)
Campos de entrada    : title + description  de  train.csv / test.csv
Dataset              : Properati Colombia — listados de Bogotá D.C.

Lógica general
--------------
1. Se combinan los campos `title` y `description` en un único texto por fila.
2. Se normaliza el texto: minúsculas y eliminación de tildes.
   - Esto hace que "depósito", "Deposito" y "deposito" sean equivalentes,
     lo que evita perder menciones por diferencias de acentuación o capitalización.
3. Sobre ese texto normalizado se aplican tres tipos de extracción:
   a) Dummies de mención  : buscar si aparece algún patrón de texto relevante (regex).
   b) Variables numéricas : longitud de descripción, piso, parqueaderos.
   c) Score continuo      : sentimiento del anuncio, score TF-IDF de términos premium.

Variables producidas (añadidas a las columnas originales del CSV)
-----------------------------------------------------------------
  Dummies de características del inmueble (0/1):
    remodelado       → menciona remodelación o estado nuevo/listo para estrenar
    vista_panoramica → menciona vistas, panorámicas o vista a los cerros
    deposito         → menciona depósito, bodega o cuarto útil
    conjunto_cerrado → menciona portería, vigilancia, áreas comunes, etc.
    balcon_terraza   → menciona balcón, terraza, patio, BBQ, etc.

  Scores NLP continuos:
    sentimiento_score → polaridad del texto (léxico pos/neg en español). Rango ≈ −1 a 1
    tfidf_premium     → suma de scores TF-IDF para vocabulario de lujo/premium

  Variables numéricas del inmueble:
    longitud_desc    → número de caracteres del campo `description`
    parqueaderos_txt → número de garajes/parqueaderos mencionados (np.nan si no hay mención)
    piso_txt         → número de piso mencionado en el texto (np.nan si no hay mención)
    banos_txt        → número de baños mencionados en el texto (np.nan si no hay mención)

  Variables de amenidades:
    gimnasio         → dummy (0/1): menciona explícitamente gimnasio/gym/fitness
    amenidades       → dummy (0/1): menciona al menos una amenidad de cualquier tipo
    num_amenidades   → entero: número de tipos distintos de amenidad mencionados
                       (piscina, jacuzzi, sauna, turco, spa, gimnasio, cancha,
                        squash, tenis, zona de trote, salón comunal, club house,
                        BBQ, salón de juegos, juegos infantiles, coworking, ascensor)

Output
------
  00_data/processed/train_texto.csv  ← train.csv original + 10 columnas nuevas
  00_data/processed/test_texto.csv   ← test.csv  original + 10 columnas nuevas

Nota sobre leakage
------------------
El vectorizador TF-IDF se ajusta ÚNICAMENTE sobre el corpus de entrenamiento
y se aplica (sin re-entrenarse) al corpus de test. Esto evita data leakage.
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================

import re                           # expresiones regulares para búsqueda de patrones
import unicodedata                  # para eliminar tildes/acentos de forma estándar
from pathlib import Path            # rutas de archivos independientes del SO

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

# __file__ apunta a 01_scripts/preparation/ → subimos tres niveles para llegar a la raíz
BASE      = Path(__file__).parent.parent.parent
RAW       = BASE / "00_data" / "raw"
PROCESSED = BASE / "00_data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)   # crea la carpeta si no existe

TRAIN = RAW / "train.csv"
TEST  = RAW / "test.csv"

# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO DE TEXTO
# =============================================================================

def normalizar(texto: str) -> str:
    """
    Transforma el texto a una forma canónica para comparación robusta:
      1. Convierte a minúsculas.
      2. Descompone cada carácter en su forma base + marcas diacríticas (NFD).
      3. Elimina las marcas diacríticas (tildes, diéresis, cedillas, etc.).

    Ejemplo:
        "Depósito" → "deposito"
        "Balcón"   → "balcon"

    Esto permite que los patrones de búsqueda funcionen sin tener que escribir
    variantes acentuadas, y que errores comunes como "Deposito" vs "depósito"
    no afecten la detección.
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


def combinar_texto(df: pd.DataFrame) -> pd.Series:
    """
    Une los campos `title` y `description` en un único texto por fila y lo
    normaliza. Los valores nulos se reemplazan por cadena vacía para evitar errores.

    Combinar ambos campos aumenta la cobertura: algunos atributos se mencionan
    sólo en el título ('Vista panorámica') y otros sólo en la descripción
    ('cuenta con depósito').
    """
    title = df["title"].fillna("").astype(str)
    desc  = df["description"].fillna("").astype(str)
    return (title + " " + desc).apply(normalizar)


def dummy_busqueda(texto: pd.Series, patrones: list) -> pd.Series:
    """
    Retorna una Serie binaria (0/1): toma valor 1 si ALGUNO de los patrones
    regex hace match en el texto de esa fila, y 0 si ninguno lo hace.

    Parámetros:
        texto    : Serie con texto ya normalizado (sin tildes, en minúsculas).
        patrones : lista de strings con expresiones regulares.
                   DEBEN escribirse sin tildes, ya que el texto fue normalizado.

    Los patrones se unen con OR (|) en una sola expresión para eficiencia.
    Se usa `str.contains` con `regex=True` para soportar expresiones complejas.
    """
    regex = "|".join(patrones)
    return texto.str.contains(regex, regex=True, na=False).astype(int)


# =============================================================================
# PATRONES DE MENCIÓN (DUMMIES)
# =============================================================================
#
# Cada clave del diccionario es el nombre de la columna resultante.
# El valor es una lista de expresiones regulares que, si alguna hace match
# en el texto del anuncio, da valor 1 a esa fila.
#
# Criterios de diseño de los patrones:
#   - \b marca límite de palabra para evitar falsos positivos
#     (e.g., "panoramico" no debe coincidir con "paisaje panoramicosss")
#   - [ao] cubre géneros gramaticales (remodelado/remodelada)
#   - (?:es)? cubre singular/plural con grupo no capturante para evitar warnings
#   - Se incluyen errores tipográficos frecuentes detectados en el corpus
#     (e.g., "depsito" en lugar de "deposito", ~1,721 ocurrencias)
# =============================================================================

PATRONES = {

    # -------------------------------------------------------------------------
    # Var 24 – Mención de 'remodelado' / 'renovado'
    #
    # Captura anuncios que indican que el inmueble fue recientemente intervenido
    # o está en condición de estreno. En modelos hedónicos, la renovación reciente
    # tiene un efecto positivo sobre el precio (Levmovitz & Munneke, 2020).
    #
    # Variaciones incluidas:
    #   - Forma verbal y adjetival: remodelado/a, renovado/a, reformado/a,
    #     restaurado/a, refaccionado/a, reconstruido/a, mejorado/a
    #   - Estado de nuevo: flamante, para estrenar, a estrenar, listo para estrenar,
    #     recién construido → indica propiedad nueva sin uso previo
    #   - Errores tipográficos: "remodeladoo", "remodeldo" rara vez aparecen;
    #     más común es omitir la tilde ("remodelado" vs "remodelâdo") → cubierto
    #     por la normalización
    # -------------------------------------------------------------------------
    "remodelado": [
        r"\bremodelad[ao]\b",          # remodelado / remodelada
        r"\bremodelacion\b",           # sustantivo: "en proceso de remodelación"
        r"\brenovad[ao]\b",            # renovado / renovada
        r"\brenovacion\b",             # sustantivo: "con renovación total"
        r"\bremodelar\b",              # infinitivo: "se puede remodelar"
        r"\brenovar\b",                # infinitivo: "listo para renovar"
        r"\breformad[ao]\b",           # reformado / reformada
        r"\brestaurad[ao]\b",          # restaurado / restaurada (más común en casas históricas)
        r"\brefaccionad[ao]\b",        # refaccionado / refaccionada (Colombia/México)
        r"\breconstruid[ao]\b",        # reconstruido / reconstruida
        r"\bmejorad[ao]\b",            # mejorado / mejorada (mejoras recientes)
        r"\bactualizado\b",            # actualizado (instalaciones, acabados)
        r"\bactualizada\b",
        r"\bpara estrenar\b",          # listo para estrenar → sin uso previo
        r"\ba estrenar\b",             # sinónimo colombiano de "para estrenar"
        r"\blisto para estrenar\b",    # frase completa frecuente en listados
        r"\brecien construid[ao]\b",   # recién construido/a
        r"\brecien terminad[ao]\b",    # recién terminado/a (obra reciente)
        r"\bacabados nuevos\b",        # menciona que los acabados son nuevos
        r"\btotalmente nuevo\b",       # "totalmente nuevo, sin uso"
        r"\btotalmente nueva\b",
        r"\barreglad[ao]\b",           # arreglado/a (uso informal en Colombia)
    ],

    # -------------------------------------------------------------------------
    # Var 25 – Mención de 'vista' / 'panorámica'
    #
    # Captura anuncios con alusión explícita a vistas desde el inmueble. Las vistas
    # se capitalizan en precio especialmente en Bogotá (cerros orientales, ciudad).
    # (Benson et al., 1998 — prima estimada 5–15%).
    #
    # Variaciones incluidas:
    #   - Directas: vista, vistas, panorámica, panorámico
    #   - Vistas contextuales bogotanas: vista a los cerros, vista a la ciudad
    #   - "exterior": en el argot inmobiliario colombiano, un apartamento "exterior"
    #     es aquel cuyas ventanas dan a la calle/exterior (no al patio interior),
    #     lo que implica luz natural y posible vista. Se captura sólo en frases
    #     como "vista exterior" o "apartamento exterior" para evitar ruido.
    #   - Horizonte, paisaje: términos menos frecuentes pero relevantes
    # -------------------------------------------------------------------------
    "vista_panoramica": [
        r"\bvista[s]?\b",                      # vista / vistas (muy frecuente)
        r"\bpanoramica[s]?\b",                 # panorámica/panorámicas
        r"\bpanoramico[s]?\b",                 # panorámico/panorámicos
        r"\bvista panoramica\b",               # frase completa
        r"\bvista[s]? a\b",                    # "vista a los cerros", "vista a la ciudad"
        r"\bvista[s]? al\b",                   # "vista al parque", "vista al norte"
        r"\bvista exterior\b",                 # ventana / balcón con vista al exterior
        r"\bapartamento exterior\b",           # aptos con ventanas a la calle = luz y vista
        r"\bapto exterior\b",                  # abreviatura frecuente
        r"\bpiso exterior\b",                  # piso cuyas habitaciones dan al exterior
        r"\bcerro[s]?\b",                      # "vista a los cerros" — muy bogotano
        r"\bmontana[s]?\b",                    # montaña/montañas (variante sin tilde)
        r"\bpaisaj\w*\b",                      # paisaje, paisajístico (sufijo variable)
        r"\bhorizonte\b",                      # "con vista al horizonte"
        r"\bvista 360\b",                      # "vista 360 grados"
    ],

    # -------------------------------------------------------------------------
    # Var 26 – Mención de 'depósito' / 'cuarto útil'
    #
    # El depósito es un espacio de almacenamiento adicional, valorado en apartamentos
    # compactos de Bogotá. Frecuentemente se omite en campos estructurados de los
    # portales, por lo que debe extraerse de texto.
    #
    # Variaciones incluidas:
    #   - "depsito": tipografía frecuente en el corpus (~1,721 ocurrencias).
    #     Ocurre porque los anunciantes escriben rápido omitiendo la vocal 'o'
    #     → d-e-p-s-i-t-o en lugar de d-e-p-o-s-i-t-o.
    #     La normalización de tildes NO corrige esto; hay que capturarlo explícitamente.
    #   - bodega: sinónimo en uso inmobiliario colombiano
    #   - cajón: espacio de almacenamiento de bicicletas o cosas pequeñas
    # -------------------------------------------------------------------------
    "deposito": [
        r"\bdeposito[s]?\b",           # depósito / depósitos (normalizado: sin tilde)
        r"\bdepsito[s]?\b",            # ERROR TIPOGRÁFICO MUY FRECUENTE (~1,721 casos)
                                       # la 'o' cae entre 'p' y 's': d-e-p-[o]-s-i-t-o
        r"\bcuarto util\b",            # cuarto útil (normalizado sin tilde)
        r"\bcuartos utiles\b",         # plural
        r"\bcuarto de deposito\b",     # frase descriptiva completa
        r"\bbodega[s]?\b",             # bodega / bodegas (sinónimo en Colombia)
        r"\balmacenamiento\b",         # zona de almacenamiento
        r"\bcajon(?:es)?\b",           # cajón / cajones (espacio pequeño de almacenaje)
        r"\bunidad de almacen\b",      # "unidad de almacenamiento"
    ],

    # -------------------------------------------------------------------------
    # Var 30 – Mención de 'conjunto cerrado' / 'portería'
    #
    # En Bogotá, vivir en un "conjunto cerrado" implica seguridad privada, portería
    # y acceso controlado, que se capitalizan en precio (Morales Zurita & Arias, 2005;
    # Gaviria et al., 2010 — disposición a pagar por seguridad en estratos 5-6).
    #
    # Variaciones incluidas:
    #   - Las "zonas comunes" y "salón comunal" son indicadores de conjunto;
    #     un inmueble independiente no tiene estas características.
    #   - "citófono": interfono/citofón de acceso → característica de conjuntos cerrados.
    #     Aparece ~271 veces en el corpus.
    #   - "cuota de administración": el pago de admin es exclusivo de PHs y conjuntos.
    # -------------------------------------------------------------------------
    "conjunto_cerrado": [
        r"\bconjunto cerrado\b",       # frase canónica en Colombia
        r"\bconjunto residencial\b",   # variante formal
        r"\bconjunto privado\b",       # énfasis en privacidad/exclusividad
        r"\bconjunto\b",               # mención simple (muy frecuente: ~6,905)
        r"\bporteria\b",               # portería (sin tilde, normalizado)
        r"\bportero\b",                # "cuenta con portero"
        r"\bvigilancia\b",             # vigilancia (privada o 24h)
        r"\bseguridad privada\b",      # explícita en muchos listados
        r"\bacceso controlado\b",      # frase alternativa
        r"\bunidad cerrada\b",         # variante de conjunto cerrado
        r"\burbanizacion\b",           # urbanización: conjunto de casas o aptos
        r"\bcopropiedad\b",            # régimen legal de conjuntos en Colombia
        r"\bzonas comunes\b",          # áreas compartidas → indica conjunto
        r"\bareas comunes\b",          # variante
        r"\bsalon comunal\b",          # salón comunal (~3,781 casos) → indicador fuerte
        r"\badministracion\b",         # cuota de administración → indica conjunto
        r"\bcuota de administracion\b",# frase específica
        r"\bcitofo(?:no|n)\b",         # citófono / citofón (~271 casos) → acceso controlado
    ],

    # -------------------------------------------------------------------------
    # Var 31 – Mención de 'balcón' / 'terraza'
    #
    # Espacios exteriores privados valorados por luz, ventilación y calidad de vida.
    # Su presencia frecuentemente no aparece en campos estructurados de los portales.
    # (Sirmans et al., 2005 — valorado consistentemente en modelos hedónicos.)
    #
    # Variaciones incluidas:
    #   - zona BBQ: espacio exterior para parrilla, muy frecuente en listados
    #     bogotanos (~6,596 ocurrencias de "bbq" en el corpus).
    #   - logia/loggia: espacio exterior cubierto integrado al apartamento.
    #   - azotea: terraza en la parte superior del edificio (cubierta plana).
    #   - solarium: terraza con exposición solar (más en segmento premium).
    #   - deck: plataforma exterior de madera (anglicismo frecuente).
    # -------------------------------------------------------------------------
    "balcon_terraza": [
        r"\bbalcon(?:es)?\b",          # balcón / balcones (grupo no capturante para evitar warning)
        r"\bterraza[s]?\b",            # terraza / terrazas
        r"\bterrraza[s]?\b",           # ERROR TIPOGRÁFICO: triple 'r' (~6 casos)
        r"\bpatio[s]?\b",              # patio / patios (casas y aptos de planta baja)
        r"\bjardin(?:es)?\b",          # jardín / jardines
        r"\bazotea[s]?\b",             # azotea (~24 casos) → terraza en techo plano
        r"\bsolarium\b",               # solarium (~36 casos) → terraza con sol
        r"\bdeck\b",                   # deck (~30 casos) → plataforma exterior de madera
        r"\blogia[s]?\b",              # logia / logias (~91 casos) → espacio cubierto exterior
        r"\bzona bbq\b",               # zona BBQ (~1,272 casos) → área exterior de parrilla
        r"\bbbq\b",                    # BBQ solo (~6,596 casos) — altísima frecuencia
        r"\bzona verde[s]?\b",         # zona verde / zonas verdes (jardines del conjunto)
    ],
}

# =============================================================================
# TÉRMINOS PREMIUM PARA TF-IDF
# =============================================================================
#
# Lista de términos y frases que señalizan calidad o lujo en listados inmobiliarios.
# El modelo TF-IDF cuantifica cuántos de estos términos aparecen en cada anuncio
# y con qué peso (ponderando por frecuencia inversa en el corpus completo).
#
# Un score TF-IDF alto indica que el anuncio usa muchos términos de lujo que
# son relativamente raros en el corpus → señal de propiedad premium.
# (Ruscheinsky, Lang & Maier, 2017 — bag-of-words captura diferenciación de producto.)
#
# Ampliaciones respecto a la versión inicial:
#   - duplex / loft: tipologías premium frecuentes en Chapinero (~2,681 / ~262)
#   - cocina integral: muy frecuente en aptos de calidad media-alta (~10,649)
#   - ascensor: edificios con ascensor tienen prima de precio (~6,924 menciones)
#   - bbq / zona bbq: amenidad exterior de conjunto premium (~6,596)
#   - cuarto de servicio: habitación para servicio doméstico → estrato alto
#   - gimnasio: amenidad de conjunto; alta correlación con precio (~8,301)
# =============================================================================

TERMINOS_PREMIUM = [
    # Tipologías premium
    "penthouse", "pent house", "ph",
    "duplex",                          # dúplex: dos pisos → más espacio, más valor
    "loft",                            # loft: espacio abierto industrial, segmento joven premium
    "club house", "clubhouse",

    # Adjetivos de lujo
    "acabados de lujo", "lujo", "lujoso", "lujosa",
    "premium", "exclusivo", "exclusiva",
    "de primera",                      # "acabados de primera"

    # Materiales premium
    "marmol", "granito", "porcelanato", "madera",

    # Amenidades de agua y bienestar
    "piscina", "jacuzzi", "sauna", "spa", "turco",

    # Amenidades deportivas y sociales
    "gimnasio",                        # muy frecuente y discriminante (~8,301)
    "salon comunal",                   # salón de eventos → conjunto premium
    "cancha",                          # cancha deportiva en el conjunto
    "zona bbq", "bbq",                 # zona de parrilla → espacio exterior premium
    "squash", "tenis",

    # Tecnología y servicios del edificio
    "ascensor",                        # ascensor → edificio con amenidades básicas (~6,924)
    "porteria 24",                     # portería 24 horas → seguridad premium
    "vigilancia 24",
    "citofono",                        # citófono → acceso moderno controlado
    "planta electrica",                # planta eléctrica de emergencia

    # Cocina y acabados interiores
    "cocina integral",                 # cocina con mobiliario integrado (~10,649)
    "cocina americana",                # cocina abierta al living → diseño moderno
    "cocina tipo americana",
    "closets",                         # "amplios closets" → espacio premium
    "cuarto de servicio",              # habitación de servicio → estrato 5-6
    "cuarto servicio",

    # Espacios especiales
    "doble altura",                    # sala con altura doble → diseño arquitectónico
    "vista 360", "vista panoramica",
    "estudio",                         # cuarto adicional de trabajo/estudio

    # Frases de marketing premium (adicionales, sin duplicar las de arriba)
    "acabados en marmol",
]

# =============================================================================
# DICCIONARIO DE AMENIDADES
# =============================================================================
#
# Cada clave es el nombre interno de una amenidad; el valor es la lista de
# patrones regex que la identifican en el texto normalizado (sin tildes).
#
# Estructura en tres categorías, alineadas con la clasificación de
# variables_hedónicas_Chapinero_v2.xlsx:
#
#   1. Agua y bienestar : amenidades de relajación y uso acuático
#   2. Deportivas        : instalaciones para actividad física
#   3. Sociales          : espacios de reunión y entretenimiento colectivo
#
# Estas tres categorías se usan en:
#   - `gimnasio`      : dummy específico para esa amenidad
#   - `amenidades`    : dummy (0/1) — ¿tiene AL MENOS UNA amenidad?
#   - `num_amenidades`: entero — número de amenidades DISTINTAS presentes
#
# Frecuencias en el corpus de train (referencia para priorización):
#   gimnasio ~8,059  |  bbq ~6,552  |  ascensor ~5,846  |  piscina ~2,209
#   sauna ~1,925  |  squash ~1,836  |  club house ~1,580  |  turco ~1,189
#   salon de juegos ~1,113  |  cancha ~2,720  |  jacuzzi ~1,041
# =============================================================================

AMENIDADES_DICT = {

    # ── Agua y bienestar ──────────────────────────────────────────────────────
    # Amenidades relacionadas con agua, relajación y salud física.
    # Suelen aparecer en conjuntos de estrato 4-6 en Chapinero.

    "piscina": [
        r"\bpiscina[s]?\b",        # piscina / piscinas (2,209 en train)
        r"\balberca[s]?\b",        # alberca: sinónimo regional (más México/Costa)
    ],
    "jacuzzi": [
        r"\bjacuzzi\b",            # jacuzzi (1,041)
        r"\bburbujas\b",           # "tina de burbujas" = jacuzzi informal
        r"\btina[s]?\b",           # tina de hidroterapia
    ],
    "sauna": [
        r"\bsauna\b",              # sauna (1,925)
    ],
    "turco": [
        r"\bturco\b",              # baño turco / turco (1,189)
        r"\bturca\b",
        r"\bbano turco\b",         # frase completa normalizada
    ],
    "spa": [
        r"\bspa\b",                # spa (157) — menos frecuente, más exclusivo
        r"\bzona humeda\b",        # zona húmeda: área con sauna+turco+jacuzzi (149)
        r"\bzona spa\b",
    ],

    # ── Deportivas ────────────────────────────────────────────────────────────
    # Instalaciones para actividad física dentro del conjunto o edificio.

    "gimnasio": [
        r"\bgimnasio\b",           # gimnasio (8,059) — la más frecuente del grupo
        r"\bgym\b",                # anglicismo ocasional
        r"\bfitness\b",            # "sala fitness"
        r"\bsala de ejercicio\b",  # denominación alternativa
        r"\bsala de gimnasio\b",
    ],
    "cancha": [
        r"\bcancha[s]?\b",         # cancha deportiva (2,720)
        r"\bcancha multiple\b",    # cancha múltiple (microfútbol + baloncesto)
        r"\bcancha deportiva\b",
        r"\bpolifuncional\b",      # cancha polifuncional
    ],
    "squash": [
        r"\bsquash\b",             # squash (1,836) — frecuente en Chapinero alto
        r"\bracquetball\b",        # variante anglicizada
    ],
    "tenis": [
        r"\bcanchas? de tenis\b",  # cancha de tenis — más específico que solo "tenis"
        r"\btenis\b",              # (654) — puede referirse al deporte o al club
    ],
    "zona_trote": [
        r"\bzona de trote\b",      # zona de trote (28) — poca frecuencia
        r"\bpista de trote\b",
        r"\bpista[s]? de correr\b",
    ],

    # ── Sociales y de entretenimiento ─────────────────────────────────────────
    # Espacios de uso colectivo para reuniones, eventos y recreación.

    "salon_comunal": [
        r"\bsalon comunal\b",      # salón comunal (3,770) — muy frecuente
        r"\bsalon de eventos\b",
        r"\bsalon social\b",
        r"\bsalon de fiestas\b",
    ],
    "club_house": [
        r"\bclub house\b",         # club house (1,580)
        r"\bclubhouse\b",
        r"\bclub de residentes\b",
    ],
    "bbq": [
        r"\bzona bbq\b",           # zona BBQ (1,269) — más específico, menos ruido
        r"\bbbq\b",                # BBQ solo (6,552) — muy frecuente en Bogotá
        r"\bparrilla[s]?\b",       # parrilla: sinónimo colombiano de BBQ
        r"\basador\b",             # asador: sinónimo menos común
    ],
    "salon_juegos": [
        r"\bsalon de juegos\b",    # salón de juegos (1,113)
        r"\bsala de juegos\b",     # (214)
        r"\bsalon de juego\b",
    ],
    "juegos_infantiles": [
        r"\bjuegos infantiles\b",  # juegos infantiles (682)
        r"\bparque infantil\b",
        r"\bzona de juegos\b",
        r"\bjuegos de ninos\b",    # normalizado sin tilde
    ],
    "coworking": [
        r"\bcoworking\b",          # coworking (47) — emergente en conjuntos nuevos
        r"\boficina virtual\b",
        r"\bsala de trabajo\b",
    ],

    # ── Infraestructura premium ───────────────────────────────────────────────
    # Servicios del edificio que representan un estándar de calidad.
    # No son "amenidades recreativas" pero son valorados como diferenciadores.

    "ascensor": [
        r"\bascensor(?:es)?\b",    # ascensor / ascensores (5,846) — alta frecuencia
    ],
}

# ── Lista de claves de amenidades para acceso programático ───────────────────
# Se usa en calcular_amenidades() para iterar sobre el diccionario de forma
# ordenada y coherente con las categorías definidas arriba.
NOMBRES_AMENIDADES = list(AMENIDADES_DICT.keys())

# =============================================================================
# LÉXICO DE SENTIMIENTO EN ESPAÑOL
# =============================================================================
#
# Se construye un léxico simple de palabras positivas y negativas para
# calcular la polaridad del texto de cada anuncio.
#
# Metodología: cada palabra del texto se compara contra el léxico.
# Score = (nro_palabras_positivas − nro_palabras_negativas) / (total_relevantes + 1)
# El denominador +1 evita división por cero cuando no hay palabras del léxico.
# Rango aproximado: −1 (muy negativo) a +1 (muy positivo).
#
# Limitación: los léxicos de dominio general pueden no capturar bien el
# sentimiento específico del mercado inmobiliario. Como alternativa robusta
# se podría usar SpaCy con un modelo en español + adaptación de dominio,
# pero esto requiere instalación adicional y es más lento.
#
# Expansiones respecto a la versión inicial:
#   - Se añaden más adjetivos típicos de anuncios inmobiliarios bogotanos
#   - Se añaden palabras negativas relacionadas con necesidad de reparación
# =============================================================================

POSITIVAS = {
    # Calificativos generales de calidad
    "excelente", "espectacular", "increible", "magnifico", "magnificа",
    "maravilloso", "maravillosa", "fantastico", "fantastica",
    "extraordinario", "extraordinaria",

    # Apariencia y estética
    "hermoso", "hermosa", "bonito", "bonita", "lindo", "linda",
    "precioso", "preciosa", "elegante", "distinguido", "distinguida",
    "disenado", "disenada",             # diseñado con cuidado

    # Luz y espacio
    "luminoso", "luminosa", "amplio", "amplia", "espacioso", "espaciosa",
    "soleado", "soleada", "iluminado", "iluminada",
    "ventilado", "ventilada", "aireado", "aireada",

    # Modernidad y estado
    "moderno", "moderna", "nuevo", "nueva", "flamante",
    "renovado", "renovada", "remodelado", "remodelada",
    "actualizado", "actualizada",

    # Ubicación y entorno
    "privilegiado", "privilegiada", "exclusivo", "exclusiva",
    "tranquilo", "tranquila", "seguro", "segura",
    "bien ubicado", "bien ubicada",
    "estrategico", "estrategica",

    # Comodidad y funcionalidad
    "comodo", "comoda", "acogedor", "acogedora",
    "funcional", "practico", "practica",
    "perfecto", "perfecta", "ideal", "optimo", "optima",
    "inigualable", "incomparable",

    # Premium / lujo
    "lujoso", "lujosa", "premium", "exquisito", "exquisita",
    "unico", "unica", "especial",

    # Adjetivos frecuentes en listings colombianos
    "agradable", "acogedor", "acogedora", "impecable",
}

NEGATIVAS = {
    # Estado deteriorado
    "deteriorado", "deteriorada", "descuidado", "descuidada",
    "abandonado", "abandonada", "viejo", "vieja",

    # Necesidad de intervención
    "necesita", "requiere", "para remodelar", "sin remodelar",
    "arreglar", "reparar", "arreglado", "reparado",   # sólo si es imperativo
    "problema", "problemas", "falla", "fallas",
    "averia", "averiado", "averiada",
    "goteras", "grietas", "humedad",

    # Ruido y contaminación
    "ruidoso", "ruidosa", "contaminado", "contaminada",

    # Tamaño reducido
    "pequeno", "pequena", "reducido", "reducida", "diminuto", "diminuta",

    # Oscuro / sin luz
    "oscuro", "oscura", "humedo", "humeda", "sombrio", "sombria",

    # Antigüedad negativa
    "anticuado", "anticuada", "obsoleto", "obsoleta",
    "original", "sin mejorar",   # "en estado original" = sin intervención

    # Urgencia de venta (señal de precio bajo)
    "urgente", "rematar", "remato", "remate",
    "liquidar", "liquidacion",

    # Expresiones de carencia
    "sin garaje", "sin parqueadero", "sin deposito", "sin ascensor",
    "dano", "danos",                # daño / daños
}


def calcular_sentimiento(texto: pd.Series) -> pd.Series:
    """
    Calcula el score de sentimiento para cada texto en la Serie.

    Proceso:
      1. Se separa el texto en palabras individuales (split por espacios).
      2. Se cuenta cuántas caen en el léxico POSITIVAS y cuántas en NEGATIVAS.
      3. Score = (pos − neg) / (pos + neg + 1)
         El +1 en el denominador sirve como suavizante para evitar división por
         cero cuando no hay palabras del léxico en el texto (el score resulta 0).

    Limitación del léxico de palabras simples: no detecta negaciones
    ("no es luminoso" se contaría como positivo). Para un modelo de producción
    se recomendaría SpaCy con análisis de dependencias.

    Retorna:
        Serie de floats en rango ≈ [−1, 1], donde:
          > 0 → texto predominantemente positivo
          = 0 → neutro o sin palabras del léxico
          < 0 → texto predominantemente negativo
    """
    def _score(txt: str) -> float:
        palabras = txt.split()
        pos = sum(1 for p in palabras if p in POSITIVAS)
        neg = sum(1 for p in palabras if p in NEGATIVAS)
        return (pos - neg) / (pos + neg + 1)

    return texto.apply(_score)


# =============================================================================
# EXTRACCIÓN DE PARQUEADEROS DESDE TEXTO
# =============================================================================

def extraer_parqueaderos(texto: pd.Series) -> pd.Series:
    """
    Extrae el número de garajes o parqueaderos mencionados en cada texto.

    El dataset no tiene un campo estructurado para parqueaderos, por lo que
    esta información debe inferirse del texto libre del anuncio.

    Estrategia de extracción (por orden de prioridad):
      1. Número arábigo seguido de keyword: "2 garajes", "1 parqueadero"
         → el número debe ser 1-2 dígitos y ≤ 10 para evitar capturar números
           de referencia de propiedades (e.g., "cod 2901 garaje" daría 2901).
         → se exige un ESPACIO entre el número y la keyword para separar
           patrones como "1parqueadero" (que no existe) de códigos pegados.
      2. Número en letras: "doble garaje", "un parqueadero", "dos garajes"
      3. Mención simple sin cantidad: se asume 1 garaje.
      4. Sin mención: retorna np.nan (distinguir "no mencionado" de "cero").

    Keywords cubiertos: garaje, parqueadero, garage (anglicismo), parking
    """
    # Mapa de palabras numéricas a enteros
    _letras = {
        "un": 1, "uno": 1, "una": 1,
        "dos": 2, "tres": 3, "cuatro": 4,
        "doble": 2,    # "doble garaje" = 2 parqueaderos
        "triple": 3,   # "triple garaje" = 3 parqueaderos
    }

    # Patrón de keywords de parqueadero (sin capturar para eficiencia)
    _kw = r"garaje[s]?|parqueadero[s]?|garage[s]?|parking|cochera[s]?|estacionamiento[s]?"

    def _extraer(txt: str):
        # ── Paso 1: número arábigo ────────────────────────────────────────────
        # Requiere espacio entre número y keyword (\s+) para evitar
        # capturar códigos internos como "X2901garaje" → sin espacio, no matchea
        m = re.search(rf"\b(\d{{1,2}})\s+(?:{_kw})", txt)
        if m:
            val = int(m.group(1))
            return val if val <= 10 else np.nan   # filtro anti-falsos-positivos

        # ── Paso 2: número en letras ──────────────────────────────────────────
        m2 = re.search(
            rf"\b(un[ao]?|dos|tres|cuatro|doble|triple)\s+(?:{_kw})", txt
        )
        if m2:
            return _letras.get(m2.group(1), np.nan)

        # ── Paso 3: mención simple sin cantidad ───────────────────────────────
        # e.g., "apartamento con garaje" → inferimos 1 parqueadero
        if re.search(rf"\b(?:{_kw})\b", txt):
            return 1

        # ── Sin mención ───────────────────────────────────────────────────────
        return np.nan

    return texto.apply(_extraer)


# =============================================================================
# EXTRACCIÓN DE PISO DESDE TEXTO
# =============================================================================

def extraer_piso(texto: pd.Series) -> pd.Series:
    """
    Extrae el número de piso del apartamento desde el texto del anuncio.

    Este campo no está disponible en los datos estructurados de Properati,
    pero aparece frecuentemente en la descripción. La extracción es importante
    porque los pisos altos tienen prima de precio por vistas y privacidad
    (Benson et al., 1998).

    Estrategia (por orden de prioridad):
      1. "piso N"  → número después de la palabra "piso": "piso 5", "piso 12"
      2. "N piso"  → número antes de "piso" con posibles sufijos ordinales:
                     "5to piso", "3er piso", "2do piso"
      3. "nivel N" → "nivel 3", "nivel 12" (alternativa a 'piso')
      4. Ordinales en texto: "primer piso", "quinto piso", etc.
      5. Sin mención: retorna np.nan

    Límite máximo (_MAX_PISO = 60):
      Bogotá tiene pocos edificios de más de 40 pisos. Valores superiores
      seguramente son números de referencia capturados por error
      (e.g., "apartamento 6012" → "piso 60" si el regex no está bien limitado).
    """
    # Piso máximo razonable para Bogotá (pocos edificios superan 40 pisos)
    _MAX_PISO = 60

    # Mapa de ordinales en texto a enteros
    _ord = {
        "primer": 1,  "primero": 1,
        "segundo": 2, "segunda": 2,
        "tercer": 3,  "tercero": 3,
        "cuarto": 4,
        "quinto": 5,
        "sexto": 6,
        "septimo": 7,
        "octavo": 8,
        "noveno": 9,
        "decimo": 10,
    }

    def _extraer(txt: str):
        # ── Paso 1: "piso N" ─────────────────────────────────────────────────
        # 1-2 dígitos después de "piso" con espacio obligatorio
        m = re.search(r"\bpiso\s+(\d{1,2})\b", txt)
        if m:
            val = int(m.group(1))
            return val if val <= _MAX_PISO else np.nan

        # ── Paso 2: "N piso" o "Nto piso" ───────────────────────────────────
        # Captura sufijos ordinales opcionales (er, do, to, vo, mo, °, º)
        m = re.search(r"\b(\d{1,2})\s*(?:er|do|to|vo|mo|[°º])?\s*piso\b", txt)
        if m:
            val = int(m.group(1))
            return val if val <= _MAX_PISO else np.nan

        # ── Paso 3: "nivel N" ────────────────────────────────────────────────
        m = re.search(r"\bnivel\s+(\d{1,2})\b", txt)
        if m:
            val = int(m.group(1))
            return val if val <= _MAX_PISO else np.nan

        # ── Paso 4: ordinales en texto ("tercer piso", "quinto piso") ────────
        m2 = re.search(
            r"\b(primer|primero|segundo|segunda|tercer|tercero|cuarto"
            r"|quinto|sexto|septimo|octavo|noveno|decimo)\s*piso\b",
            txt,
        )
        if m2:
            return _ord.get(m2.group(1), np.nan)

        # ── Sin mención ───────────────────────────────────────────────────────
        return np.nan

    return texto.apply(_extraer)


# =============================================================================
# EXTRACCIÓN DE BAÑOS DESDE TEXTO
# =============================================================================

def extraer_banos(texto: pd.Series) -> pd.Series:
    """
    Extrae el número de baños mencionado en el texto del anuncio.

    Contexto:
        El campo estructurado `bathrooms` del dataset tiene muchos valores
        faltantes. Esta función extrae el conteo desde el texto libre como
        variable complementaria / de respaldo.

    Nota sobre baños en listados colombianos:
        - "baño completo"  : baño con sanitario, lavamanos y ducha (baño entero).
        - "baño social"    : baño de visitas con sanitario y lavamanos (medio baño).
        - "baño auxiliar"  : igual que baño social; a veces en habitaciones.
        - Cuando el anuncio dice "3 baños 1 social", el total puede ser 3 o 4
          según la interpretación. Aquí extraemos el primer número mencionado
          antes de "baño(s)" como aproximación del total de baños completos.

    Estrategia (por orden de prioridad):
        1. Número arábigo seguido de "baño(s)":
               "3 baños" → 3,  "2 banos" → 2
        2. Número arábigo precede con sufijo ordinal:
               "banos: 2" → 2 (número DESPUÉS de la keyword)
        3. Número en letras antes de "baño(s)":
               "dos baños" → 2,  "tres banos" → 3
        4. Mención de "baño social" o "baño auxiliar" sin número explícito:
               → al menos 1 baño (retorna 1 como piso mínimo)
        5. Sin mención de baño alguno → np.nan

    Límite máximo:
        Se aplica un tope de 15 baños para filtrar falsos positivos
        (números grandes que el regex captura fuera de contexto).
    """
    _MAX_BANOS = 15

    # Mapa de números en letras a enteros (géneros masculino y femenino)
    _letras = {
        "un": 1, "uno": 1, "una": 1,
        "dos": 2, "tres": 3, "cuatro": 4,
        "cinco": 5, "seis": 6, "siete": 7,
    }

    def _extraer(txt: str):
        # ── Paso 1: "N baño(s)" — número arábigo ANTES de la keyword ─────────
        # Se exige espacio (\s+) entre número y keyword para no capturar
        # códigos de propiedad como "ref123banos".
        m = re.search(r"\b(\d{1,2})\s+bano[s]?\b", txt)
        if m:
            val = int(m.group(1))
            if val == 0:
                return np.nan          # "0 baños" no es informativo
            return val if val <= _MAX_BANOS else np.nan

        # ── Paso 2: "baño(s) N" — número arábigo DESPUÉS de la keyword ───────
        # Patrón: "baños: 2", "baños 2" (menos común pero presente)
        m = re.search(r"\bbano[s]?\s*[:\-]?\s*(\d{1,2})\b", txt)
        if m:
            val = int(m.group(1))
            return val if 1 <= val <= _MAX_BANOS else np.nan

        # ── Paso 3: número en letras ANTES de "baño(s)" ──────────────────────
        # "dos baños", "tres banos", "cuatro baños"
        m2 = re.search(
            r"\b(un[ao]?|dos|tres|cuatro|cinco|seis|siete)\s+bano[s]?\b", txt
        )
        if m2:
            return _letras.get(m2.group(1), np.nan)

        # ── Paso 4: mención de baño social/auxiliar sin número ────────────────
        # Si el anuncio menciona "baño social" o "baño auxiliar" sin indicar
        # la cantidad total, asumimos al menos 1 baño en el inmueble.
        if re.search(r"\bbano\b", txt):
            return 1

        # ── Sin mención ───────────────────────────────────────────────────────
        return np.nan

    return texto.apply(_extraer)


# =============================================================================
# CÁLCULO DE AMENIDADES
# =============================================================================

def calcular_amenidades(texto: pd.Series) -> pd.DataFrame:
    """
    Calcula tres variables de amenidades a partir de AMENIDADES_DICT:

      1. `gimnasio`      : dummy (0/1) — ¿menciona explícitamente un gimnasio?
                           Se extrae por separado porque es la amenidad más
                           frecuente (~8,059 menciones) y con mayor impacto
                           en precio en conjuntos de Chapinero.

      2. `amenidades`    : dummy (0/1) — ¿menciona AL MENOS UNA amenidad
                           de cualquier categoría (agua/bienestar, deportiva,
                           social, infraestructura)?

      3. `num_amenidades`: entero — número de TIPOS DE AMENIDAD distintos
                           presentes en el anuncio.
                           Cada clave de AMENIDADES_DICT cuenta como 1 si
                           alguno de sus patrones hace match, sin importar
                           cuántos patrones distintos de esa misma amenidad
                           aparezcan.
                           Ejemplos:
                             piscina + jacuzzi + gimnasio → num_amenidades = 3
                             solo bbq                     → num_amenidades = 1
                             ninguna                      → num_amenidades = 0

    Diseño:
        Se itera sobre AMENIDADES_DICT aplicando dummy_busqueda() a cada
        amenidad. Esto produce una matriz binaria (n_filas × n_amenidades)
        cuya suma por fila es `num_amenidades`. El resultado de gimnasio
        se extrae directamente de esa matriz.

    Parámetros:
        texto : Serie con texto combinado y normalizado (salida de combinar_texto).

    Retorna:
        DataFrame con tres columnas: 'gimnasio', 'amenidades', 'num_amenidades'.
    """
    # Construimos un DataFrame donde cada columna es una amenidad (0/1)
    # Usamos dict-comprehension para aplicar dummy_busqueda a cada amenidad
    dummies = pd.DataFrame(
        {nombre: dummy_busqueda(texto, patrones)
         for nombre, patrones in AMENIDADES_DICT.items()}
    )

    # `num_amenidades`: suma de las columnas binarias por fila
    # Cada amenidad con al menos un patrón activo aporta 1 al total
    num_amenidades = dummies.sum(axis=1).astype(int)

    # `amenidades`: 1 si hay al menos una amenidad presente, 0 si ninguna
    amenidades = (num_amenidades > 0).astype(int)

    # `gimnasio`: extraído directamente de la columna correspondiente
    # Se mantiene como variable separada por su relevancia individual
    gimnasio = dummies["gimnasio"]

    return pd.DataFrame({
        "gimnasio":       gimnasio,
        "amenidades":     amenidades,
        "num_amenidades": num_amenidades,
    }, index=texto.index)


# =============================================================================
# TF-IDF PREMIUM
# =============================================================================

def ajustar_tfidf(corpus_train: pd.Series) -> TfidfVectorizer:
    """
    Ajusta (fit) el vectorizador TF-IDF sobre el corpus de entrenamiento.

    Parámetros del vectorizador:
      - vocabulary : sólo se consideran los términos en TERMINOS_PREMIUM.
        Esto fuerza al modelo a calcular pesos SÓLO para nuestros términos
        de interés, ignorando el resto del vocabulario.
      - ngram_range=(1,3) : permite capturar frases de hasta 3 palabras,
        necesario para términos como "acabados de lujo" o "cocina integral".

    Por qué fit sólo sobre train:
      Si usáramos el corpus completo (train+test) para ajustar el IDF,
      estaríamos usando información del test para calcular los pesos →
      data leakage. El IDF del test debe estimarse como si fuera desconocido.

    Retorna:
        TfidfVectorizer ajustado, listo para transformar cualquier corpus.
    """
    # Normalizamos los términos premium por coherencia con el texto procesado
    vocab_norm = [normalizar(t) for t in TERMINOS_PREMIUM]

    tfidf = TfidfVectorizer(
        vocabulary=vocab_norm,
        ngram_range=(1, 3),
    )
    tfidf.fit(corpus_train)
    return tfidf


def calcular_tfidf(tfidf: TfidfVectorizer, corpus: pd.Series) -> np.ndarray:
    """
    Transforma el corpus usando el vectorizador ajustado y devuelve,
    para cada documento, la SUMA de los scores TF-IDF de todos los términos
    premium que aparecen en él.

    Interpretar el score:
      - Score = 0   → ningún término premium aparece en el anuncio.
      - Score > 0   → aparecen uno o más términos premium; mayor score significa
                      más términos y/o términos más raros en el corpus (mayor IDF).
      - Score alto  → anuncio con vocabulario de lujo concentrado → señal premium.
    """
    # transform() devuelve una matriz dispersa (n_docs × n_términos)
    matrix = tfidf.transform(corpus)

    # Sumamos por filas: cada fila es un documento, obtenemos un escalar por doc
    return np.asarray(matrix.sum(axis=1)).flatten()


# =============================================================================
# PIPELINE PRINCIPAL DE EXTRACCIÓN
# =============================================================================

def agregar_variables_texto(df: pd.DataFrame, tfidf: TfidfVectorizer) -> pd.DataFrame:
    """
    Aplica todas las extracciones de texto sobre el DataFrame y retorna
    una copia del mismo con las 14 columnas nuevas añadidas al final.

    El DataFrame resultante mantiene TODAS las columnas originales del CSV
    (property_id, city, price, surface_total, rooms, etc.) más las nuevas.
    Esto permite usar el archivo resultante directamente como input de modelos.

    Parámetros:
        df    : DataFrame original (train o test), con columnas title y description.
        tfidf : vectorizador TF-IDF ya ajustado sobre el corpus de train.
    """
    # Texto combinado y normalizado: base de todas las extracciones
    texto = combinar_texto(df)

    # Partimos de una copia del DataFrame original para no modificarlo
    out = df.copy()

    # ── 1. Dummies de mención ─────────────────────────────────────────────────
    # Para cada variable dummy, se aplican todos sus patrones regex en una sola
    # pasada con OR (|). El resultado es 1 si algún patrón hace match, 0 si no.
    for nombre, patrones in PATRONES.items():
        out[nombre] = dummy_busqueda(texto, patrones)

    # ── 2. Sentimiento del anuncio ────────────────────────────────────────────
    # Score de polaridad léxica: (positivas − negativas) / (total_relevantes + 1)
    out["sentimiento_score"] = calcular_sentimiento(texto)

    # ── 3. Longitud de la descripción ─────────────────────────────────────────
    # Se mide ANTES de normalizar, sobre el campo original, para preservar la
    # longitud real del texto tal como lo escribió el anunciante.
    # Anuncios más largos tienden a correlacionar con precios más altos
    # (proxy de esfuerzo del vendedor — Levmovitz & Munneke, 2020).
    out["longitud_desc"] = df["description"].fillna("").str.len()

    # ── 4. Score TF-IDF de términos premium ───────────────────────────────────
    # Se usa el vectorizador ajustado sobre train para transformar el corpus
    # actual (puede ser train o test). La suma por filas da un escalar por doc.
    out["tfidf_premium"] = calcular_tfidf(tfidf, texto)

    # ── 5. Parqueaderos extraídos del texto ───────────────────────────────────
    # np.nan cuando no hay mención → permite distinguir "sin garaje" de "no mencionado"
    out["parqueaderos_txt"] = extraer_parqueaderos(texto)

    # ── 6. Piso extraído del texto ────────────────────────────────────────────
    # np.nan cuando no hay mención → el imputador del modelo puede manejarlo
    out["piso_txt"] = extraer_piso(texto)

    # ── 7. Baños extraídos del texto ──────────────────────────────────────────
    # Complementa el campo estructurado `bathrooms` del CSV, que tiene NAs.
    # np.nan cuando no hay mención del número de baños en el texto.
    out["banos_txt"] = extraer_banos(texto)

    # ── 8. Amenidades ─────────────────────────────────────────────────────────
    # `gimnasio`      : dummy — menciona gimnasio/gym/fitness específicamente
    # `amenidades`    : dummy — menciona al menos una amenidad de cualquier tipo
    # `num_amenidades`: entero — cuántos tipos distintos de amenidad se mencionan
    # Las tres columnas se calculan en una sola pasada sobre AMENIDADES_DICT.
    amen = calcular_amenidades(texto)
    out["gimnasio"]       = amen["gimnasio"]
    out["amenidades"]     = amen["amenidades"]
    out["num_amenidades"] = amen["num_amenidades"]

    return out


# =============================================================================
# RESUMEN ESTADÍSTICO EN CONSOLA
# =============================================================================

def resumen(df: pd.DataFrame, nombre: str) -> None:
    """
    Imprime en consola un resumen de las variables de texto extraídas.
    Útil para verificar que las tasas de mención son razonables y
    detectar posibles problemas (e.g., tasa de 0% = ningún patrón funcionó,
    tasa de 99% = patrón demasiado amplio).
    """
    print(f"\n{'='*60}")
    print(f"  {nombre}  — {len(df):,} filas")
    print(f"{'='*60}")

    # Dummies de mención de características del inmueble
    cols_dummy_prop = ["remodelado", "vista_panoramica", "deposito",
                       "conjunto_cerrado", "balcon_terraza"]
    print("\nCaracterísticas del inmueble — tasa de mención:")
    for c in cols_dummy_prop:
        tasa = df[c].mean() * 100
        print(f"  {c:<22}: {tasa:5.1f}%")

    # Dummies de amenidades
    cols_dummy_amen = ["gimnasio", "amenidades"]
    print("\nAmenidades — tasa de mención:")
    for c in cols_dummy_amen:
        tasa = df[c].mean() * 100
        print(f"  {c:<22}: {tasa:5.1f}%")

    # Numéricas: estadísticas descriptivas básicas
    cols_num = ["sentimiento_score", "longitud_desc", "tfidf_premium",
                "parqueaderos_txt", "piso_txt", "banos_txt", "num_amenidades"]
    print("\nVariables numéricas — estadísticas básicas:")
    print(df[cols_num].describe().round(3).to_string())


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """
    Flujo principal:
      1. Cargar train.csv y test.csv desde 00_data/raw/
      2. Ajustar TF-IDF sobre corpus de train (SOLO train para evitar leakage)
      3. Extraer variables de texto sobre train y test por separado
      4. Guardar los DataFrames enriquecidos en 00_data/processed/
    """
    print("Cargando train.csv y test.csv...")
    train = pd.read_csv(TRAIN)
    test  = pd.read_csv(TEST)

    # El TF-IDF se ajusta sobre el corpus de entrenamiento solamente.
    # Usar train+test para ajustar el IDF sería data leakage.
    print("Ajustando TF-IDF sobre corpus de train (sin test)...")
    corpus_train = combinar_texto(train)
    tfidf = ajustar_tfidf(corpus_train)

    print("Extrayendo variables de texto — TRAIN...")
    train_out = agregar_variables_texto(train, tfidf)
    resumen(train_out, "TRAIN")

    print("\nExtrayendo variables de texto — TEST...")
    test_out = agregar_variables_texto(test, tfidf)
    resumen(test_out, "TEST")

    # Guardar resultados: versión enriquecida de cada conjunto
    out_train = PROCESSED / "train_texto.csv"
    out_test  = PROCESSED / "test_texto.csv"
    train_out.to_csv(out_train, index=False)
    test_out.to_csv(out_test,  index=False)

    print(f"\nArchivos guardados en 00_data/processed/:")
    print(f"  train_texto.csv  ({len(train_out):,} filas, {len(train_out.columns)} columnas)")
    print(f"  test_texto.csv   ({len(test_out):,} filas, {len(test_out.columns)} columnas)")


if __name__ == "__main__":
    main()
