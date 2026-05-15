#!/usr/bin/env python3.11
"""
Script: 02_add_osm_features.py
=================================
Agrega variables de localización y accesibilidad desde OpenStreetMap (OSM)
a las bases procesadas, alineado con variables_hedónicas_Chapinero_v2.xlsx.

Lógica central
--------------
1. BigQuery descarga POIs de Bogotá (puntos + polígonos) → poi_bogota.csv
2. BigQuery descarga segmentos viales de Bogotá → roads_bogota.csv
3. Se construyen KDTrees en memoria a partir de ambos CSVs.
4. Para cada vivienda (lat/lon del dataset) se consulta el KDTree local.
   BigQuery no interviene después de los pasos 1 y 2.

Nota: se eliminó osmnx porque usa Overpass API que está bloqueada en esta red.
      Toda la descarga de datos OSM ocurre vía BigQuery (cloud.google.com).

Costo estimado BigQuery:
  Query 1 (POIs):      ~47 GB  →  ~$0.29 USD
  Query 2 (Vías):      ~22 GB  →  ~$0.14 USD
  Total primera vez:   ~69 GB  →  ~$0.43 USD  (o $0 dentro del free tier)
  Ejecuciones futuras: $0 (ambas queries cacheadas en CSV local)

Variables producidas:
  Var  8  dist_cbd_km          — Distancia al CBD (Plaza de Bolívar) [km]
  Var  9  dist_transmilenio_m  — Distancia a portal/estación TM más cercana [m]
  Var 10  dist_via_arterial_m  — Distancia a segmento arterial más cercano [m]
  Var 11  dist_hospital_m      — Distancia a hospital o clínica [m]
  Var 13  dist_centro_com_m    — Distancia a centro comercial [m]
  Var 14  dist_parque_m        — Distancia a parque o área verde [m]
  Var 15  n_restaurantes_500m  — Restaurantes y cafés en radio 500 m
  Var 18  n_bancos_500m        — Bancos y cajeros en radio 500 m
  Var 16  walkability_score    — Categorías de servicio accesibles en 800 m (0–100)
  Var 17  densidad_vial        — Segmentos viales por km² en radio 500 m

Cache local (00_data/raw/osm/):
  poi_bogota.csv   — todos los POIs de Bogotá (~23K filas, ~1.2 MB)
  roads_bogota.csv — centroides de segmentos viales de Bogotá

Input:  00_data/processed/train_texto.csv  /  test_texto.csv
Output: 00_data/processed/train_osm.csv   /  test_osm.csv

Ejecución:
  python3.11 01_scripts/preparation/02_add_osm_features.py
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================

import os
import time
import warnings
from math import cos, pi, radians
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from scipy.spatial import KDTree

warnings.filterwarnings("ignore")

# =============================================================================
# RUTAS Y CONSTANTES
# =============================================================================

BASE      = Path(__file__).parent.parent.parent   # raíz del proyecto
PROCESSED = BASE / "00_data" / "processed"
OSM_CACHE = BASE / "00_data" / "raw" / "osm"
OSM_CACHE.mkdir(parents=True, exist_ok=True)

# Ruta al service account de Google Cloud.
# Se lee de la variable de entorno estándar GOOGLE_APPLICATION_CREDENTIALS.
# Para configurarla antes de correr el script:
#   export GOOGLE_APPLICATION_CREDENTIALS="/ruta/a/tu/credentials.json"
_creds_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
if not _creds_env:
    raise EnvironmentError(
        "Define la variable de entorno GOOGLE_APPLICATION_CREDENTIALS con la ruta "
        "a tu archivo de credenciales de Google Cloud.\n"
        "Ejemplo:\n"
        "  export GOOGLE_APPLICATION_CREDENTIALS='/ruta/al/archivo.json'"
    )
CREDS_FILE = Path(_creds_env)

# Bounding box Bogotá D.C. (WGS84) — polígono en orden (lon, lat) para BigQuery
BBOX = dict(south=4.45, north=4.85, west=-74.25, east=-73.95)
BBOX_WKT = (
    f"POLYGON(({BBOX['west']} {BBOX['south']}, "
    f"{BBOX['east']} {BBOX['south']}, "
    f"{BBOX['east']} {BBOX['north']}, "
    f"{BBOX['west']} {BBOX['north']}, "
    f"{BBOX['west']} {BBOX['south']}))"
)

# CBD de Bogotá: Plaza de Bolívar
CBD_LAT, CBD_LON = 4.5981, -74.0758

# Conversión grados → metros (lat de referencia ≈ 4.65°N, error < 0.05%)
LAT0          = (BBOX["south"] + BBOX["north"]) / 2
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * cos(radians(LAT0))

# Tags de highway considerados "arteriales" (tráfico intenso, ruido)
HIGHWAY_ARTERIAL = frozenset([
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link",
])

# =============================================================================
# CLIENTE BIGQUERY
# =============================================================================

def crear_cliente_bq() -> bigquery.Client:
    """Autentica con service account y retorna el cliente BigQuery."""
    creds = service_account.Credentials.from_service_account_file(
        str(CREDS_FILE),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(credentials=creds, project=creds.project_id)

# =============================================================================
# CONSULTAS BIGQUERY
# =============================================================================

# ── Query 1: POIs (puntos + centroides de polígonos) ──────────────────────────
# Escanea ~47 GB total (planet_features_points + planet_features_multipolygons)
QUERY_POIS = f"""
SELECT 'point' AS fuente,
  ST_Y(geometry) AS lat, ST_X(geometry) AS lon,
  (SELECT value FROM UNNEST(all_tags) WHERE key='amenity'          LIMIT 1) AS amenity,
  (SELECT value FROM UNNEST(all_tags) WHERE key='shop'             LIMIT 1) AS shop,
  (SELECT value FROM UNNEST(all_tags) WHERE key='network'          LIMIT 1) AS network,
  (SELECT value FROM UNNEST(all_tags) WHERE key='operator'         LIMIT 1) AS operator,
  (SELECT value FROM UNNEST(all_tags) WHERE key='public_transport' LIMIT 1) AS public_transport,
  CAST(NULL AS STRING) AS leisure,
  CAST(NULL AS STRING) AS landuse
FROM `bigquery-public-data.geo_openstreetmap.planet_features_points`
WHERE ST_WITHIN(geometry, ST_GEOGFROMTEXT('{BBOX_WKT}'))
  AND (
    EXISTS(SELECT 1 FROM UNNEST(all_tags) WHERE key='amenity'
           AND value IN ('bank','atm','bureau_de_change',
                         'hospital','clinic','pharmacy','doctors',
                         'restaurant','cafe','fast_food','food_court',
                         'bar','pub','bakery','bus_station','marketplace',
                         'school','university','college'))
    OR EXISTS(SELECT 1 FROM UNNEST(all_tags) WHERE key='shop'
              AND value IN ('mall','supermarket','grocery','convenience'))
    OR EXISTS(SELECT 1 FROM UNNEST(all_tags) WHERE key='network'
              AND LOWER(value) LIKE '%transmilenio%')
    OR EXISTS(SELECT 1 FROM UNNEST(all_tags) WHERE key='operator'
              AND LOWER(value) LIKE '%transmilenio%')
  )
UNION ALL
SELECT 'polygon' AS fuente,
  ST_Y(ST_CENTROID(geometry)) AS lat, ST_X(ST_CENTROID(geometry)) AS lon,
  (SELECT value FROM UNNEST(all_tags) WHERE key='amenity'  LIMIT 1) AS amenity,
  (SELECT value FROM UNNEST(all_tags) WHERE key='shop'     LIMIT 1) AS shop,
  CAST(NULL AS STRING) AS network, CAST(NULL AS STRING) AS operator,
  CAST(NULL AS STRING) AS public_transport,
  (SELECT value FROM UNNEST(all_tags) WHERE key='leisure'  LIMIT 1) AS leisure,
  (SELECT value FROM UNNEST(all_tags) WHERE key='landuse'  LIMIT 1) AS landuse
FROM `bigquery-public-data.geo_openstreetmap.planet_features_multipolygons`
WHERE ST_WITHIN(ST_CENTROID(geometry), ST_GEOGFROMTEXT('{BBOX_WKT}'))
  AND (
    EXISTS(SELECT 1 FROM UNNEST(all_tags) WHERE key='leisure'
           AND value IN ('park','garden','playground',
                         'recreation_ground','nature_reserve','pitch'))
    OR EXISTS(SELECT 1 FROM UNNEST(all_tags) WHERE key='landuse'
              AND value IN ('recreation_ground','grass','forest',
                            'village_green','greenfield'))
    OR EXISTS(SELECT 1 FROM UNNEST(all_tags) WHERE key='shop' AND value='mall')
    OR EXISTS(SELECT 1 FROM UNNEST(all_tags) WHERE key='amenity'
              AND value IN ('bus_station','marketplace','hospital','clinic'))
  )
"""

# ── Query 2: Red vial — centroides de segmentos de carretera ─────────────────
# Escanea ~22 GB (planet_features_lines).
# Se usa ST_CENTROID de cada segmento vial como punto de referencia:
#   dist_via_arterial_m : centroides de vías arteriales (trunk, primary, motorway)
#   densidad_vial       : conteo de todos los segmentos dentro de radio 500 m
QUERY_ROADS = f"""
SELECT
  ST_Y(ST_CENTROID(geometry))  AS lat,
  ST_X(ST_CENTROID(geometry))  AS lon,
  (SELECT value FROM UNNEST(all_tags) WHERE key = 'highway' LIMIT 1) AS highway
FROM `bigquery-public-data.geo_openstreetmap.planet_features_lines`
WHERE ST_WITHIN(ST_CENTROID(geometry), ST_GEOGFROMTEXT('{BBOX_WKT}'))
  AND EXISTS(
    SELECT 1 FROM UNNEST(all_tags)
    WHERE key = 'highway'
      AND value IN (
        'motorway','motorway_link','trunk','trunk_link',
        'primary','primary_link','secondary','secondary_link',
        'tertiary','tertiary_link','residential','living_street',
        'unclassified','service'
      )
  )
"""

# =============================================================================
# DESCARGA CON CACHE
# =============================================================================

def descargar_con_cache(
    client: bigquery.Client,
    query: str,
    cache: Path,
    nombre: str,
    costo_gb: float,
) -> pd.DataFrame:
    """
    Ejecuta la query en BigQuery y guarda el resultado como CSV.
    En ejecuciones posteriores lee el CSV local sin consultar BigQuery.

    Parámetros:
        client   : cliente BigQuery autenticado.
        query    : SQL a ejecutar.
        cache    : ruta del CSV de cache.
        nombre   : descripción para el log.
        costo_gb : GB escaneados estimados (solo informativo).
    """
    if cache.exists():
        print(f"  ✓ {nombre} desde CSV local (sin BigQuery).")
        return pd.read_csv(cache)

    costo_usd = costo_gb / 1024 * 6.25
    print(f"  → Descargando {nombre} de BigQuery ...")
    print(f"    (~{costo_gb:.0f} GB scan, ≤${costo_usd:.2f} USD, 1–3 min)")
    t0 = time.time()
    df = client.query(query).to_dataframe()
    df.to_csv(cache, index=False)
    print(f"  ✓ {len(df):,} filas en {time.time()-t0:.0f}s → CSV guardado")
    return df

# =============================================================================
# UTILIDADES ESPACIALES
# =============================================================================

def en_metros(lats, lons) -> np.ndarray:
    """
    Proyección equirectangular (lat, lon) → metros.
    Error < 0.05% para distancias < 50 km en Bogotá.
    Retorna array (n, 2): [y_m, x_m].
    """
    return np.column_stack([
        np.asarray(lats, float) * M_PER_DEG_LAT,
        np.asarray(lons, float) * M_PER_DEG_LON,
    ])


def kdtree(df: pd.DataFrame) -> KDTree | None:
    """KDTree en espacio métrico. Retorna None si df está vacío."""
    d = df.dropna(subset=["lat", "lon"])
    return KDTree(en_metros(d["lat"], d["lon"])) if len(d) > 0 else None


def dist_min(props_m: np.ndarray, tree: KDTree) -> np.ndarray:
    """Distancia en metros al POI más cercano (k=1, paralelo)."""
    d, _ = tree.query(props_m, k=1, workers=-1)
    return d


def count_r(props_m: np.ndarray, tree: KDTree, r: float) -> np.ndarray:
    """Número de POIs dentro del radio `r` metros."""
    return np.array([
        len(v) for v in tree.query_ball_point(props_m, r=r, workers=-1)
    ])

# =============================================================================
# PREPARACIÓN DE CATEGORÍAS
# =============================================================================

def preparar_categorias_poi(df: pd.DataFrame) -> dict:
    """
    Divide el DataFrame de POIs en subconjuntos por categoría.
    Retorna dict {nombre: DataFrame(lat, lon)}.
    """
    def f(col: str, vals: list) -> pd.DataFrame:
        return df.loc[df[col].isin(vals), ["lat", "lon"]].dropna()

    return {
        "bancos":        f("amenity", ["bank", "atm", "bureau_de_change"]),
        "hospitales":    f("amenity", ["hospital", "clinic", "doctors"]),
        "restaurantes":  f("amenity", ["restaurant", "cafe", "fast_food",
                                       "food_court", "bar", "pub", "bakery"]),
        "farmacias":     f("amenity", ["pharmacy"]),
        "supermercados": f("shop",    ["supermarket", "grocery", "convenience"]),
        "educacion":     f("amenity", ["school", "university", "college"]),
        "centros_com":   f("shop",    ["mall"]),

        # TransMilenio: portales (bus_station) + nodos con red/operador TM
        "transmilenio": pd.concat([
            f("amenity", ["bus_station"]),
            df.loc[df["network"].str.lower().str.contains("transmilenio", na=False),
                   ["lat", "lon"]],
            df.loc[df["operator"].str.lower().str.contains("transmilenio", na=False),
                   ["lat", "lon"]],
        ], ignore_index=True).dropna().drop_duplicates(),

        # Parques: tags leisure y landuse
        "parques": pd.concat([
            f("leisure", ["park", "garden", "playground",
                          "recreation_ground", "nature_reserve", "pitch"]),
            f("landuse", ["recreation_ground", "grass", "forest",
                          "village_green", "greenfield"]),
        ], ignore_index=True).dropna().drop_duplicates(),
    }


def preparar_categorias_vial(df_roads: pd.DataFrame) -> dict:
    """
    Divide el DataFrame de vías en:
      'todas'    : todos los segmentos viales (para densidad)
      'arterial' : sólo autopistas y avenidas primarias (para distancia arterial)
    """
    return {
        "todas":    df_roads[["lat", "lon"]].dropna(),
        "arterial": df_roads.loc[
            df_roads["highway"].isin(HIGHWAY_ARTERIAL), ["lat", "lon"]
        ].dropna(),
    }

# =============================================================================
# WALKABILITY SCORE
# =============================================================================

def calc_walkability(props_m: np.ndarray, trees: dict, r: float = 800.0) -> np.ndarray:
    """
    Proxy de walkability: fracción de 6 categorías de servicio con al menos
    un representante dentro de `r` metros, escalado a 0–100.

    Categorías: alimentación, salud, financiero, recreación, transporte, servicios.
    """
    CATS = {
        "alimentacion": ["restaurantes"],
        "salud":        ["hospitales", "farmacias"],
        "financiero":   ["bancos"],
        "recreacion":   ["parques"],
        "transporte":   ["transmilenio"],
        "servicios":    ["supermercados", "educacion"],
    }
    score = np.zeros(len(props_m))
    for claves in CATS.values():
        presente = np.zeros(len(props_m), dtype=bool)
        for c in claves:
            if trees.get(c) is not None:
                presente |= count_r(props_m, trees[c], r) > 0
        score += presente
    return (score / len(CATS)) * 100.0

# =============================================================================
# CÁLCULO DE LAS 10 VARIABLES OSM
# =============================================================================

def calcular_features(
    df: pd.DataFrame,
    trees_poi: dict,
    trees_vial: dict,
) -> pd.DataFrame:
    """
    Calcula las 10 variables OSM para cada fila de `df` usando KDTrees locales.
    No realiza ninguna llamada a BigQuery.
    Retorna DataFrame con [property_id, var1, …, var10].
    """
    pm  = en_metros(df["lat"].values, df["lon"].values)
    out = pd.DataFrame({"property_id": df["property_id"].values})

    # Var 8 — distancia al CBD (cálculo directo, sin árbol)
    print("    dist_cbd_km ...")
    cbd_m = en_metros([CBD_LAT], [CBD_LON])
    out["dist_cbd_km"] = np.sqrt(((pm - cbd_m) ** 2).sum(axis=1)) / 1_000.0

    # Var 9 — distancia a portal/estación TransMilenio
    print("    dist_transmilenio_m ...")
    out["dist_transmilenio_m"] = dist_min(pm, trees_poi["transmilenio"])

    # Var 10 — distancia al segmento arterial más cercano
    # (centroide del segmento como referencia espacial)
    print("    dist_via_arterial_m ...")
    out["dist_via_arterial_m"] = dist_min(pm, trees_vial["arterial"])

    # Var 11 — distancia a hospital o clínica
    print("    dist_hospital_m ...")
    out["dist_hospital_m"] = dist_min(pm, trees_poi["hospitales"])

    # Var 13 — distancia a centro comercial
    print("    dist_centro_com_m ...")
    out["dist_centro_com_m"] = dist_min(pm, trees_poi["centros_com"])

    # Var 14 — distancia a parque o área verde
    print("    dist_parque_m ...")
    out["dist_parque_m"] = dist_min(pm, trees_poi["parques"])

    # Var 15 — restaurantes y cafés en radio 500 m
    print("    n_restaurantes_500m ...")
    out["n_restaurantes_500m"] = count_r(pm, trees_poi["restaurantes"], 500.0)

    # Var 18 — bancos y cajeros en radio 500 m
    print("    n_bancos_500m ...")
    out["n_bancos_500m"] = count_r(pm, trees_poi["bancos"], 500.0)

    # Var 16 — walkability: categorías de servicio en radio 800 m (0–100)
    print("    walkability_score ...")
    out["walkability_score"] = calc_walkability(pm, trees_poi, r=800.0)

    # Var 17 — densidad vial: segmentos viales por km² en radio 500 m
    # Fórmula: segmentos_en_radio / área_círculo_500m = conteo / (π × 0.25)
    # Nota: usa centroides de segmentos como proxy de densidad de red vial.
    print("    densidad_vial ...")
    out["densidad_vial"] = count_r(pm, trees_vial["todas"], 500.0) / (pi * 0.25)

    return out


def imprimir_resumen(df: pd.DataFrame, nombre: str) -> None:
    dist = ["dist_cbd_km", "dist_transmilenio_m", "dist_via_arterial_m",
            "dist_hospital_m", "dist_centro_com_m", "dist_parque_m"]
    num  = ["n_restaurantes_500m", "n_bancos_500m", "walkability_score", "densidad_vial"]
    print(f"\n{'='*65}\n  {nombre}  —  {len(df):,} filas\n{'='*65}")
    print("\nDistancias:")
    print(df[dist].describe().round(1).to_string())
    print("\nConteos y scores:")
    print(df[num].describe().round(2).to_string())

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """
    Flujo:
      1. BigQuery → POIs de Bogotá         → poi_bogota.csv   (una vez)
      2. BigQuery → Red vial de Bogotá     → roads_bogota.csv (una vez)
      3. KDTrees en memoria por categoría.
      4. 10 variables OSM para train y test usando lat/lon → KDTree local.
      5. Guardar train_osm.csv y test_osm.csv.
    """
    client = crear_cliente_bq()

    # ── 1. POIs ───────────────────────────────────────────────────────────────
    print("\n[1/5] POIs de Bogotá (BigQuery) ...")
    df_poi = descargar_con_cache(
        client, QUERY_POIS, OSM_CACHE / "poi_bogota.csv",
        "POIs (puntos + polígonos)", costo_gb=47,
    )
    print(f"  {len(df_poi):,} POIs | "
          f"puntos={df_poi['fuente'].eq('point').sum():,} | "
          f"polígonos={df_poi['fuente'].eq('polygon').sum():,}")

    # ── 2. Red vial ───────────────────────────────────────────────────────────
    print("\n[2/5] Red vial (BigQuery) ...")
    df_roads = descargar_con_cache(
        client, QUERY_ROADS, OSM_CACHE / "roads_bogota.csv",
        "Segmentos viales", costo_gb=22,
    )
    print(f"  {len(df_roads):,} segmentos | "
          f"arteriales={df_roads['highway'].isin(HIGHWAY_ARTERIAL).sum():,}")

    # ── 3. KDTrees ────────────────────────────────────────────────────────────
    print("\n[3/5] Construyendo KDTrees ...")
    cats_poi  = preparar_categorias_poi(df_poi)
    cats_vial = preparar_categorias_vial(df_roads)

    trees_poi = {}
    for nombre, d in cats_poi.items():
        trees_poi[nombre] = kdtree(d)
        n = len(d.dropna())
        print(f"  POI '{nombre}': {n:,}" if trees_poi[nombre] else f"  POI '{nombre}': ⚠ vacío")

    trees_vial = {}
    for nombre, d in cats_vial.items():
        trees_vial[nombre] = kdtree(d)
        print(f"  Vial '{nombre}': {len(d):,} segmentos")

    vacias = [k for k in ["bancos","hospitales","restaurantes","transmilenio","parques"]
              if trees_poi.get(k) is None]
    if vacias:
        print(f"\n  ⚠ Categorías críticas vacías: {vacias}")

    # ── 4. Calcular variables ─────────────────────────────────────────────────
    print("\n[4/5] Calculando 10 variables OSM ...")
    train = pd.read_csv(PROCESSED / "train_texto.csv")
    test  = pd.read_csv(PROCESSED / "test_texto.csv")
    print(f"  train: {len(train):,} viviendas | test: {len(test):,} viviendas\n")

    print("  TRAIN:")
    ft = calcular_features(train, trees_poi, trees_vial)
    imprimir_resumen(ft, "TRAIN")

    print("\n  TEST:")
    ftest = calcular_features(test, trees_poi, trees_vial)
    imprimir_resumen(ftest, "TEST")

    # ── 5. Guardar ────────────────────────────────────────────────────────────
    print("\n[5/5] Guardando archivos ...")
    train_out = train.merge(ft,    on="property_id", how="left")
    test_out  = test.merge(ftest,  on="property_id", how="left")
    train_out.to_csv(PROCESSED / "train_osm.csv", index=False)
    test_out.to_csv( PROCESSED / "test_osm.csv",  index=False)

    print(f"  train_osm.csv : {len(train_out):,} × {len(train_out.columns)} col")
    print(f"  test_osm.csv  : {len(test_out):,}  × {len(test_out.columns)} col")
    print("\n  Variables OSM añadidas:")
    for c in [x for x in train_out.columns if x not in train.columns]:
        print(f"    + {c}")


if __name__ == "__main__":
    main()
