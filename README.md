# Problem Set 3 — Making Money with ML

Modelos de predicción de precios de vivienda para Bogotá D.C. usando datos de
listados inmobiliarios de Properati y variables de accesibilidad urbana de
OpenStreetMap.

---

## Estructura del proyecto

```
├── 00_data/
│   ├── raw/
│   │   ├── train.csv          # Datos de entrenamiento (Properati)
│   │   ├── test.csv           # Datos de evaluación
│   │   └── osm/               # Cache de descargas OSM — generado automáticamente
│   └── processed/             # Outputs de los scripts — generado automáticamente
│
├── 01_scripts/
│   └── preparation/
│       ├── 01_extract_text_features.py   # Variables hedónicas desde texto
│       └── 02_add_osm_features.py        # Variables de localización desde OSM
│
├── .env.example               # Plantilla de variables de entorno requeridas
├── requirements.txt           # Dependencias del proyecto
└── README.md
```

---

## Configuración del entorno

### Requisitos previos

- Python 3.11 (`python3.11 --version`)
- Cuenta de Google Cloud con acceso a BigQuery y un archivo de credenciales
  (service account JSON)

### 1. Crear el entorno virtual

```bash
python3.11 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar credenciales de Google Cloud

Copiar `.env.example` como `.env` y completar la ruta al JSON del service account:

```bash
cp .env.example .env
```

Luego exportar la variable antes de correr los scripts:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/ruta/completa/a/tu/credentials.json"
```

> El archivo `.env` y el JSON de credenciales están en `.gitignore` y **nunca
> deben subirse al repositorio**.

---

## Scripts de preparación de datos

### `01_extract_text_features.py` — Variables hedónicas de texto

Extrae variables del campo `title` y `description` de cada listado.

**Input:** `00_data/raw/train.csv` y `test.csv`

**Output:** `00_data/processed/train_texto.csv` y `test_texto.csv`

**Variables producidas:**

| Variable | Tipo | Descripción |
|---|---|---|
| `remodelado` | 1 / NaN | Menciona remodelación o inmueble nuevo |
| `vista_panoramica` | 1 / NaN | Menciona vistas, cerros o panorámica |
| `deposito` | 1 / NaN | Menciona depósito, bodega o cuarto útil |
| `conjunto_cerrado` | 1 / NaN | Menciona portería, vigilancia o áreas comunes |
| `balcon_terraza` | 1 / NaN | Menciona balcón, terraza, BBQ o patio |
| `tfidf_premium` | float | Score TF-IDF de términos de lujo en el anuncio |
| `parqueaderos_txt` | int / NaN | Número de garajes mencionados en el texto |
| `piso_txt` | int / NaN | Piso del apartamento mencionado en el texto |
| `gimnasio` | 1 / NaN | Menciona gimnasio, gym o fitness |
| `amenidades` | 1 / NaN | Menciona al menos una amenidad |
| `num_amenidades` | float / NaN | Cantidad de tipos de amenidad mencionados |

> **Convención de missings:** los dummies usan `1` cuando el patrón aparece en
> el texto y `NaN` cuando no aparece. Un `NaN` no significa que el inmueble no
> tenga esa característica — significa que el anuncio no la mencionó.

**Ejecución:**
```bash
python 01_scripts/preparation/01_extract_text_features.py
```

---

### `02_add_osm_features.py` — Variables de localización (OpenStreetMap)

Descarga POIs y red vial de Bogotá desde BigQuery y calcula variables de
accesibilidad para cada inmueble usando sus coordenadas `lat`/`lon`.

**Input:** `00_data/processed/train_texto.csv` y `test_texto.csv`

**Output:** `00_data/processed/train_osm.csv` y `test_osm.csv`

**Variables producidas:**

| Variable | Unidad | Descripción |
|---|---|---|
| `dist_cbd_km` | km | Distancia al CBD (Plaza de Bolívar) |
| `dist_transmilenio_m` | m | Distancia al portal/estación TM más cercana |
| `dist_via_arterial_m` | m | Distancia a autopista o avenida principal |
| `dist_hospital_m` | m | Distancia al hospital o clínica más cercana |
| `dist_centro_com_m` | m | Distancia al centro comercial más cercano |
| `dist_parque_m` | m | Distancia al parque o área verde más cercana |
| `n_restaurantes_500m` | conteo | Restaurantes y cafés en radio 500 m |
| `n_bancos_500m` | conteo | Bancos y cajeros en radio 500 m |
| `walkability_score` | 0–100 | Categorías de servicio accesibles en 800 m |
| `densidad_vial` | seg/km² | Segmentos viales por km² en radio 500 m |

**Costo BigQuery (primera ejecución):**
- ~47 GB (POIs) + ~22 GB (vías) = ~69 GB totales ≈ **$0.43 USD**
- Dentro del free tier mensual de 1 TB → **$0 si no se ha superado el límite**
- A partir de la segunda ejecución: **$0** (lee desde cache local en `00_data/raw/osm/`)

**Ejecución:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/ruta/a/tu/credentials.json"
python 01_scripts/preparation/02_add_osm_features.py
```

---

## Orden de ejecución

```bash
source venv/bin/activate
python 01_scripts/preparation/01_extract_text_features.py
python 01_scripts/preparation/02_add_osm_features.py
```

---

## Fuentes de datos

| Fuente | Uso |
|---|---|
| Properati Colombia | Listados de vivienda con precio, características y texto |
| BigQuery — `bigquery-public-data.geo_openstreetmap` | POIs y red vial de Bogotá |
| variables_hedónicas_Chapinero_v2.xlsx | Marco teórico de variables hedónicas |
