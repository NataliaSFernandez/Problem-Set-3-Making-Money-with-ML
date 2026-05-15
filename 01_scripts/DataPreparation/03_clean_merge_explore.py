"""
Script: 03_clean_merge_explore.py
=================================
Limpieza, merge y exploración de datos OSM, text y de properaty para Chapinero.

Inputs:
    - train.csv / test.csv          → Properati base data
    - train_osm.csv / test_osm.csv  → Base + OSM spatial features
    - train_texto.csv / test_texto.csv → Base + text-derived features
 
Output:
    - train_final.csv / test_final.csv guardados en /00_data/processed  Merged y limpios 
    - Outputs guardadas en /02_outputs/DataPreparation
"""
# =============================================================================
# 0. LIBRERÍAS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")