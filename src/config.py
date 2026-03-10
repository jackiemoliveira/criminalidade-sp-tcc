# variáveis explicativas
FEATURES_BASE = [
    "per_cent_urban",
    "infant_mortality",
    "per_cent_male",
    "per_cent_youth",
    "per_cent_unemployed",
    "log_pib",
    "gini",
    "log_populacao"
]

# =========================
# MODELO ORIGINAL
# =========================

FEATURES_ORIGINAL = [
    "per_cent_urban",
    "infant_mortality",
    "per_cent_male",
    "per_cent_youth",
    "per_cent_unemployed",
    "log_pib",
    "gini",
    "log_populacao"
]

# =========================
# MODELO COM HOMENS JOVENS
# =========================

FEATURES_YOUNG_MALE = [
    "per_cent_urban",
    "infant_mortality",
    "per_cent_unemployed",
    "log_pib",
    "gini",
    "log_populacao",
    "per_cent_young_male"
]

# variável resposta
TARGET = "tx_homicidios_bayes"

# parâmetros de modelagem
TEST_SIZE = 0.3
RANDOM_STATE = 42

# caminhos
DATA_PATH = "data/raw/db_homicidios_sp_onu.csv"
GEO_PATH = "data/geodata/municipios_sp.geojson"