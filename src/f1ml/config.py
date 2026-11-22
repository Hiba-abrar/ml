"""Central configuration: paths, split, class labels, feature list, final model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CLEAN_FILE = DATA_DIR / "race_results_clean.csv"
INTERIM_DIR = DATA_DIR / "interim"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "results"
FROZEN_DIR = ROOT / "provenance" / "frozen_reference"

RANDOM_STATE = 42
N_CV_SPLITS = 5

# Time-ordered split by season (never random)
TRAIN_SEASONS = list(range(2010, 2022))   # 2010-2021
VALID_SEASONS = [2022, 2023]
TEST_SEASONS = [2024, 2025]

TARGET = "finish_category"
TOP3, POINTS, OUTSIDE = "Top 3", "Points", "Outside Points"
CLASS_ORDER = [TOP3, POINTS, OUTSIDE]
CLASS_COLORS = {TOP3: "#2a78d6", POINTS: "#eb6834", OUTSIDE: "#1baf7a"}

# The 20 model inputs, in the order the final model was fitted
FEATURES = [
    "grid", "quali_position", "quali_gap_to_pole_pct", "season", "driver_age",
    "grid_penalty_places", "pit_lane_start", "driver_avg_finish_last5",
    "constructor_points_last5", "driver_avg_finish_at_circuit",
    "driver_circuit_appearances", "driver_standing_before",
    "constructor_standing_before", "driver_points_before",
    "constructor_points_before", "driver_dnf_rate_last5", "driver_prev_finish",
    "driver_avg_quali_last5", "driver_career_races", "teammate_prev_finish",
]

# Final model: Random Forest
FINAL_RF_PARAMS = dict(
    n_estimators=400, max_depth=6, min_samples_leaf=40, max_features=0.3,
    random_state=RANDOM_STATE, class_weight=None,
)

# Constructors renamed over time share one "team lineage"
CONSTRUCTOR_LINEAGE = {
    "toro_rosso": "faenza", "alphatauri": "faenza", "rb": "faenza",
    "force_india": "silverstone", "racing_point": "silverstone", "aston_martin": "silverstone",
    "renault": "enstone", "lotus_f1": "enstone", "alpine": "enstone",
    "sauber": "hinwil", "alfa": "hinwil",
    "virgin": "marussia", "marussia": "marussia", "manor": "marussia",
    "lotus_racing": "caterham", "caterham": "caterham", "hrt": "hrt",
    "red_bull": "red_bull", "mercedes": "mercedes", "ferrari": "ferrari",
    "mclaren": "mclaren", "williams": "williams", "haas": "haas",
}


def ensure_dirs():
    for d in (MODELS_DIR, FIG_DIR, RES_DIR):
        d.mkdir(parents=True, exist_ok=True)
