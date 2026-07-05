"""
Central configuration for the Afficionado Coffee Roasters forecasting project.
Edit paths / constants here rather than scattering magic values across scripts.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "transactions.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

for _d in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DAILY_STORE_PATH = PROCESSED_DIR / "daily_store.parquet"
HOURLY_STORE_PATH = PROCESSED_DIR / "hourly_store.parquet"
DAILY_CATEGORY_PATH = PROCESSED_DIR / "daily_category.parquet"

# ---------------------------------------------------------------------------
# Data reconstruction
# ---------------------------------------------------------------------------
# The raw export only contains `year` (constant, 2025) and `transaction_time`
# (HH:MM:SS) -- there is no explicit calendar date column. The row order in
# the file is chronological (verified: transaction_time resets to an early
# morning value 180 times across the whole file, i.e. 181 distinct trading
# days), so we reconstruct a synthetic `transaction_date` by treating every
# reset (time going backwards vs. the previous row) as a new day, starting
# from BASE_DATE. This is an assumption -- if you have the real dates,
# replace `rebuild_dates()` in data_preprocessing.py with a direct read of
# the real date column instead.
BASE_DATE = "2025-01-01"

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
SHORT_HORIZON_DAYS = 7
MEDIUM_HORIZON_DAYS = 30
TEST_HOLDOUT_DAYS = 30          # last N days held out for evaluation
LAG_DAYS = [1, 7]                 # daily lag features (t-1, t-7)
LAG_HOURS = [1, 24, 168]          # hourly lag features (t-1, t-24, t-168)
ROLLING_WINDOWS_DAYS = [3, 7]
RANDOM_STATE = 42

# Rush-hour definition used for the "Peak Error Rate" KPI (local store hours)
PEAK_HOURS = list(range(7, 10))  # 7am-9:59am morning rush
