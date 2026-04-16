from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
#Cbm
CBM_DIR = BASE_DIR / "cbm"

# Data Paths
LOOKUP_DIR = BASE_DIR / "data" /"lookup"
DATABASE_DIR = BASE_DIR / "data" /"database"
FEATURES_DIR = BASE_DIR / "data" /"features"

# Live Path
JOURNAL_DIR = BASE_DIR / "journal"

# Reports Path
BACKTEST_TRADELOG_DIR = BASE_DIR / "reports" /"backtest_tradelogs"
PREDICTION_REPORT_DIR = BASE_DIR / "reports" /"monthly_predictions"
PLOTS_REPORT_DIR = BASE_DIR / "reports" /"plots"

#dashboard path
DASHBOARD_DIR = BASE_DIR / "dashboard"

