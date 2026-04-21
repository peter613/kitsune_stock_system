from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#=============================
#benchmark
#=============================
KITSUNE_BENCHMARK_DIR = BASE_DIR / "benchmark_data"

#=============================
#kitsune_quant_management
#=============================
KITSUNE_QUANT_MANAGEMENT_DIR = BASE_DIR / "kitsune_quant" / "management"

#Cbm
CBM_DIR = KITSUNE_QUANT_MANAGEMENT_DIR / "cbm"

# Data Paths
LOOKUP_DIR = KITSUNE_QUANT_MANAGEMENT_DIR / "data" /"lookup"
DATABASE_DIR = KITSUNE_QUANT_MANAGEMENT_DIR / "data" /"database"
FEATURES_DIR = KITSUNE_QUANT_MANAGEMENT_DIR / "data" /"features"

# Live Path
JOURNAL_DIR = KITSUNE_QUANT_MANAGEMENT_DIR / "journal"

# Reports Path
BACKTEST_TRADELOG_DIR = KITSUNE_QUANT_MANAGEMENT_DIR / "reports" /"backtest_tradelogs"
PREDICTION_REPORT_DIR = KITSUNE_QUANT_MANAGEMENT_DIR / "reports" /"monthly_predictions"
PLOTS_REPORT_DIR = KITSUNE_QUANT_MANAGEMENT_DIR / "reports" /"plots"

#dashboard path
DASHBOARD_QUANT_DIR = KITSUNE_QUANT_MANAGEMENT_DIR / "dashboard"

#=============================
#kitsune_quant_administration
#=============================
KITSUNE_QUANT_ADMINISTRATION_DIR = BASE_DIR / "kitsune_quant" / "administration"

#=============================
#kitsune_portfolio
#=============================
KITSUNE_PORT_DIR = BASE_DIR / "kitsune_portfolio"
#dashboard path
DASHBOARD_PORT_DIR = KITSUNE_PORT_DIR / "dashboard"
