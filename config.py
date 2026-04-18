from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#=============================
#kitsune_fund_management
#=============================
KITSUNE_FUND_MANAGEMENT_DIR = BASE_DIR / "kitsune_fund" / "management"

#Cbm
CBM_DIR = KITSUNE_FUND_MANAGEMENT_DIR / "cbm"

# Data Paths
LOOKUP_DIR = KITSUNE_FUND_MANAGEMENT_DIR / "data" /"lookup"
DATABASE_DIR = KITSUNE_FUND_MANAGEMENT_DIR / "data" /"database"
FEATURES_DIR = KITSUNE_FUND_MANAGEMENT_DIR / "data" /"features"

# Live Path
JOURNAL_DIR = KITSUNE_FUND_MANAGEMENT_DIR / "journal"

# Reports Path
BACKTEST_TRADELOG_DIR = KITSUNE_FUND_MANAGEMENT_DIR / "reports" /"backtest_tradelogs"
PREDICTION_REPORT_DIR = KITSUNE_FUND_MANAGEMENT_DIR / "reports" /"monthly_predictions"
PLOTS_REPORT_DIR = KITSUNE_FUND_MANAGEMENT_DIR / "reports" /"plots"

#dashboard path
DASHBOARD_FUND_DIR = KITSUNE_FUND_MANAGEMENT_DIR / "dashboard"

#=============================
#kitsune_fund_administration
#=============================
KITSUNE_FUND_ADMINISTRATION_DIR = BASE_DIR / "kitsune_fund" / "administration"

#=============================
#kitsune_portfolio
#=============================
KITSUNE_PORT_DIR = BASE_DIR / "kitsune_portfolio"
#dashboard path
DASHBOARD_PORT_DIR = KITSUNE_PORT_DIR / "dashboard"
