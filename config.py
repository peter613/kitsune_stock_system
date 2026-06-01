from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#=============================
# fund_stock_management
#=============================
FUND_STOCK_MANAGEMENT_DIR = BASE_DIR / "fund_stock" / "management"

# Data Paths
LOOKUP_DIR = FUND_STOCK_MANAGEMENT_DIR / "data" /"lookup"
DATABASE_DIR = FUND_STOCK_MANAGEMENT_DIR / "data" /"database" / "data"
FEATURES_DIR = FUND_STOCK_MANAGEMENT_DIR / "data" /"features"
WHITE_LIST_DIR = FUND_STOCK_MANAGEMENT_DIR / "data" /"cleaning"

# Live Path
JOURNAL_DIR = FUND_STOCK_MANAGEMENT_DIR / "journal"

# Reports Path (base directories — 實際使用時透過 get_version_dir 取得版本子資料夾)
PLOTS_DIR = FUND_STOCK_MANAGEMENT_DIR / "reports" / "plots"
BACKTEST_TRADELOG_DIR = FUND_STOCK_MANAGEMENT_DIR / "reports" / "backtest_tradelogs"
PREDICTION_REPORT_DIR = FUND_STOCK_MANAGEMENT_DIR / "reports" / "predictions"

# Cbm (base directory)
CBM_DIR = FUND_STOCK_MANAGEMENT_DIR / "cbm"

# Dashboard path
DASHBOARD_QUANT_DIR = FUND_STOCK_MANAGEMENT_DIR / "dashboard"

#=============================
# fund_stock_administration
#=============================
FUND_STOCK_ADMINISTRATION_DIR = BASE_DIR / "fund_stock" / "administration"

#=============================
# portfolio
#=============================
PORTFOLIO_DIR = BASE_DIR / "portfolio"
# Dashboard path
DASHBOARD_PORT_DIR = PORTFOLIO_DIR / "dashboard"


#=============================
# 版本路徑工具
#=============================
def get_version_dir(base_dir: Path, version: str) -> Path:
    """
    取得版本子資料夾路徑，不存在時自動建立。
    用法範例:
        plot_dir = config.get_version_dir(config.PLOTS_DIR, 'v27')
        # -> .../reports/plots/v27/
    """
    version_dir = base_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir
