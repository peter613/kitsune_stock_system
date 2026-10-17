from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#=============================
# market_data (共用報價中心)
#=============================
MARKET_DATA_DIR = BASE_DIR / "market_data" / "data"
MARKET_UPDATER_DIR = BASE_DIR / "market_data" / "updater"
MARKET_DIAGNOSE_DIR = MARKET_DATA_DIR  # 相容舊路徑設定，現 holidays.csv 與報價統一置於 MARKET_DATA_DIR

#=============================
# fund_stock_management
#=============================
FUND_STOCK_MANAGEMENT_DIR = BASE_DIR / "fund_stock" / "management"

# Pipeline & Features Data Paths
STOCK_PIPELINE_DIR = FUND_STOCK_MANAGEMENT_DIR / "pipeline"
STOCK_FEATURES_DIR = FUND_STOCK_MANAGEMENT_DIR / "features_data"
STOCK_TRAIN_FEATURES_DIR = STOCK_FEATURES_DIR / "train"
STOCK_BT_FEATURES_DIR = STOCK_FEATURES_DIR / "bt"
STOCK_WHITE_LIST_DIR = MARKET_DATA_DIR

# Live Path
STOCK_JOURNAL_DIR = FUND_STOCK_MANAGEMENT_DIR / "journal"

# Reports Path (base directories — 實際使用時透過 get_version_dir 取得版本子資料夾)
STOCK_PLOTS_DIR = FUND_STOCK_MANAGEMENT_DIR / "reports" / "plots"
STOCK_BACKTEST_TRADELOG_DIR = FUND_STOCK_MANAGEMENT_DIR / "reports" / "backtest_tradelogs"
STOCK_PREDICTION_REPORT_DIR = FUND_STOCK_MANAGEMENT_DIR / "reports" / "predictions"

# Cbm (base directory)
STOCK_CBM_DIR = FUND_STOCK_MANAGEMENT_DIR / "cbm"

# Dashboard path
STOCK_DASHBOARD_DIR = FUND_STOCK_MANAGEMENT_DIR / "dashboard"

#=============================
# fund_stock_administration
#=============================
FUND_STOCK_ADMINISTRATION_DIR = BASE_DIR / "fund_stock" / "administration"


#=============================
# fund_etf_management
#=============================
FUND_ETF_MANAGEMENT_DIR = BASE_DIR / "fund_etf" / "management"

# Data Paths
ETF_FEATURES_DIR = FUND_ETF_MANAGEMENT_DIR / "data" / "features"
ETF_WHITE_LIST_DIR = FUND_ETF_MANAGEMENT_DIR / "data" / "cleaning"

# Live Path
ETF_JOURNAL_DIR = FUND_ETF_MANAGEMENT_DIR / "journal"

# Dashboard path
ETF_DASHBOARD_DIR = FUND_ETF_MANAGEMENT_DIR / "dashboard"

#=============================
# fund_etf_administration
#=============================
FUND_ETF_ADMINISTRATION_DIR = BASE_DIR / "fund_etf" / "administration"


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
        plot_dir = config.get_version_dir(config.STOCK_PLOTS_DIR, 'v27')
        # -> .../reports/plots/v27/
    """
    version_dir = base_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir
