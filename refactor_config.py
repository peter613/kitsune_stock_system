import os

replacements_stock = {
    "config.DATABASE_DIR": "config.STOCK_DATABASE_DIR",
    "config.FEATURES_DIR": "config.STOCK_FEATURES_DIR",
    "config.WHITE_LIST_DIR": "config.STOCK_WHITE_LIST_DIR",
    "config.JOURNAL_DIR": "config.STOCK_JOURNAL_DIR",
    "config.PLOTS_DIR": "config.STOCK_PLOTS_DIR",
    "config.BACKTEST_TRADELOG_DIR": "config.STOCK_BACKTEST_TRADELOG_DIR",
    "config.PREDICTION_REPORT_DIR": "config.STOCK_PREDICTION_REPORT_DIR",
    "config.CBM_DIR": "config.STOCK_CBM_DIR",
    "config.DASHBOARD_QUANT_DIR": "config.STOCK_DASHBOARD_DIR",
}

replacements_etf = {
    "config.DATABASE_DIR": "config.ETF_DATABASE_DIR",
    "config.FEATURES_DIR": "config.ETF_FEATURES_DIR",
    "config.WHITE_LIST_DIR": "config.ETF_WHITE_LIST_DIR",
    "config.JOURNAL_DIR": "config.ETF_JOURNAL_DIR",
    "config.JOURNAL_ETf_DIR": "config.ETF_JOURNAL_DIR",
    "config.DASHBOARD_QUANT_DIR": "config.ETF_DASHBOARD_DIR",
    "config.FUND_STOCK_ADMINISTRATION_DIR": "config.FUND_ETF_ADMINISTRATION_DIR",
}

def replace_in_dir(directory, replacements):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = content
                for old, new in replacements.items():
                    new_content = new_content.replace(old, new)
                
                if new_content != content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated {path}")

def replace_in_file(path, replacements):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
    if new_content != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {path}")

base = r"c:\Users\Lynn610\Documents\_kitsune_Stock_System"
replace_in_dir(os.path.join(base, "fund_stock"), replacements_stock)
replace_in_dir(os.path.join(base, "fund_etf"), replacements_etf)

# portfolio may just use FUND_STOCK_ADMINISTRATION_DIR which didn't change name.
# main.py might use config.DASHBOARD_QUANT_DIR for quant dashboard. Let's fix main.py
replace_in_file(os.path.join(base, "main.py"), {
    "config.DASHBOARD_QUANT_DIR": "config.STOCK_DASHBOARD_DIR",
    "config.DATABASE_DIR": "config.STOCK_DATABASE_DIR",
})

print("Done")
