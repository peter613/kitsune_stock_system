import os

replacements = {
    "config.STOCK_DATABASE_DIR": "config.MARKET_DATA_DIR",
    "config.ETF_DATABASE_DIR": "config.MARKET_DATA_DIR",
    "from fund_stock.management.data.database.updater": "from market_data.updater",
    "from fund_etf.management.data.database.updater": "from market_data.updater"
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
replace_in_dir(os.path.join(base, "fund_stock"), replacements)
replace_in_dir(os.path.join(base, "fund_etf"), replacements)
replace_in_dir(os.path.join(base, "market_data"), replacements)
replace_in_file(os.path.join(base, "main.py"), replacements)

print("Done")
