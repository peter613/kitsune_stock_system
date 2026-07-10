import sys
from datetime import datetime
import time
import subprocess
import os
import config
import questionary
import shutil
""""
未來投組比例，etf投組 : 量化投組 : 現金 = 60% : 30% : 10%
etf投組， 00731(高息低波) : 00919 : 0056 = 3 : 1 : 1

jornal action:
buy / sell : ai，other，etf
in /out : cash，div(只限etf的股利，若要提款，優先提股利(特別股概念))

jornal cash label:
div  (id = cash, label = div)
cash (id = cash, label = 隨意(wage/capital))
"""

# 匯入模組
from backup_db import backup
from market_data.updater.etf import update_etf
from market_data.updater.revenue import update_revenue
from market_data.diagnose.diagnose import diagnose_all, diagnose_market
from market_data.sync_data import sync
from fund_stock.management.core.prediction import prediction
from fund_stock.management.core.backtest.simulator import backtester
from fund_stock.management.core.backtest.analysis import bt_analysis
from fund_stock.management.data.cleaning.train import create_training
from fund_stock.management.data.cleaning.pred import create_prediction
from fund_stock.management.core.create_cbm import create_cbm
from fund_stock.administration.update_navpu import update_quant

# =================
# 系統工具定義(指令)
# =================
def sync_database():
    market_ok = diagnose_market()

    print('\n\n[1.] 下載 ETF')
    update_etf()
    
    print('\n\n[2.] 同步 股市')
    sync()
    
    print('\n\n[3.] 資料庫診斷')
    if market_ok:
        print("market.parquet 完美無缺漏！")
    else:
        print("market.parquet 異常!")
        return f"\n\n資料庫問題，中斷程序。"

    print('\n\n[4.] 更新基金帳務')
    update_quant()

    print('\n\n[5.] 備份資料庫')
    backup()

    return f"\n\n資料庫已全面更新完畢。"


def open_dashboard_fund() -> str:
    """啟動 Streamlit fund 視覺化儀表板/戰情室/dashboard。"""
    print("啟動基金儀表板...")
    # /k 代表執行完後保持視窗開啟 (方便 Debug)，若要啟動後自動關閉視窗可改為 /c

    target_path = r"C:\Users\Lynn610\Documents\_kitsune_stock_system"

    cmd = (
        f'start /D "{target_path}" cmd /k '
        f'"set PYTHONPATH=. && '
        f'venv\\Scripts\\activate && '
        f'streamlit run {config.STOCK_DASHBOARD_DIR}/app.py --server.port 8501"'
    )
        
    subprocess.Popen(cmd, shell=True)
    return "\n儀表板已成功在背景啟動。"

def open_dashboard_port() -> str:
    print("啟動投組儀表板...")
    # /k 代表執行完後保持視窗開啟 (方便 Debug)，若要啟動後自動關閉視窗可改為 /c

    target_path = r"C:\Users\Lynn610\Documents\_kitsune_stock_system"

    cmd = (
        f'start /D "{target_path}" cmd /k '
        f'"set PYTHONPATH=. && '
        f'venv\\Scripts\\activate && '
        f'streamlit run {config.DASHBOARD_PORT_DIR}/app.py --server.port 8502"'
    )
        
    subprocess.Popen(cmd, shell=True)
    return "儀表板已成功在背景啟動。"

# =================
# 系統工具定義(AI)
# =================
def train_model(version: str) -> str:
    """更新(CBM)模型。必須輸入版本號 (例如 v11)。"""
    print(f"準備訓練版本: {version}")
    create_training(version=version, years_10=True, stock_filter=True)
    create_cbm(version)
    return f"版本 {version} 特徵與模型訓練完成。"

def run_prediction(version: str, update_features: bool = True) -> str:
    """執行特定版本模型的明日預測。預設update_features為True"""
    if update_features:
        print("更新預測特徵...")
        create_prediction() 
    
    print(f"啟動 {version} 模型預測...")
    prediction_result_string = prediction(version) 


def run_backtest(version: str, benchmark: str = '0050.TW') -> str:
    """執行模型歷史回測。
    參數 benchmark 可選 '0050.TW' 或 '00631L.TW'。
    參數 version 需請使用者輸入。
    """
    select_rank = [1, 2, 3]
    print(f"啟動 {version} 回測 (對照: {benchmark}, 範圍: Rank 1~3)...")
    backtester(version=version, benchmark=benchmark, select_rank=select_rank)
    bt_analysis(version=version, top_n=3)
    
    return f"版本 {version} 回測執行完畢，績效報告已產出。"

# ==========================================
# 主程式 (終端機介面)
# ==========================================
def type_effect(text, delay=0.023, end_new_line=True):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    if end_new_line:
        print()

def get_available_versions():
    if not os.path.exists(config.STOCK_CBM_DIR): return []
    versions = [d for d in os.listdir(config.STOCK_CBM_DIR)
                if os.path.isdir(config.STOCK_CBM_DIR / d) and d.startswith('v')]
    return sorted(versions)[-3:]

def main():
    print('')
    type_effect(" >> INITIALIZING KITSUNE KERNEL...")
    time.sleep(0.3)
    type_effect(" >> WELCOME ARCHITECT")
    time.sleep(0.3)

    while True:
        print('\n')
        width = shutil.get_terminal_size().columns
        print('.' * width)
        print('\n')
        action_home = questionary.select(
            "請選擇要執行的模組：\n",
            choices=[
                "資料庫模組",
                "量化模組",
                "儀表板模組",
                "清除畫面",
            ],
            qmark="",
            instruction=" "
        ).ask()

        if action_home == "資料庫模組":
            action_database = questionary.select(
                "請選擇要執行的資料庫模組：\n",
                choices=[
                    "同步與備份資料庫",
                    "更新營收資料庫",
                    "診斷資料庫",
                    "返回主選單"
                ],
                qmark="",
                instruction=" "
            ).ask()
            
            if action_database == "同步與備份資料庫":
                print("\n[處理中...]")
                res = sync_database()

                print(f"{res}")

            elif action_database == "更新營收資料庫":
                print("\n[更新中...]")
                update_revenue()

            elif action_database == "診斷資料庫":
                print("\n[診斷中...]")
                diagnose_all()

        elif action_home == "量化模組":
            action_quant = questionary.select(
                "請選擇要執行的量化模組：\n",
                choices=[
                    "回測",
                    "預測",
                    "模型訓練",
                    "返回主選單"
                ],
                qmark="",
                instruction=" "
            ).ask()
            
            
            if action_quant in ["回測", "預測"]:
                versions = get_available_versions()
                if not versions:
                    print("目前無可用模型版本，請先執行模型訓練。")
                    continue
                
                version = questionary.select("請選擇模型版本：", choices=versions,qmark="",instruction=" ").ask()
                
                if version:
                    if action_quant == "回測":
                        benchmark = questionary.select(
                            "請選擇 Benchmark：",
                            choices=["0050.TW", "00631L.TW"],
                        qmark="",
                        instruction=" "
                        ).ask()
                        
                        if benchmark:
                            res = run_backtest(version, benchmark=benchmark)
                            print(f"{res}")
                            
                    elif action_quant == "預測":
                        run_prediction(version)

            elif action_quant == "模型訓練":
                # 模型訓練通常會建立新版本，因此保留輸入欄位
                version = questionary.text(
                ("請輸入要訓練的模型版本號 (例如 v11):"),
                qmark="",
                instruction=" ").ask()
                if version:
                    res = train_model(version)
                    print(f"{res}")

        elif action_home == "儀表板模組":
            action_dashboard = questionary.select(
                "請選擇要執行的儀表板模組：\n",
                choices=[
                    "經理人儀表板",
                    "投資人儀表板",
                    "全部儀表板",
                    "返回主選單"
                ],
                qmark="",
                instruction=" "
            ).ask()
            
            if action_dashboard != "返回主選單":
                SANDBOX_MODE = True

                if not SANDBOX_MODE and datetime.now().weekday() < 5:
                    print('\n為平日，暫停打開儀表板喔')
                else:
                    if action_dashboard == "經理人儀表板":
                        print("\n[開啟經理人儀表板]")
                        print(f"{open_dashboard_fund()}")
                    elif action_dashboard == "投資人儀表板":
                        print("\n[開啟投資者儀表板]")
                        print(f"{open_dashboard_port()}")
                    elif action_dashboard == "全部儀表板":
                        print("\n[開啟所有儀表板]")
                        print(f"{open_dashboard_fund()}")
                        print(f"{open_dashboard_port()}")

        elif action_home == "清除畫面":
            os.system('cls')

if __name__ == "__main__":
    main()