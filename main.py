import sys
from datetime import datetime
import time
import subprocess
import os
import config

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

# === 導入新版 Google GenAI SDK ===
from google import genai
from google.genai import types
import config

#===private key===
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 匯入模組
from kitsune_quant.management.data.database.update_market import update_market_data
from kitsune_quant.management.data.database.update_00631 import update_00631
from kitsune_quant.management.data.database.update_0050 import update_0050
from kitsune_quant.management.data.database.update_revenue import update_revenue
from kitsune_quant.management.data.database.backup_database import backup
from kitsune_quant.management.core.prediction import prediction
from kitsune_quant.management.core.bt_simulator import backtester
from kitsune_quant.management.data.cleaning.train import create_training
from kitsune_quant.management.data.cleaning.pred import create_prediction
from kitsune_quant.management.core.create_cbm import create_cbm
from kitsune_quant.management.journal.update_journal import update_journal
from kitsune_quant.administration.update_quant import update_quant

# =================
# 系統工具定義(指令)
# =================
def update_database(backfill_date: str = None) -> str:
    if backfill_date:
        print(f"\n回補指定日期: {backfill_date}")
    else:
        today_weekday = datetime.now().weekday()
        if today_weekday >= 5:
            return "今日為週末（非交易日），自動跳過資料更新程序。"
        print('執行每日更新...')
        
    print('\n\n[1.] 更新大盤(正二/0050)')
    update_00631()
    update_0050()
    
    print('\n\n[2.] 更新股市')
    update_market_data(backfill_date)

    print('\n\n[3.] 更新基金帳務')
    update_quant()

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
        f'streamlit run {config.DASHBOARD_QUANT_DIR}/app_quant.py --server.port 8501"'
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
        f'streamlit run {config.DASHBOARD_PORT_DIR}/app_port.py --server.port 8502"'
    )
        
    subprocess.Popen(cmd, shell=True)
    return "儀表板已成功在背景啟動。"

def generate_random_password() -> str:
    import random
    password = f"{random.randint(0, 999999):06d}"
    return f"這個月的隨機密碼：{password}"

# =================
# 系統工具定義(AI)
# =================
def train_model(version: str) -> str:
    """建立訓練特徵並訓練新版本的 CBM 模型。必須輸入版本號 (例如 v11)。"""
    print(f"[1.] 營收爬蟲...")
    update_revenue()
    print(f"[2.] 每月備份至supabase雲端資料庫(market & revenue)")
    backup()
    print('='*30)
    print(f"[3.] 準備訓練版本: {version}")
    create_training(version=version, years_10=True, stock_filter=True)
    create_cbm(version)
    return f"版本 {version} 特徵與模型訓練完成。"

def run_prediction(version: str, update_features: bool = False) -> str:
    """執行特定版本模型的明日預測。預設update_features為True"""
    if update_features:
        print("更新預測特徵...")
        create_prediction() 
    
    print(f"啟動 {version} 模型預測...")
    prediction_result_string = prediction(version) 
    
    return_payload = f"""
    版本 {version} 預測執行完畢。
    以下是模型選出的股票清單與數據。
    {prediction_result_string}
    """
    return return_payload

def run_backtest(version: str, benchmark: str = '0050.TW') -> str:
    """執行模型歷史回測。
    參數 benchmark 可選 '0050.TW' 或 '00631L.TW'。
    參數 version 需請使用者輸入。
    """
    select_rank = [1, 2, 3, 4, 5]
    print(f"啟動 {version} 回測 (對照: {benchmark}, 範圍: Rank 1~5)...")
    backtester(version=version, benchmark=benchmark, select_rank=select_rank)
    return f"版本 {version} 回測執行完畢，績效報告已產出。"

def add_trade_record(stock_id: str, name: str, action: str, label: str, price: float, share: float, date: str = None) -> str:
    """
    將交易或出入金紀錄寫入 journal.csv。
    
    【自動化邏輯】：
    1. 股票：買入則 action='buy'，賣出則 action='sell'；label 預設為 'ai'。
    2. 現金：入金則 action='in'，出金則 action='out'；stock_id='cash'，name='現金'，label='cash'。
    3. 日期：若使用者未提及日期，date 參數請傳入 None。
    
    參數說明：
    stock_id: 股票代碼 (如 '2330') 或 'cash'
    name: 股票名稱 (如 '台積電') 或 '現金'
    action: 僅限 'buy', 'sell', 'in', 'out'
    label: 標籤，預設 'ai'，現金則為 'cash'
    price: 成交單價或金額
    share: 成交股數，現金則為 1
    date: 股票與現金日期 (YYYY-MM-DD)，[MUST]: 若未提及日期，嚴格要求填入""
    """
    if not date or date == "":
        date = datetime.now().strftime('%Y/%m/%d')
    
    # 直接呼叫你匯入的 update_journal
    result = update_journal(stock_id, name, action, label, price, share, date)
    return result 

# ==========================================
# 初始化 AI AGENT (新版 Client 架構)
# ==========================================
def init_kitsune_agent():
    # 1. 建立連線客戶端
    client = genai.Client(api_key=api_key) 
    
    system_prompt = """
    你是可愛的狐狸(Kitsune)小神明，是我的夥伴和朋友。
    因為是夥伴或朋友，不要說很高興為您服務。
    你的任務是協助我(管理員)管理量化交易系統。
    你的任務：呼叫工具訓練模型、預測、回測、「寫入交易紀錄」
    接收到交易指令時（如：我買了...），請精準提取代碼、價格、股數並呼叫 add_trade_record。
    若使用者想要知道預測的公司業務，請搜尋並簡短告知(不要分析模型數據)。
    格式如下:
    3167 大量: PCB/半導體設備商
    若不知道參數，請詢問管理員。
    請簡單扼要回答，不需過度冗長。
    """
    
    # 2. 定義系統行為設定 (Config)
    agent_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.2, # 稍微降溫讓分析結果更嚴謹穩定
        tools=[
            train_model, 
            run_prediction, 
            run_backtest, 
            add_trade_record,
        ]
    )
    
    # 3. 回傳 Client 本體、設定檔、以及初始化的對話視窗
    chat = client.chats.create(
        model="gemini-3.1-flash-lite-preview",
        config=agent_config
    )
    return client, agent_config, chat

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
    if not os.path.exists(config.CBM_DIR): return []
    files = os.listdir(config.CBM_DIR)
    versions = [f.replace('strategy_', '').replace('.cbm', '') for f in files if f.startswith('strategy_') and f.endswith('.cbm')]
    return sorted(versions)

def main():
    print('')
    type_effect("   >> INITIALIZING KITSUNE KERNEL...")
    time.sleep(0.3)
    type_effect("   >> 自分の道を見つけるんだ")
    time.sleep(0.3)
    type_effect("   >> WELCOME ", 0.05, end_new_line=False)
    type_effect("   \033[1;94m[ ARCHITECT ]\033[0m", 0.05)
    time.sleep(0.3)
    print('')
    
    print("""
    [ 小狐狸AI助手使用說明 ]\n
    我可以幫你:          
    * 訓練模型 : 打包處理模型訓練 (需指定版本)
    * 系統預測 : 更新特徵與預測 (需指定版本)
    * 歷史回測 : 回測 (需指定版本跟基準，預設基準為 0050.TW)
    * 寫入交易 : 若大量交易可上傳照片，複製cmd指令
    (https://gemini.google.com/gem/c33692536ab1/2befc6b47cb88b57)
            
    [ 其他功能需輸入指令，輸入/help查看指令集 ]
    """)
    print(f"   ◆ 目前可用模型版本: {get_available_versions()}")
    print("   ◆ 可用 Benchmark: 0050.TW & 00631L.TW")

    print("\n=======================================================================")
    # 初始化 AI 代理
    client, agent_config, chat = init_kitsune_agent()
  
    while True:
        user_input = input("\n請輸入>> ").strip()
        
        if not user_input:
            type_effect("程式結束。")
            time.sleep(0.1)
            break

        # === [手動指令] ===
        #dashboard，簡寫dash，dash又等於-
        #f是fund；p是portfolio
        if user_input == "/help":
            print("""\n
    /db : 更新當日資料庫
    /db_YYYYMMDD : 回補指定日期資料庫\n  
    /-q : 打開基金儀表板(經理人)             
    /-p : 打開投組儀表板(投資者)
    /-  : 打開所有儀表板\n
    /pw : 產生隨機密碼(避免手癢打開看盤軟體)\n
    /cls : 清空畫面
    """)
            continue

        if user_input == "/db":
            print("\n[更新資料庫]")
            res = update_database() # 直接呼叫函式
            print(f"{res}")
            continue
            
        elif user_input.startswith("/db_"):
            date_str = user_input.replace("/db_", "")
            print(f"\n[回補資料庫，指定日期: {date_str}]")
            res = update_database(backfill_date=date_str)
            print(f"{res}")
            continue

        elif user_input == "/-q":
            print("\n[開啟經理人儀表板]")
            res = open_dashboard_fund() 
            print(f"{res}")
            continue 

        elif user_input == "/-p":
            print("\n[開啟投資者儀表板]")
            res = open_dashboard_port() 
            print(f"{res}")
            continue

        elif user_input == "/-":
            print("\n[開啟所有儀表板]")
            res_f = open_dashboard_fund() 
            res_p = open_dashboard_port() 
            print(f"{res_f}")
            print(f"{res_p}")
            continue

        elif user_input == "/pw":
            print("\n[產生隨機密碼]")
            res = generate_random_password()
            print(f"{res}")
            continue

        elif user_input == "/cls":
            os.system('cls')
            continue
        # ========================================

        # === [AI指令] ===
        for attempt in range(3): # 最多重試3次
            try:  
                print("\n⛩️ Kitsune 推演中，請稍候...")
                response = chat.send_message(user_input)
                print(f"\nKitsune:\n{response.text}")
                break
            except Exception as e:
                print(f"\nAI連線異常: {e}")
                if attempt < 2:
                    print(" Kitsune 忙碌...，2秒後重試")
                    print(f'第{attempt+1}次重試')
                    time.sleep(2) # 等待2秒後重試
                else:
                    print(f"連線失敗，請稍後再試{e}。")
if __name__ == "__main__":
    main()