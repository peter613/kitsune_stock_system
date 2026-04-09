import sys
from datetime import datetime
import time
import subprocess
import os
import config

# === 導入新版 Google GenAI SDK ===
from google import genai
from google.genai import types
import config

#===private key===
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# 匯入模組
try:
    from data.scripts.update.market import update_market_data
    from data.scripts.update.gold import update_gold
    from data.scripts.update._00631 import update_00631
    from data.scripts.update._0050 import update_0050
    from data.scripts.update.revenue import update_revenue
    from core.model.prediction import prediction
    from core.backtest.simulator import backtester
    from data.scripts.cleaning.train import create_training
    from data.scripts.cleaning.pred import create_prediction
    from core.model.create_cbm import create_cbm
    from journal.update_journal import update_journal

except ImportError as e:
    print(f"錯誤：找不到檔案\n詳細錯誤: {e}")
    sys.exit()

# -----------------
# 系統工具定義 
# -----------------
def update_database(backfill_date: str = None) -> str:
    """
    [核心功能]: 更新交易資料庫至最新狀態。
    
    [邏輯決策規則]:
    1. 預設行為: 若使用者要求「更新資料」、「同步今日資料」、「執行每日更新」，請務必將 specific_date 設為 None。
    2. 例外行為: 只有當使用者「明確指定」某個過去的日期（例如：回補 2025-05-20、修正舊資料），才需傳入 'YYYY-MM-DD'。
    
    [參數說明]:
    specific_date: 僅用於「回補歷史」。若無明確日期，請保持預設值 None 以執行今日自動更新。
    """
    if backfill_date:
        print(f"\n回補指定日期: {backfill_date}")
    else:
        today_weekday = datetime.now().weekday()
        if today_weekday >= 5:
            return "今日為週末（非交易日），自動跳過資料更新程序。"
        print('\n執行每日更新...')
        
    print('\n更新大盤(正二/0050)....')
    update_00631(backfill_date)
    update_0050(backfill_date)
    
    print('\n更新股市....')
    update_market_data(backfill_date)
    
    target_date = backfill_date if backfill_date else "今日"
    return f"{target_date} 之資料庫已全面更新與回補完畢。"

def train_model(version: str) -> str:
    """建立訓練特徵並訓練新版本的 CBM 模型。必須輸入版本號 (例如 v11)。"""
    print(f"準備訓練版本: {version}")
    print(f"營收爬蟲...")
    update_revenue()
    create_training(version=version, years_10=True, stock_filter=True)
    create_cbm(version)
    return f"版本 {version} 特徵與模型訓練完成。"

def run_prediction(version: str, update_features: bool = False) -> str:
    """執行特定版本模型的明日預測。若 update_features 為 True 則先更新特徵。"""
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

def open_dashboard() -> str:
    """啟動 Streamlit 視覺化儀表板/戰情室/dashboard。"""
    print("啟動儀表板...")
    # /k 代表執行完後保持視窗開啟 (方便 Debug)，若要啟動後自動關閉視窗可改為 /c

    target_path = r"C:\Users\Lynn610\Documents\kitsune_stock_system"
    cmd = f'start /D "{target_path}" cmd /k "set PYTHONPATH=. && streamlit run {config.DASHBOARD_DIR}/app.py"'
    
    subprocess.Popen(cmd, shell=True)
    return "儀表板已成功在背景啟動。"

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
    你的任務：呼叫工具更新資料、訓練模型、預測、回測、開啟儀表板、以及「寫入交易紀錄」。
    接收到交易指令時（如：我買了...），請精準提取代碼、價格、股數並呼叫 add_trade_record。
    若使用者想要知道預測的公司業務，請搜尋並簡短告知(不要分析模型數據)。
    格式如下:
    3167 大量: PCB/半導體設備商
    若不知道參數，請詢問管理員。
    請簡單扼要回答，不需過度冗長。
    當使用者提到「更新資料庫」且未提及具體日期時，視為「每日例行任務」，呼叫 update_database(specific_date=None)。禁止擅自推測或填入今日日期。
    """
    
    # 2. 定義系統行為設定 (Config)
    agent_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.2, # 稍微降溫讓分析結果更嚴謹穩定
        tools=[
            update_database, 
            train_model, 
            run_prediction, 
            run_backtest, 
            open_dashboard, 
            add_trade_record,
        ]
    )
    
    # 3. 回傳 Client 本體、設定檔、以及初始化的對話視窗
    chat = client.chats.create(
        model="gemini-3.1-flash-lite-preview",
        config=agent_config
    )
    # print([m.name for m in client.models.list()])
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
    type_effect("\n>> INITIALIZING KITSUNE KERNEL...")
    time.sleep(0.3)
    type_effect(">> 自分の道を見つけるんだ")
    time.sleep(0.3)
    type_effect(">> WELCOME ", 0.05, end_new_line=False)
    type_effect("\033[1;94m[ ARCHITECT ]\033[0m", 0.05)
    time.sleep(0.3)
    type_effect("="*45, 0.01)
    
    print("""
[ 系統使用說明 ]
* 更新資料：更新今日或指定日期之 database
* 訓練模型：打包處理模型訓練 (需指定版本)
* 系統預測：更新特徵與預測 (需指定版本)
* 歷史回測：回測 (需指定版本跟基準，預設基準為 0050.TW)
* 開啟儀表：啟動 DASHBOARD 戰情室
* 寫入交易筆記
""")
    
    print(f"◆ 目前可用模型版本: {get_available_versions()}")
    print("◆ 可用 Benchmark: 0050.TW & 00631L.TW\n")

    # 初始化 AI 代理
    client, agent_config, chat = init_kitsune_agent()
    print("小狐狸初始化...")
    while True:
        user_input = input("\n請輸入>> ").strip()
        
        if not user_input:
            type_effect("程式結束。")
            time.sleep(1.5)
            break

        # === [手動指令] ===
        if user_input == "/update":
            print("\n[系統] 啟動手動強制更新流程...")
            res = update_database() # 直接呼叫函式
            print(f"結果: {res}")
            continue # 跳過後面的 AI 連線
            
        elif user_input == "/dashboard":
            print("\n[系統] 啟動手動強制開啟儀表板...")
            res = open_dashboard() # 直接呼叫函式
            print(f"結果: {res}")
            continue # 跳過後面的 AI 連線
        # ========================================

        # === [AI指令] ===
        try:  
            print("\n⛩️ Kitsune 推演中，請稍候...")
            response = chat.send_message(user_input)
            print(f"\nKitsune:\n{response.text}")
        
            if len(response.text) > 500:
                print("\n\033[93m本次分析回傳字數較多，為避免後續指令耗費過多Token，建議輸入 'c' 清空記憶。\033[0m")
        except Exception as e:
            print(f"\nAI連線異常: {e}")
if __name__ == "__main__":
    main()