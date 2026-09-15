from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
import config

def scraper(url):
    data_list = []
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36'
    options.add_argument(f'user-agent={user_agent}')
    
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(url)
        time.sleep(1)
        
        tbody = driver.find_element(By.TAG_NAME, "tbody")
        rows = tbody.find_elements(By.TAG_NAME, "tr")
        for tr in rows[1:]: 
            tds = tr.find_elements(By.TAG_NAME, "td")
            row = [tds[i].text for i in [2, 3, 4, 6]]
            data_list.append(row)
        return data_list

    except Exception as e:
        print(f"爬取出錯: {e}")
        return None
        
    finally:
        driver.quit()
        time.sleep(3)

if __name__ == "__main__":
    # market=1上市；=2上櫃

    # industry_code=
    # 24:半導體
    # 25:電腦周邊
    # 26:光電業
    # 27:通信網路
    # 28:電子零組件
    # 29:電子通路
    # 30:資訊服務
    # 31:其他電子

    final_data_list = []
    for market in range(1, 3):
        for code in ['24', '25', '26', '27', '28', '29', '30', '31']:
            url = f'https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market={market}&issuetype=&industry_code={code}&Page=1&chklike=Y'
            print(f"爬取中: market={market}, code={code}")  # 方便追蹤進度
            data = scraper(url)
            if data:
                final_data_list.extend(data)  # extend 而非 append

    df_new = pd.DataFrame(final_data_list, columns=['id', 'name', 'market', 'industry'])
    df_new['id'] = df_new['id'].astype(str)
    #去掉市場別
    df_new = df_new[~(df_new['market']=='市場別')]

    file_path = config.STOCK_WHITE_LIST_DIR / 'white_list.csv'
    if file_path.exists():
        # 讀取舊資料
        df_old = pd.read_csv(file_path, dtype={'id': str})
        
        # 將舊資料與新資料上下拼接
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        
        # 針對 id 欄位進行去重
        # keep='last' 代表若有重複，保留最後一筆（即 df_new 的最新狀態）
        # 下市股因為只存在於 df_old，沒有重複問題，會自然被保留
        df_final = df_combined.drop_duplicates(subset=['id'], keep='last')
    else:
        # 若沒有舊檔案，則直接使用新資料
        df_final = df_new

    # 存檔
    df_final.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"完成，資料庫目前共 {len(df_final)} 筆標的")