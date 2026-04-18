from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import datetime
import os
import pandas as pd
import config

def update_gold():
    now = datetime.now().strftime('%Y-%m-%d')

    file_path = config.DATABASE_DIR/"gold.csv"

    def crawler():
        url = "https://ebank.taipeifubon.com.tw/B2C/cfhqu/cfhqu002/CFHQU002_Home.faces"
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        now_price = driver.find_elements(By.XPATH, '//*[@id="form1:today_DataGridBody"]/tbody/tr[4]/td[2]')
        now_price = int(float(now_price[0].text.replace(',', '')))
        return now_price

    current_price = crawler()

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='utf-8')
    else:
        print('檔案不存在')

    if not df.empty:
        df = df[df['date'] != now]

    # 建立新資料列
    new_row = pd.DataFrame({'date': [now], 'price': [current_price]})
    df = pd.concat([df, new_row], ignore_index=True)

    # 4. 存回 CSV
    df.to_csv(file_path, index=False, encoding='utf-8')

    print(f'黃金資料庫更新成功! 現價{current_price}')
    print(df.tail(3))

if __name__ == "__main__":
    update_gold()

