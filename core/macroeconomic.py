from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

def sp_500_pe():
    url = 'https://www.multpl.com/s-p-500-pe-ratio'
    options = webdriver.ChromeOptions()  
    options.add_argument("--headless") #全螢幕
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    pe = driver.find_element(By.ID,'current')

    pe_stock_market = float(pe.text.split(' ')[5])
    f = 1/pe_stock_market-0.045
    driver.quit()
    print(f'現在s&p500 pe : {pe_stock_market}')
    # print(f'{'買股票優於買債券' if f>0 else '買債券優於買股票'} {f:.2%}')

    