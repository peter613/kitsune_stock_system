import sys
from pathlib import Path

# 確保定位專案根目錄
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fund_stock.management.pipeline._features import build_dataset


def create_prediction(stock_filter=True, specified_date=None):
    """
    預測特徵集建置入口
    """
    return build_dataset(mode='pred', stock_filter=stock_filter, specified_date=specified_date)


if __name__ == "__main__":
    create_prediction(specified_date='20260910')
