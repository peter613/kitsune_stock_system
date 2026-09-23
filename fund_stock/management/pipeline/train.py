import sys
from pathlib import Path

# 確保定位專案根目錄
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fund_stock.management.pipeline._features import build_dataset


def create_training(version='v51', years_10=True, stock_filter=True):
    """
    訓練特徵集建置入口
    """
    return build_dataset(mode='train', version=version, years_10=years_10, stock_filter=stock_filter)


if __name__ == "__main__":
    create_training(version='v52', years_10=True, stock_filter=True)
