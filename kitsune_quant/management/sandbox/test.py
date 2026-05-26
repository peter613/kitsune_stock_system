import pandas as pd
import config
df = pd.read_parquet(config.DATABASE_DIR/'market.parquet')
print(df.head(2))