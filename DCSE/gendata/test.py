
import pandas as pd

load_raw_dir = "src/load/load_normalize_clean.csv"
load_raw = pd.read_csv(load_raw_dir)
print(load_raw)