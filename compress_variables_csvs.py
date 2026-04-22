import os
import pandas as pd
from pathlib import Path
 
DIR = "model_variables_csv_sparse"
 
csv_files = list(Path(DIR).glob("*.csv"))
 
if not csv_files:
    print(f"No CSV files found in '{DIR}'.")
else:
    total = len(csv_files)
    for i, filepath in enumerate(csv_files, 1):
        df = pd.read_csv(filepath)
        original_len = len(df)
        df = df[df["value"] != 0]
        df.to_csv(filepath, index=False)
        print(f"[{i}/{total}] {filepath.name}: {original_len} → {len(df)} rows")
 
    print(f"\nDone. Processed {total} file(s).")
 