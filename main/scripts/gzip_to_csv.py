import pandas as pd
import gzip
from pathlib import Path

src = Path(r"C:\Users\hibra\Downloads\archive\Loan_status_2007-2020Q3.gzip")
dst = src.with_suffix(".csv")


with open(src, "rb") as f:
    magic = f.read(2)

if magic == b"\x1f\x8b":
    # Real gzip
    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, low_memory=False)
else:
    # Plain text misnamed as .gzip
    df = pd.read_csv(src, low_memory=False)

print("Shape:", df.shape)
print(df.head())


df.to_csv(dst, index=False)
print("Saved:", dst)