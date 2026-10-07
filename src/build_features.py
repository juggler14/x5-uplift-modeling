"""Builds data/features.parquet (one row per client) from the raw X5 RetailHero files using DuckDB."""
import os
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # paths in features.sql are relative to the project root

start = time.time()
con = duckdb.connect(str(ROOT / "data" / "x5.duckdb"))
con.execute("SET memory_limit = '4GB'; SET preserve_insertion_order = false;")
con.execute((ROOT / "src" / "features.sql").read_text())
con.execute(f"COPY features TO '{ROOT / 'data' / 'features.parquet'}' (FORMAT parquet)")

stats = con.execute("""
    SELECT (SELECT COUNT(*) FROM items)        AS receipt_lines,
           (SELECT COUNT(*) FROM transactions) AS transactions,
           (SELECT COUNT(*) FROM features)     AS clients,
           (SELECT MIN(ts) FROM transactions)  AS first_ts,
           (SELECT MAX(ts) FROM transactions)  AS last_ts
""").fetchdf()
print(stats.T.to_string(header=False))
print(f"done in {time.time() - start:.0f}s")
