"""
box_db_info.py — 檢視 box_db.npz 內容
用法：python box_db_info.py [npz_path]
"""

import sys
from collections import defaultdict
import numpy as np

npz_path = sys.argv[1] if len(sys.argv) > 1 else "box_db.npz"

data        = np.load(npz_path, allow_pickle=True)
feats       = data["feats"]
type_names  = data["type_names"].astype(str)
type_brands = data["type_brands"].astype(str)

# 統計每個 (brand, name) 有幾筆 embedding
counts: dict[tuple, int] = defaultdict(int)
for brand, name in zip(type_brands, type_names):
    counts[(brand, name)] += 1

print(f"資料庫：{npz_path}")
print(f"embedding 維度：{feats.shape[1]}")
print(f"總 embedding 數：{feats.shape[0]}")
print(f"品項數：{len(counts)}")
print()
print(f"{'#':<4} {'品牌':<15} {'品名':<25} {'embedding 數'}")
print("-" * 60)
for i, ((brand, name), count) in enumerate(counts.items(), 1):
    print(f"{i:<4} {brand:<15} {name:<25} {count}")

print(feats[0])