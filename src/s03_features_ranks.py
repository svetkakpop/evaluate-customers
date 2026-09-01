"""Стадия 03. Перцентильные ранги внутри якоря.

Таргет центрируется по каждому якорю, а признаки оставались абсолютными: модель
экстраполировала тренд роста платформы. Ранг внутри якоря делает признак
инвариантным к дрейфу уровня - это не монотонное преобразование одной колонки,
к которому деревья невосприимчивы, а своё отображение в каждом якоре.

Побочный эффект оказался крупнее прямого: ранги лежат в [0,1] и это идеальный
вход для MLP, который после них почти сравнялся с GBDT и наконец дал ансамблю
разнообразие.

Пишет features/X4.npy и meta4.json.
"""

import paths
import json
import time
from pathlib import Path

import numpy as np

OUT = Path("features")
meta = json.load(open(OUT / "meta.json"))
NAMES = meta["names"]
N_USERS = paths.dataset()[0]

RANK_FEATURES = [
    "log_gmv_30d", "log_gmv_60d", "log_gmv_90d", "log_gmv_180d", "log_gmv_365d", "log_gmv_all",
    "gmv_30d", "gmv_90d", "gmv_all",
    "to_ord_60d", "to_ord_90d", "to_ord_180d", "to_ord_all", "to_cart_90d",
    "d_gmv_30d", "d_gmv_90d", "d_gmv_all", "d_ord_90d",
    "searches_7d", "searches_14d", "searches_90d",
    "active_days_14d", "active_days_30d", "active_days_90d", "active_days_all",
    "days_since_active", "days_since_gmv", "days_since_ord", "days_since_first",
    "aov_90d", "gmv_all_per_hist_day", "recency_over_freq",
    "log_gmv_7d", "log_gmv_14d", "gmv_60d", "to_ord_30d",
    "gmv_search_30d", "gmv_search_90d", "to_cart_30d", "d_gmv_60d",
    "d_ord_30d", "d_ord_60d", "searches_30d", "searches_60d",
    "active_days_7d", "active_days_60d", "active_days_180d", "max_gmv_ever",
]
assert len(set(RANK_FEATURES)) == len(RANK_FEATURES), "duplicate rank source"
SRC = [NAMES.index(n) for n in RANK_FEATURES]
names4 = [f"rank_{n}" for n in RANK_FEATURES]

X = np.load(OUT / "X.npy", mmap_mode="r")
n_rows = X.shape[0]
n_blocks = n_rows // N_USERS
X4 = np.empty((n_rows, len(SRC)), dtype=np.float32)

t0 = time.time()
for b in range(n_blocks):
    lo, hi = b * N_USERS, (b + 1) * N_USERS
    blk = np.asarray(X[lo:hi])
    for j, c in enumerate(SRC):
        v = blk[:, c]
        s = np.sort(v)
        X4[lo:hi, j] = np.searchsorted(s, v, "left") / N_USERS
    print(f"  block {b}/{n_blocks} ({time.time()-t0:.0f}s)", flush=True)

np.save(OUT / "X4.npy", X4)
json.dump({"names": names4, "source": RANK_FEATURES}, open(OUT / "meta4.json", "w"))
print(f"saved X4{X4.shape} {time.time()-t0:.1f}s", flush=True)
