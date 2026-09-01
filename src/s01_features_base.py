"""Стадия 01. Агрегаты по окнам - базовая матрица признаков.

Данные отсортированы по (user_id, event_date). Строится глобальный ключ

    KEY = user_rank * 1024 + day_idx

так что окно [lo, hi] сразу для всех пользователей находится двумя вызовами
np.searchsorted по всему массиву из 30.6 млн строк, а суммы по окну берутся из
глобальных кумулятивных сумм. Цикла по пользователям нет вообще.

Восемь окон (7/14/30/60/90/180/365/весь период) на пятнадцать метрик, плюс
давности, максимумы и производные отношения.

Строки сгруппированы по якорю: блок a занимает [a*n_users, (a+1)*n_users).

Пишет features/X.npy, y.npy, meta.json, anchor.npy, user_id.npy.
"""

import paths
import gc
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

DATA_PATH = "train.parquet"
OUT_DIR = Path("features")
OUT_DIR.mkdir(exist_ok=True)

HORIZON = 30
STRIDE = 14
FIRST_ANCHOR = 43
SENTINEL = 999.0

t_start = time.time()

df = pl.read_parquet(DATA_PATH).sort(["user_id", "event_date"])
N = df.height

day_raw = df["event_date"].to_physical().to_numpy().astype(np.int64)
DAY0 = int(day_raw.min())
day_idx = (day_raw - DAY0).astype(np.int32)
MAX_DAY = int(day_idx.max())
del day_raw

uid_raw = df["user_id"].to_numpy()
uniq_users, user_rank = np.unique(uid_raw, return_inverse=True)
user_rank = user_rank.astype(np.int32)
n_users = len(uniq_users)
uniq_users = uniq_users.astype(np.int64)
del uid_raw

print(f"rows={N} users={n_users} max_day={MAX_DAY} load={time.time()-t_start:.1f}s", flush=True)

gmv = df["gmv"].to_numpy().astype(np.float64)
raw = {
    "gmv": gmv,
    "gmv_search": df["gmv_search"].to_numpy().astype(np.float64),
    "gmv_cat": df["gmv_cat"].to_numpy().astype(np.float64),
    "log_gmv": np.log1p(gmv),
    "searches": df["searches"].to_numpy().astype(np.int32),
    "to_ord": df["to_ord"].to_numpy().astype(np.int32),
    "to_cart": df["to_cart"].to_numpy().astype(np.int32),
    "s2ord": df["search_to_ord"].to_numpy().astype(np.int32),
    "c2ord": df["cat_to_ord"].to_numpy().astype(np.int32),
    "s2cart": df["search_to_cart"].to_numpy().astype(np.int32),
    "c2cart": df["cat_to_cart"].to_numpy().astype(np.int32),
    "d_gmv": (gmv > 0).astype(np.int32),
    "d_ord": (df["to_ord"].to_numpy() > 0).astype(np.int32),
    "d_search": df["search"].to_numpy().astype(np.int32),
    "d_cat": df["cat"].to_numpy().astype(np.int32),
}
del df
gc.collect()

FLOAT_METRICS = {"gmv", "gmv_search", "gmv_cat", "log_gmv"}

CS = {}
for name, arr in raw.items():
    dt = np.float64 if name in FLOAT_METRICS else np.int32
    out = np.empty(N + 1, dtype=dt)
    out[0] = 0
    np.cumsum(arr, dtype=dt, out=out[1:])
    CS[name] = out
print(f"cumsums built {time.time()-t_start:.1f}s", flush=True)

OFF = user_rank * np.int32(1024)
cm_ord = np.maximum.accumulate(OFF + np.where(raw["d_ord"] > 0, day_idx, np.int32(-1)))
cm_gmv = np.maximum.accumulate(OFF + np.where(raw["d_gmv"] > 0, day_idx, np.int32(-1)))
BIG = 1e7
cmax_gmv = np.maximum.accumulate(user_rank * BIG + gmv)

KEY = OFF + day_idx
first_day = day_idx[np.searchsorted(KEY, np.arange(n_users, dtype=np.int32) * 1024, "left")]

del raw, gmv, OFF
gc.collect()
print(f"helpers built {time.time()-t_start:.1f}s", flush=True)

u_all = np.arange(n_users, dtype=np.int32)
u_base = u_all * np.int32(1024)
latest_anchor = MAX_DAY - HORIZON
TRAIN_ANCHORS = list(range(FIRST_ANCHOR, latest_anchor + 1, STRIDE))
if TRAIN_ANCHORS[-1] != latest_anchor:
    TRAIN_ANCHORS.append(latest_anchor)
FINAL_ANCHOR = MAX_DAY
ALL_ANCHORS = TRAIN_ANCHORS + [FINAL_ANCHOR]
K = len(ALL_ANCHORS)
print(f"{K-1} train anchors {TRAIN_ANCHORS[0]}..{TRAIN_ANCHORS[-1]} + final {FINAL_ANCHOR}", flush=True)

FULL_METRICS = ["gmv", "gmv_search", "gmv_cat", "log_gmv", "searches", "to_ord",
                "to_cart", "s2ord", "c2ord", "s2cart", "c2cart",
                "d_gmv", "d_ord", "d_search", "d_cat"]
SLIM_METRICS = ["gmv", "gmv_search", "log_gmv", "searches", "to_ord", "d_gmv"]

WINDOWS = [("7d", 7), ("14d", 14), ("30d", 30), ("60d", 60), ("90d", 90),
           ("180d", 180), ("365d", 365), ("all", 10_000)]
WIN_METRICS = {w: (FULL_METRICS if L <= 90 else SLIM_METRICS) for w, L in WINDOWS}
WIN_LEN = dict(WINDOWS)

names = []
for wname, _ in WINDOWS:
    for m in WIN_METRICS[wname]:
        names.append(f"{m}_{wname}")
    names.append(f"active_days_{wname}")
names += ["days_since_active", "days_since_ord", "days_since_gmv", "days_since_first",
          "max_gmv_ever", "history_days", "anchor_dow", "anchor_doy"]
n_raw = len(names)

derived = []
for wname, L in WINDOWS:
    ms = WIN_METRICS[wname]
    derived += [f"aov_{wname}", f"gmv_per_day_{wname}", f"logmv_per_day_{wname}",
                f"buy_day_rate_{wname}", f"search_gmv_share_{wname}", f"activity_rate_{wname}"]
    if "to_cart" in ms:
        derived += [f"cart2ord_{wname}", f"cart_per_day_{wname}", f"search_ord_share_{wname}"]
    derived += [f"searches_per_day_{wname}", f"ord_per_buy_day_{wname}"]
derived += ["trend_7_30", "trend_14_60", "trend_30_90", "trend_90_365",
            "act_trend_7_30", "act_trend_30_90",
            "gmv_share_30_of_all", "gmv_share_90_of_all",
            "ord_share_30_of_all", "recency_over_freq", "gmv_all_per_hist_day",
            "ever_bought", "gap_ratio", "max_over_mean_gmv"]

all_names = names + derived
n_feat = len(all_names)
col = {n: i for i, n in enumerate(all_names)}
name_col = col

TOTAL_ROWS = n_users * K
print(f"raw features={n_raw}  total features={n_feat}  rows={TOTAL_ROWS}", flush=True)

Xraw = np.zeros((TOTAL_ROWS, n_feat), dtype=np.float32)
y = np.full(TOTAL_ROWS, np.nan, dtype=np.float32)
out_anchor = np.empty(TOTAL_ROWS, dtype=np.int32)
out_uid = np.empty(TOTAL_ROWS, dtype=np.int32)

for ai, a in enumerate(ALL_ANCHORS):
    lo_row, hi_row = ai * n_users, (ai + 1) * n_users
    out_anchor[lo_row:hi_row] = a
    out_uid[lo_row:hi_row] = uniq_users

    pos_anchor = np.searchsorted(KEY, u_base + min(a, MAX_DAY) + 1, "left")

    for wname, L in WINDOWS:
        lo_day = max(a - L + 1, 0)
        lo_pos = np.searchsorted(KEY, u_base + lo_day, "left")
        hi_pos = pos_anchor
        for m in WIN_METRICS[wname]:
            C = CS[m]
            Xraw[lo_row:hi_row, name_col[f"{m}_{wname}"]] = C[hi_pos] - C[lo_pos]
        Xraw[lo_row:hi_row, name_col[f"active_days_{wname}"]] = hi_pos - lo_pos

    has_hist = pos_anchor > np.searchsorted(KEY, u_base, "left")
    p = np.maximum(pos_anchor - 1, 0)

    last_active = np.where(has_hist, day_idx[p], -1)
    last_ord = np.where(has_hist, cm_ord[p] - u_base, -1)
    last_gmv = np.where(has_hist, cm_gmv[p] - u_base, -1)
    mx_gmv = np.where(has_hist, cmax_gmv[p] - u_all * BIG, 0.0)

    Xraw[lo_row:hi_row, name_col["days_since_active"]] = np.where(last_active >= 0, a - last_active, SENTINEL)
    Xraw[lo_row:hi_row, name_col["days_since_ord"]] = np.where(last_ord >= 0, a - last_ord, SENTINEL)
    Xraw[lo_row:hi_row, name_col["days_since_gmv"]] = np.where(last_gmv >= 0, a - last_gmv, SENTINEL)
    Xraw[lo_row:hi_row, name_col["days_since_first"]] = np.where(has_hist, np.maximum(a - first_day, 0), SENTINEL)
    Xraw[lo_row:hi_row, name_col["max_gmv_ever"]] = mx_gmv
    Xraw[lo_row:hi_row, name_col["history_days"]] = a + 1
    Xraw[lo_row:hi_row, name_col["anchor_dow"]] = (a + DAY0) % 7
    Xraw[lo_row:hi_row, name_col["anchor_doy"]] = a % 365

    if a != FINAL_ANCHOR:
        t_lo = np.searchsorted(KEY, u_base + a + 1, "left")
        t_hi = np.searchsorted(KEY, u_base + a + HORIZON + 1, "left")
        y[lo_row:hi_row] = CS["gmv"][t_hi] - CS["gmv"][t_lo]

    print(f"  anchor {a} done ({time.time()-t_start:.1f}s)", flush=True)

del CS, cm_ord, cm_gmv, cmax_gmv, KEY, day_idx, user_rank
gc.collect()
print(f"raw matrix built {time.time()-t_start:.1f}s", flush=True)

X = Xraw
EPS = np.float32(1e-3)
rate = {}
act_rate = {}
for wname, L in WINDOWS:
    ms = WIN_METRICS[wname]
    g = X[:, col[f"gmv_{wname}"]]
    lg = X[:, col[f"log_gmv_{wname}"]]
    gs = X[:, col[f"gmv_search_{wname}"]]
    o = X[:, col[f"to_ord_{wname}"]]
    sr = X[:, col[f"searches_{wname}"]]
    ad = X[:, col[f"active_days_{wname}"]]
    bd = X[:, col[f"d_gmv_{wname}"]]
    eff_len = np.float32(L) if L < 10_000 else np.maximum(X[:, col["history_days"]], 1)

    X[:, col[f"aov_{wname}"]] = g / np.maximum(o, 1)
    X[:, col[f"gmv_per_day_{wname}"]] = g / np.maximum(ad, 1)
    X[:, col[f"logmv_per_day_{wname}"]] = lg / np.maximum(ad, 1)
    X[:, col[f"buy_day_rate_{wname}"]] = bd / np.maximum(ad, 1)
    X[:, col[f"search_gmv_share_{wname}"]] = gs / np.maximum(g, EPS)
    X[:, col[f"activity_rate_{wname}"]] = ad / eff_len
    X[:, col[f"searches_per_day_{wname}"]] = sr / np.maximum(ad, 1)
    X[:, col[f"ord_per_buy_day_{wname}"]] = o / np.maximum(bd, 1)
    if "to_cart" in ms:
        c = X[:, col[f"to_cart_{wname}"]]
        X[:, col[f"cart2ord_{wname}"]] = o / np.maximum(c, 1)
        X[:, col[f"cart_per_day_{wname}"]] = c / np.maximum(ad, 1)
        X[:, col[f"search_ord_share_{wname}"]] = X[:, col[f"s2ord_{wname}"]] / np.maximum(o, 1)
    rate[wname] = g / eff_len
    act_rate[wname] = ad / eff_len

X[:, col["trend_7_30"]] = (rate["7d"] + EPS) / (rate["30d"] + EPS)
X[:, col["trend_14_60"]] = (rate["14d"] + EPS) / (rate["60d"] + EPS)
X[:, col["trend_30_90"]] = (rate["30d"] + EPS) / (rate["90d"] + EPS)
X[:, col["trend_90_365"]] = (rate["90d"] + EPS) / (rate["365d"] + EPS)
X[:, col["act_trend_7_30"]] = (act_rate["7d"] + EPS) / (act_rate["30d"] + EPS)
X[:, col["act_trend_30_90"]] = (act_rate["30d"] + EPS) / (act_rate["90d"] + EPS)

gmv_all = X[:, col["gmv_all"]]
X[:, col["gmv_share_30_of_all"]] = X[:, col["gmv_30d"]] / np.maximum(gmv_all, EPS)
X[:, col["gmv_share_90_of_all"]] = X[:, col["gmv_90d"]] / np.maximum(gmv_all, EPS)
X[:, col["ord_share_30_of_all"]] = X[:, col["to_ord_30d"]] / np.maximum(X[:, col["to_ord_all"]], 1)
X[:, col["recency_over_freq"]] = X[:, col["days_since_gmv"]] / np.maximum(X[:, col["d_gmv_all"]], 1)
X[:, col["gmv_all_per_hist_day"]] = gmv_all / np.maximum(X[:, col["history_days"]], 1)
X[:, col["ever_bought"]] = (gmv_all > 0).astype(np.float32)
mean_gap = np.maximum(X[:, col["days_since_first"]], 1) / np.maximum(X[:, col["d_gmv_all"]], 1)
X[:, col["gap_ratio"]] = X[:, col["days_since_gmv"]] / np.maximum(mean_gap, 1)
X[:, col["max_over_mean_gmv"]] = X[:, col["max_gmv_ever"]] / np.maximum(
    gmv_all / np.maximum(X[:, col["d_gmv_all"]], 1), EPS)

np.nan_to_num(X, copy=False, nan=0.0, posinf=1e7, neginf=-1e7)

np.save(OUT_DIR / "X.npy", X)
np.save(OUT_DIR / "y.npy", y)
np.save(OUT_DIR / "anchor.npy", out_anchor)
np.save(OUT_DIR / "user_id.npy", out_uid)
json.dump({"names": all_names, "n_users": n_users,
           "final_anchor": FINAL_ANCHOR, "day0": DAY0,
           "train_anchors": TRAIN_ANCHORS}, open(OUT_DIR / "meta.json", "w"))

print(f"saved X{X.shape}  total {time.time()-t_start:.1f}s", flush=True)
