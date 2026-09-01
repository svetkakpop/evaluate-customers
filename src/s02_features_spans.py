"""Стадия 02. Непересекающиеся интервалы.

Оконные суммы вложены друг в друга: 90-дневная включает 30-дневную. Здесь те же
метрики берутся по непересекающимся интервалам (0-7, 8-30, 31-90, 91-180,
181-365), поэтому деревьям не приходится вычитать одну колонку из другой, чтобы
увидеть, что происходило именно в том промежутке.

Пишет features/X2.npy и meta2.json.
"""

import paths
import gc
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

OUT_DIR = Path("features")
meta = json.load(open(OUT_DIR / "meta.json"))
TRAIN_ANCHORS = meta["train_anchors"]
FINAL_ANCHOR = meta["final_anchor"]
ALL_ANCHORS = TRAIN_ANCHORS + [FINAL_ANCHOR]
HORIZON = 30

t_start = time.time()
df = pl.read_parquet("train.parquet").sort(["user_id", "event_date"])
N = df.height
day_raw = df["event_date"].to_physical().to_numpy().astype(np.int64)
DAY0 = int(day_raw.min())
day_idx = (day_raw - DAY0).astype(np.int32)
MAX_DAY = int(day_idx.max())
del day_raw

uniq_users, user_rank = np.unique(df["user_id"].to_numpy(), return_inverse=True)
user_rank = user_rank.astype(np.int32)
n_users = len(uniq_users)

gmv = df["gmv"].to_numpy().astype(np.float64)
metrics = {
    "gmv": gmv,
    "log_gmv": np.log1p(gmv),
    "log_gmv_sq": np.log1p(gmv) ** 2,
    "to_ord": df["to_ord"].to_numpy().astype(np.int32),
    "to_cart": df["to_cart"].to_numpy().astype(np.int32),
    "searches": df["searches"].to_numpy().astype(np.int32),
    "d_gmv": (gmv > 0).astype(np.int32),
}
del df, gmv
gc.collect()

CS = {}
for k, v in metrics.items():
    dt = np.float64 if v.dtype.kind == "f" else np.int32
    o = np.empty(N + 1, dtype=dt)
    o[0] = 0
    np.cumsum(v, dtype=dt, out=o[1:])
    CS[k] = o
del metrics
gc.collect()

KEY = user_rank * np.int32(1024) + day_idx
u_all = np.arange(n_users, dtype=np.int32)
u_base = u_all * np.int32(1024)
del user_rank, day_idx
gc.collect()
print(f"prep {time.time()-t_start:.1f}s", flush=True)

SEGMENTS = [("s0_7", 0, 6), ("s7_14", 7, 13), ("s14_30", 14, 29), ("s30_60", 30, 59),
            ("s60_90", 60, 89), ("s90_180", 90, 179), ("s180_365", 180, 364)]
SEG_METRICS = ["gmv", "log_gmv", "to_ord", "searches", "d_gmv"]
GLOBAL_NAMES = ["g_gmv_30", "g_gmv_prev30", "g_gmv_ratio", "g_active_30",
                "g_buyers_30", "g_gmv_90", "g_gmv_30_over_90"]

names2 = []
for sname, _, _ in SEGMENTS:
    for m in SEG_METRICS:
        names2.append(f"{m}_{sname}")
    names2.append(f"active_days_{sname}")
names2 += ["logmv_std_30d", "logmv_std_90d", "logmv_std_all"]
names2 += GLOBAL_NAMES
names2 += ["gmv30_over_global", "gmv90_over_global", "logmv30_over_global"]
c2 = {n: i for i, n in enumerate(names2)}
n_feat2 = len(names2)

K = len(ALL_ANCHORS)
X2 = np.zeros((n_users * K, n_feat2), dtype=np.float32)
print(f"extra features={n_feat2} rows={n_users*K}", flush=True)


def win(lo_day, hi_day):
    lo = np.searchsorted(KEY, u_base + max(lo_day, 0), "left")
    hi = np.searchsorted(KEY, u_base + min(hi_day, MAX_DAY) + 1, "left")
    return lo, hi


for ai, a in enumerate(ALL_ANCHORS):
    r0, r1 = ai * n_users, (ai + 1) * n_users

    for sname, off_lo, off_hi in SEGMENTS:
        lo, hi = win(a - off_hi, a - off_lo)
        for m in SEG_METRICS:
            C = CS[m]
            X2[r0:r1, c2[f"{m}_{sname}"]] = C[hi] - C[lo]
        X2[r0:r1, c2[f"active_days_{sname}"]] = hi - lo

    for wname, L in [("30d", 30), ("90d", 90), ("all", 10_000)]:
        lo, hi = win(a - L + 1, a)
        n = (hi - lo).astype(np.float64)
        s1 = CS["log_gmv"][hi] - CS["log_gmv"][lo]
        s2 = CS["log_gmv_sq"][hi] - CS["log_gmv_sq"][lo]
        var = np.maximum(s2 / np.maximum(n, 1) - (s1 / np.maximum(n, 1)) ** 2, 0.0)
        X2[r0:r1, c2[f"logmv_std_{wname}"]] = np.where(n > 1, np.sqrt(var), 0.0)

    lo30, hi30 = win(a - 29, a)
    lo60, hi60 = win(a - 59, a - 30)
    lo90, hi90 = win(a - 89, a)
    g30 = float((CS["gmv"][hi30] - CS["gmv"][lo30]).sum()) / n_users
    gp30 = float((CS["gmv"][hi60] - CS["gmv"][lo60]).sum()) / n_users
    g90 = float((CS["gmv"][hi90] - CS["gmv"][lo90]).sum()) / n_users
    gact = float((hi30 - lo30).sum()) / n_users
    gbuy = float((CS["d_gmv"][hi30] - CS["d_gmv"][lo30]).sum()) / n_users
    for nm, v in zip(GLOBAL_NAMES,
                     [g30, gp30, g30 / max(gp30, 1e-6), gact, gbuy, g90, g30 / max(g90 / 3, 1e-6)]):
        X2[r0:r1, c2[nm]] = v

    ugmv30 = (CS["gmv"][hi30] - CS["gmv"][lo30]).astype(np.float32)
    ugmv90 = (CS["gmv"][hi90] - CS["gmv"][lo90]).astype(np.float32)
    ulog30 = (CS["log_gmv"][hi30] - CS["log_gmv"][lo30]).astype(np.float32)
    X2[r0:r1, c2["gmv30_over_global"]] = ugmv30 / max(g30, 1e-6)
    X2[r0:r1, c2["gmv90_over_global"]] = ugmv90 / max(g90, 1e-6)
    X2[r0:r1, c2["logmv30_over_global"]] = ulog30 / max(g30, 1e-6)

    print(f"  anchor {a} ({time.time()-t_start:.1f}s) g30={g30:.2f}", flush=True)

np.nan_to_num(X2, copy=False, nan=0.0, posinf=1e7, neginf=-1e7)
np.save(OUT_DIR / "X2.npy", X2)
json.dump({"names": names2, "global_names": GLOBAL_NAMES}, open(OUT_DIR / "meta2.json", "w"))
print(f"saved X2{X2.shape} {time.time()-t_start:.1f}s", flush=True)
