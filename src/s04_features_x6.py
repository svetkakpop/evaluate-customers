"""Стадия 04. Затухания, последние покупки, максимумы окон - 79 колонок X6.

Определения всех 79 колонок выведены поколоночной сверкой, а не угаданы. Четыре
из них назывались не тем, чем являются:

    has_*        оконные СЧЁТЧИКИ, а не флаги
    wknd_gmv_*   считается на log1p(gmv)
    rank_log_*   ранг СУММЫ log1p, а не log1p суммы
    dec_ratio    (num/tau_a + e)/(den/tau_b + e), где e = 1e-3/tau_a;
                 именно e даёт пользователю без истории ровно 1.0

Плюс два порога: prc_lgmv_trend равен нулю при менее чем четырёх покупках,
prc_gap_last_over_mean - при менее чем двух.

Результат сверки: 74 колонки совпадают до точности float32, пять ранговых
расходятся на 2-5 позиций из 250 000 (0.025% на уровне корзин LightGBM).

Пишет features/X6.npy и meta6.json.
"""
import gc
import json
import sys
import time

import numpy as np
import polars as pl

import paths

SENT = 999.0
HORIZON = 30
TAUS = (7, 21, 60, 180)
EPS = 1e-3
VERIFY = "--verify" in sys.argv

t0 = time.time()
df = pl.read_parquet("train.parquet").sort(["user_id", "event_date"])
day_raw = df["event_date"].to_physical().to_numpy().astype(np.int64)
DAY0 = int(day_raw.min())
day = (day_raw - DAY0).astype(np.int32)
MAX_DAY = int(day.max())
uniq, ur = np.unique(df["user_id"].to_numpy(), return_inverse=True)
ur = ur.astype(np.int64)
n = len(uniq)
KEY = ur * 1024 + day
u_base = np.arange(n, dtype=np.int64) * 1024

gmv = df["gmv"].to_numpy().astype(np.float64)
lgmv = np.log1p(gmv)
to_ord = df["to_ord"].to_numpy().astype(np.float64)
searches = df["searches"].to_numpy().astype(np.float64)
act = np.ones(len(day), dtype=np.float64)
d_gmv = (gmv > 0).astype(np.float64)
HAS = {k: df[k].to_numpy().astype(np.float64) for k in
       ("has_search_to_ord", "has_cat_to_ord",
        "has_search_to_cart", "has_cat_to_cart")}
dow = (day + DAY0) % 7
wknd = ((dow == 5) | (dow == 6)).astype(np.float64)
del df, day_raw
gc.collect()

META = {"gmv": gmv, "log_gmv": lgmv, "d_gmv": d_gmv, "to_ord": to_ord,
        "searches": searches, "act": act}

names = []
for tau in TAUS:
    names += [f"dec{tau}_{m}" for m in META]
for m, a, b in (("gmv", 7, 60), ("log_gmv", 7, 60), ("act", 7, 60),
                ("gmv", 21, 180), ("log_gmv", 21, 180), ("act", 21, 180)):
    names.append(f"dec_ratio_{m}_{a}_{b}")
for i in range(1, 6):
    names += [f"prc_rec_{i}", f"prc_lgmv_{i}"]
names += [f"prc_gap_{i}" for i in range(1, 5)]
names += ["prc_gap_mean", "prc_gap_std", "prc_gap_last_over_mean",
          "prc_lgmv_mean5", "prc_lgmv_max5", "prc_lgmv_trend"]
names += [f"act_rec_{i}" for i in range(1, 4)] + [f"act_gap_{i}" for i in (1, 2)]
SHORT = ("search_to_ord", "cat_to_ord", "search_to_cart", "cat_to_cart")
for tag in ("30d", "90d"):
    names += [f"max_lgmv_{tag}", f"max_ord_{tag}", f"max_srch_{tag}"]
    names += [f"has_{s}_{tag}" for s in SHORT]
names += ["wknd_act_share_90d", "wknd_gmv_share_90d",
          "rank_log_gmv_90d", "rank_log_gmv_30d", "rank_log_gmv_365d",
          "rank_gmv_all", "rank_days_since_gmv", "rank_active_days_90d",
          "rank_to_ord_90d", "rank_dec21_log_gmv"]
col = {c: i for i, c in enumerate(names)}
assert len(names) == 79, len(names)

TA = json.load(open("features/meta.json"))["train_anchors"]
ALL = TA + [MAX_DAY]
K = len(ALL)
print(f"users {n}  rows {len(day)}  anchors {K}  columns 79  "
      f"load {time.time()-t0:.0f}s", flush=True)

X = np.lib.format.open_memmap("features/X6.npy", mode="w+",
                              dtype=np.float32, shape=(n * K, 79))


def rank01(v):
    o = np.argsort(v, kind="stable")
    r = np.empty(n)
    r[o] = np.arange(n)
    return r / (n - 1)


for ai, A in enumerate(ALL):
    ta = time.time()
    B = np.zeros((n, 79), dtype=np.float64)
    upto = day <= A
    pos = np.searchsorted(KEY, u_base + A + 1, "left")
    start = np.searchsorted(KEY, u_base, "left")

    def wsum(v, lo_day):
        g = np.where(upto & (day >= lo_day), v, 0.0)
        cs = np.concatenate([[0.0], np.cumsum(g)])
        return cs[pos] - cs[np.searchsorted(KEY, u_base + lo_day, "left")]

    def wmax(v, lo_day):
        sel = np.flatnonzero(upto & (day >= lo_day))
        out = np.zeros(n)
        if sel.size:
            us = ur[sel]
            st = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
            out[us[st]] = np.maximum.reduceat(v[sel], st)
        return out

    DEC = {}
    for tau in TAUS:
        w = np.exp(-(A - day) / tau)
        for m, v in META.items():
            g = np.where(upto, v * w, 0.0)
            cs = np.concatenate([[0.0], np.cumsum(g)])
            DEC[(tau, m)] = cs[pos] - cs[start]
            B[:, col[f"dec{tau}_{m}"]] = DEC[(tau, m)]
    for m, a, b in (("gmv", 7, 60), ("log_gmv", 7, 60), ("act", 7, 60),
                    ("gmv", 21, 180), ("log_gmv", 21, 180), ("act", 21, 180)):
        e = EPS / a
        B[:, col[f"dec_ratio_{m}_{a}_{b}"]] = \
            (DEC[(a, m)] / a + e) / (DEC[(b, m)] / b + e)

    pm = (gmv > 0) & upto
    pk, pday, plg = KEY[pm], day[pm], lgmv[pm]
    pe = np.searchsorted(pk, u_base + 1024, "left")
    ps = np.searchsorted(pk, u_base, "left")
    cnt = pe - ps
    rec = np.full((n, 5), SENT)
    lg5 = np.zeros((n, 5))
    for i in range(5):
        ok = cnt > i
        idx = pe[ok] - 1 - i
        rec[ok, i] = A - pday[idx]
        lg5[ok, i] = plg[idx]
        B[:, col[f"prc_rec_{i+1}"]] = rec[:, i]
        B[:, col[f"prc_lgmv_{i+1}"]] = lg5[:, i]
    gaps = np.full((n, 4), SENT)
    for i in range(4):
        ok = cnt > i + 1
        gaps[ok, i] = rec[ok, i + 1] - rec[ok, i]
        B[:, col[f"prc_gap_{i+1}"]] = gaps[:, i]
    gm = np.where(gaps == SENT, np.nan, gaps)
    with np.errstate(invalid="ignore"):
        gmean = np.nan_to_num(np.nanmean(gm, 1), nan=0.0)
        gstd = np.nan_to_num(np.nanstd(gm, 1), nan=0.0)
        lmn = np.where(rec == SENT, np.nan, lg5)
        lmean = np.nan_to_num(np.nanmean(lmn, 1), nan=0.0)
        lmax = np.nan_to_num(np.nanmax(lmn, 1), nan=0.0)
    B[:, col["prc_gap_mean"]] = np.where(cnt >= 2, gmean, SENT)
    B[:, col["prc_gap_std"]] = gstd
    B[:, col["prc_gap_last_over_mean"]] = np.where(
        cnt >= 2, rec[:, 0] / np.maximum(gmean, 1e-12), 0.0)
    B[:, col["prc_lgmv_mean5"]] = lmean
    B[:, col["prc_lgmv_max5"]] = lmax
    B[:, col["prc_lgmv_trend"]] = np.where(
        cnt >= 4, (lg5[:, 0] + lg5[:, 1]) - (lg5[:, 2] + lg5[:, 3]), 0.0)

    acnt = pos - start
    arec = np.full((n, 3), SENT)
    for i in range(3):
        ok = acnt > i
        arec[ok, i] = A - day[pos[ok] - 1 - i]
        B[:, col[f"act_rec_{i+1}"]] = arec[:, i]
    for i in range(2):
        g = np.full(n, SENT)
        ok = acnt > i + 1
        g[ok] = arec[ok, i + 1] - arec[ok, i]
        B[:, col[f"act_gap_{i+1}"]] = g

    for L, tag in ((30, "30d"), (90, "90d")):
        lo = A - L + 1
        B[:, col[f"max_lgmv_{tag}"]] = wmax(lgmv, lo)
        B[:, col[f"max_ord_{tag}"]] = wmax(to_ord, lo)
        B[:, col[f"max_srch_{tag}"]] = wmax(searches, lo)
        for s in SHORT:
            B[:, col[f"has_{s}_{tag}"]] = wsum(HAS[f"has_{s}"], lo)

    lo90 = A - 89
    na, ng = wsum(act, lo90), wsum(gmv, lo90)
    B[:, col["wknd_act_share_90d"]] = np.where(
        na > 0, wsum(wknd, lo90) / np.maximum(na, 1e-12), 0.0)
    nl = wsum(lgmv, lo90)
    B[:, col["wknd_gmv_share_90d"]] = np.where(
        nl > 0, wsum(lgmv * wknd, lo90) / np.maximum(nl, 1e-12), 0.0)

    B[:, col["rank_log_gmv_90d"]] = rank01(wsum(lgmv, A - 89))
    B[:, col["rank_log_gmv_30d"]] = rank01(wsum(lgmv, A - 29))
    B[:, col["rank_log_gmv_365d"]] = rank01(wsum(lgmv, A - 364))
    B[:, col["rank_gmv_all"]] = rank01(wsum(gmv, 0))
    B[:, col["rank_days_since_gmv"]] = rank01(rec[:, 0])
    B[:, col["rank_active_days_90d"]] = rank01(wsum(act, A - 89))
    B[:, col["rank_to_ord_90d"]] = rank01(wsum(to_ord, A - 89))
    B[:, col["rank_dec21_log_gmv"]] = rank01(DEC[(21, "log_gmv")])

    X[ai * n:(ai + 1) * n] = B.astype(np.float32)
    print(f"  anchor {A:3d} ({ai+1}/{K})  {time.time()-ta:.0f}s", flush=True)
    del B
    gc.collect()

X.flush()
json.dump({"names": names}, open("features/meta6.json", "w"))
print(f"\nwrote features/X6.npy {X.shape} and features/meta6.json "
      f"({time.time()-t0:.0f}s)")
