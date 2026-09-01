"""Стадия 05. Панель остатков вне времени - честная база для остаточной модели.

Каждому якорю a своя база, обученная только на якорях <= a-42, фиксированные 250
раундов. Раунды выбраны без оценочных якорей: плато при допуске 2e-4 на якорях
253/281/309/337 пересекаются ровно в 250.

    z_a = log1p(y_a) - mu_a      p~_a = p_a - mean(p_a)      r_a = z_a - p~_a

Якорь считается пригодным, если его базе досталось не меньше MIN_BASE якорей:
ниже этого «остаток» - в основном невежество самой базы.

Пишет oof_panel.npz и features/oof_panel_meta.json.
"""

import paths
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

N = paths.dataset()[0]
ROUNDS = 250
PURGE = 42
MIN_BASE = 6
OUT = "oof_panel.npz"

meta = json.load(open("features/meta.json"))
meta2 = json.load(open("features/meta2.json"))
meta4 = json.load(open("features/meta4.json"))
TA = meta["train_anchors"]
ANCHOR_ONLY = {"history_days", "anchor_dow", "anchor_doy"}
C1 = [i for i, n in enumerate(meta["names"]) if n not in ANCHOR_ONLY]
C2 = [i for i, n in enumerate(meta2["names"]) if n not in set(meta2["global_names"])]
C4 = list(range(len(meta4["names"])))
NAMES = ([meta["names"][i] for i in C1] + [meta2["names"][i] for i in C2]
         + [meta4["names"][i] for i in C4])
P = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127,
         min_data_in_leaf=500, feature_fraction=0.6, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=20.0, max_bin=127, num_threads=12,
         force_row_wise=True, verbose=-1, seed=42)
if "--gpu" in sys.argv:
    probe = dict(P, device_type="gpu", gpu_use_dp=False)
    try:
        lgb.train(probe, lgb.Dataset(np.random.rand(200, 4).astype(np.float32),
                                     label=np.random.rand(200)), num_boost_round=2)
        P = probe
        print("LightGBM: сборка с GPU, device_type=gpu")
    except Exception as e:
        print(f"LightGBM собран без GPU ({str(e)[:60]}...), считаем на CPU")

X = np.load("features/X.npy", mmap_mode="r")
X2 = np.load("features/X2.npy", mmap_mode="r")
X4 = np.load("features/X4.npy", mmap_mode="r")
logy = np.log1p(np.load("features/y.npy"))
mu = np.array([float(logy[b * N:(b + 1) * N].mean()) for b in range(len(TA))])

ELIGIBLE = [a for a in TA
            if len([t for t in TA if t <= a - PURGE]) >= MIN_BASE]
print(f"eligible residual anchors ({len(ELIGIBLE)}, base >= {MIN_BASE} anchors): "
      f"{ELIGIBLE}", flush=True)


def gather(blocks):
    """Строки блока берутся СРЕЗОМ, а не массивом индексов: индексация memmap
    массивом уходит на путь случайного сбора.
    Блок якоря - непрерывный диапазон [b*N, (b+1)*N), так что срез эквивалентен."""
    idx = np.concatenate([np.arange(b * N, (b + 1) * N) for b in blocks])
    out = np.empty((len(idx), len(NAMES)), dtype=np.float32)
    for i, b in enumerate(blocks):
        lo, hi = b * N, (b + 1) * N
        r0 = i * N
        k = 0
        out[r0:r0 + N, k:k + len(C1)] = X[lo:hi][:, C1]; k += len(C1)
        out[r0:r0 + N, k:k + len(C2)] = X2[lo:hi][:, C2]; k += len(C2)
        out[r0:r0 + N, k:k + len(C4)] = X4[lo:hi][:, C4]
    gc.collect()
    return out, idx


store = dict(np.load(OUT)) if os.path.exists(OUT) else {}
t0 = time.time()
for a in ELIGIBLE:
    if f"resid_{a}" in store:
        continue
    tr = [b for b, an in enumerate(TA) if an <= a - PURGE]
    Xtr, idx = gather(tr)
    ztr = (logy[idx] - mu[idx // N]).astype(np.float32)
    m = lgb.train(P, lgb.Dataset(Xtr, label=ztr, feature_name=NAMES),
                  num_boost_round=ROUNDS)
    del Xtr, ztr
    gc.collect()
    b = TA.index(a)
    Xo, _ = gather([b])
    p = m.predict(Xo)
    del Xo, m
    gc.collect()
    z = logy[b * N:(b + 1) * N] - mu[b]
    r = z - (p - p.mean())
    store[f"pred_{a}"] = (p - p.mean()).astype(np.float32)
    store[f"resid_{a}"] = r.astype(np.float32)
    store[f"z_{a}"] = z.astype(np.float32)
    np.savez_compressed(OUT, **store)
    print(f"anchor {a}: base on {len(tr)} anchors (<= {a-PURGE}), "
          f"rmse {np.sqrt(np.mean(r**2)):.6f}  resid sd {r.std():.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

json.dump({"eligible": ELIGIBLE, "rounds": ROUNDS, "purge": PURGE,
           "min_base": MIN_BASE,
           "note": "base for anchor a trained only on anchors <= a-42"},
          open("features/oof_panel_meta.json", "w"), indent=1)
print(f"\nwrote {OUT} for {len(ELIGIBLE)} anchors ({time.time()-t0:.0f}s)")
