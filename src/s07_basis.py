"""Стадия 07. Шесть направлений - локальный базис для Stage C.

Все обучаются на якорях 43..337 и предсказывают якорь 378. Окно якоря 337
закрывается на дне 367, то есть до начала окна 379..408: пересечения нет.

Число раундов у всех фиксировано заранее, и это принципиально. 

    A  гиперпараметры заморожены заранее   ranksonly, tweedie, nd_plain
    B  случайное внутреннее разбиение      mlp, mlp2  (внутри разрешённых якорей)

Пишет artifacts/basis/hold_<имя>.npy и artifacts/basis/meta.json.
"""
import gc
import json
import time

import lightgbm as lgb
import numpy as np
from sklearn.neural_network import MLPRegressor

import paths

N, TA_ALL, _, _ = paths.dataset()
HOLD = TA_ALL[-1]
HORIZON = 30
ROUNDS = {"ranksonly": 250, "tweedie": 300, "tweedie_sub": 300,
          "nd_plain": 565}
LEVELS = {"ranksonly": "A", "tweedie": "A", "tweedie_sub": "A",
          "nd_plain": "A", "mlp": "B", "mlp2": "B"}

LGB = dict(objective="regression", metric="rmse", learning_rate=0.05,
           num_leaves=127, min_data_in_leaf=500, feature_fraction=0.6,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0, max_bin=127,
           num_threads=12, force_row_wise=True, verbose=-1, seed=42)
LGB_ND = dict(LGB, learning_rate=0.02)

paths.require("features/meta6.json", "сначала запустите src/s04_features_x6.py")

meta = json.load(open("features/meta.json"))
meta2 = json.load(open("features/meta2.json"))
meta4 = json.load(open("features/meta4.json"))
meta6 = json.load(open("features/meta6.json"))
TA = meta["train_anchors"]
DEV = [b for b, a in enumerate(TA) if a + HORIZON <= HOLD]
HOLD_B = TA.index(HOLD)

DROP = {"history_days", "anchor_dow", "anchor_doy"}
C1 = [i for i, n in enumerate(meta["names"]) if n not in DROP]
C2 = [i for i, n in enumerate(meta2["names"]) if n not in set(meta2["global_names"])]
C4 = list(range(len(meta4["names"])))
NAMES = ([meta["names"][i] for i in C1] + [meta2["names"][i] for i in C2]
         + [meta4["names"][i] for i in C4])
seen = set(meta["names"]) | set(meta2["names"]) | set(meta4["names"])
C6 = [i for i, n in enumerate(meta6["names"]) if n not in seen]
RANK_COLS = [i for i, n in enumerate(NAMES) if n.startswith("rank_")]

MM = [(np.load("features/X.npy", mmap_mode="r"), C1),
      (np.load("features/X2.npy", mmap_mode="r"), C2),
      (np.load("features/X4.npy", mmap_mode="r"), C4)]
X6 = np.load("features/X6.npy", mmap_mode="r")

y = np.load("features/y.npy")
logy = np.log1p(y)
mu = np.array([float(logy[b * N:(b + 1) * N].mean()) for b in range(len(TA))])
y_hold = logy[HOLD_B * N:(HOLD_B + 1) * N]

print(f"обучающие якоря {TA[DEV[0]]}..{TA[DEV[-1]]} ({len(DEV)} шт)")
print(f"holdout {HOLD}, признаков {len(NAMES)} (+{len(C6)} из X6)\n", flush=True)


def gather(blocks, with6=False, chunk=250_000):
    idx = np.concatenate([np.arange(b * N, (b + 1) * N) for b in blocks])
    w = len(NAMES) + (len(C6) if with6 else 0)
    out = np.empty((len(idx), w), dtype=np.float32)
    for i in range(0, len(idx), chunk):
        lo, hi = idx[i], idx[min(i + chunk, len(idx)) - 1] + 1
        off = 0
        for X, c in MM:
            out[i:i + (hi - lo), off:off + len(c)] = X[lo:hi][:, c]
            off += len(c)
        if with6:
            out[i:i + (hi - lo), off:] = X6[lo:hi][:, C6]
    gc.collect()
    return out


def fit(X, label, params, rounds):
    ds = lgb.Dataset(X, label=label, params=params).construct()
    m = lgb.train(params, ds, num_boost_round=rounds)
    del ds
    gc.collect()
    return m


def shift_free(pred):
    p = pred - pred.mean() + mu[HOLD_B]
    return float(np.sqrt(np.mean((p - (p.mean() - y_hold.mean()) - y_hold) ** 2)))


def save(tag, pred):
    np.save(paths.basis(f"hold_{tag}.npy"),
            (pred - pred.mean()).astype(np.float32))
    s = shift_free(pred)
    print(f"  {tag:10s} [{LEVELS[tag]}]  shift_free {s:.6f}  ({time.time()-t0:.0f}с)",
          flush=True)
    return s


t0 = time.time()
scores = {}

ztr = np.concatenate([logy[b * N:(b + 1) * N] - mu[b] for b in DEV]).astype(np.float32)
Xtr = gather(DEV)
Xh = gather([HOLD_B])

m = fit(Xtr[:, RANK_COLS], ztr, LGB, ROUNDS["ranksonly"])
scores["ranksonly"] = save("ranksonly", m.predict(Xh[:, RANK_COLS]))
del m
gc.collect()

m = fit(Xtr, ztr, LGB, 250)
p0 = m.predict(Xh)
np.save(paths.basis("p0_causal30_378.npy"), (p0 - p0.mean()).astype(np.float32))
print(f"  {'p0_causal30':10s} [--]  shift_free {shift_free(p0):.6f}  "
      f"({time.time()-t0:.0f}с)", flush=True)
del m, p0
gc.collect()

ytr = np.concatenate([y[b * N:(b + 1) * N] for b in DEV])
m = fit(Xtr, ytr, dict(LGB, objective="tweedie", tweedie_variance_power=1.5,
                       metric="tweedie"), ROUNDS["tweedie"])
scores["tweedie"] = save("tweedie", np.log1p(np.clip(m.predict(Xh), 0, None)))
del m, ytr
gc.collect()

rng = np.random.RandomState(0)
sub3 = np.sort(rng.choice(len(Xtr), 3_000_000, replace=False))
ysub = np.concatenate([y[b * N:(b + 1) * N] for b in DEV])[sub3]
m = fit(Xtr[sub3], ysub, dict(LGB, objective="tweedie",
                              tweedie_variance_power=1.5, metric="tweedie"),
        ROUNDS["tweedie_sub"])
scores["tweedie_sub"] = save("tweedie_sub",
                             np.log1p(np.clip(m.predict(Xh), 0, None)))
del m, ysub, sub3
gc.collect()

sub = np.sort(rng.choice(len(Xtr), 1_500_000, replace=False))
Xs = np.sign(Xtr[sub]) * np.log1p(np.abs(Xtr[sub]))
mean_, std_ = Xs.mean(0), Xs.std(0) + 1e-6
Xs = (Xs - mean_) / std_
Xhs = (np.sign(Xh) * np.log1p(np.abs(Xh)) - mean_) / std_
for tag, arch, seed in (("mlp", (256, 128), 42), ("mlp2", (512,), 7)):
    net = MLPRegressor(hidden_layer_sizes=arch, batch_size=8192,
                       learning_rate_init=1.5e-3, alpha=1e-4, max_iter=30,
                       early_stopping=True, n_iter_no_change=3,
                       validation_fraction=0.05, random_state=seed)
    net.fit(Xs, ztr[sub])
    scores[tag] = save(tag, net.predict(Xhs) + mu[HOLD_B])
    del net
    gc.collect()
del Xs, Xhs, Xtr, Xh
gc.collect()

Xtr6 = gather(DEV, with6=True)
Xh6 = gather([HOLD_B], with6=True)
m = fit(Xtr6, ztr, LGB_ND, ROUNDS["nd_plain"])
scores["nd_plain"] = save("nd_plain", m.predict(Xh6))
del m, Xtr6, Xh6
gc.collect()

json.dump({"directions": list(scores), "levels": LEVELS, "shift_free": scores,
           "hold_anchor": HOLD, "train_anchors": [TA[b] for b in DEV],
           "rounds": ROUNDS,
           "core": [k for k, v in LEVELS.items() if v == "A"],
           "extended": list(LEVELS)},
          open(paths.basis("meta.json"), "w"), indent=1)
print(f"\n{len(scores)} направлений записано в artifacts/basis ({time.time()-t0:.0f}с)")
