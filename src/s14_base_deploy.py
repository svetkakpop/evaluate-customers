"""Стадия 14. База на границе развёртывания - табличное предсказание якоря 408.

Конвейер производит поправку к базе, а не всё предсказание целиком. В Конкурсе
базой был отправленный файл, собранный табличным стеком, и стадии его не
пересобирают: он лежит в submissions/ как sub_stage12_base.csv. На другом наборе
данных такого файла нет, и собрать отправку не из чего. Эта стадия закрывает
разрыв.

Что она делает: одна LightGBM на всех якорях, легальных для границы 408
(правило a + 30 <= 408, то есть все двадцать пять), цель - z = log1p(y) - mu_a,
предсказание последнего блока признаков. Гиперпараметры и 250 раундов те же, что
у p0_causal30 в стадии 07, и зафиксированы так же - до обучения.

Чего она НЕ делает: не воспроизводит тот стек и не претендует на его качество.
Это самодостаточная база, чтобы конвейер собирался на любых данных.

УРОВЕНЬ. Модель предсказывает z, центрированный по якорям; сам уровень окна
409..438 данные не показывают - окно лежит за концом истории. Он берётся
допущением: среднее mu по последним --mu-anchors якорям. Это единственное место
конвейера, где число берётся не из данных и не из измерения, и стадия говорит об
этом вслух: расхождение с уровнем отправленной базы печатается, если она рядом.

    py -3.11 src/s14_base_deploy.py                    среднее по 3 якорям
    py -3.11 src/s14_base_deploy.py --mu-anchors 6     по шести
    py -3.11 src/s14_base_deploy.py --mu 2.3294        задать уровень прямо

Пишет submissions/base_deploy.csv и artifacts/base_deploy.json.
"""
import paths
import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=250)
ap.add_argument("--mu-anchors", type=int, default=3,
                help="по скольким последним якорям усредняется уровень")
ap.add_argument("--mu", type=float, default=None,
                help="задать уровень явно, в лог-пространстве")
ap.add_argument("--out", default=paths.submission("base_deploy.csv"))
ap.add_argument("--reference", default=paths.submission("sub_stage12_base.csv"),
                help="отправленная база для сверки, если она есть")
args = ap.parse_args()

HORIZON = 30
LGB = dict(objective="regression", metric="rmse", learning_rate=0.05,
           num_leaves=127, min_data_in_leaf=500, feature_fraction=0.6,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0, max_bin=127,
           num_threads=12, force_row_wise=True, verbose=-1, seed=42)

N, TA, _, FINAL = paths.dataset()
paths.require("features/X.npy", "сначала запустите src/s01_features_base.py")
paths.require("features/X2.npy", "сначала запустите src/s02_features_spans.py")
paths.require("features/X4.npy", "сначала запустите src/s03_features_ranks.py")

meta = json.load(open("features/meta.json"))
meta2 = json.load(open("features/meta2.json"))
meta4 = json.load(open("features/meta4.json"))
DROP = {"history_days", "anchor_dow", "anchor_doy"}
C1 = [i for i, n in enumerate(meta["names"]) if n not in DROP]
C2 = [i for i, n in enumerate(meta2["names"]) if n not in set(meta2["global_names"])]
C4 = list(range(len(meta4["names"])))
NAMES = ([meta["names"][i] for i in C1] + [meta2["names"][i] for i in C2]
         + [meta4["names"][i] for i in C4])

MM = [(np.load("features/X.npy", mmap_mode="r"), C1),
      (np.load("features/X2.npy", mmap_mode="r"), C2),
      (np.load("features/X4.npy", mmap_mode="r"), C4)]
DEPLOY_B = len(TA)                      # последний блок - якорь FINAL
logy = np.log1p(np.load("features/y.npy"))
mu = np.array([float(logy[b * N:(b + 1) * N].mean()) for b in range(len(TA))])

DEV = [b for b, a in enumerate(TA) if a + HORIZON <= FINAL]
if not DEV:
    raise SystemExit(f"нет якорей с правилом a+{HORIZON} <= {FINAL}: "
                     f"данным не хватает истории")
print(f"граница {FINAL}, обучающие якоря {TA[DEV[0]]}..{TA[DEV[-1]]} "
      f"({len(DEV)} шт), признаков {len(NAMES)}")
print(f"правило: a + {HORIZON} <= {FINAL}, максимальный якорь "
      f"{TA[DEV[-1]]} + {HORIZON} = {TA[DEV[-1]] + HORIZON}", flush=True)


def gather(blocks, chunk=250_000):
    """Строки берутся срезом: индексация memmap массивом уходит на путь
    случайного сбора и стоит минуты вместо секунд."""
    out = np.empty((len(blocks) * N, len(NAMES)), dtype=np.float32)
    for i, b in enumerate(blocks):
        lo, hi = b * N, (b + 1) * N
        r0, off = i * N, 0
        for X, c in MM:
            out[r0:r0 + N, off:off + len(c)] = X[lo:hi][:, c]
            off += len(c)
    gc.collect()
    return out


t0 = time.time()
ztr = np.concatenate([logy[b * N:(b + 1) * N] - mu[b] for b in DEV]).astype(np.float32)
Xtr = gather(DEV)
print(f"обучающая матрица {Xtr.shape} собрана ({time.time() - t0:.0f}с)", flush=True)

ds = lgb.Dataset(Xtr, label=ztr, feature_name=NAMES, params=LGB).construct()
model = lgb.train(LGB, ds, num_boost_round=args.rounds)
del Xtr, ztr, ds
gc.collect()
print(f"обучено {args.rounds} раундов ({time.time() - t0:.0f}с)", flush=True)

Xd = gather([DEPLOY_B])
p = model.predict(Xd).astype(np.float64)
del Xd, model
gc.collect()

k = max(1, min(args.mu_anchors, len(DEV)))
mu_data = float(mu[[b for b in DEV][-k:]].mean())
mu_hat = args.mu if args.mu is not None else mu_data
src = ("задан ключом --mu" if args.mu is not None
       else f"среднее mu по последним {k} якорям {[TA[b] for b in DEV[-k:]]}")
print(f"\nуровень {mu_hat:.6f}  ({src})")
print(f"  для сравнения: mu последнего якоря {TA[DEV[-1]]} = {mu[DEV[-1]]:.6f}, "
      f"по всем якорям {mu[DEV].mean():.6f}")

z = (p - p.mean()) + mu_hat
pred = np.clip(np.expm1(z), 0, None)
uid = np.load("features/user_id.npy")[DEPLOY_B * N:(DEPLOY_B + 1) * N].astype(np.int64)
frame = pd.DataFrame({"user_id": uid, "predict": pred}).sort_values("user_id")
frame.to_csv(args.out, index=False)
print(f"\n-> {args.out}")
print(f"   sd(z) {z.std():.6f}   мин. predict {pred.min():.4f}   "
      f"обрезано {(z < 0).sum()}")

info = {"cutoff": FINAL, "train_anchors": [TA[b] for b in DEV],
        "rounds": args.rounds, "mu": mu_hat, "mu_source": src,
        "mu_last_anchor": float(mu[DEV[-1]]), "mu_all_anchors": float(mu[DEV].mean()),
        "sd_z": float(z.std()), "clipped": int((z < 0).sum()),
        "note": "самодостаточная база; уровень окна за концом истории - допущение"}

if os.path.exists(args.reference):
    ref = pd.read_csv(args.reference).sort_values("user_id")
    if np.array_equal(ref["user_id"].to_numpy(), frame["user_id"].to_numpy()):
        rz = np.log1p(ref["predict"].to_numpy(dtype=np.float64))
        zc, rc = z - z.mean(), rz - rz.mean()
        corr = float(np.corrcoef(zc, rc)[0, 1])
        info.update(ref=os.path.basename(args.reference), corr_shape=corr,
                    mu_reference=float(rz.mean()), mu_error=float(mu_hat - rz.mean()),
                    rms_log=float(np.sqrt(np.mean((z - rz) ** 2))))
        print(f"\nпротив {os.path.basename(args.reference)} (отправленная база):")
        print(f"   форма: corr {corr:.6f}   sd наша {zc.std():.4f} против "
              f"{rc.std():.4f}")
        print(f"   уровень: наш {mu_hat:.6f} против {rz.mean():.6f}, "
              f"промах {mu_hat - rz.mean():+.6f}")
        print(f"   rms(log) {np.sqrt(np.mean((z - rz) ** 2)):.6f}")
        print("   форма и уровень - разные вещи: первую даёт модель, второй допущение")
    else:
        print(f"\n{os.path.basename(args.reference)}: другие user_id, сверка пропущена")

json.dump(info, open(paths.artifact("base_deploy.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"\nзаписано artifacts/base_deploy.json ({time.time() - t0:.0f}с)")
