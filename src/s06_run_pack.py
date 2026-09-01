"""Стадия 06. Три факта, которые нужны обучению, и ничего больше.

    day0           начало календаря, чтобы event_date лёг на сетку якорей
    final_anchor   граница развёртывания, 408
    user_id        канонический порядок, который предполагают все массивы


Пишет run_pack.npz.
"""
import json

import numpy as np
import pandas as pd

import paths

paths.require("train.parquet", "положите train.parquet в data/")
paths.require("features/meta.json", "сначала запустите src/s01_features_base.py")
paths.check_parquet()

meta = json.load(open("features/meta.json"))
TRAIN_ANCHORS = meta["train_anchors"]
FINAL_ANCHOR = TRAIN_ANCHORS[-1] + 30

df = pd.read_parquet("train.parquet", columns=["user_id", "event_date"])
uniq = np.unique(df["user_id"].to_numpy())
days = df["event_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
DAY0 = int(days.min())
del df

print(f"пользователей {len(uniq):,}")
print(f"day0          {DAY0}   (начало календаря; индекс дня = event_date - day0)")
print(f"сетка якорей  {TRAIN_ANCHORS[0]} .. {TRAIN_ANCHORS[-1]}  "
      f"({len(TRAIN_ANCHORS)} якорей, шаг {TRAIN_ANCHORS[1]-TRAIN_ANCHORS[0]})")
print(f"граница      {FINAL_ANCHOR}   окно таргета "
      f"{FINAL_ANCHOR+1}..{FINAL_ANCHOR+30} (не наблюдается)")
print(f"дней в данных {int(days.max()) - DAY0 + 1}")

np.savez_compressed("run_pack.npz", user_id=uniq, day0=np.int64(DAY0),
                    final_anchor=np.int32(FINAL_ANCHOR),
                    train_anchors=np.array(TRAIN_ANCHORS, dtype=np.int32))
print("\nзаписано data/run_pack.npz")
