"""Пути репозитория и форма набора данных.

Скрипты стадий написаны в расчёте на плоский рабочий каталог: они открывают
"train.parquet" и "features/X.npy". Импорт этого модуля переводит процесс в
data/, поэтому относительные пути разрешаются туда, а выходы кладутся в
artifacts/ по абсолютному пути.

Импортировать первым, до любого обращения к файлам.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FEATURES = os.path.join(DATA, "features")
ARTIFACTS = os.path.join(ROOT, "artifacts")
BASIS = os.path.join(ARTIFACTS, "basis")
SNAPSHOTS = os.path.join(ARTIFACTS, "snapshots")
SUBMISSIONS = os.path.join(ROOT, "submissions")
CONFIG = os.path.join(ROOT, "config")

for _d in (DATA, FEATURES, ARTIFACTS, BASIS, SNAPSHOTS):
    os.makedirs(_d, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(DATA)


def artifact(name):
    return os.path.join(ARTIFACTS, name)


def basis(name):
    return os.path.join(BASIS, name)


def snapshot(name):
    return os.path.join(SNAPSHOTS, name)


def submission(name):
    return os.path.join(SUBMISSIONS, name)


def config(name):
    return os.path.join(CONFIG, name)


def is_lfs_pointer(path):
    """Файл выгружен как указатель Git LFS, а не как содержимое."""
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs.github.com/spec/")
    except OSError:
        return False


def require(path, hint=""):
    if not os.path.exists(path):
        where = path if os.path.isabs(path) else os.path.join(DATA, path)
        raise SystemExit(f"нет файла: {where}" + (f"\n  {hint}" if hint else ""))
    if is_lfs_pointer(path):
        raise SystemExit(
            f"{path} - указатель Git LFS, а не сам файл.\n"
            "  установите git-lfs и выгрузите содержимое:\n"
            "    git lfs install && git lfs pull")
    return path


def dataset():
    import numpy as np
    m = json.load(open(os.path.join(FEATURES, "meta.json")))
    ta = m["train_anchors"]
    n = m.get("n_users")
    if n is None:
        n = len(np.load(os.path.join(FEATURES, "y.npy"))) // (len(ta) + 1)
    return int(n), ta, int(m["day0"]), int(m["final_anchor"])


def check_parquet(path="train.parquet"):
    import numpy as np
    import pyarrow.parquet as pq
    exp_n, _, exp_day0, final = dataset()
    t = pq.read_table(path, columns=["user_id", "event_date"])
    n = len(np.unique(t.column("user_id").to_numpy()))
    d = t.column("event_date").to_numpy().astype("datetime64[D]").astype(np.int64)
    day0, span = int(d.min()), int(d.max() - d.min())
    bad = []
    if n != exp_n:
        bad.append(f"пользователей {n}, признаки построены на {exp_n}")
    if day0 != exp_day0:
        bad.append(f"day0 {day0}, в meta.json {exp_day0}")
    if span != final:
        bad.append(f"диапазон дней {span}, final_anchor {final}")
    if bad:
        raise SystemExit(
            "train.parquet не соответствует features/:\n  " + "\n  ".join(bad)
            + "\n  пересоберите признаки: py -3.11 run_all.py --from 01 --force")
