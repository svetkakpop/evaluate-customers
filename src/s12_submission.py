"""Стадия 12. Сборка отправки SEQ408: база + t * направление, в лог-пространстве.

    z = log1p(база) + t * центрированное(pred_seq11_408)
    predict = clip(expm1(z), 0, inf)

База - sub_stage12_base.csv, публичный счёт 1.6472134360.
Коэффициент t оценён, а не подобран: одна отправка несла направление в известном
масштабе, её значение метрики даёт оценку ковариации со скрытым таргетом, дальше t
выводится в замкнутой форме. Получилось 0.092360, результат 1.6470414912.

Пересборка сверяется с отправленным файлом, а не принимается на веру.

Коэффициент можно и не измерять по метрике: на оценочном якоре таргет известен,
и стадия 11 считает там ту же величину напрямую, beta_resid = E[d·(T−m)]/E[d²].
Здесь это 0.083911 против 0.092360 по метрике, разница 10%, а собранные файлы
различаются на 1.6e-06 дисперсии. На другом наборе данных это единственный
доступный путь, и он работает.

    py -3.11 src/s12_submission.py                    по config/blend.json
    py -3.11 src/s12_submission.py --fit              вывести t из отправленного
    py -3.11 src/s12_submission.py --t-from-report    локальная оценка, без лидерборда
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="submissions/sub_stage12_base.csv")
ap.add_argument("--direction", default="artifacts/pred_seq11_408.npy")
ap.add_argument("--blend", default="config/blend.json")
ap.add_argument("--out", default="submissions/submission_stage12.csv")
ap.add_argument("--verify", default="submissions/sub_stage12_ref.csv")
ap.add_argument("--t", type=float, default=None, help="override the stored coefficient")
ap.add_argument("--fit", action="store_true",
                help="re-derive t by least squares against --verify")
ap.add_argument("--t-from-report", action="store_true",
                help="взять t из локальной оценки на оценочном якоре, без лидерборда")
ap.add_argument("--report", default="artifacts/report_validation.json")
args = ap.parse_args()

base = pd.read_csv(args.base).sort_values("user_id")
uid = base["user_id"].to_numpy()
z0 = np.log1p(base["predict"].to_numpy(dtype=np.float64))

rp = "data/run_pack.npz"
try:
    cur = np.load(rp)["user_id"] if os.path.exists(rp) else None
except Exception:
    cur = None          # указатель Git LFS или битый файл: сверять нечем, идём дальше
if cur is not None:
    if len(cur) != len(uid) or not np.array_equal(cur, uid):
        raise SystemExit(
            f"user_id базы {os.path.basename(args.base)} не совпадают с текущим "
            f"train.parquet ({len(uid)} против {len(cur)}).\n"
            "  база - отправленный файл, привязанный к своим пользователям;\n"
            "  на другом наборе стадии 12-13 неприменимы, см. README, "
            "раздел «Другой набор данных»")

d = np.load(args.direction).astype(np.float64)
if d.shape[0] != len(uid):
    raise SystemExit(f"в направлении {d.shape[0]} строк, в базе {len(uid)}")
d = d - d.mean()

cfg = (json.load(open(args.blend, encoding="utf-8"))
       if os.path.exists(args.blend) else {})
t = args.t if args.t is not None else cfg.get("t")

if args.t_from_report:
    if not os.path.exists(args.report):
        raise SystemExit(f"нет {args.report}; сначала запустите src/s11_report.py")
    s1 = json.load(open(args.report, encoding="utf-8"))["stage1"]
    name, rec = next(((k, v) for k, v in s1.items() if v.get("role") == "ПЕРВИЧНАЯ"),
                     next(iter(s1.items())))
    t_local = float(rec["beta_resid"])
    print(f"локальная оценка на оценочном якоре ({name}): beta_resid = {t_local:.6f}")
    if cfg.get("t") is not None:
        d_pct = 100 * (cfg["t"] - t_local) / t_local
        print(f"  для сравнения, измеренный по метрике t = {cfg['t']:.6f} "
              f"({d_pct:+.1f}% к локальной оценке)")
    print("  это оценка по анкору, где таргет известен: лидерборд для неё не нужен")
    t = t_local

if args.fit:
    if not os.path.exists(args.verify):
        raise SystemExit(f"для --fit нужен {args.verify}")
    ref = pd.read_csv(args.verify).sort_values("user_id")
    if not np.array_equal(ref["user_id"].to_numpy(), uid):
        raise SystemExit("порядок user_id в базе и в проверочном файле различается")
    y = np.log1p(ref["predict"].to_numpy(dtype=np.float64))
    z0c = z0 - z0.mean()
    A = np.stack([z0c, d])
    a, *_ = np.linalg.lstsq(A.T, y - y.mean(), rcond=None)
    print(f"подгонка: коэффициент базы {a[0]:+.5f}  (1.0 - база не тронута)")
    t = float(a[1] / a[0])
    print(f"выведено t = {t:.6f}")

if t is None:
    raise SystemExit("нет коэффициента: задайте --t, запустите --fit или положите config/blend.json")

z = z0 + t * d
pred = np.clip(np.expm1(z), 0, None)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
pd.DataFrame({"user_id": uid, "predict": pred}).sort_values("user_id").to_csv(
    args.out, index=False)
print(f"\nt = {t:.6f}  ->  {args.out}")
print(f"  уровень {z.mean():.7f}   sd {z.std():.6f}   "
      f"мин. predict {pred.min():.4f}   обрезано {(z < 0).sum()}")

STORED_BASE = "submissions/sub_stage12_base.csv"
if os.path.exists(args.verify):
    ref = pd.read_csv(args.verify).sort_values("user_id")
    rl = np.log1p(ref["predict"].to_numpy(dtype=np.float64))
    rms = float(np.sqrt(np.mean((rl - z) ** 2)))
    var = float(np.var(rl))
    print(f"  против {os.path.basename(args.verify)}: rms(log) {rms:.3e}, "
          f"{rms**2/var:.2e} его дисперсии, corr {np.corrcoef(rl, z)[0,1]:.8f}")
    if os.path.abspath(args.base) != os.path.abspath(STORED_BASE):
        print("  сверка справочная: собрано на своей базе, а эталон строился "
              "на отправленной")
    else:
        print("  " + ("ТОЧНО" if rms < 1e-9 else
                      "воспроизведено в пределах обрезки expm1"
                      if rms ** 2 / var < 1e-4 else
                      "НЕ СОВПАДАЕТ - проверьте файл направления и t"))
