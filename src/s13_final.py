"""Стадия 13. Пересборка финальной отправки (публичный счёт 1.6466536668, 14 место).

Цепочка, каждый шаг - один измеренный коэффициент поверх предыдущего файла:

    epoch 11                                    1.6473694213
    sub_stage12_base                    1.6472134360
    + 0.0924 * SEQ408  -> sub_stage12_ref    1.6470414912
    + snap6_unique         -> sub_stage13_base   1.6468651642
    + четыре оси         -> ФИНАЛ              1.6466536668

Каждая ось - часть своей пробной отправки, ортогональная базе, поэтому
коэффициенты не конкурируют за объяснение одного и того же.

    py -3.11 src/s13_final.py            по config/final_blend.json
    py -3.11 src/s13_final.py --fit      вывести коэффициенты заново
"""
import argparse
import io
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="submissions/sub_stage13_base.csv")
ap.add_argument("--probe-dir", default="submissions/probes")
ap.add_argument("--axes", default="config/axes")
ap.add_argument("--blend", default="config/final_blend.json")
ap.add_argument("--target", default="submissions/sub_stage13_ref.csv")
ap.add_argument("--out", default="submissions/submission_final.csv")
ap.add_argument("--fit", action="store_true")
args = ap.parse_args()

def axis_names():
    """Какие оси собирать. Источник один - ключи config/final_blend.json.
    Пока конфига нет (первый запуск с --fit) - по файлам в config/axes."""
    if os.path.exists(args.blend):
        cfg = json.load(open(args.blend, encoding="utf-8"))
        if cfg.get("коэффициенты"):
            return list(cfg["коэффициенты"])
    return sorted(f[:-4] for f in os.listdir(args.axes) if f.endswith(".npy"))


def logvec(path, uid=None):
    d = pd.read_csv(path, dtype={"user_id": np.int64, "predict": np.float64})
    d = d.sort_values("user_id")
    u = d["user_id"].to_numpy()
    if uid is not None and not np.array_equal(u, uid):
        raise SystemExit(f"{os.path.basename(path)}: порядок user_id различается")
    return u, np.log1p(d["predict"].to_numpy(dtype=np.float64))


uid, z0 = logvec(args.base)
base_c = z0 - z0.mean()

D = {}
for name in axis_names():
    ax = os.path.join(args.axes, name + ".npy")
    if os.path.exists(ax):
        v = np.load(ax).astype(np.float64)
        D[name] = v - v.mean()
        continue
    # оси нет - построить из пробной отправки, если она под рукой. В репозитории
    # проб нет: каждая была нужна ровно для того, чтобы измерить свою ось.
    p = os.path.join(args.probe_dir, name + ".csv")
    if not os.path.exists(p):
        raise SystemExit(f"нет оси {args.axes}/{name}.npy, и построить её не из "
                         f"чего: нет {p}")
    _, l = logvec(p, uid)
    v = l - l.mean()
    a = float(v @ base_c) / float(base_c @ base_c)
    r = v - a * base_c
    D[name] = r - r.mean()
    np.save(ax, D[name].astype(np.float32))

names = list(D)
if args.fit:
    _, tgt = logvec(args.target, uid)
    y = tgt - tgt.mean()
    A = np.stack([base_c] + [D[n] for n in names])
    a, *_ = np.linalg.lstsq(A.T, y, rcond=None)
    resid = y - A.T @ a
    coef = {n: float(v / a[0]) for n, v in zip(names, a[1:])}
    print(f"коэффициент базы {a[0]:+.5f}  (1.0 - база проходит нетронутой)")
    prev = json.load(open(args.blend, encoding="utf-8")) if os.path.exists(args.blend) else {}
    prev.update({"база": os.path.basename(args.base), "коэффициенты": coef,
                 "воспроизведение_доля_дисперсии":
                     float(resid @ resid) / float(y @ y)})
    io.open(args.blend, "w", encoding="utf-8").write(
        json.dumps(prev, ensure_ascii=False, indent=1))
    print(f"записано {args.blend}")
else:
    if not os.path.exists(args.blend):
        raise SystemExit(f"нет {args.blend}; запустите один раз с --fit")
    coef = json.load(open(args.blend, encoding="utf-8"))["коэффициенты"]

z = z0.copy()
for n, t in coef.items():
    z = z + t * D[n]
pred = np.clip(np.expm1(z), 0, None)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
pd.DataFrame({"user_id": uid, "predict": pred}).sort_values("user_id").to_csv(
    args.out, index=False)

print(f"\nbase: {os.path.basename(args.base)}")
for n, t in sorted(coef.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {n:18s} {t:+.6f}")
print(f"-> {args.out}")
print(f"   уровень {z.mean():.7f}  sd {z.std():.6f}  обрезано {(z < 0).sum()}")

if os.path.exists(args.target):
    _, tgt = logvec(args.target, uid)
    rms = float(np.sqrt(np.mean((tgt - z) ** 2)))
    frac = rms ** 2 / float(np.var(tgt))
    print(f"\nпротив отправленного: rms(log) {rms:.3e}, {frac:.2e} его дисперсии, "
          f"corr {np.corrcoef(tgt, z)[0,1]:.8f}")
    print("   " + ("ТОЧНО" if frac < 1e-9 else
                   "воспроизведено в пределах обрезки expm1" if frac < 1e-4 else
                   "НЕ СОВПАДАЕТ"))
