"""Стадия 11. Чтение результата SEQ408 в две стадии, по заранее заданной шкале.

Стадия 1 спрашивает, выучила ли последовательная модель ошибку базового GBDT
вообще. Считается против двух баз, и это не формальность:

    ПЕРВИЧНАЯ      oof_panel pred_378   purge 42, якоря <=336, последний 323
    ЧУВСТВИТЕЛЬНОСТЬ p0_causal30_378    правило s+30<=a, последний 337

Ct у них совпадает по построению, расходится Cp. Если сигнал есть только против
той базы, на остатке которой модель обучалась, это не residual-архитектура, а
поправка под конкретную базу, и большой результат стадии 1 ничего не доказывает.

Стадия 2 - вердикт: что переживает проекцию на базис.

    dMSE_novel  < 0.0005        CLOSE
                0.0005-0.0015   MARGINAL
                0.0015-0.0030   STRONG
                0.0030-0.0060   BREAKTHROUGH
                >= 0.0060       TARGET-SCALE

Плюс retention относительно 0.015992 - замера с загрязнённым протоколом против
того же базиса, сделанного на этапе разработки.

Первична эпоха 11 и только она.
"""
import json
import os

import numpy as np

import paths

N = 250_000
DIRTY_REF = 0.015992
SHIP_EPOCH = 11
EVAL = None
BANDS = [(0.0005, "CLOSE"), (0.0015, "MARGINAL"), (0.0030, "STRONG"),
         (0.0060, "BREAKTHROUGH"), (float("inf"), "TARGET-SCALE")]
RET = [(0.10, "от старого эффекта почти ничего не осталось"),
       (0.25, "часть архитектурного сигнала реальна"),
       (0.50, "очень сильно"),
       (float("inf"), "сопоставимо со всем оставшимся разрывом")]

rp = paths.artifact("report_pack.npz")
pack = np.load(rp if os.path.exists(rp)
               else paths.require("validation_pack.npz",
                                  "сначала запустите src/s08_packs.py"))
TA = ([int(a) for a in pack["all_anchors"]] if "all_anchors" in pack.files
      else json.load(open("features/meta.json"))["train_anchors"])
EVAL = int(pack["eval_anchor"])
N = len(pack["T"])
z = pack["T"].astype(np.float64)
z = z - z.mean()
V_T = float(np.mean(z * z))
BASIS = pack["basis_ext"].astype(np.float64)
BNAMES = [str(x) for x in pack["basis_ext_names"]]


def c(a):
    a = np.asarray(a, dtype=np.float64)
    return a - a.mean()


def band(v, table):
    for hi, name in table:
        if v < hi:
            return name
    return table[-1][1]


def stage1(d, p0):
    V = float(np.mean(d * d))
    r0 = z - p0
    Cr = float(np.mean(d * r0))
    V_r0 = float(np.mean(r0 * r0))
    return dict(Ct=float(np.mean(d * z)), Cp=float(np.mean(d * p0)), Cr_base=Cr,
                rho_resid=Cr / np.sqrt(V * V_r0), beta_resid=Cr / V,
                dMSE_base=Cr * Cr / V, rmse_base=float(np.sqrt(V_r0)),
                rmse_corr=float(np.sqrt(np.mean((r0 - (Cr / V) * d) ** 2))))


def stage2(d):
    G = BASIS @ BASIS.T / N
    w = np.linalg.pinv(G) @ (BASIS @ d / N)
    q = d - w @ BASIS
    Vq = float(np.mean(q * q))
    Cn = float(np.mean(q * z))
    return dict(kept=Vq / float(np.mean(d * d)), C_novel=Cn,
                rho_novel=Cn / np.sqrt(Vq * V_T), beta_novel=Cn / Vq,
                dMSE_novel=Cn * Cn / Vq)


f = paths.artifact("hold_seq11.npy")
if not os.path.exists(f):
    f = paths.require(paths.snapshot(f"resid_snap{SHIP_EPOCH}.npy"),
                      "сначала запустите src/s10_train_validation.py")
d = c(np.load(f))
tr = [int(a) for a in pack["train_anchors"]]
print(f"SEQ408, эпоха {SHIP_EPOCH} (терминальная, заморожена)")
print(f"обучающие якоря остатка {tr[0]}..{tr[-1]} ({len(tr)} шт), оценка {EVAL}")
print(f"V(d) {float(np.mean(d * d)):.6f}\n")

purge = (int(pack["base_purge"]) if "base_purge" in pack.files
         else json.load(open("features/oof_panel_meta.json"))["purge"])
rounds = (int(pack["base_rounds"]) if "base_rounds" in pack.files
          else json.load(open("features/oof_panel_meta.json"))["rounds"])
bases = [(f"oof_panel pred_{EVAL}", "ПЕРВИЧНАЯ", c(pack["pred_eval"]),
          f"purge {purge}, якоря <= {EVAL - purge}, "
          f"последний {max(a for a in TA if a <= EVAL - purge)}, "
          f"{rounds} раундов")]
p2 = paths.basis("p0_causal30_378.npy")
if os.path.exists(p2):
    bases.append(("p0_causal30_378", "ЧУВСТВИТЕЛЬНОСТЬ", c(np.load(p2)),
                  f"правило s+30<=a, последний "
                  f"{max(a for a in TA if a + 30 <= EVAL)}, 250 раундов"))

print("--- стадия 1: выучена ли ошибка базового GBDT ---")
s1 = {}
for name, role, p0, desc in bases:
    r = stage1(d, p0)
    s1[name] = dict(r, role=role, description=desc)
    print(f"  {role}: {name}")
    print(f"    {desc}")
    print(f"    Ct {r['Ct']:+.6f}   Cp {r['Cp']:+.6f}   Cr {r['Cr_base']:+.6f}")
    print(f"    rho_resid {r['rho_resid']:+.5f}   beta_resid {r['beta_resid']:+.4f}"
          f"   dMSE_base {r['dMSE_base']:.6f}")
    print(f"    rmse базы {r['rmse_base']:.6f} -> {r['rmse_corr']:.6f} "
          f"при оптимальном весе  ({r['rmse_base'] - r['rmse_corr']:+.6f})")
if len(s1) == 2:
    a, b = (v["dMSE_base"] for v in s1.values())
    ratio = min(a, b) / max(a, b) if max(a, b) > 0 else float("nan")
    print(f"\n  чувствительность к базе: dMSE_base {a:.6f} против {b:.6f}, "
          f"отношение {ratio:.3f}")

print("\n--- стадия 2: что переживает проекцию на базис ---")
print(f"  базис k={len(BNAMES)}: {BNAMES}")
sc = stage2(d)
ret = sc["dMSE_novel"] / DIRTY_REF
print(f"  kept        {sc['kept']:.4f}")
print(f"  C_novel     {sc['C_novel']:+.6f}")
print(f"  rho_novel   {sc['rho_novel']:+.5f}")
print(f"  dMSE_novel  {sc['dMSE_novel']:.6f}   -> {band(sc['dMSE_novel'], BANDS)}")
print(f"  retention против загрязнённого протокола ({DIRTY_REF:.6f}): {100 * ret:.1f}%"
      f"   -> {band(ret, RET)}")
print(f"  для масштаба: нужный dMSE ~0.0078, шум корреляции на {N} строках "
      f"{1 / np.sqrt(N):.5f}")

print("\n--- траектория ---")
print(f"  {'эп':>3} {'kept':>7} {'dMSE_base':>10} {'rho_resid':>10} "
      f"{'rho_novel':>10} {'dMSE_novel':>11}")
traj = {}
p0p = bases[0][2]
for ep in range(12):
    g = paths.snapshot(f"resid_snap{ep}.npy")
    if not os.path.exists(g):
        continue
    de = c(np.load(g))
    s, bs = stage2(de), stage1(de, p0p)
    traj[ep] = dict(s, dMSE_base=bs["dMSE_base"], rho_resid=bs["rho_resid"])
    print(f"  {ep:>3} {s['kept']:>7.4f} {bs['dMSE_base']:>10.6f} "
          f"{bs['rho_resid']:>+10.5f} {s['rho_novel']:>+10.5f} "
          f"{s['dMSE_novel']:>11.6f}")

print(f"\nПЕРВИЧНАЯ: эпоха {SHIP_EPOCH}")
if traj:
    print(f"диагностический argmax: эпоха "
          f"{max(traj, key=lambda e: traj[e]['dMSE_novel'])}")

json.dump({"ship_epoch": SHIP_EPOCH, "stage1": s1, "stage2": sc,
           "retention_vs_dirty": ret, "dirty_reference": DIRTY_REF,
           "band": band(sc["dMSE_novel"], BANDS),
           "retention_band": band(ret, RET),
           "basis": BNAMES, "train_anchors": tr,
           "trajectory": {str(k): v for k, v in traj.items()}},
          open(paths.artifact("report_validation.json"), "w"), indent=1)
print("\nзаписано artifacts/report_validation.json")
