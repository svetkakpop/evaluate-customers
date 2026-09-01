"""Стадия 08. Два пакета данных для обучения последовательной модели.

Одно причинное правило, два разных среза:

    target_end(s) = s + 30 <= cutoff

    validation_pack   cutoff = 378   якоря 155..337 (14)   есть holdout
    deploy_pack       cutoff = 408   якоря 155..378 (17)   holdout невозможен

Якоря 351, 365 и 378 запрещены, когда 378 - оценочная цель, и являются обычной
наблюдённой историей, когда 408 - граница признаков. В этом вся разница между
моделью, которую можно измерить, и моделью, которую можно отправить.

Веса по правилу, зафиксированному до обучения: 0.5 там, где база остатка
обучалась меньше чем на 13 якорях, иначе 1.0.

Пишет data/validation_pack.npz, data/deploy_pack.npz и их json-описания.
"""
import json
import os

import numpy as np

import paths

N, TA_ALL, _, FINAL = paths.dataset()
HORIZON = 30
PURGE_BASE = 42
MIN_STRONG_BASE = 13
BASE_ROUNDS = 250
EVAL = TA_ALL[-1]
CUTOFF = FINAL

paths.require("oof_panel.npz", "сначала запустите src/s05_oof_panel.py")
paths.require(paths.basis("meta.json"), "сначала запустите src/s07_basis.py")

meta = json.load(open("features/meta.json"))
TA = meta["train_anchors"]
panel = np.load("oof_panel.npz")
PA = sorted(int(k.split("_")[1]) for k in panel.files if k.startswith("resid_"))
bmeta = json.load(open(paths.basis("meta.json")))

logy = np.log1p(np.load("features/y.npy"))
b_eval = TA.index(EVAL)
y_hold = logy[b_eval * N:(b_eval + 1) * N].astype(np.float32)
T = (y_hold - y_hold.mean()).astype(np.float32)


def base_size(a):
    return len([t for t in TA if t <= a - PURGE_BASE])


def weights_for(anchors):
    return np.array([0.5 if base_size(a) < MIN_STRONG_BASE else 1.0
                     for a in anchors], dtype=np.float32)


def stack(anchors, key):
    return np.stack([panel[f"{key}_{a}"] for a in anchors]).astype(np.float32)


core = bmeta["core"]
ext = bmeta["extended"]
B_core = np.stack([np.load(paths.basis(f"hold_{d}.npy")) for d in core])
B_ext = np.stack([np.load(paths.basis(f"hold_{d}.npy")) for d in ext])

tr = [a for a in PA if a + HORIZON <= EVAL]
assert EVAL in PA and EVAL not in tr
np.savez_compressed(
    "validation_pack.npz",
    train_anchors=np.array(tr, dtype=np.int32),
    residuals=stack(tr, "resid"), z=stack(tr, "z"), pred=stack(tr, "pred"),
    weights=weights_for(tr),
    eval_anchor=np.int32(EVAL),
    resid_eval=panel[f"resid_{EVAL}"].astype(np.float32),
    z_eval=panel[f"z_{EVAL}"].astype(np.float32),
    pred_eval=panel[f"pred_{EVAL}"].astype(np.float32),
    y_hold_log=y_hold, T=T,
    basis_core=B_core, basis_ext=B_ext,
    basis_core_names=np.array(core), basis_ext_names=np.array(ext),
)
json.dump({"eval_anchor": EVAL, "train_anchors": tr,
           "excluded": [a for a in PA if a + HORIZON > EVAL and a != EVAL],
           "weights": weights_for(tr).tolist(),
           "basis_core": core, "basis_ext": ext,
           "base_rounds": BASE_ROUNDS, "base_purge": PURGE_BASE},
          open(paths.artifact("validation_pack.json"), "w"), indent=1)
print(f"validation_pack: {len(tr)} якорей {tr[0]}..{tr[-1]}, "
      f"исключены {[a for a in PA if a + HORIZON > EVAL and a != EVAL]}")
print(f"  базис CORE {len(core)} {core}")
print(f"        EXT  {len(ext)} {ext}")

tr2 = [a for a in PA if a + HORIZON <= CUTOFF]
np.savez_compressed(
    "deploy_pack.npz",
    train_anchors=np.array(tr2, dtype=np.int32),
    residuals=stack(tr2, "resid"), weights=weights_for(tr2),
    cutoff=np.int32(CUTOFF),
)
json.dump({"cutoff": CUTOFF, "train_anchors": tr2,
           "newly_legal_vs_eval378": [a for a in tr2 if a + HORIZON > EVAL],
           "weights": weights_for(tr2).tolist(),
           "base_rounds": BASE_ROUNDS, "base_purge": PURGE_BASE,
           "holdout": None,
           "note": "локальный гейт невозможен: 378 внутри обучающего набора"},
          open(paths.artifact("deploy_pack.json"), "w"), indent=1)
print(f"deploy_pack:     {len(tr2)} якорей {tr2[0]}..{tr2[-1]}, "
      f"впервые легальны {[a for a in tr2 if a + HORIZON > EVAL]}")

np.savez_compressed(
    paths.artifact("report_pack.npz"),
    T=T, pred_eval=panel[f"pred_{EVAL}"].astype(np.float32),
    basis_ext=B_ext, basis_ext_names=np.array(ext),
    basis_core_names=np.array(core),
    train_anchors=np.array(tr, dtype=np.int32),
    eval_anchor=np.int32(EVAL),
    all_anchors=np.array(TA, dtype=np.int32),
    base_purge=np.int32(PURGE_BASE), base_rounds=np.int32(BASE_ROUNDS),
)
print(f"report_pack:     {os.path.getsize(paths.artifact('report_pack.npz'))/1e6:.1f} МБ "
      f"-> artifacts/, для стадии 11 на свежем клоне")
