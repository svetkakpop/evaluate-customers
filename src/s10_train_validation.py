"""Стадия 10. Обучение SEQ408 на оценочном якоре 378. Это доказательство.

Обучение только на якорях 155..337: окно каждого закрывается не позже дня 367,
то есть до начала окна 379..408. Пересечения с оценочной целью нет.

На этапе разработки обучение шло на якорях, включавших 351 и 365, чьи окна
перекрывали 379..408 на 3 и 17 дней, а остаточные цели брали число раундов из
файла, подобранного early stopping'ом на самом 378. Тот замер давал dMSE 0.015992
против того же базиса. После исправления протокола остаётся 6.2%.

Эпоха 11 - терминальная эпоха OneCycleLR: SHIP_EPOCH = EPOCHS - 1, правило
зафиксировано до обучения. Диагностический argmax по dMSE считается и
печатается, но ни на что не влияет - в отличие от эталонного скрипта, где отбор
эпохи делался именно по argmax.

Пишет artifacts/snapshots/resid_snap*.npy, artifacts/hold_seq11.npy,
artifacts/pred_seq11.npy.
"""

import paths
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOW = 364
HORIZON = 30
CHANNELS = 96
BLOCKS = 7
DROPOUT = 0.2
WD = 3e-4
BATCH = 768
ANCHORS_PER_BATCH = 6
PAIRS_PER_EPOCH = 2_000_000
EPOCHS = 12
LR_MAX = 2e-3
W_BUYER, W_BUY7, W_POSVAL = 0.20, 0.15, 0.10
SHIP_EPOCH = EPOCHS - 1
CKPT = paths.artifact("seq_ckpt.pt")
REFERENCE = paths.config("frozen_config_reference.py")

if os.path.exists(REFERENCE):
    src = open(REFERENCE, encoding="utf-8").read()
    FROZEN = dict(WINDOW=WINDOW, HORIZON=HORIZON, CHANNELS=CHANNELS, BLOCKS=BLOCKS,
                  DROPOUT=DROPOUT, WD=WD, BATCH=BATCH,
                  ANCHORS_PER_BATCH=ANCHORS_PER_BATCH,
                  PAIRS_PER_EPOCH=PAIRS_PER_EPOCH, EPOCHS=EPOCHS, LR_MAX=LR_MAX)
    bad = []
    for k, v in FROZEN.items():
        m = re.search(rf"^{k}\s*=\s*([0-9_.e-]+)", src, re.M)
        if not m or abs(float(m.group(1).replace("_", "")) - float(v)) > 1e-12:
            bad.append(f"{k}: в эталоне {m.group(1) if m else 'нет'}, здесь {v}")
    m = re.search(r"^W_BUYER, W_BUY7, W_POSVAL\s*=\s*([0-9.,\s]+)$", src, re.M)
    if m:
        got = [float(x) for x in m.group(1).split(",")]
        if got != [W_BUYER, W_BUY7, W_POSVAL]:
            bad.append(f"веса лоссов: в эталоне {got}, здесь {[W_BUYER, W_BUY7, W_POSVAL]}")
    if bad:
        raise SystemExit("конфиг разошёлся с эталоном:\n  " + "\n  ".join(bad))
    print(f"конфиг совпадает с эталоном {REFERENCE}")
else:
    print(f"ВНИМАНИЕ: {REFERENCE} не найден, заморозка конфига не проверена")

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    raise SystemExit("no GPU visible")
print(torch.cuda.get_device_name(0), flush=True)

gp = np.load(paths.require("run_pack.npz", "сначала запустите src/s06_run_pack.py"))
pack = np.load(paths.require("validation_pack.npz", "сначала запустите src/s08_packs.py"))
DAY0 = int(gp["day0"])
FINAL_ANCHOR = int(gp["final_anchor"])
DEV_ANCHORS = [int(a) for a in pack["train_anchors"]]
HOLD_ANCHOR = int(pack["eval_anchor"])
OOF_R = pack["residuals"]
OOF_W = pack["weights"]
assert max(DEV_ANCHORS) + HORIZON <= HOLD_ANCHOR, "causal rule violated"
print(f"train anchors {DEV_ANCHORS}\nholdout {HOLD_ANCHOR}   "
      f"max train target_end {max(DEV_ANCHORS) + HORIZON} <= {HOLD_ANCHOR}", flush=True)

paths.check_parquet()

t0 = time.time()
df = pd.read_parquet("train.parquet")
day = ((df["event_date"].to_numpy().astype("datetime64[D]").astype(np.int64)) - DAY0).astype(np.int32)
n_days = int(day.max()) + 1
uniq, urank = np.unique(df["user_id"].to_numpy(), return_inverse=True)
urank = urank.astype(np.int64)
n_users = len(uniq)
assert np.array_equal(uniq, gp["user_id"])
gmv = df["gmv"].to_numpy().astype(np.float32)
chan = {
    "log_gmv": np.log1p(gmv),
    "log_gmv_search": np.log1p(df["gmv_search"].to_numpy().astype(np.float32)),
    "log_to_ord": np.log1p(df["to_ord"].to_numpy().astype(np.float32)),
    "log_searches": np.log1p(df["searches"].to_numpy().astype(np.float32)),
    "log_to_cart": np.log1p(df["to_cart"].to_numpy().astype(np.float32)),
    "active": np.ones(len(df), dtype=np.float32),
    "purchase": (gmv > 0).astype(np.float32),
}
del df
C_USER = len(chan)
flat = urank * n_days + day
seq = np.zeros((n_users, n_days, C_USER), dtype=np.float16)
for ci, nm in enumerate(chan):
    plane = np.zeros(n_users * n_days, dtype=np.float32)
    plane[flat] = chan[nm]
    seq[:, :, ci] = plane.reshape(n_users, n_days)
    del plane
gmv_dense = np.zeros(n_users * n_days, dtype=np.float32)
gmv_dense[flat] = gmv
gmv_dense = gmv_dense.reshape(n_users, n_days)
cum_gmv = np.concatenate([np.zeros((n_users, 1), np.float64), gmv_dense.cumsum(1, dtype=np.float64)], 1)
cum_buy = np.concatenate([np.zeros((n_users, 1), np.float32),
                          (gmv_dense > 0).astype(np.float32).cumsum(1)], 1)
del gmv_dense, chan, gmv, flat
print(f"tensors built {time.time()-t0:.0f}s", flush=True)

R, BUY, BUY7, POS, WT = {}, {}, {}, {}, {}
for i, a in enumerate(DEV_ANCHORS):
    yy = (cum_gmv[:, min(a + HORIZON + 1, n_days)] - cum_gmv[:, a + 1]).astype(np.float32)
    ly = np.log1p(yy)
    pos = yy > 0
    R[a] = torch.from_numpy(OOF_R[i].astype(np.float32)).to(dev)
    BUY[a] = torch.from_numpy(pos.astype(np.float32)).to(dev)
    BUY7[a] = torch.from_numpy(((cum_buy[:, min(a + 8, n_days)] - cum_buy[:, a + 1]) > 0)
                               .astype(np.float32)).to(dev)
    POS[a] = torch.from_numpy(np.where(pos, ly - float(ly[pos].mean()), 0.0).astype(np.float32)).to(dev)
    WT[a] = float(OOF_W[i])
assert np.allclose(np.log1p((cum_gmv[:, HOLD_ANCHOR + HORIZON + 1] - cum_gmv[:, HOLD_ANCHOR + 1])),
                   pack["y_hold_log"], atol=1e-3), "holdout target mismatch"
print("holdout target matches the CPU pipeline", flush=True)
del cum_gmv, cum_buy

seq_gpu = torch.from_numpy(seq).to(dev)
del seq
dows = torch.arange(n_days, device=dev, dtype=torch.float32) + DAY0
dow_sin, dow_cos = torch.sin(2 * math.pi * (dows % 7) / 7), torch.cos(2 * math.pi * (dows % 7) / 7)
C_IN = C_USER + 3
samp = seq_gpu[::97, -WINDOW:, :].float().reshape(-1, C_USER)
ch_mean, ch_std = samp.mean(0), samp.std(0) + 1e-3
del samp


def make_window(u, a):
    lo = a - WINDOW + 1
    src_lo = max(lo, 0)
    pad = src_lo - lo
    x = torch.zeros(u.shape[0], WINDOW, C_IN, device=dev)
    x[:, pad:, :C_USER] = (seq_gpu[u, src_lo:a + 1, :].float() - ch_mean) / ch_std
    x[:, pad:, C_USER] = 1.0
    x[:, pad:, C_USER + 1] = dow_sin[src_lo:a + 1]
    x[:, pad:, C_USER + 2] = dow_cos[src_lo:a + 1]
    return x.permute(0, 2, 1).contiguous()


class Block(nn.Module):
    def __init__(self, cin, cout, d):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, padding=d, dilation=d)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=d, dilation=d)
        self.n1, self.n2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return F.gelu(self.n2(self.c2(h)) + self.skip(x))


class ResidSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(*[Block(C_IN if i == 0 else CHANNELS, CHANNELS, 2 ** i)
                                    for i in range(BLOCKS)])
        self.attn = nn.Conv1d(CHANNELS, 1, 1)
        self.trunk = nn.Sequential(nn.Linear(CHANNELS * 4, 192), nn.GELU(),
                                   nn.Dropout(DROPOUT), nn.Linear(192, 96), nn.GELU())
        self.h_r = nn.Linear(96, 1)
        self.h_buy = nn.Linear(96, 1)
        self.h_b7 = nn.Linear(96, 1)
        self.h_pos = nn.Linear(96, 1)

    def forward(self, x):
        h = self.body(x)
        w = torch.softmax(self.attn(h), -1)
        e = self.trunk(torch.cat([h.mean(-1), h.amax(-1), h[:, :, -1], (h * w).sum(-1)], 1))
        return (self.h_r(e).squeeze(-1), self.h_buy(e).squeeze(-1),
                self.h_b7(e).squeeze(-1), self.h_pos(e).squeeze(-1))


model = ResidSeq().to(dev)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WD)
spe = PAIRS_PER_EPOCH // BATCH
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR_MAX, total_steps=EPOCHS * spe, pct_start=0.2)
scaler = torch.amp.GradScaler("cuda")
start_ep = 0
if os.path.exists(CKPT):
    st = torch.load(CKPT, map_location=dev)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
    start_ep = st["epoch"] + 1
    print(f"resumed from epoch {start_ep}", flush=True)

T = pack["T"].astype(np.float64)
BASIS = pack["basis_ext"].astype(np.float64)
BASIS_NAMES = [str(x) for x in pack["basis_ext_names"]]
V_T = float(np.mean(T ** 2))
NU = len(T)


def stage_c(dnew):
    d = dnew - dnew.mean()
    G = BASIS @ BASIS.T / NU
    w = np.linalg.pinv(G) @ (BASIS @ d / NU)
    q = d - w @ BASIS
    Vq = float(np.mean(q * q))
    Cn = float(np.mean(q * T))
    return dict(kept=Vq / float(np.mean(d * d)), C_novel=Cn,
                rho_novel=Cn / np.sqrt(Vq * V_T), dMSE_novel=Cn ** 2 / Vq)


@torch.no_grad()
def predict(a, bs=4096):
    model.eval()
    out = torch.empty(n_users, device=dev)
    for i in range(0, n_users, bs):
        u = torch.arange(i, min(i + bs, n_users), device=dev)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out[i:i + u.shape[0]] = model(make_window(u, a))[0].float()
    model.train()
    return out.cpu().numpy().astype(np.float64)


per = BATCH // ANCHORS_PER_BATCH
t0 = time.time()
for ep in range(start_ep, EPOCHS):
    run = seen = 0
    for step in range(spe):
        picks = [DEV_ANCHORS[i] for i in np.random.randint(0, len(DEV_ANCHORS), ANCHORS_PER_BATCH)]
        xs, rs, bs_, b7s, ps, ws = [], [], [], [], [], []
        for a in picks:
            u = torch.randint(0, n_users, (per,), device=dev)
            xs.append(make_window(u, a))
            rs.append(R[a][u]); bs_.append(BUY[a][u]); b7s.append(BUY7[a][u]); ps.append(POS[a][u])
            ws.append(torch.full((per,), WT[a], device=dev))
        x = torch.cat(xs); r = torch.cat(rs); b = torch.cat(bs_)
        b7 = torch.cat(b7s); pv = torch.cat(ps); w = torch.cat(ws)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pr, pb, p7, pp = model(x)
            loss = (w * (pr - r) ** 2).mean() / w.mean()
            loss = loss + W_BUYER * F.binary_cross_entropy_with_logits(pb, b) \
                + W_BUY7 * F.binary_cross_entropy_with_logits(p7, b7)
            m = b > 0
            if m.any():
                loss = loss + W_POSVAL * F.mse_loss(pp[m], pv[m])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        run += float(loss.item()) * r.shape[0]; seen += r.shape[0]
        if step % 400 == 0:
            print(f"  ep{ep} {step}/{spe} loss={run/max(seen,1):.4f} ({time.time()-t0:.0f}s)", flush=True)
    zh = predict(HOLD_ANCHOR)
    sc = stage_c(zh)
    print(f"эпоха {ep}  kept={sc['kept']:.4f}  rho_novel={sc['rho_novel']:+.5f}  "
          f"dMSE_novel={sc['dMSE_novel']:.5f}  ({time.time()-t0:.0f}с)", flush=True)
    np.save(paths.snapshot(f"resid_snap{ep}.npy"), zh)
    np.save(paths.snapshot(f"pred_snap{ep}.npy"), predict(FINAL_ANCHOR))
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": ep}, CKPT)

print(f"\nbasis for Stage C: {BASIS_NAMES}")
print(f"\n{'epoch':>5} {'kept':>7} {'rho_novel':>10} {'dMSE_novel':>11}")
hist = {}
for ep in range(EPOCHS):
    f = paths.snapshot(f"resid_snap{ep}.npy")
    if not os.path.exists(f):
        continue
    zh = np.load(f)
    sc = stage_c(zh)
    hist[ep] = dict(sc)
    mark = "  <- отправлена, эпоха по правилу" if ep == SHIP_EPOCH else ""
    print(f"{ep:>5} {sc['kept']:>7.4f} {sc['rho_novel']:>+10.5f} "
          f"{sc['dMSE_novel']:>11.6f}{mark}")

if hist:
    arg = max(hist, key=lambda e: hist[e]["dMSE_novel"])
    print(f"\nтолько диагностика: argmax dMSE на 378 указал бы на эпоху {arg}, "
          f"не используется")

ship = np.load(paths.snapshot(f"resid_snap{SHIP_EPOCH}.npy"))
sc = stage_c(ship)
print(f"\nОТПРАВЛЕНА эпоха {SHIP_EPOCH} (терминальная, правило SHIP_EPOCH = EPOCHS-1)")
print(f"  Stage C vs EXT6    : kept {sc['kept']:.4f}  C_novel {sc['C_novel']:+.6f}  "
      f"rho_novel {sc['rho_novel']:+.5f}  dMSE_novel {sc['dMSE_novel']:.6f}")
print(f"\nсправочно, замер с загрязнённым протоколом на том же базисе: "
      f"rho_novel +0.05526  dMSE_novel 0.015992")
print(f"retention = dMSE_novel / 0.015992 = "
      f"{sc['dMSE_novel'] / 0.015992:.3f}")
np.save(paths.artifact("hold_seq11.npy"), ship)
np.save(paths.artifact("pred_seq11.npy"), np.load(paths.snapshot(f"pred_snap{SHIP_EPOCH}.npy")))
json.dump({"ship_epoch": SHIP_EPOCH, "train_anchors": DEV_ANCHORS,
           "eval_anchor": HOLD_ANCHOR, "basis": BASIS_NAMES,
           "history": {str(k): v for k, v in hist.items()},
           "shipped": dict(sc)},
          open(paths.artifact("train_validation.json"), "w"), indent=1)
print("wrote hold_seq11.npy, pred_seq11.npy, train_validation.json")
