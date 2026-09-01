"""Стадия 09. Обучение SEQ408 на границе развёртывания 408. Только инференс.

Тот же замороженный принцип, что и в стадии 10, с одним содержательным отличием -
обучающий набор. На границе 408 легален каждый якорь, чьё 30-дневное окно уже
закрылось, поэтому в обучение возвращаются 351, 365 и 378, и якорей становится
семнадцать вместо четырнадцати.

Отправляется эпоха 11 - терминальная эпоха расписания OneCycleLR, то есть
SHIP_EPOCH = EPOCHS - 1, а не эпоха, выбранная по какому-либо результату.
Скрипт при старте перечитывает config/frozen_config_reference.py, разбирает
оттуда все гиперпараметры и отказывается работать при любом расхождении.
Сверяются именно гиперпараметры: отбор эпохи по argmax dMSE, как в эталонном
скрипте, здесь намеренно не воспроизводится.

Пишет artifacts/snapshots/deploy_snap*.npy и artifacts/pred_seq11_408.npy.
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
CKPT = paths.artifact("seq_deploy_ckpt.pt")
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
    if m and [float(x) for x in m.group(1).split(",")] != [W_BUYER, W_BUY7, W_POSVAL]:
        bad.append("веса вспомогательных лоссов расходятся с эталоном")
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

gp = np.load(paths.require("run_pack.npz", "run src/build_run_pack.py first"))
pack = np.load(paths.require("deploy_pack.npz", "сначала запустите src/s08_packs.py"))
DAY0 = int(gp["day0"])
FINAL_ANCHOR = int(gp["final_anchor"])
CUTOFF = int(pack["cutoff"])
DEV_ANCHORS = [int(a) for a in pack["train_anchors"]]
OOF_R = pack["residuals"]
OOF_W = pack["weights"]
assert FINAL_ANCHOR == CUTOFF, "pack cutoff and gate_pack final_anchor disagree"
assert max(DEV_ANCHORS) + HORIZON <= CUTOFF, "causal rule violated"
print(f"train anchors {DEV_ANCHORS}")
print(f"cutoff {CUTOFF}   max train target_end {max(DEV_ANCHORS)+HORIZON} <= {CUTOFF}")

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
    pf = predict(FINAL_ANCHOR)
    np.save(paths.snapshot(f"deploy_snap{ep}.npy"), pf)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": ep}, CKPT)
    print(f"epoch {ep}  sd(pred_408) {pf.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

ship = np.load(paths.snapshot(f"deploy_snap{SHIP_EPOCH}.npy"))
np.save(paths.artifact("pred_seq11_408.npy"), (ship - ship.mean()).astype(np.float32))
shipc = ship - ship.mean()
vp = paths.artifact("pred_seq11.npy")
val = None
if os.path.exists(vp):
    val = np.load(vp).astype(np.float64)
    val = val - val.mean()
print(f"\nshipped epoch {SHIP_EPOCH}, centred, to pred_seq11_408.npy")
if val is not None:
    print(f"correlation with the validation twin's 408 prediction "
          f"(trained on 14 anchors): {np.corrcoef(shipc, val)[0,1]:.5f}")
    print(f"sd deploy {shipc.std():.5f}   sd validation twin {val.std():.5f}")
else:
    print(f"sd deploy {shipc.std():.5f}   (validation twin absent, not compared)")
json.dump({"ship_epoch": SHIP_EPOCH, "cutoff": CUTOFF,
           "train_anchors": DEV_ANCHORS,
           "newly_legal_vs_eval378": [a for a in DEV_ANCHORS if a + HORIZON > 378],
           "corr_with_validation_twin": (float(np.corrcoef(shipc, val)[0, 1])
                                        if val is not None else None),
           "sd_deploy": float(shipc.std()), "sd_validation_twin": (float(val.std()) if val is not None else None),
           "local_verdict": "none possible; 378 is in-sample here"},
          open(paths.artifact("seq_deploy.json"), "w"), indent=1)
print("wrote seq_deploy.json")
