"""Весь конвейер одной командой.

    py -3.11 run_all.py                 всё, пропуская уже посчитанное
    py -3.11 run_all.py --list          показать стадии и их состояние
    py -3.11 run_all.py --only 09       одна стадия
    py -3.11 run_all.py --from 07       с этой стадии и дальше
    py -3.11 run_all.py --force         пересчитать даже готовое
    py -3.11 run_all.py --skip-train    без двух GPU-обучений

Стадия пропускается, если её выход уже на месте: конвейер можно останавливать и
продолжать. Каждая стадия - самостоятельный скрипт в src/, который можно
запустить и отдельно.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts")
SUB = os.path.join(ROOT, "submissions")

STAGES = [
    ("01", "s01_features_base.py", "агрегаты по окнам -> X.npy",
     f"{DATA}/features/X.npy", [], False),
    ("02", "s02_features_spans.py", "непересекающиеся интервалы -> X2.npy",
     f"{DATA}/features/X2.npy", [], False),
    ("03", "s03_features_ranks.py", "ранги внутри якоря -> X4.npy",
     f"{DATA}/features/X4.npy", [], False),
    ("04", "s04_features_x6.py", "затухания, покупки, максимумы -> X6.npy",
     f"{DATA}/features/X6.npy", [], False),
    ("05", "s05_oof_panel.py", "панель остатков без утечки (purge 42)",
     f"{DATA}/oof_panel.npz", ["--gpu"], True),
    ("06", "s06_run_pack.py", "календарь и порядок пользователей",
     f"{DATA}/run_pack.npz", [], False),
    ("07", "s07_basis.py", "шесть направлений базиса + вторая база",
     f"{ART}/basis/meta.json", [], True),
    ("08", "s08_packs.py", "пакеты для обучения: 378 и 408",
     f"{DATA}/deploy_pack.npz", [], False),
    ("09", "s09_train_deploy.py", "SEQ408 на границе 408 -> отправка",
     f"{ART}/pred_seq11_408.npy", [], True),
    ("10", "s10_train_validation.py", "SEQ408 на 378 -> доказательство",
     f"{ART}/hold_seq11.npy", [], True),
    ("11", "s11_report.py", "двухстадийный отчёт по валидации",
     f"{ART}/report_validation.json", [], False),
    ("12", "s12_submission.py", "база + t * SEQ408 -> submission_stage12.csv",
     f"{SUB}/submission_stage12.csv", [], False),
    ("13", "s13_final.py", "сборка из измеренных осей -> submission_final.csv",
     f"{SUB}/submission_final.csv", [], False),
    ("14", "s14_base_deploy.py", "самодостаточная база на границе 408",
     f"{SUB}/base_deploy.csv", [], True),
]
TRAIN_STAGES = {"09", "10"}

ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true")
ap.add_argument("--only")
ap.add_argument("--from", dest="start")
ap.add_argument("--force", action="store_true")
ap.add_argument("--skip-train", action="store_true")
args = ap.parse_args()

if args.list:
    print(f"{'ст':>3} {'скрипт':26s} {'готово':>7s}  описание")
    for num, script, desc, out, _, _ in STAGES:
        print(f"{num:>3} {script:26s} {'да' if os.path.exists(out) else '-':>7s}  {desc}")
    raise SystemExit(0)

todo = STAGES
if args.only:
    todo = [s for s in STAGES if s[0] == args.only]
    if not todo:
        raise SystemExit(f"нет стадии {args.only}")
elif args.start:
    idx = [i for i, s in enumerate(STAGES) if s[0] == args.start]
    if not idx:
        raise SystemExit(f"нет стадии {args.start}")
    todo = STAGES[idx[0]:]

NEEDS_DATA = {"01", "02", "04", "06", "09", "10"}
# 14 работает от features/, сырые данные ей не нужны
if (any(n in NEEDS_DATA for n, *_ in todo)
        and not os.path.exists(os.path.join(DATA, "train.parquet"))):
    need = sorted(n for n, *_ in todo if n in NEEDS_DATA)
    msg = ["стадиям " + ", ".join(need) + " нужен train.parquet в " + DATA,
           "  без него доступны стадии 11-13: отчёт и сборка отправок",
           "  из артефактов, без сырых данных"]
    raise SystemExit(chr(10).join(msg))

t_all = time.time()
done = skipped = 0
for num, script, desc, out, extra, heavy in todo:
    if args.skip_train and num in TRAIN_STAGES:
        print(f"[{num}] пропуск (--skip-train): {desc}")
        skipped += 1
        continue
    if os.path.exists(out) and not args.force:
        print(f"[{num}] готово, пропуск: {desc}")
        skipped += 1
        continue
    print(f"\n[{num}] {desc}" + ("   (долго)" if heavy else ""))
    print(f"      py -3.11 src/{script} {' '.join(extra)}".rstrip(), flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", script)] + extra,
                       cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"\n[{num}] {script} завершился с кодом {r.returncode}")
    if not os.path.exists(out):
        raise SystemExit(f"\n[{num}] {script} отработал, но не создал {out}")
    print(f"[{num}] готово за {time.time() - t:.0f}с")
    done += 1

print(f"\nвыполнено {done}, пропущено {skipped}, всего {time.time() - t_all:.0f}с")
if not args.only:
    print("\nрезультат:")
    for p, what in ((f"{ART}/pred_seq11_408.npy", "направление SEQ408"),
                    (f"{SUB}/submission_stage12.csv", "сборка стадии 12"),
                    (f"{SUB}/submission_final.csv", "финальная отправка"),
                    (f"{ART}/report_validation.json", "отчёт валидации")):
        print(f"  {'есть' if os.path.exists(p) else 'нет '}  "
              f"{os.path.relpath(p, ROOT)}  - {what}")
