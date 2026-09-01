"""Обход графа зависимостей: что из чего выводится при старте с нуля.
    py -3.11 check_pipeline.py
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = {"data/train.parquet"}

GRAPH = {
    "features/X.npy":          ("s01_features_base.py", ["data/train.parquet"]),
    "features/meta.json":      ("s01_features_base.py", ["data/train.parquet"]),
    "features/y.npy":          ("s01_features_base.py", ["data/train.parquet"]),
    "features/X2.npy":         ("s02_features_spans.py", ["data/train.parquet"]),
    "features/X4.npy":         ("s03_features_ranks.py", ["features/X.npy"]),
    "features/X6.npy":         ("s04_features_x6.py",
                                ["data/train.parquet", "features/meta.json"]),
    "oof_panel.npz":           ("s05_oof_panel.py",
                                ["features/X.npy", "features/X2.npy",
                                 "features/X4.npy", "features/y.npy"]),
    "run_pack.npz":            ("s06_run_pack.py",
                                ["data/train.parquet", "features/meta.json"]),
    "artifacts/basis/":        ("s07_basis.py",
                                ["features/X.npy", "features/X2.npy",
                                 "features/X4.npy", "features/X6.npy"]),
    "artifacts/basis/p0_causal30_378.npy": ("s07_basis.py",
                                            ["features/X.npy"]),
    "deploy_pack.npz":         ("s08_packs.py",
                                ["oof_panel.npz", "artifacts/basis/"]),
    "validation_pack.npz":     ("s08_packs.py",
                                ["oof_panel.npz", "artifacts/basis/"]),
    "deploy_snap*.npy":    ("s09_train_deploy.py",
                                ["data/train.parquet", "deploy_pack.npz"]),
    "pred_seq11_408.npy": ("s09_train_deploy.py", ["deploy_snap*.npy"]),
    "resid_snap*.npy":     ("s10_train_validation.py",
                                ["data/train.parquet", "run_pack.npz",
                                 "validation_pack.npz"]),
    "hold_seq11.npy":  ("s10_train_validation.py", ["resid_snap*.npy"]),
    "report_validation.json":  ("s11_report.py",
                                ["resid_snap*.npy", "validation_pack.npz",
                                 "artifacts/basis/p0_causal30_378.npy"]),
    "submissions/base_deploy.csv": ("s14_base_deploy.py",
                                ["features/X.npy", "features/X2.npy",
                                 "features/X4.npy", "features/y.npy"]),
    "submission_seq408":     ("s12_submission.py",
                                ["pred_seq11_408.npy",
                                 "submissions/sub_stage12_base.csv",
                                 "config/blend.json"]),
    "submission_FINAL":        ("s13_final.py",
                                ["submissions/sub_stage13_base.csv",
                                 "config/final_blend.json", "config/axes"]),
    "submissions/sub_stage12_base.csv": (None, []),
    "submissions/sub_stage13_base.csv": (None, []),
    "config/blend.json":               (None, []),
    "config/final_blend.json":         (None, []),
    "config/axes":                     (None, []),
}
# лежат в репозитории и ниоткуда не выводятся: отправленные файлы и коэффициенты,
# измеренные по обратной связи метрики
IN_REPO = {"submissions/sub_stage12_base.csv", "submissions/sub_stage13_base.csv",
           "config/blend.json", "config/final_blend.json", "config/axes"}

MODEL = ["features/X.npy", "features/X2.npy", "features/X4.npy",
         "oof_panel.npz", "run_pack.npz", "deploy_pack.npz",
         "deploy_snap*.npy", "pred_seq11_408.npy",
         "submission_seq408", "submission_FINAL"]
EVIDENCE = ["submissions/base_deploy.csv", "features/X6.npy", "artifacts/basis/",
            "artifacts/basis/p0_causal30_378.npy", "validation_pack.npz",
            "resid_snap*.npy", "hold_seq11.npy",
            "report_validation.json"]

state, why = {}, {}


def resolve(a, stack=()):
    if a in state:
        return state[a]
    if a in RAW:
        state[a], why[a] = True, "исходные данные"
        return True
    if a in IN_REPO:
        ok = os.path.exists(os.path.join(ROOT, a))
        state[a], why[a] = ok, "лежит в репозитории" if ok else "НЕТ В РЕПОЗИТОРИИ"
        return ok
    if a in stack:
        state[a], why[a] = False, "циклическая зависимость"
        return False
    script, inputs = GRAPH.get(a, (None, []))
    if script is None:
        state[a], why[a] = False, "СБОРЩИКА НЕТ"
        return False
    if not os.path.exists(os.path.join(ROOT, "src", script)):
        state[a], why[a] = False, f"нет src/{script}"
        return False
    bad = [i for i in inputs if not resolve(i, stack + (a,))]
    state[a] = not bad
    why[a] = f"заблокировано: {bad[0]}" if bad else f"src/{script}"
    return state[a]


for group, title in ((MODEL, "МОДЕЛЬ  (то, что отправлялось)"),
                     (EVIDENCE, "ДОКАЗАТЕЛЬСТВА  (валидация)")):
    print(f"\n=== {title} ===")
    w = max(len(a) for a in group)
    for a in group:
        ok = resolve(a)
        print(f"  {'да ' if ok else 'НЕТ'}  {a:{w}s}  {why[a]}")

nm = sum(1 for a in MODEL if state[a])
ne = sum(1 for a in EVIDENCE if state[a])
print(f"\nмодель {nm}/{len(MODEL)}, доказательства {ne}/{len(EVIDENCE)}")
