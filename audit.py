"""Полная проверка репозитория: синтаксис, импорты, файлы, ссылки, выходы.
    py -3.11 audit.py
"""
import ast
import importlib.util
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
fails, warns = [], []

ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def check(name, ok, detail="", note=""):
    tail = note if ok else detail
    print(f"  {'ok  ' if ok else 'ОШИБКА'}  {name}" + (f"  - {tail}" if tail else ""))
    if not ok:
        fails.append(name)


def warn(name, detail):
    print(f"  ?     {name}  - {detail}")
    warns.append(name)


PY = ([os.path.join("src", f) for f in sorted(os.listdir("src")) if f.endswith(".py")]
      + ["run_all.py", "check_pipeline.py", "audit.py"])

print("\n=== 1. синтаксис ===")
trees = {}
for p in PY:
    try:
        trees[p] = ast.parse(io.open(p, encoding="utf-8").read())
    except SyntaxError as e:
        check(p, False, f"строка {e.lineno}: {e.msg}")
if len(trees) == len(PY):
    check(f"{len(PY)} файлов разбираются", True)

print("\n=== 2. сторонние импорты ===")
mods = set()
for t in trees.values():
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split(".")[0])
STD = set(sys.stdlib_module_names) | {"paths"}
PYPI = {"sklearn": "scikit-learn"}
req = io.open("requirements.txt", encoding="utf-8").read().lower()
for m in sorted(mods - STD):
    check(m, importlib.util.find_spec(m) is not None, "НЕ УСТАНОВЛЕН")
    pkg = PYPI.get(m, m)
    check(f"{pkg} в requirements.txt",
          re.search(rf"^\s*{re.escape(pkg)}\b", req, re.M) is not None,
          "импортируется, но не объявлен - свежий клон его не поставит")

print("\n=== 3. докстринги и язык ===")
for p in PY:
    d = ast.get_docstring(trees[p])
    if not d:
        warn(p, "нет докстринга")
    elif not re.search(r"[а-яА-Я]", d.split("\n")[0]):
        warn(p, "первая строка докстринга не на русском")
if not warns:
    check("у всех стадий есть русский докстринг", True)

print("\n=== 4. файлы, на которые ссылается код ===")
PRODUCED = {
    "features/X.npy", "features/X2.npy", "features/X4.npy", "features/X6.npy",
    "features/y.npy", "features/meta.json", "features/meta2.json",
    "features/meta4.json", "features/meta6.json", "features/anchor.npy",
    "features/user_id.npy", "features/oof_panel_meta.json", "features/names.json",
    "oof_panel.npz", "run_pack.npz", "validation_pack.npz", "deploy_pack.npz",
    "report_pack.npz", "train.parquet",
}
OUTPUTS = {"submissions/submission_stage12.csv", "submissions/submission_final.csv",
           "submissions/base_deploy.csv", "submissions/probes"}
PAT = re.compile(r'["\']((?:submissions|config|artifacts)/[A-Za-z0-9_./{}-]+'
                 r'|features/[A-Za-z0-9_.{}-]+|[a-z0-9_]+\.(?:npz|parquet))["\']')
missing = []
for p in PY:
    src = io.open(p, encoding="utf-8").read()
    for m in PAT.findall(src):
        if "{" in m or m in PRODUCED:
            continue
        if m in OUTPUTS:
            continue
        if not os.path.exists(os.path.join(ROOT, m)):
            missing.append(f"{p} -> {m}")
check("все статические пути существуют", not missing,
      "; ".join(missing[:4]) if missing else "")

print("\n=== 5. конфиги ===")
for p, keys in (("config/blend.json", ["t", "база"]),
                ("config/final_blend.json", ["база", "коэффициенты"])):
    try:
        d = json.load(io.open(p, encoding="utf-8"))
        miss = [k for k in keys if k not in d]
        check(p, not miss, f"нет ключей {miss}")
    except Exception as e:
        check(p, False, str(e))
axes = json.load(io.open("config/final_blend.json", encoding="utf-8"))["коэффициенты"]
for a in axes:
    check(f"config/axes/{a}.npy", os.path.exists(f"config/axes/{a}.npy"))

print("\n=== 6. ссылки в README ===")
md = io.open("README.md", encoding="utf-8").read()
refs = set(re.findall(r"\]\(([^)#][^)]*)\)", md))
refs |= set(re.findall(r'(?:src|srcset)="([^"]+)"', md))
for link in sorted(refs):
    if link.startswith("http"):
        continue
    check(f"README -> {link}", os.path.exists(os.path.join(ROOT, link)))

print("\n=== 7. .gitignore ===")
gi = io.open(".gitignore", encoding="utf-8").read().split()
for must in ("data/train.parquet", "data/features/", "__pycache__/"):
    check(f".gitignore: {must}", must in gi)
for out in ("submissions/submission_final.csv", "submissions/submission_stage12.csv"):
    check(f".gitignore: {out}", out in gi, "выход конвейера не должен уезжать в git")

print("\n=== 8. отправки на месте ===")
for f in ("sub_stage12_base.csv", "sub_stage12_ref.csv",
          "sub_stage13_base.csv", "sub_stage13_ref.csv", "MANIFEST.md"):
    check(f"submissions/{f}", os.path.exists(f"submissions/{f}"))

print("\n=== 9. артефакты для свежего клона ===")
for f in ("report_pack.npz", "pred_seq11_408.npy", "hold_seq11.npy",
          "basis/meta.json", "snapshots/resid_snap11.npy"):
    check(f"artifacts/{f}", os.path.exists(f"artifacts/{f}"))
b = json.load(io.open("artifacts/basis/meta.json", encoding="utf-8"))
for d in b["extended"]:
    check(f"artifacts/basis/hold_{d}.npy", os.path.exists(f"artifacts/basis/hold_{d}.npy"))
check("artifacts/basis/p0_causal30_378.npy",
      os.path.exists("artifacts/basis/p0_causal30_378.npy"))
check("12 эпох валидации", all(os.path.exists(f"artifacts/snapshots/resid_snap{i}.npy")
                               for i in range(12)))
check("12 эпох развёртывания", all(os.path.exists(f"artifacts/snapshots/deploy_snap{i}.npy")
                                   for i in range(12)))

print("\n=== 10. стадии без сырых данных отрабатывают ===")
for st, script in (("11", "s11_report.py"), ("12", "s12_submission.py"),
                   ("13", "s13_final.py")):
    r = subprocess.run([sys.executable, os.path.join("src", script)],
                       capture_output=True, text=True, cwd=ROOT,
                       encoding="utf-8", errors="replace", env=ENV)
    out = (r.stdout or "") + (r.stderr or "")
    ok = r.returncode == 0
    tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
    check(f"стадия {st} ({script})", ok, "" if ok else tail[0][:70])

print("\n=== 11. сборки совпадают с эталонами ===")
import numpy as np
import pandas as pd


def logv(p):
    d = pd.read_csv(p, dtype={"user_id": np.int64, "predict": np.float64})
    d = d.sort_values("user_id")
    return np.log1p(d["predict"].to_numpy())


SCORE = {}
for cfg in ("config/blend.json", "config/final_blend.json"):
    try:
        d = json.load(io.open(cfg, encoding="utf-8"))
        SCORE[d.get("результат") or ""] = d.get("счёт_результата")
    except Exception:
        pass
SCORE.setdefault("sub_stage13_ref.csv",
                 json.load(io.open("config/final_blend.json",
                                   encoding="utf-8")).get("счёт_результата"))

for out, ref, lim in (("submission_stage12.csv", "sub_stage12_ref.csv", 1e-4),
                      ("submission_final.csv", "sub_stage13_ref.csv", 1e-4)):
    po, pr = f"submissions/{out}", f"submissions/{ref}"
    if not os.path.exists(po):
        check(f"{out}", False, "не создан")
        continue
    a, b_ = logv(po), logv(pr)
    frac = float(np.mean((a - b_) ** 2)) / float(np.var(b_))
    check(f"{out} vs {ref}", frac < lim, f"доля дисперсии {frac:.2e}",
          f"corr {np.corrcoef(a, b_)[0, 1]:.8f}, расхождение {frac:.2e} дисперсии")

print("\n=== 12. граф зависимостей ===")
r = subprocess.run([sys.executable, "check_pipeline.py"], capture_output=True,
                   text=True, cwd=ROOT, encoding="utf-8", errors="replace", env=ENV)
txt = r.stdout or ""
m = re.search(r"модель (\d+)/(\d+), доказательства (\d+)/(\d+)", txt)
check("check_pipeline.py", m is not None and m.group(1) == m.group(2)
      and m.group(3) == m.group(4), m.group(0) if m else "не разобран",
      m.group(0) if m else "")

print(f"\n{'=' * 62}")
if fails:
    print(f"ОШИБОК: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
else:
    print("все проверки пройдены")
if warns:
    print(f"предупреждений: {len(warns)}")
sys.exit(1 if fails else 0)
