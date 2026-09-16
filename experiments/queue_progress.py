"""KUYRUK9: kalan işlerin listesi ve ilerleme hesabı (yüzde, kalan süre, tahmini bitiş).

İş listesinin tek kaynağı burasıdır; `experiments/run_fovea_final.ps1` listeyi `--plan` çıktısından okur ve iş sürerken
her 30 saniyede `--poll` ile yüzdeyi alır (pencerenin üstündeki çubuk ve pencere başlığı).

Hesap: iş birimi sonuç tablosundaki satırdır. Kuyruk başında (`--plan`) her işin eksik satırları ve satır başına tahmini
süresi (aynı veri/yapılandırma/model için daha önce ölçülmüş `sure_s`) results/logs/kuyruk9_plan.json'a yazılır.
Betikler var olan satırları atladığı için kuyruk yarıda kesilip yeniden başlatılırsa plan yalnızca kalan işi kapsar.
Yoklamada tabloya eklenen satırlar sayılır; Adım 1 işlerinde o anki satırın epoch ilerlemesi de eklenir. Geçen gerçek süre
ile tamamlanan tahmini süre arasındaki oran (0.6–2.5) kalan tahmine uygulanır. Yüzde hiç geri gitmez.

Yoklama yalnızca standart kütüphaneyi kullanır (torch/pandas yüklenmez); `--plan` sabitleri asıl modüllerle karşılaştırır.
Kuyruğa giden metinler ASCII'dir (PowerShell konsol kod sayfası).

Çalıştırma: python -m experiments.queue_progress --plan [--deneme]
            python -m experiments.queue_progress --poll --job 3 --done=0,1,2 --log results/logs/kuyruk9_x.out.log
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"

DISPLAY = {"brain": "beyin MR (Cheng)", "covidqu": "akciğer grafisi (COVID-QU-Ex)"}  # foveahe.data.DISPLAY
ASCII_NAME = {"brain": "beyin", "covidqu": "akciger"}
RAKIP = "şifreli özet (ImageNet ResNet-18, 512)"  # experiments.fovea_baselines.METHOD
SEEDS_ALL, SEEDS_NEW = [0, 1, 2, 3, 4], [1, 2, 3, 4]
BRAIN_INFO = ["tam", "tam_pencere", "F32_G16", "F64_G32", "U32", "U32_pencere", "U64", "U64_pencere", "U90",
              "U90_pencere"]
CXR_INFO = ["tam", "F32_G16", "F64_G32", "U32", "U64"]
MODEL_CFG = {"brain": ["U512", "F32_G16", "F64_G32", "U64"], "covidqu": ["U256", "F32_G16", "F64_G32", "U64"]}
MODELS = ["D", "D2", "C"]
EPOCH_LINES = {"brain": 5 * 12, "covidqu": 5}  # fovea_info satır başına: beyinde 5 kat × 12 epoch, CXR 1 bölme × 5
INFO_COLS = ["veri", "yapilandirma", "tohum"]
MODEL_COLS = ["veri", "model", "temsil", "tohum"]
RAKIP_COLS = ["veri", "yontem", "model", "tohum"]


def norm(v) -> str:
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def keys(rows: list[dict], cols) -> set:
    return {tuple(norm(r.get(c)) for c in cols) for r in rows}


def median_or(values, default: float) -> float:
    values = [v for v in values if v > 0]
    return float(statistics.median(values)) if values else default


def seconds(rows, col="sure_s", **match) -> list[float]:
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()) and r.get(col):
            try:
                out.append(float(r[col]))
            except ValueError:
                pass
    return out


def job(key, label, module, args=(), table=None, cols=None, units=(), est=(), fixed=0.0, epochs=None,
        optional=False) -> dict:
    units = [[norm(p) for p in u] for u in units]
    return {"key": key, "label": label, "module": module, "args": [str(a) for a in args],
            "table": str(table) if table else None, "cols": cols, "units": units, "est": [float(e) for e in est],
            "fixed": float(fixed), "epochs": epochs, "optional": optional}


def real_jobs() -> list[dict]:
    from experiments.fovea_baselines import METHOD
    from experiments.fovea_budget import BUDGETS, grid
    from foveahe.data import DISPLAY as display
    assert display == DISPLAY and METHOD == RAKIP, "queue_progress sabitleri asıl modüllerle uyuşmuyor"

    info, models = read_rows(TABLES / "cozum_bilgi.csv"), read_rows(TABLES / "cozum_modeller.csv")
    rakip, butce = read_rows(TABLES / "cozum_rakipler.csv"), read_rows(TABLES / "cozum_butce.csv")
    jobs = []

    for name, configs, seeds in (("brain", BRAIN_INFO, SEEDS_ALL), ("covidqu", CXR_INFO, SEEDS_NEW)):
        veri, have = DISPLAY[name], keys(info, INFO_COLS)
        timed = [(float(r["sure_s"]), norm(r["tohum"])) for r in info
                 if r["veri"] == veri and r.get("sure_s") and r["yapilandirma"] != "sabit"]
        recent = [t for t, s in timed if s != "0"]  # dünkü koşu: GPU paylaşımsız ölçüm
        per = median_or(recent if len(recent) >= 3 else [t for t, _ in timed], 600.0)
        units = [(veri, c, s) for c in configs for s in seeds if (veri, c, str(s)) not in have]
        jobs.append(job(f"adim1_{name}", f"Adim 1 bilgi kaybi: {ASCII_NAME[name]}", "experiments.fovea_info",
                        ["--dataset", name, "--configs", *configs, "--seeds", *seeds], TABLES / "cozum_bilgi.csv",
                        INFO_COLS, units, [per] * len(units), fixed=20 * len({u[1] for u in units}),
                        epochs=EPOCH_LINES[name]))
    jobs.append(job("adim1_sekil", "Adim 1 sekil ve tablo", "experiments.fovea_info", ["--plot-only"], fixed=60))

    for name in ("brain", "covidqu"):
        veri, have = DISPLAY[name], keys(models, MODEL_COLS)
        units, est = [], []
        for cfg in MODEL_CFG[name]:
            for m in MODELS:
                for s in SEEDS_NEW:
                    if (veri, m, cfg, str(s)) not in have:
                        units.append((veri, m, cfg, s))
                        est.append(median_or(seconds(models, veri=veri, model=m, temsil=cfg),
                                             median_or(seconds(models, veri=veri, model=m), 300.0)))
        jobs.append(job(f"adim2_{name}", f"Adim 2 sifreli modeller: {ASCII_NAME[name]}", "experiments.fovea_models",
                        ["--dataset", name, "--configs", *MODEL_CFG[name], "--seeds", *SEEDS_NEW, "--device", "cuda"],
                        TABLES / "cozum_modeller.csv", MODEL_COLS, units, est, fixed=40 * len({u[2] for u in units})))

    for s in SEEDS_NEW:  # rakip betiği satır atlamaz: satırları tamamsa iş atlanır (plan birimi yok)
        units = [(DISPLAY[n], RAKIP, m, s) for n in ("brain", "covidqu") for m in ("D", "D2")
                 if (DISPLAY[n], RAKIP, m, str(s)) not in keys(rakip, RAKIP_COLS)]
        est = [median_or(seconds(rakip, "egitim_s", veri=u[0], model=u[2]), 120.0) for u in units]
        jobs.append(job(f"adim5_rakip_t{s}", f"Adim 5a sifreli ozet rakibi: tohum {s}", "experiments.fovea_baselines",
                        ["--seed", s, "--skip-he"], TABLES / "cozum_rakipler.csv", RAKIP_COLS, units, est,
                        fixed=90 if units else 0))

    configs = list(dict.fromkeys(grid(BUDGETS).yapilandirma))
    have = keys(butce, MODEL_COLS)
    units, est = [], []
    for name in ("brain", "covidqu"):
        for cfg in configs:
            for m in ("C", "D"):
                if (DISPLAY[name], m, cfg, "0") not in have:
                    units.append((DISPLAY[name], m, cfg, 0))
                    est.append(median_or(seconds(models, veri=DISPLAY[name], model=m), 300.0))
    jobs.append(job("bonus_butce", "Bonus: butce duyarli odak", "experiments.fovea_budget", [],
                    TABLES / "cozum_butce.csv", MODEL_COLS, units, est, fixed=45 * len(configs)))

    jobs.append(job("bonus_genelleme", "Bonus: Kaggle capraz kaynak genelleme", "experiments.fovea_transfer", [],
                    fixed=1200, optional=True))
    jobs.append(job("adim6_sekil", "Adim 6 birlesik sekil ve tablo", "experiments.fovea_pareto", [], fixed=30))
    jobs.append(job("ozet", "OZET.md", "experiments.summarize", [], fixed=20))
    return jobs


def fake_jobs() -> list[dict]:
    table = Path(tempfile.gettempdir()) / f"kuyruk9_deneme_{int(time.time())}.csv"
    units = [("deneme", "sahte", u) for u in range(3)]
    return [job("deneme_tablo", "Deneme: tablo satirlari", "experiments.queue_progress",
                ["--sahte-is", 3, "--sahte-tablo", table], table, INFO_COLS, units, [2.4] * 3, fixed=1, epochs=4),
            job("deneme_hata", "Deneme: hatali cikis", "experiments.queue_progress", ["--sahte-is", 0, "--cikis", 3],
                fixed=1),
            job("deneme_yok", "Deneme: olmayan betik", "experiments.boyle_bir_betik_yok", fixed=1, optional=True)]


def plan_path(deneme: bool) -> Path:
    return LOGS / ("kuyruk9_deneme_plan.json" if deneme else "kuyruk9_plan.json")


def dur(s: float) -> str:
    h, m = divmod(int(round(s / 60)), 60)
    return f"{h} sa {m:02d} dk" if h else f"{m} dk"


def make_plan(deneme: bool) -> None:
    jobs = fake_jobs() if deneme else real_jobs()
    for i, j in enumerate(jobs):
        j["idx"] = i
    plan = {"baslangic": time.time(), "jobs": jobs}
    path = plan_path(deneme)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    total = sum(sum(j["est"]) + j["fixed"] for j in jobs)
    for j in jobs:
        print(f"# {j['idx'] + 1:2d}. {j['label']:42s} {len(j['units']):3d} satir  ~{dur(sum(j['est']) + j['fixed'])}")
        print("\t".join(["IS", str(j["idx"]), j["key"], j["label"], j["module"], " ".join(j["args"]),
                         "1" if j["optional"] else "0", "1" if (j["table"] and not j["units"]) else "0"]))
    print("\t".join(["PLAN", f"{len(jobs)} is, {sum(len(j['units']) for j in jobs)} satir, tahmini ~{dur(total)}"]))


def row_fraction(log: Path, epoch_lines: int) -> float:
    """Günlükte son tamamlanan satırdan sonraki epoch satırları / satır başına epoch satırı."""
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return 0.0
    n = 0
    for line in lines:
        if "tohum=" in line and "AUC=" in line:
            n = 0
        elif "epoch " in line:
            n += 1
    return min(0.98, n / epoch_lines)


def poll(deneme: bool, idx: int, done_jobs: set, log: str | None) -> str:
    path = plan_path(deneme)
    plan = json.loads(path.read_text(encoding="utf-8"))
    jobs, now = plan["jobs"], time.time()
    total = sum(sum(j["est"]) + j["fixed"] for j in jobs) or 1.0
    done = sum(sum(j["est"]) + j["fixed"] for j in jobs if j["idx"] in done_jobs)
    cur = jobs[idx]
    n_done, n_all = 0, len(cur["units"])
    if idx not in done_jobs and cur["table"] and cur["units"]:
        present = keys(read_rows(Path(cur["table"])), cur["cols"])
        finished = [e for u, e in zip(cur["units"], cur["est"]) if tuple(u) in present]
        n_done = len(finished)
        done += sum(finished) + cur["fixed"] * n_done / n_all
        if cur["epochs"] and log and n_done < n_all:
            nxt = next(e for u, e in zip(cur["units"], cur["est"]) if tuple(u) not in present)
            done += nxt * row_fraction(Path(log), cur["epochs"])
    state_path = path.with_name(path.stem + "_son.json")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if state.get("baslangic") == plan["baslangic"]:
        done = max(done, state.get("done", 0.0))  # tablo yazılırken okunursa yüzde geri gitmesin
    state_path.write_text(json.dumps({"baslangic": plan["baslangic"], "done": done}), encoding="utf-8")

    pct = min(100.0, 100.0 * done / total)
    elapsed = now - plan["baslangic"]
    factor = min(2.5, max(0.6, elapsed / done)) if done > 600 else 1.0
    remaining = max(0.0, total - done) * factor
    eta = datetime.fromtimestamp(now + remaining)
    eta_txt = f"{eta:%H:%M}" if eta.date() == datetime.fromtimestamp(now).date() else f"yarin {eta:%H:%M}"
    label = f"is {idx + 1}/{len(jobs)}: {cur['label']}" + (f" ({n_done}/{n_all} satir)" if n_all else "")
    status = f"%{pct:.1f} | kalan ~{dur(remaining)} | bitis ~{eta_txt} | {label}"
    return "\t".join([str(int(pct)), status, f"%{pct:.0f} | bitis ~{eta_txt} | KUYRUK9", "ILERLEME " + status])


def fake_job(units: int, table: str | None, code: int) -> None:
    for u in range(units):
        for e in range(4):
            print(f"    epoch {e + 1}/4 kayip=0.1 (1 s)", flush=True)
            time.sleep(0.6)
        path = Path(table)
        new = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(INFO_COLS + ["sure_s"])
            w.writerow(["deneme", "sahte", u, 2.4])
        print(f"[deneme] sahte           tohum={u} deger=1 AUC=0.5000 2 s", flush=True)
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--deneme", action="store_true", help="kuyruk betiğini sınamak için kısa sahte işler")
    ap.add_argument("--job", type=int, default=0)
    ap.add_argument("--done", default="")
    ap.add_argument("--log", default=None)
    ap.add_argument("--sahte-is", type=int, default=None)
    ap.add_argument("--sahte-tablo", default=None)
    ap.add_argument("--cikis", type=int, default=0)
    args = ap.parse_args()
    if args.sahte_is is not None:
        fake_job(args.sahte_is, args.sahte_tablo, args.cikis)
    elif args.plan:
        make_plan(args.deneme)
    elif args.poll:
        done = {int(x) for x in args.done.split(",") if x.strip()}
        print(poll(args.deneme, args.job, done, args.log))


if __name__ == "__main__":
    main()
