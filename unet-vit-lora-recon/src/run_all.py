#!/usr/bin/env python
"""prep -> train -> test -> eval (+efficiency) -> ablation.  Resumable (DONE markers) with a wall-clock guard."""
import argparse, json, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import ConcatDataset
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from models import build_model, model_summary_text, count_params
from train import train_model, test_model, dev_of
import evaluate as ev, efficiency as eff, make_readme as mr
T_START = time.time()

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DS)); p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--stages", default="prep,train,test,eval,ablation")
    for d, r in DEFAULT_ROOTS.items(): p.add_argument("--%s_root" % d, default=r)
    p.add_argument("--work_dir", default="/kaggle/working/recon_work"); p.add_argument("--out_dir", default="/kaggle/working/results_recon")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--test_seed", type=int, default=2024)
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--iters", type=int, default=100)
    p.add_argument("--pre_epochs", type=int, default=100); p.add_argument("--adapt_epochs", type=int, default=50); p.add_argument("--abl_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--lr_adapt", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=16); p.add_argument("--ablation_ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--abl_datasets", nargs="+", default=["chestxray"])
    p.add_argument("--keep_frac", type=float, default=0.30); p.add_argument("--center_frac", type=float, default=0.08); p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--val_n", type=int, default=48); p.add_argument("--n_vis", type=int, default=6); p.add_argument("--n_html", type=int, default=2)
    p.add_argument("--max_cases", type=int, default=0); p.add_argument("--force_prep", action="store_true"); p.add_argument("--force_train", action="store_true")
    p.add_argument("--time_budget_h", type=float, default=10.3); p.add_argument("--synthetic", action="store_true"); p.add_argument("--quick", action="store_true")
    a, _ = p.parse_known_args()
    if a.quick:
        a.epochs, a.iters, a.pre_epochs, a.adapt_epochs, a.abl_epochs, a.val_n, a.n_vis = 1, 3, 1, 1, 1, 4, 3
        a.max_cases = a.max_cases or 16; a.ablation_ranks = [4]
    return a

def deadline(a): return T_START + a.time_budget_h * 3600
def over(a): return time.time() > deadline(a)

def build_datasets(ds, paths, a):
    tr, _ = load_split(paths, ds, "train"); va, _ = load_split(paths, ds, "val"); te, ids = load_split(paths, ds, "test")
    return dict(train=ArrDataset(tr, ds, True), val=ArrDataset(va[: a.val_n], ds), test=ArrDataset(te, ds), ids=ids)

def run_one(name, ds, builder, D, cfg, paths, a, deg, log, stages, res_dir, epochs, lr, init_ckpt=None, stage=None, pred_name=None, first=False):
    run = paths.run(ds, name); ckpt = run / "best.pt"; res_dir = Path(res_dir); res_dir.mkdir(parents=True, exist_ok=True); bs = cfg["batch"]
    if "train" in stages and (not (run / "DONE").exists() or a.force_train):
        if over(a): log.info("[%s/%s] SKIPPED (time budget exhausted)" % (ds, name)); return None
        model = builder()
        if init_ckpt: model.load_state_dict(torch.load(init_ckpt, map_location="cpu"), strict=False)
        if stage and hasattr(model, "set_stage"): model.set_stage(stage)
        (res_dir / "model_summary.txt").write_text(model_summary_text(model, "%s | %s | dim=%d" % (name, cfg["name"], cfg["dim"])))
        train_model(model, D["train"], D["val"], bs, a.iters, epochs, lr, deg, a, run, res_dir, "%s/%s" % (ds, name), log, deadline(a))
    else: log.info("[%s/%s] training skipped/cached" % (ds, name))
    if "test" in stages and ckpt.exists():
        return test_model(builder(), ckpt, D["test"], D["ids"], ds, res_dir, paths.preds(pred_name or name, ds), deg, a, log, "%s/%s" % (ds, name),
                          zf_res=paths.res(ds, "zerofilled") if first else None, inputs_dir=paths.preds("_inputs", ds) if first else None)

def main():
    a = parse(); stages = set(a.stages.split(",")); paths = Paths(a.work_dir, a.out_dir); log = get_logger(paths.out / "pipeline.log")
    log.info("args: %s" % vars(a)); deg = dict(keep_frac=a.keep_frac, center_frac=a.center_frac, sigma=a.sigma)
    log.info("device: %s" % dev_of()); torch.manual_seed(a.seed); np.random.seed(a.seed)
    data = {}
    for ds in a.datasets:
        try:
            if "prep" in stages: prepare_dataset(ds, getattr(a, ds + "_root"), DS[ds], paths, a, log)
            data[ds] = build_datasets(ds, paths, a)
        except Exception: log.info("[%s] PREP FAILED\n%s" % (ds, traceback.format_exc()))
    for dim in (2, 3):
        dss = [d for d in data if DS[d]["dim"] == dim]
        if not dss: continue
        firsts = {d: True for d in dss}
        pre_ckpt = paths.run("pooled%dd" % dim, "ours_pretrain") / "best.pt"
        # ---------- ours: stage-1 pooled pretraining, then per-dataset LoRA adaptation (run FIRST so it is never skipped)
        if "ours" in a.models:
            try:
                cfg0 = DS[dss[0]]; pr = paths.run("pooled%dd" % dim, "ours_pretrain"); rd = paths.out / ("pooled_%dd" % dim) / "ours_pretrain"
                if "train" in stages and (not (pr / "DONE").exists() or a.force_train) and not over(a):
                    m0 = build_model("ours", dim, cfg0["size"], enable_lora=False); m0.set_stage("pretrain"); rd.mkdir(parents=True, exist_ok=True)
                    (rd / "model_summary.txt").write_text(model_summary_text(m0, "ours stage-1 (pooled %dD: %s)" % (dim, dss)))
                    train_model(m0, ConcatDataset([data[d]["train"] for d in dss]), ConcatDataset([data[d]["val"] for d in dss]), cfg0["batch"], a.iters, a.pre_epochs, a.lr, deg, a, pr, rd,
                                "pooled%dd/ours_pretrain" % dim, log, deadline(a))
                if pre_ckpt.exists():
                    for ds in dss:
                        cfg = DS[ds]
                        run_one("ours", ds, lambda cfg=cfg: build_model("ours", cfg["dim"], cfg["size"], rank=a.rank, enable_lora=True), data[ds], cfg, paths, a, deg, log, stages,
                                paths.res(ds, "ours"), a.adapt_epochs, a.lr_adapt, init_ckpt=pre_ckpt, stage="adapt", first=firsts[ds]); firsts[ds] = False
            except Exception: log.info("[ours dim%d] FAILED\n%s" % (dim, traceback.format_exc()))
        # ---------- baselines + ours_scratch (per dataset, from scratch, identical budget)
        for ds in dss:
            cfg = DS[ds]
            for m in a.models:
                if m == "ours": continue
                try:
                    run_one(m, ds, (lambda m=m, cfg=cfg: build_model(m, cfg["dim"], cfg["size"], enable_lora=False)), data[ds], cfg, paths, a, deg, log, stages, paths.res(ds, m), a.epochs, a.lr,
                            stage="pretrain" if m == "ours_scratch" else None, first=firsts[ds])
                    firsts[ds] = False
                except Exception: log.info("[%s/%s] FAILED\n%s" % (ds, m, traceback.format_exc()))
    # ---------- efficiency (params / FLOPs / sec per scan): independent of training
    if "eval" in stages:
        for ds in data:
            try:
                te, _ = load_split(paths, ds, "test"); eff.profile_dataset(ds, [m for m in a.models], paths, te, deg, log)
            except Exception: log.info("[%s] EFFICIENCY FAILED\n%s" % (ds, traceback.format_exc()))
    # ---------- ablations (last; skipped automatically when out of time)
    if "ablation" in stages and "ours" in a.models:
        for dim in (2, 3):
            pre_ckpt = paths.run("pooled%dd" % dim, "ours_pretrain") / "best.pt"
            for ds in [d for d in data if DS[d]["dim"] == dim and d in a.abl_datasets]:
                if pre_ckpt.exists() and not over(a):
                    try: run_ablation(ds, DS[ds], data[ds], paths, a, deg, log, pre_ckpt)
                    except Exception: log.info("[%s] ABLATION FAILED\n%s" % (ds, traceback.format_exc()))
    if "eval" in stages:
        for ds in data:
            try: ev.evaluate_dataset(ds, DS[ds], MODELS, paths, a, log)
            except Exception: log.info("[%s] EVAL FAILED\n%s" % (ds, traceback.format_exc()))
    ev.write_summary(paths.out, list(data), MODELS); ev.write_efficiency(paths.out, list(data), MODELS)
    ev.write_global_qualitative_grid(paths, {d: DS[d] for d in data}, MODELS); mr.write_readme(paths.out)
    log.info("DONE in %.2f h. Summary: %s" % ((time.time() - T_START) / 3600, paths.out / "summary_all.md"))

def run_ablation(ds, cfg, D, paths, a, deg, log, pre_ckpt):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = paths.out / ds / "ablation"; out.mkdir(parents=True, exist_ok=True); dim, size = cfg["dim"], cfg["size"]; rows = []
    specs = [dict(name="lora_rank%d" % r, group="adaptation", mode="lora", rank=r) for r in a.ablation_ranks]
    specs += [dict(name="no_lora_frozen", group="adaptation", mode="frozen"), dict(name="full_finetune", group="adaptation", mode="full")]
    specs += [dict(name="full_scratch", group="architecture", kw={}), dict(name="no_conditioning", group="architecture", kw=dict(use_cond=False)),
              dict(name="no_local_vit", group="architecture", kw=dict(use_vit=False)), dict(name="no_global_bottleneck", group="architecture", kw=dict(use_bottleneck=False)),
              dict(name="no_data_consistency", group="architecture", kw=dict(use_dc=False))]
    for s in specs:
        if over(a): log.info("[%s] ablation stopped: time budget" % ds); break
        nm = s["name"]; tag = "abl_" + nm; rd = out / nm
        if "mode" in s:
            lora = s["mode"] == "lora"; rank = s.get("rank", a.rank)
            builder = lambda lora=lora, rank=rank: build_model("ours", dim, size, rank=rank, enable_lora=lora)
            summ = run_one(tag, ds, builder, D, cfg, paths, a, deg, log, {"train", "test"}, rd, a.abl_epochs, a.lr_adapt if s["mode"] != "full" else a.lr,
                           init_ckpt=pre_ckpt, stage="pretrain" if s["mode"] == "full" else "adapt", pred_name=tag)
            m_ = builder(); m_.set_stage("pretrain" if s["mode"] == "full" else "adapt")
        else:
            builder = lambda kw=s["kw"]: build_model("ours", dim, size, enable_lora=False, **kw)
            summ = run_one(tag, ds, builder, D, cfg, paths, a, deg, log, {"train", "test"}, rd, a.abl_epochs, a.lr, stage="pretrain", pred_name=tag)
            m_ = builder()
        tot, tr = count_params(m_)
        if summ is None: continue
        mm = summ["metrics"]
        rows.append(dict(group=s["group"], variant=nm, params_total=tot, params_trainable=tr, trainable_pct=round(100.0 * tr / tot, 2),
                         PSNR=mm["PSNR"]["mean"], SSIM=mm["SSIM"]["mean"], MAE=mm["MAE"]["mean"], RMSE=mm["RMSE"]["mean"], NMSE=mm["NMSE"]["mean"]))
    if not rows: return
    df = pd.DataFrame(rows); df.to_csv(out / "ablation_results.csv", index=False)
    try: (out / "ablation_results.md").write_text(df.to_markdown(index=False))
    except Exception: (out / "ablation_results.md").write_text(df.to_string(index=False))
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for a_, k, c in zip(ax, ("PSNR", "SSIM", "trainable_pct"), ("tab:blue", "tab:green", "tab:orange")):
        a_.bar(df["variant"], df[k], color=c); a_.set_title("%s - %s (test)" % (k, cfg["name"])); a_.tick_params(axis="x", rotation=60); a_.grid(axis="y", alpha=.3)
        if k != "trainable_pct": lo, hi = df[k].min(), df[k].max(); pad = max(hi - lo, 1e-3) * 0.3; a_.set_ylim(lo - pad, hi + pad)
    fig.tight_layout(); fig.savefig(out / "ablation_results.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    log.info("[%s] ablation table:\n%s" % (ds, df.to_string(index=False)))

if __name__ == "__main__":
    main()
