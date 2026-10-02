"""Tables, statistics and figures for the paper."""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import LABEL, MODELS, BASELINES, DS, METRICS, HIGHER_BETTER, ArrDataset, load_split

COL = {"unet": "#1f77b4", "swinunetr": "#ff7f0e", "admmnet": "#2ca02c", "dncnn": "#9467bd", "varnet": "#e377c2",
       "ours_scratch": "#17becf", "ours": "#d62728", "zerofilled": "#7f7f7f"}


def _mid(v, dim):
    v = v.astype(np.float32)
    if dim == 2: return v
    return np.rot90(v[v.shape[0] // 2])

def _planes(v, dim):
    if dim == 2: return [v]
    d, h, w = v.shape; return [np.rot90(v[d // 2]), np.rot90(v[:, h // 2, :]), np.rot90(v[:, :, w // 2])]

def _save(fig, path, dpi=140):
    Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------------- qualitative figures
def plot_qualitative_grid(zf, gt, pred, dim, title, path, n=6, seed=0):
    N = len(gt); rng = np.random.RandomState(seed); n = min(n, N)
    idx = sorted(rng.choice(N, size=n, replace=False).tolist()) if N > n else list(range(N))
    rows = [("Degraded", zf), ("Reconstruction", pred), ("Ground Truth", gt)]
    fig, ax = plt.subplots(3, n, figsize=(2.6 * n, 8.1), squeeze=False)
    for r, (label, vol) in enumerate(rows):
        for c, i in enumerate(idx):
            ax[r, c].imshow(np.clip(_mid(vol[i], dim), 0, 1), cmap="gray", vmin=0, vmax=1); ax[r, c].axis("off")
        ax[r, 0].text(-0.08, 0.5, label, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=11)
    fig.suptitle(title, fontsize=12); fig.tight_layout(); _save(fig, path, 130)

def plot_dataset_comparison(ds, cfg, zf, gt, preds, models, dfs, idx, path):
    """ROWS = test cases (image row + |error| row), COLUMNS = Degraded | every model | Ground truth."""
    dim = cfg["dim"]; cols = ["zf"] + models + ["gt"]; nc = len(cols)
    fig, ax = plt.subplots(2 * len(idx), nc, figsize=(2.15 * nc, 4.5 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        g = _mid(gt[i], dim)
        for c, k in enumerate(cols):
            v = zf[i] if k == "zf" else gt[i] if k == "gt" else preds[k][i]
            im = _mid(v, dim); a1, a2 = ax[2 * r, c], ax[2 * r + 1, c]
            a1.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1); a1.axis("off")
            a2.imshow(np.abs(im - g), cmap="inferno", vmin=0, vmax=0.2); a2.axis("off")
            # FIX: LABEL[k] was evaluated eagerly as the .get() default and raised KeyError for 'zf'/'gt'
            if r == 0: a1.set_title({"zf": "Degraded", "gt": "Ground truth"}.get(k) or LABEL[k], fontsize=8, color=COL.get(k, "k"), fontweight="bold" if k == "ours" else "normal")
            if k in models and k in dfs:
                a1.text(0.5, -0.02, "%.2f dB / %.3f" % (dfs[k].PSNR.iloc[i], dfs[k].SSIM.iloc[i]), transform=a1.transAxes, ha="center", va="top", fontsize=7)
        ax[2 * r, 0].text(-0.05, 0.5, "image", transform=ax[2 * r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
        ax[2 * r + 1, 0].text(-0.05, 0.5, "|error|", transform=ax[2 * r + 1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("%s - all models vs ground truth (test set, evenly spaced cases, error colour range 0-0.2)" % cfg["name"], fontsize=11)
    fig.tight_layout(); _save(fig, path, 130)

def _recompute_inputs(ds, cfg, models, paths, args, log):
    from models import build_model
    from train import run_eval, dev_of
    dev = dev_of(); deg = dict(keep_frac=args.keep_frac, center_frac=args.center_frac, sigma=args.sigma)
    te, ids0 = load_split(paths, ds, "test"); test_ds = ArrDataset(te, ds)
    bs = DS[ds]["batch"] if DS[ds]["dim"] == 2 else 1
    for m in models:
        ck = paths.run(ds, m) / "best.pt"
        if ck.exists(): break
    else: raise FileNotFoundError("[%s] no checkpoint to recompute gt/zf" % ds)
    probe = build_model(m, cfg["dim"], cfg["size"]); probe.load_state_dict(torch.load(ck, map_location="cpu")); probe.to(dev)
    out = run_eval(probe, test_ds, bs, deg, dev, seed=args.test_seed, collect=True)
    d = paths.preds("_inputs", ds); d.mkdir(parents=True, exist_ok=True); np.save(d / "gt.npy", out["gt"]); np.save(d / "zf.npy", out["zf"])
    log.info("[%s] recomputed _inputs using model=%s" % (ds, m))

def evaluate_dataset(ds, cfg, models, paths, args, log):
    dim = cfg["dim"]
    dfs = {m: pd.read_csv(paths.res(ds, m) / "metrics_per_case.csv") for m in models if (paths.res(ds, m) / "metrics_per_case.csv").exists()}
    if not dfs: log.info("[%s] no results to evaluate" % ds); return
    avail = list(dfs); d = paths.preds("_inputs", ds)
    if not (d / "gt.npy").exists():
        try: _recompute_inputs(ds, cfg, avail, paths, args, log)
        except Exception as e: log.info("[%s] cannot build inputs, skipping figures: %s" % (ds, e)); return
    gt = np.load(d / "gt.npy"); zf = np.load(d / "zf.npy")
    preds = {m: np.load(paths.preds(m, ds) / "preds.npy") for m in avail if (paths.preds(m, ds) / "preds.npy").exists()}
    avail = [m for m in avail if m in preds]
    if not avail: return
    N = len(gt); out = paths.out / ds / "model_comparison"; out.mkdir(parents=True, exist_ok=True)
    vis = sorted(set(np.linspace(0, N - 1, min(args.n_vis, N)).astype(int).tolist()))
    # ONE figure per dataset: every model + GT (+ error maps)
    for k in range(0, len(vis), 3):
        plot_dataset_comparison(ds, cfg, zf, gt, preds, avail, dfs, vis[k:k + 3], paths.out / ds / ("comparison_all_models_%d.png" % (k // 3 + 1)))
    for i in vis:   # per-case folders with three planes for 3-D
        cid = dfs[avail[0]]["case"].iloc[i]
        for m in avail:
            cd = paths.res(ds, m) / "cases" / cid; cd.mkdir(parents=True, exist_ok=True)
            for k, (a, b) in enumerate(zip(_planes(gt[i].astype(np.float32), dim), _planes(preds[m][i].astype(np.float32), dim))):
                plt.imsave(cd / ("recon_%d.png" % k), np.clip(b, 0, 1), cmap="gray", vmin=0, vmax=1)
                plt.imsave(cd / ("error_%d.png" % k), np.abs(a - b), cmap="inferno", vmin=0, vmax=0.2)
                plt.imsave(cd / ("gt_%d.png" % k), np.clip(a, 0, 1), cmap="gray", vmin=0, vmax=1)
    for m in avail:   # per-model Degraded / Reconstruction / Ground truth
        try: plot_qualitative_grid(zf, gt, preds[m], dim, "%s | %s (test set)" % (cfg["name"], LABEL[m]), paths.res(ds, m) / ("%s_%s_qualitative.png" % (ds, m)), n=min(6, N))
        except Exception as e: log.info("[%s/%s] qualitative grid failed: %s" % (ds, m, e))
    fig, ax = plt.subplots(1, 3, figsize=(20, 4))
    for a, k in zip(ax, ("PSNR", "SSIM", "NMSE")):
        bp = a.boxplot([dfs[m][k].values for m in avail], patch_artist=True)
        for patch, m in zip(bp["boxes"], avail): patch.set_facecolor(COL[m]); patch.set_alpha(0.7)
        a.set_xticklabels([LABEL[m].replace(" (U-Net-ViT-LoRA)", "") for m in avail], rotation=35, ha="right"); a.set_title("%s per test case - %s" % (k, cfg["name"])); a.grid(alpha=.3)
    _save(fig, paths.out / ds / "boxplot_metrics.png", 120)
    # validation curves of all models
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for m in avail:
        f = list((paths.res(ds, m) / "logs").glob("*_train_log.csv"))
        if not f: continue
        L = pd.read_csv(f[0]); ax[0].plot(L.epoch, L.val_psnr, label=LABEL[m], c=COL[m], lw=2.2 if m == "ours" else 1.2); ax[1].plot(L.epoch, L.val_ssim, c=COL[m], lw=2.2 if m == "ours" else 1.2)
    ax[0].set_title("val PSNR vs epoch - %s" % cfg["name"]); ax[1].set_title("val SSIM vs epoch"); ax[0].legend(fontsize=7)
    for a in ax: a.grid(alpha=.3); a.set_xlabel("epoch")
    _save(fig, paths.out / ds / "training_curves_all_models.png", 120)
    log.info("[%s] evaluation figures written" % ds)


# ------------------------------------------------------------------------- summary tables
def _load_summary(out_dir, datasets, models):
    rows = []
    for d in datasets:
        for m in ["zerofilled"] + list(models):
            f = Path(out_dir) / d / m / "metrics_summary.json"
            if not f.exists(): continue
            s = json.loads(f.read_text()); r = dict(dataset=d, model=m, label=LABEL[m], n_test=s.get("n_test", 0))
            for k in METRICS: r[k + "_mean"] = s["metrics"][k]["mean"]; r[k + "_std"] = s["metrics"][k]["std"]
            e = Path(out_dir) / d / m / "efficiency.json"
            if e.exists(): r.update(json.loads(e.read_text()))
            r["params_M_total"] = s["params_total"] / 1e6; rows.append(r)
    return pd.DataFrame(rows)

def _best(df, ds, k):
    sub = df[(df.dataset == ds) & (df.model != "zerofilled")][["model", k + "_mean"]].dropna()
    if sub.empty: return None
    return sub.loc[sub[k + "_mean"].idxmax() if HIGHER_BETTER[k] else sub[k + "_mean"].idxmin(), "model"]

def write_summary(out_dir, datasets, models):
    out_dir = Path(out_dir); df = _load_summary(out_dir, datasets, models)
    if df.empty: return None
    df.to_csv(out_dir / "summary_all.csv", index=False)
    md, tex = [], []
    for ds in dict.fromkeys(df.dataset):
        sub = df[df.dataset == ds]; md.append("### %s (n_test=%d)\n" % (DS[ds]["name"], int(sub.n_test.max())))
        md.append("| Model | " + " | ".join(METRICS) + " |"); md.append("|---|" + "---|" * len(METRICS))
        tex.append("\\begin{table}[t]\\centering\\caption{%s}\\begin{tabular}{l%s}\\hline\nModel & %s \\\\\\hline" % (DS[ds]["name"], "c" * len(METRICS), " & ".join(METRICS)))
        for _, r in sub.iterrows():
            cells, tc = [], []
            for k in METRICS:
                s = "%.4f +- %.4f" % (r[k + "_mean"], r[k + "_std"]); t = "$%.4f \\pm %.4f$" % (r[k + "_mean"], r[k + "_std"])
                if r.model != "zerofilled" and _best(df, ds, k) == r.model: s, t = "**%s**" % s, "\\textbf{%s}" % t
                cells.append(s); tc.append(t)
            md.append("| %s | %s |" % (r.label, " | ".join(cells))); tex.append("%s & %s \\\\" % (r.label, " & ".join(tc)))
        tex.append("\\hline\\end{tabular}\\end{table}\n"); md.append("")
    (out_dir / "summary_all.md").write_text("\n".join(md)); (out_dir / "summary_latex.tex").write_text("\n".join(tex))
    for k in METRICS:
        df.pivot_table(index="label", columns="dataset", values=k + "_mean").to_csv(out_dir / ("table_%s.csv" % k))
    # win table: does OURS win against the six published baselines?
    wins, rows = 0, []
    for ds in dict.fromkeys(df.dataset):
        for k in METRICS:
            sub = df[(df.dataset == ds) & (df.model.isin(BASELINES + ["ours"]))]
            if "ours" not in set(sub.model): continue
            b = _best(sub.assign(dataset=ds), ds, k); w = b == "ours"; wins += int(w)
            rows.append(dict(dataset=ds, metric=k, best_model=LABEL[b], ours_wins=w))
    if rows:
        W = pd.DataFrame(rows); W.to_csv(out_dir / "win_table.csv", index=False)
        print("\nOURS wins %d / %d (dataset x metric) against the 6 baselines" % (wins, len(W)))
    # bars per dataset
    dss = list(dict.fromkeys(df.dataset)); ms = [m for m in ["zerofilled"] + list(models) if m in set(df.model)]; w = 0.85 / len(ms)
    fig, ax = plt.subplots(1, 5, figsize=(30, 4.6))
    for a, k in zip(ax, METRICS):
        for i, m in enumerate(ms):
            v = [df[(df.dataset == d) & (df.model == m)][k + "_mean"].mean() for d in dss]; e = [df[(df.dataset == d) & (df.model == m)][k + "_std"].mean() for d in dss]
            a.bar(np.arange(len(dss)) + i * w, v, w, yerr=e, capsize=1.5, label=LABEL[m], color=COL[m])
        a.set_xticks(np.arange(len(dss)) + w * (len(ms) - 1) / 2); a.set_xticklabels([DS[d]["name"] for d in dss]); a.set_title(k + (" (higher better)" if HIGHER_BETTER[k] else " (lower better)")); a.grid(axis="y", alpha=.3)
        if k == "PSNR": a.legend(fontsize=6, ncol=2); a.set_ylim(max(0, df.PSNR_mean.min() - 2), None)
        if k == "SSIM": a.set_ylim(max(0, df.SSIM_mean.min() - 0.05), 1)
    fig.tight_layout(); _save(fig, out_dir / "comparison_bar.png", 120)
    # heatmaps
    fig, ax = plt.subplots(1, 3, figsize=(1.7 * len(dss) * 3 + 4, 0.55 * len(ms) + 2.4))
    for a_, k in zip(ax, ("PSNR_mean", "SSIM_mean", "NMSE_mean")):
        mat = np.full((len(ms), len(dss)), np.nan)
        for i, m in enumerate(ms):
            for j, d in enumerate(dss):
                v = df[(df.model == m) & (df.dataset == d)][k]
                if len(v): mat[i, j] = v.values[0]
        im = a_.imshow(mat, cmap="viridis" if k != "NMSE_mean" else "viridis_r", aspect="auto")
        a_.set_xticks(range(len(dss))); a_.set_xticklabels([DS[d]["name"] for d in dss], rotation=35, ha="right"); a_.set_yticks(range(len(ms))); a_.set_yticklabels([LABEL[m] for m in ms])
        a_.set_title(k.replace("_mean", "")); fig.colorbar(im, ax=a_, fraction=0.046)
        for i in range(len(ms)):
            for j in range(len(dss)):
                if not np.isnan(mat[i, j]): a_.text(j, i, "%.3f" % mat[i, j] if k != "PSNR_mean" else "%.2f" % mat[i, j], ha="center", va="center", fontsize=7, color="w")
    fig.suptitle("All models x all datasets"); fig.tight_layout(); _save(fig, out_dir / "global_metric_heatmap.png", 130)
    # per-model across datasets
    ncol = min(len(ms), 4); nrow = 2 * ((len(ms) + ncol - 1) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
    for idx, m in enumerate(ms):
        r, c = (idx // ncol) * 2, idx % ncol; sub = df[df.model == m]
        axes[r][c].bar([DS[d]["name"] for d in sub.dataset], sub.PSNR_mean, yerr=sub.PSNR_std, capsize=2, color=COL[m]); axes[r][c].set_title(LABEL[m], fontsize=9); axes[r][c].tick_params(axis="x", rotation=40)
        axes[r + 1][c].bar([DS[d]["name"] for d in sub.dataset], sub.SSIM_mean, yerr=sub.SSIM_std, capsize=2, color=COL[m]); axes[r + 1][c].tick_params(axis="x", rotation=40)
        if c == 0: axes[r][c].set_ylabel("PSNR"); axes[r + 1][c].set_ylabel("SSIM")
    for idx in range(len(ms), nrow // 2 * ncol): r, c = (idx // ncol) * 2, idx % ncol; axes[r][c].axis("off"); axes[r + 1][c].axis("off")
    fig.suptitle("Per-model performance across datasets"); fig.tight_layout(); _save(fig, out_dir / "per_model_across_datasets.png", 120)
    # gain of ours over best baseline
    if "ours" in set(df.model):
        g = []
        for d in dss:
            b = df[(df.dataset == d) & (df.model.isin(BASELINES))]; o = df[(df.dataset == d) & (df.model == "ours")]
            if len(b) and len(o): g.append((DS[d]["name"], float(o.PSNR_mean.iloc[0] - b.PSNR_mean.max()), float(o.SSIM_mean.iloc[0] - b.SSIM_mean.max())))
        if g:
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
            for a, j, t in zip(ax, (1, 2), ("PSNR gain (dB)", "SSIM gain")):
                a.bar([x[0] for x in g], [x[j] for x in g], color=["#2ca02c" if x[j] > 0 else "#d62728" for x in g]); a.axhline(0, c="k", lw=.8); a.set_title("Ours minus best baseline: " + t); a.grid(axis="y", alpha=.3); a.tick_params(axis="x", rotation=25)
            fig.tight_layout(); _save(fig, out_dir / "ours_vs_best_baseline.png", 130)
    # statistics
    try:
        from scipy.stats import wilcoxon
        S = []
        for d in dss:
            po = Path(out_dir) / d / "ours" / "metrics_per_case.csv"
            if not po.exists(): continue
            o = pd.read_csv(po).set_index("case")
            for b in BASELINES + ["ours_scratch"]:
                pb = Path(out_dir) / d / b / "metrics_per_case.csv"
                if not pb.exists(): continue
                bb = pd.read_csv(pb).set_index("case").reindex(o.index)
                for k in ("PSNR", "SSIM"):
                    diff = (o[k] - bb[k]).values
                    try: p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                    except Exception: p = float("nan")
                    S.append(dict(dataset=d, baseline=LABEL[b], metric=k, mean_diff=float(np.mean(diff)), frac_cases_ours_better=float(np.mean(diff > 0)), wilcoxon_p=p))
        if S: pd.DataFrame(S).to_csv(out_dir / "stats_ours_vs_baselines.csv", index=False)
    except Exception as e: print("stats skipped:", e)
    return df

def write_efficiency(out_dir, datasets, models):
    out_dir = Path(out_dir); f = out_dir / "summary_all.csv"
    if not f.exists(): return
    df = pd.read_csv(f)
    if "GFLOPs" not in df: return
    df = df[(df.model != "zerofilled") & df.GFLOPs.notna()]
    if df.empty: return
    cols = ["dataset", "label", "params_M", "trainable_M", "GFLOPs", "sec_per_scan", "sec_per_scan_std", "peak_mem_MB", "PSNR_mean", "SSIM_mean"]
    T = df[cols].rename(columns={"label": "model"}); T.to_csv(out_dir / "efficiency_table.csv", index=False)
    try: (out_dir / "efficiency_table.md").write_text(T.round(4).to_markdown(index=False))
    except Exception: (out_dir / "efficiency_table.md").write_text(T.round(4).to_string(index=False))
    dss = list(dict.fromkeys(df.dataset)); ms = [m for m in models if m in set(df.model)]; w = 0.85 / len(ms)
    for col, ttl, fn, lg in (("GFLOPs", "GFLOPs per scan (batch 1)", "flops_chart.png", True), ("sec_per_scan", "Inference time per scan (sec, median of 30)", "time_per_scan_chart.png", True), ("params_M", "Parameters (M)", "params_chart.png", True)):
        fig, ax = plt.subplots(figsize=(1.9 * len(dss) + 5, 4.6))
        for i, m in enumerate(ms):
            v = [df[(df.dataset == d) & (df.model == m)][col].mean() for d in dss]
            ax.bar(np.arange(len(dss)) + i * w, v, w, label=LABEL[m], color=COL[m])
        ax.set_xticks(np.arange(len(dss)) + w * (len(ms) - 1) / 2); ax.set_xticklabels([DS[d]["name"] for d in dss]); ax.set_title(ttl); ax.grid(axis="y", alpha=.3)
        if lg: ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2); _save(fig, out_dir / fn, 130)
    fig, ax = plt.subplots(2, len(dss), figsize=(4.4 * len(dss), 7.5), squeeze=False)   # accuracy vs cost
    for j, d in enumerate(dss):
        sub = df[df.dataset == d]
        for r, (xc, xl) in enumerate((("GFLOPs", "GFLOPs / scan"), ("sec_per_scan", "sec / scan"))):
            for _, q in sub.iterrows():
                ax[r][j].scatter(q[xc], q.PSNR_mean, s=90 if q.model == "ours" else 45, c=COL[q.model], marker="*" if q.model == "ours" else "o", label=q.label)
            ax[r][j].set_xscale("log"); ax[r][j].set_xlabel(xl); ax[r][j].set_ylabel("PSNR (dB)"); ax[r][j].set_title(DS[d]["name"]); ax[r][j].grid(alpha=.3)
    ax[0][0].legend(fontsize=6); fig.tight_layout(); _save(fig, out_dir / "psnr_vs_cost.png", 130)

def write_global_qualitative_grid(paths, datasets_cfg, models, seed=0):
    out_dir = Path(paths.out); data = {}
    for ds, cfg in datasets_cfg.items():
        d = paths.preds("_inputs", ds)
        if not (d / "gt.npy").exists(): continue
        preds = {m: np.load(paths.preds(m, ds) / "preds.npy") for m in models if (paths.preds(m, ds) / "preds.npy").exists()}
        if preds: data[ds] = dict(cfg=cfg, gt=np.load(d / "gt.npy"), zf=np.load(d / "zf.npy"), preds=preds)
    if not data: return
    am = [m for m in models if any(m in v["preds"] for v in data.values())]
    nc, nr = 2 + len(am), len(data); fig, ax = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.5 * nr), squeeze=False); rng = np.random.RandomState(seed)
    for r, (ds, v) in enumerate(data.items()):
        i = int(rng.randint(0, len(v["gt"]))); dim = v["cfg"]["dim"]
        ax[r][0].imshow(np.clip(_mid(v["zf"][i], dim), 0, 1), cmap="gray", vmin=0, vmax=1)
        for c, m in enumerate(am, 1):
            if m in v["preds"]: ax[r][c].imshow(np.clip(_mid(v["preds"][m][i], dim), 0, 1), cmap="gray", vmin=0, vmax=1)
        ax[r][-1].imshow(np.clip(_mid(v["gt"][i], dim), 0, 1), cmap="gray", vmin=0, vmax=1)
        for a in ax[r]: a.axis("off")
        if r == 0:
            ax[0][0].set_title("Degraded", fontsize=8)
            for c, m in enumerate(am, 1): ax[0][c].set_title(LABEL[m], fontsize=7, color=COL[m])
            ax[0][-1].set_title("Ground truth", fontsize=8)
        ax[r][0].text(-0.12, 0.5, v["cfg"]["name"], transform=ax[r][0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle("Global qualitative comparison (all datasets x all models, 1 test case each)", fontsize=11); fig.tight_layout(); _save(fig, out_dir / "global_qualitative_comparison.png", 140)
