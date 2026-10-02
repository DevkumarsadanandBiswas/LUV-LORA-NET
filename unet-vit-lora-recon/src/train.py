"""Training / validation / test loops (degradation is generated on the GPU from the clean image)."""
import time, json
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from common import make_mask, measure, make_gen, compute_metrics, ssim_torch, DS, safe_tag, _dims
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def dev_of(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loader(ds, bs, train, iters=None):
    if train:
        return DataLoader(ds, batch_size=bs, sampler=RandomSampler(ds, replacement=True, num_samples=iters * bs), drop_last=True, num_workers=0)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

def step_inputs(batch, dev, deg, gen):
    x = batch["x"].to(dev); mask = make_mask(x.shape[0], x.shape[2:], deg["keep_frac"], deg["center_frac"], gen, dev)
    y, zf = measure(x, mask, deg["sigma"], gen)
    return x, zf, y, mask, batch["mod"].to(dev), batch["ana"].to(dev), batch["dim"].to(dev)

def loss_fn(pred, x):
    """Identical for every model: L1 + 0.2 (1-SSIM) + 0.1 k-space L1 (keeps high frequencies -> sharper images)."""
    pred = pred.float(); dims = _dims(x)
    fl = (torch.fft.fftn(pred, dim=dims, norm="ortho") - torch.fft.fftn(x, dim=dims, norm="ortho")).abs().mean()
    return F.l1_loss(pred, x) + 0.2 * (1 - ssim_torch(pred.clamp(0, 1), x).mean()) + 0.1 * fl

@torch.no_grad()
def run_eval(model, ds, bs, deg, dev, seed, collect=False, use_amp=True):
    model.eval(); gen = make_gen(dev, seed); R = defaultdict(list); P, Z, G, T = [], [], [], []
    amp = use_amp and getattr(model, "use_amp", True) and dev.type == "cuda"
    for batch in make_loader(ds, bs, False):
        x, zf, y, mask, mod, ana, dt = step_inputs(batch, dev, deg, gen)
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", enabled=amp): pred = model(zf, y, mask, mod, ana, dt)
        if dev.type == "cuda": torch.cuda.synchronize()
        T.append((time.time() - t0) / x.shape[0])
        for k, v in compute_metrics(pred.float(), x).items(): R[k] += v.tolist()
        for k, v in compute_metrics(zf, x).items(): R["zf_" + k] += v.tolist()
        R["idx"] += batch["idx"].tolist()
        if collect:
            P.append(pred.float().clamp(0, 1).cpu().half().numpy()); Z.append(zf.clamp(0, 1).cpu().half().numpy()); G.append(x.cpu().half().numpy())
    out = dict(R); out["ms"] = 1000 * float(np.mean(T[1:] if len(T) > 1 else T))
    if collect: out.update(preds=np.concatenate(P)[:, 0], zf=np.concatenate(Z)[:, 0], gt=np.concatenate(G)[:, 0])
    return out


def plot_curves(df, path, title):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(df["epoch"], df["train_loss"], label="train"); ax[0].plot(df["epoch"], df["val_loss"], label="val"); ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(df["epoch"], df["val_psnr"], c="green"); ax[1].set_title("val PSNR")
    ax[2].plot(df["epoch"], df["val_ssim"], c="purple"); ax[2].set_title("val SSIM")
    for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.suptitle(title); fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=120); plt.close(fig)


def train_model(model, train_ds, val_ds, bs, iters, epochs, lr, deg, args, run_dir, res_dir, tag, log, deadline=None):
    dev = dev_of(); model.to(dev); run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(res_dir); res_dir.mkdir(parents=True, exist_ok=True)
    ftag = safe_tag(tag)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    amp = getattr(model, "use_amp", True) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ckpt = run_dir / "best.pt"; best, rows, t0 = -1e9, [], time.time()
    log.info("[%s] trainable params %s / %s  lr=%g epochs=%d iters=%d bs=%d amp=%s" % (
        tag, format(sum(p.numel() for p in params), ","), format(sum(p.numel() for p in model.parameters()), ","), lr, epochs, iters, bs, amp))
    log_csv = res_dir / "logs" / (ftag + "_train_log.csv"); log_csv.parent.mkdir(parents=True, exist_ok=True)
    df = None
    for ep in range(epochs):
        if deadline is not None and time.time() > deadline:
            log.info("[%s] wall-clock budget reached -> stopping early at epoch %d (best checkpoint is kept)" % (tag, ep)); break
        model.train(); tl = []
        for batch in make_loader(train_ds, bs, True, iters):
            x, zf, y, mask, mod, ana, dt = step_inputs(batch, dev, deg, None)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp): pred = model(zf, y, mask, mod, ana, dt)
            loss = loss_fn(pred, x)
            if not torch.isfinite(loss): continue
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
            tl.append(loss.item())
        sched.step()
        v = run_eval(model, val_ds, bs, deg, dev, seed=1234)
        vl = float(np.mean(v["MAE"]) + 0.2 * (1 - np.mean(v["SSIM"]))); vp = float(np.mean(v["PSNR"])); vs = float(np.mean(v["SSIM"]))
        rows.append(dict(epoch=ep + 1, train_loss=float(np.mean(tl)) if tl else float("nan"), val_loss=vl, val_psnr=vp, val_ssim=vs, min=(time.time() - t0) / 60))
        if vp > best: best = vp; torch.save(model.state_dict(), ckpt)
        log.info("[%s] ep %3d/%d train %.4f val_loss %.4f val_PSNR %.2f val_SSIM %.4f (%.1f min)" % (tag, ep + 1, epochs, rows[-1]["train_loss"], vl, vp, vs, rows[-1]["min"]))
        df = pd.DataFrame(rows); log_csv.parent.mkdir(parents=True, exist_ok=True); df.to_csv(log_csv, index=False)
    if not ckpt.exists(): torch.save(model.state_dict(), ckpt)
    if df is not None: plot_curves(df, res_dir / "curves" / (ftag + "_loss_curve.png"), "%s (best val PSNR %.2f)" % (tag, best))
    (run_dir / "DONE").write_text("ok"); log.info("[%s] finished. best val PSNR %.2f" % (tag, best))
    return ckpt


def test_model(model, ckpt, test_ds, ids, ds_key, res_dir, pred_dir, deg, args, log, tag, zf_res=None, inputs_dir=None):
    dev = dev_of(); model.load_state_dict(torch.load(ckpt, map_location="cpu")); model.to(dev)
    bs = DS[ds_key]["batch"] if DS[ds_key]["dim"] == 2 else 1
    out = run_eval(model, test_ds, bs, deg, dev, seed=args.test_seed, collect=True)
    res_dir, pred_dir = Path(res_dir), Path(pred_dir); res_dir.mkdir(parents=True, exist_ok=True); pred_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({k: out[k] for k in ["PSNR", "SSIM", "MAE", "RMSE", "NMSE"]}); df.insert(0, "case", [ids[i] for i in out["idx"]])
    df.to_csv(res_dir / "metrics_per_case.csv", index=False)
    tot, tr = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
    summ = dict(metrics={k: dict(mean=float(df[k].mean()), std=float(df[k].std(ddof=0))) for k in ["PSNR", "SSIM", "MAE", "RMSE", "NMSE"]},
                params_total=tot, params_trainable_at_end=tr, ms_per_sample=out["ms"], n_test=len(df))
    (res_dir / "metrics_summary.json").write_text(json.dumps(summ, indent=2)); np.save(pred_dir / "preds.npy", out["preds"])
    log.info("[%s] TEST  PSNR %.2f  SSIM %.4f  MAE %.4f  RMSE %.4f  NMSE %.4f  (%d cases)" % (tag, df.PSNR.mean(), df.SSIM.mean(), df.MAE.mean(), df.RMSE.mean(), df.NMSE.mean(), len(df)))
    if inputs_dir is not None and not (Path(inputs_dir) / "gt.npy").exists():
        Path(inputs_dir).mkdir(parents=True, exist_ok=True); np.save(Path(inputs_dir) / "gt.npy", out["gt"]); np.save(Path(inputs_dir) / "zf.npy", out["zf"])
    if zf_res is not None and not (Path(zf_res) / "metrics_per_case.csv").exists():
        Path(zf_res).mkdir(parents=True, exist_ok=True)
        z = pd.DataFrame({k: out["zf_" + k] for k in ["PSNR", "SSIM", "MAE", "RMSE", "NMSE"]}); z.insert(0, "case", df["case"]); z.to_csv(Path(zf_res) / "metrics_per_case.csv", index=False)
        (Path(zf_res) / "metrics_summary.json").write_text(json.dumps(dict(metrics={k: dict(mean=float(z[k].mean()), std=float(z[k].std(ddof=0))) for k in ["PSNR", "SSIM", "MAE", "RMSE", "NMSE"]},
                                                                           params_total=0, params_trainable_at_end=0, ms_per_sample=0.0, n_test=len(z)), indent=2))
    return summ
