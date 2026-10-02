"""Parameters, FLOPs and time-per-scan (seconds) for every model / dataset."""
import json, math, time
from pathlib import Path
import numpy as np, torch
from common import DS, make_mask, measure, make_gen, MODALITY_IDS, ANATOMY_IDS
from models import build_model
from train import dev_of


class FFTCount:
    """Adds an analytic 5 N log2 N cost for every torch.fft.fftn / ifftn call (FlopCounterMode does not see FFTs)."""
    def __enter__(self):
        self.flops = 0.0; self._o = (torch.fft.fftn, torch.fft.ifftn); outer = self
        def wrap(f):
            def g(x, *a, **k):
                per = max(2, x.numel() // max(1, x.shape[0])); outer.flops += 5.0 * x.numel() * math.log2(per)
                return f(x, *a, **k)
            return g
        torch.fft.fftn, torch.fft.ifftn = wrap(self._o[0]), wrap(self._o[1]); return self
    def __exit__(self, *a): torch.fft.fftn, torch.fft.ifftn = self._o


def profile_model(name, ds_key, x_np, deg, dev, reps=30, warmup=5):
    from torch.utils.flop_counter import FlopCounterMode
    cfg = DS[ds_key]
    model = build_model(name, cfg["dim"], cfg["size"], **({"enable_lora": True} if name == "ours" else {}))
    if hasattr(model, "set_stage") and name == "ours": model.set_stage("adapt")
    model.to(dev).eval()
    total, trainable = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.from_numpy(np.asarray(x_np[:1], dtype=np.float32)).unsqueeze(1).to(dev)
    gen = make_gen(dev, 0); mask = make_mask(1, x.shape[2:], deg["keep_frac"], deg["center_frac"], gen, dev)
    y, zf = measure(x, mask, deg["sigma"], gen)
    m = torch.tensor([MODALITY_IDS[cfg["modality"]]], device=dev); a = torch.tensor([ANATOMY_IDS[cfg["anatomy"]]], device=dev); d = torch.tensor([cfg["dim"] - 2], device=dev)
    args = (zf, y, mask, m, a, d)
    with torch.no_grad(), FlopCounterMode(display=False) as fc, FFTCount() as ff: model(*args)
    gflops = (fc.get_total_flops() + ff.flops) / 1e9
    amp = getattr(model, "use_amp", True) and dev.type == "cuda"
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats()
    ts = []
    with torch.no_grad():
        for i in range(warmup + reps):
            if dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", enabled=amp): model(*args)
            if dev.type == "cuda": torch.cuda.synchronize()
            if i >= warmup: ts.append(time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else float("nan")
    del model
    return dict(params_M=total / 1e6, trainable_M=trainable / 1e6, GFLOPs=gflops, sec_per_scan=float(np.median(ts)), sec_per_scan_mean=float(np.mean(ts)),
                sec_per_scan_std=float(np.std(ts)), peak_mem_MB=mem)


def profile_dataset(ds_key, models, paths, x_np, deg, log):
    dev = dev_of()
    for m in models:
        try:
            r = profile_model(m, ds_key, x_np, deg, dev)
            out = paths.res(ds_key, m); out.mkdir(parents=True, exist_ok=True); (out / "efficiency.json").write_text(json.dumps(r, indent=2))
            log.info("[%s/%s] params %.3fM  GFLOPs %.3f  sec/scan %.4f  mem %.0fMB" % (ds_key, m, r["params_M"], r["GFLOPs"], r["sec_per_scan"], r["peak_mem_MB"]))
        except Exception as e:
            log.info("[%s/%s] efficiency profiling failed: %s" % (ds_key, m, e))
