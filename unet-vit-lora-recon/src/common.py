"""Config, IO, masked-Fourier degradation, datasets, metrics."""
import os, sys, json, glob, random, logging
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch.utils.data import Dataset

DEFAULT_ROOTS = {
    "chestxray":  "/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray/chest_xray",
    "busi":       "/kaggle/input/datasets/sabahesaraki/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT",
    "brats":      "/kaggle/input/datasets/darksteeldragon/brats2020-nifti-format-for-deepmedic/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    "mriusbrain": "/kaggle/input/datasets/shubhamcodez/3d-mri-ultrasound-brain-images",
}
MARKERS = {"chestxray": "chest_xray", "busi": "Dataset_BUSI_with_GT",
           "brats": "MICCAI_BraTS2020_TrainingData", "mriusbrain": "3d-mri-ultrasound-brain-images"}
DS = {
    "chestxray":  dict(name="ChestXray",  dim=2, modality="xray",       anatomy="chest",  size=128, max_cases=1200, batch=16),
    "busi":       dict(name="BUSI",       dim=2, modality="ultrasound", anatomy="breast", size=128, max_cases=800,  batch=16),
    "brats":      dict(name="BraTS2020",  dim=3, modality="mri",        anatomy="brain",  size=64,  max_cases=150,  batch=2),
    "mriusbrain": dict(name="MRIUSBrain", dim=3, modality="mri",        anatomy="brain",  size=64,  max_cases=60,   batch=2),
}
MODALITY_IDS = {"xray": 0, "ultrasound": 1, "mri": 2, "ct": 3}
ANATOMY_IDS = {"chest": 0, "breast": 1, "brain": 2, "heart": 3, "other": 4}
BASELINES = ["unet", "swinunetr", "admmnet", "dncnn", "varnet"]
MODELS = BASELINES + ["ours_scratch", "ours"]
LABEL = {"unet": "U-Net", "swinunetr": "Swin-UNETR", "admmnet": "ADMM-Net", "dncnn": "DnCNN",
         "varnet": "VarNet", "ours_scratch": "Ours (scratch)", "ours": "Ours (U-Net-ViT-LoRA)", "zerofilled": "Zero-filled"}
METRICS = ["PSNR", "SSIM", "MAE", "RMSE", "NMSE"]
HIGHER_BETTER = {"PSNR": True, "SSIM": True, "MAE": False, "RMSE": False, "NMSE": False}


def safe_tag(tag):
    """'<dataset>/<model>' -> '<dataset>_<model>' so it is a bare filename."""
    return str(tag).replace("/", "_").replace("\\", "_")


class Paths:
    def __init__(self, work, out):
        self.work, self.out = Path(work), Path(out)
    def prep(self, ds):          return self.work / "prep" / ds
    def run(self, ds, name):     return self.work / "runs" / ds / name
    def preds(self, name, ds):   return self.work / "preds" / name / ds
    def res(self, ds, name):     return self.out / ds / name


def get_logger(logfile=None):
    lg = logging.getLogger("recon"); lg.setLevel(logging.INFO)
    if not any(type(h) is logging.StreamHandler for h in lg.handlers):
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")); lg.addHandler(sh)
    for h in [h for h in lg.handlers if isinstance(h, logging.FileHandler)]:
        lg.removeHandler(h); h.close()
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode="a"); fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s")); lg.addHandler(fh)
    return lg


# ----------------------------------------------------------------- data discovery / prep
def resolve_root(ds, root):
    if os.path.isdir(root): return root
    hits = [h for h in glob.glob("/kaggle/input/**/" + MARKERS[ds], recursive=True) if os.path.isdir(h)]
    if hits: return hits[0]
    raise FileNotFoundError("[%s] data root not found: %s" % (ds, root))


def discover(ds, root):
    root = Path(root); fs = []
    if ds == "chestxray":
        for sp in ("train", "test", "val"):
            for cl in ("NORMAL", "PNEUMONIA"):
                fs += sorted(glob.glob(str(root / sp / cl / "*.jp*g")))
    elif ds == "busi":
        for cl in ("benign", "malignant", "normal"):
            fs += [f for f in sorted(glob.glob(str(root / cl / "*.png"))) if "_mask" not in f]
    elif ds == "brats":
        for p in sorted(glob.glob(str(root / "BraTS20_Training_*"))):
            t = sorted(glob.glob(os.path.join(p, "*_t1.nii*")))
            if t: fs.append(t[0])
    elif ds == "mriusbrain":
        fs = sorted(glob.glob(str(root / "**" / "MRI" / "*.nii*"), recursive=True))
    return fs


def load_2d(path, size):
    from PIL import Image
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_3d(path, size):
    import nibabel as nib
    v = np.squeeze(np.nan_to_num(nib.load(path).get_fdata(dtype=np.float32)))
    if v.ndim != 3: raise ValueError("expected 3-D volume, got %s" % (v.shape,))
    nz = v[v > 0]
    lo, hi = np.percentile(nz if nz.size > 100 else v, (0.5, 99.5))
    v = np.clip(v, lo, hi); v = (v - lo) / (hi - lo + 1e-8)
    t = torch.from_numpy(v.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(size,) * 3, mode="trilinear", align_corners=False)
    return t[0, 0].numpy()


def synth_case(dim, size, rng):
    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, size)] * dim, indexing="ij"))
    img = np.zeros((size,) * dim, np.float32)
    for _ in range(rng.randint(4, 9)):
        c = rng.uniform(-.6, .6, dim); r = rng.uniform(.08, .4, dim); v = rng.uniform(.2, 1)
        d = sum(((grid[i] - c[i]) / r[i]) ** 2 for i in range(dim)); img += v * (d < 1)
    return np.clip(img, 0, 1)


def make_splits(n, seed=42, val=0.1, test=0.2):
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    nt, nv = max(1, round(n * test)), max(1, round(n * val))
    return dict(test=sorted(idx[:nt]), val=sorted(idx[nt:nt + nv]), train=sorted(idx[nt + nv:]))


def prepare_dataset(ds, root, cfg, paths, args, log):
    pd_ = paths.prep(ds); man = pd_ / "manifest.json"
    if man.exists() and not args.force_prep:
        log.info("[%s] prep cached -> %s" % (ds, pd_)); return json.loads(man.read_text())
    arrs, ids = [], []
    if args.synthetic:
        rng = np.random.RandomState(0)
        for i in range(args.max_cases or 24):
            arrs.append(synth_case(cfg["dim"], cfg["size"], rng).astype(np.float16)); ids.append("%s_%04d_synth" % (ds, i))
    else:
        root = resolve_root(ds, root)
        files = discover(ds, root)
        if not files: raise RuntimeError("[%s] no files found under %s" % (ds, root))
        mx = args.max_cases or cfg["max_cases"]
        if len(files) > mx: files = sorted(random.Random(args.seed).sample(files, mx))
        log.info("[%s] %d files (root=%s)" % (ds, len(files), root))
        for i, f in enumerate(files):
            try:
                a = load_2d(f, cfg["size"]) if cfg["dim"] == 2 else load_3d(f, cfg["size"])
            except Exception as e:
                log.info("  SKIP %s: %s" % (f, e)); continue
            arrs.append(a.astype(np.float16))
            stem = Path(f).parent.name if ds == "brats" else Path(f).stem.replace(".nii", "")
            ids.append("%s_%04d_%s" % (ds, i, stem))
    data = np.stack(arrs); splits = make_splits(len(ids), args.seed)
    pd_.mkdir(parents=True, exist_ok=True); np.save(pd_ / "data.npy", data)
    man.write_text(json.dumps(dict(ds=ds, dim=cfg["dim"], ids=ids, splits=splits), indent=1))
    log.info("[%s] data %s  train/val/test = %d/%d/%d" % (ds, data.shape, len(splits["train"]), len(splits["val"]), len(splits["test"])))
    return json.loads(man.read_text())


def load_split(paths, ds, split):
    man = json.loads((paths.prep(ds) / "manifest.json").read_text())
    data = np.load(paths.prep(ds) / "data.npy", mmap_mode="r")
    idx = man["splits"][split]
    return np.asarray(data[idx]), [man["ids"][i] for i in idx]


class ArrDataset(Dataset):
    def __init__(self, arr, ds_key, augment=False):
        c = DS[ds_key]; self.arr, self.augment = arr, augment
        self.mod, self.ana, self.dim_id = MODALITY_IDS[c["modality"]], ANATOMY_IDS[c["anatomy"]], c["dim"] - 2
    def __len__(self): return len(self.arr)
    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.arr[i], dtype=np.float32)).unsqueeze(0)
        if self.augment:
            for ax in range(1, x.dim()):
                if random.random() < 0.5: x = torch.flip(x, [ax])
        return {"x": x, "mod": self.mod, "ana": self.ana, "dim": self.dim_id, "idx": i}


# ----------------------------------------------------------------- forward operator  A = mask . FFT (ortho)
def _dims(x): return tuple(range(2, x.dim()))

def A_op(x, mask):
    with torch.autocast("cuda", enabled=False):
        return torch.fft.fftn(x.float(), dim=_dims(x), norm="ortho") * mask

def AH_op(y, mask):
    with torch.autocast("cuda", enabled=False):
        return torch.fft.ifftn(y * mask, dim=_dims(y), norm="ortho").real

def AHA_op(x, mask): return AH_op(A_op(x, mask), mask)

def make_gen(device, seed):
    g = torch.Generator(device=device); g.manual_seed(int(seed)); return g

def make_mask(B, spatial, keep_frac, center_frac, gen, device):
    """Random phase-encode line mask (axis -2) with fully sampled centre; un-shifted FFT layout."""
    H = spatial[-2]
    ncent = max(2, int(round(H * center_frac))); nkeep = max(ncent + 1, int(round(H * keep_frac)))
    sc = torch.rand(B, H, generator=gen, device=device)
    c0 = H // 2 - ncent // 2; sc[:, c0:c0 + ncent] = 2.0
    idx = sc.topk(nkeep, dim=1).indices
    m = torch.zeros(B, H, device=device); m.scatter_(1, idx, 1.0)
    m = torch.fft.ifftshift(m, dim=1)
    shape = [B, 1] + [1] * len(spatial); shape[-2] = H
    return m.view(*shape)

def measure(x, mask, sigma, gen):
    import math
    k = A_op(x, mask)
    n = torch.complex(torch.randn(k.shape, generator=gen, device=k.device), torch.randn(k.shape, generator=gen, device=k.device)) * (sigma / math.sqrt(2))
    y = k + n * mask
    return y, AH_op(y, mask)


# ----------------------------------------------------------------- metrics
def ssim_torch(p, t, win=7):
    d = p.dim() - 2; pool = F.avg_pool2d if d == 2 else F.avg_pool3d
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mp, mt = pool(p, win, 1), pool(t, win, 1)
    sp, st = pool(p * p, win, 1) - mp ** 2, pool(t * t, win, 1) - mt ** 2
    cv = pool(p * t, win, 1) - mp * mt
    m = ((2 * mp * mt + C1) * (2 * cv + C2)) / ((mp ** 2 + mt ** 2 + C1) * (sp + st + C2))
    return m.flatten(1).mean(1)

def compute_metrics(pred, tgt):
    p, t = pred.float().clamp(0, 1), tgt.float(); dims = tuple(range(1, p.dim()))
    se = (p - t) ** 2; mse = se.mean(dims)
    return dict(PSNR=10 * torch.log10(1.0 / (mse + 1e-12)), SSIM=ssim_torch(p, t), MAE=(p - t).abs().mean(dims),
                RMSE=mse.sqrt(), NMSE=se.sum(dims) / ((t ** 2).sum(dims) + 1e-12))
