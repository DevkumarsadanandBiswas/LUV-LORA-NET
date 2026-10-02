"""Baselines and OURS (unified U-Net-ViT + LoRA + FiLM + data consistency) following the diagram."""
import math, inspect
import torch, torch.nn as nn, torch.nn.functional as F
from common import A_op, AH_op


def convnd(dim): return nn.Conv2d if dim == 2 else nn.Conv3d
def _gn(c): return nn.GroupNorm(min(8, c), c)


# =============================================================== baselines
class DoubleConv(nn.Sequential):
    def __init__(self, dim, i, o):
        C = convnd(dim)
        super().__init__(C(i, o, 3, padding=1), _gn(o), nn.LeakyReLU(0.1, True), C(o, o, 3, padding=1), _gn(o), nn.LeakyReLU(0.1, True))


class UNetBL(nn.Module):
    use_amp = True
    def __init__(self, dim, base=32, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth)]; self.depth = depth
        self.enc = nn.ModuleList([DoubleConv(dim, 1 if i == 0 else ch[i - 1], ch[i]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2) if dim == 2 else nn.MaxPool3d(2)
        T = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        self.up = nn.ModuleList([T(ch[i + 1], ch[i], 2, stride=2) for i in range(depth - 1)])
        self.dec = nn.ModuleList([DoubleConv(dim, 2 * ch[i], ch[i]) for i in range(depth - 1)])
        self.head = convnd(dim)(ch[0], 1, 1)
    def forward(self, zf, y=None, mask=None, mod=None, ana=None, dimt=None):
        x, sk = zf, []
        for i, b in enumerate(self.enc):
            x = b(x if i == 0 else self.pool(x)); sk.append(x)
        x = sk.pop()
        for i in reversed(range(self.depth - 1)):
            x = self.dec[i](torch.cat([self.up[i](x), sk.pop()], 1))
        return zf + self.head(x)


class DnCNN(nn.Module):
    use_amp = True
    def __init__(self, dim, ch=64, layers=15):
        super().__init__()
        C = convnd(dim)
        # BatchNorm with batch size 2 (3-D) is unstable at eval time -> GroupNorm for 3-D (fairer to the baseline)
        NORM = (lambda c: nn.BatchNorm2d(c)) if dim == 2 else (lambda c: nn.GroupNorm(8, c))
        L = [C(1, ch, 3, padding=1), nn.ReLU(True)]
        for _ in range(layers - 2): L += [C(ch, ch, 3, padding=1, bias=False), NORM(ch), nn.ReLU(True)]
        L += [C(ch, 1, 3, padding=1)]; self.net = nn.Sequential(*L)
    def forward(self, zf, y=None, mask=None, mod=None, ana=None, dimt=None):
        return zf - self.net(zf)


class SwinUNETRBL(nn.Module):
    use_amp = True
    def __init__(self, dim, img_size, feature=24):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        sig = inspect.signature(SwinUNETR.__init__); kw = dict(in_channels=1, out_channels=1, feature_size=feature)
        if "spatial_dims" in sig.parameters: kw["spatial_dims"] = dim
        if "img_size" in sig.parameters: kw["img_size"] = (img_size,) * dim
        self.net = SwinUNETR(**kw)
    def forward(self, zf, y=None, mask=None, mod=None, ana=None, dimt=None):
        return zf + self.net(zf)


class SmallCNN(nn.Module):
    def __init__(self, dim, ch=32, layers=4):
        super().__init__()
        C = convnd(dim); L = [C(1, ch, 3, padding=1), nn.ReLU(True)]
        for _ in range(layers - 2): L += [C(ch, ch, 3, padding=1), nn.ReLU(True)]
        L += [C(ch, 1, 3, padding=1)]; self.net = nn.Sequential(*L)
    def forward(self, x): return self.net(x)


class VarNet(nn.Module):
    use_amp = False
    def __init__(self, dim, ch=32, T=6):
        super().__init__()
        self.regs = nn.ModuleList([SmallCNN(dim, ch, 3) for _ in range(T)]); self.lams = nn.Parameter(torch.full((T,), 0.5)); self.T = T
    def forward(self, zf, y, mask, mod=None, ana=None, dimt=None):
        x = zf
        for t in range(self.T):
            x = x - self.lams[t] * AH_op(A_op(x, mask) - y, mask) + self.regs[t](x)
        return x


class ADMMNet(nn.Module):
    use_amp = False
    def __init__(self, dim, ch=32, K=6):
        super().__init__()
        C = convnd(dim); self.K = K
        self.ta = nn.ModuleList([C(1, ch, 3, padding=1) for _ in range(K)]); self.tb = nn.ModuleList([C(ch, 1, 3, padding=1) for _ in range(K)])
        self.thr = nn.Parameter(torch.full((K,), 0.01)); self.rho_raw = nn.Parameter(torch.full((K,), -1.0))
    def forward(self, zf, y, mask, mod=None, ana=None, dimt=None):
        dims = tuple(range(2, zf.dim())); x, z, u = zf, zf, torch.zeros_like(zf)
        for k in range(self.K):
            rho = F.softplus(self.rho_raw[k]) + 1e-3
            Fb = torch.fft.fftn(z - u, dim=dims, norm="ortho")
            x = torch.fft.ifftn((y + rho * Fb) / (mask + rho), dim=dims, norm="ortho").real
            v = x + u; h = self.ta[k](v); h = torch.sign(h) * F.relu(h.abs() - self.thr[k].abs())
            z = v + self.tb[k](h); u = u + x - z
        return x


# =============================================================== OURS (matches the architecture diagram)
class LoRALinear(nn.Module):
    """Linear (+ optional low-rank adapter). LoRA lives ONLY in ViT attention q,k,v,out-proj."""
    def __init__(self, i, o, rank=16, enable_lora=True, alpha=16.0):
        super().__init__()
        self.base = nn.Linear(i, o); self.enable_lora = enable_lora; self.use_lora = enable_lora; self.scale = alpha / max(rank, 1)
        if enable_lora:
            self.lora_down = nn.Linear(i, rank, bias=False); self.lora_up = nn.Linear(rank, o, bias=False)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5)); nn.init.zeros_(self.lora_up.weight)
    def forward(self, x):
        out = self.base(x)
        if self.enable_lora and self.use_lora: out = out + self.lora_up(self.lora_down(x)) * self.scale
        return out


class PEG(nn.Module):
    """Positional Encoding Generator: depth-wise 3x3 conv (works at every resolution)."""
    def __init__(self, c, dim): super().__init__(); self.pe = convnd(dim)(c, c, 3, padding=1, groups=c)
    def forward(self, x): return x + self.pe(x)


class ViTBlock(nn.Module):
    """PEG -> [window partition] -> LN -> MHSA(q,k,v,proj with LoRA) -> res -> LN -> MLP(ratio 4) -> res.
    window=None -> global attention (bottleneck)."""
    def __init__(self, c, dim, rank, enable_lora, heads=4, mlp_ratio=4, window=None):
        super().__init__()
        assert c % heads == 0
        self.dim, self.c, self.h, self.window = dim, c, heads, window
        self.peg = PEG(c, dim); self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.q, self.k, self.v, self.proj = [LoRALinear(c, c, rank, enable_lora) for _ in range(4)]
        self.fc1, self.fc2, self.act = nn.Linear(c, c * mlp_ratio), nn.Linear(c * mlp_ratio, c), nn.GELU()
    def _tok(self, t):
        Bw, N, C = t.shape; hd = C // self.h; hh = self.n1(t)
        sp = lambda m: m(hh).view(Bw, N, self.h, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(sp(self.q), sp(self.k), sp(self.v)).transpose(1, 2).reshape(Bw, N, C)
        t = t + self.proj(a)
        return t + self.fc2(self.act(self.fc1(self.n2(t))))
    def forward(self, x):
        x = self.peg(x); B, C = x.shape[:2]; sp = x.shape[2:]; w = self.window
        if w is None or all(s <= w for s in sp):
            t = self._tok(x.flatten(2).transpose(1, 2)); return t.transpose(1, 2).reshape(B, C, *sp)
        pads = []
        for s in reversed(sp): pads += [0, (w - s % w) % w]
        xp = F.pad(x, pads) if any(pads) else x; ps = xp.shape[2:]; g = [s // w for s in ps]
        if self.dim == 2:
            t = xp.reshape(B, C, g[0], w, g[1], w).permute(0, 2, 4, 3, 5, 1).reshape(-1, w * w, C)
            t = self._tok(t).reshape(B, g[0], g[1], w, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, *ps)
        else:
            t = xp.reshape(B, C, g[0], w, g[1], w, g[2], w).permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(-1, w ** 3, C)
            t = self._tok(t).reshape(B, g[0], g[1], g[2], w, w, w, C).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, C, *ps)
        return t[(slice(None), slice(None)) + tuple(slice(0, s) for s in sp)].contiguous()


class ConvBlock(nn.Sequential):
    """2 x [Conv3x3 -> GroupNorm(8) -> SiLU]"""
    def __init__(self, dim, i, o):
        C = convnd(dim); super().__init__(C(i, o, 3, padding=1), _gn(o), nn.SiLU(), C(o, o, 3, padding=1), _gn(o), nn.SiLU())


class CondEmbedding(nn.Module):
    """Modality(32) + Anatomy(32) + Dimension(16) + Degradation(16) -> concat(96) -> MLP -> Condition Embedding(128)."""
    def __init__(self, n_mod, n_ana, hid=128):
        super().__init__()
        self.m, self.a, self.d = nn.Embedding(n_mod, 32), nn.Embedding(n_ana, 32), nn.Embedding(2, 16)
        self.deg = nn.Sequential(nn.Linear(3, 16), nn.SiLU())              # degradation token: [keep, centre, sigma*10]
        self.mlp = nn.Sequential(nn.Linear(96, hid), nn.SiLU())
    def forward(self, mod, ana, dimt, deg):
        dv = deg.to(mod.device).unsqueeze(0).expand(mod.shape[0], -1)
        return self.mlp(torch.cat([self.m(mod), self.a(ana), self.d(dimt), self.deg(dv)], -1))


class DataConsistency(nn.Module):
    """x = x0 + lambda * A^H (y - A x0),  lambda = sigmoid(w[modality]) (learnable). k-space DC (masked Fourier)."""
    def __init__(self, n_mod=4, init=0.5):
        super().__init__()
        self.w = nn.Embedding(n_mod, 1); nn.init.constant_(self.w.weight, math.log(init / (1 - init)))
    def forward(self, x0, y, mask, mod):
        lam = torch.sigmoid(self.w(mod)).view(-1, *([1] * (x0.dim() - 1)))
        return x0.float() + lam * AH_op(y - A_op(x0, mask), mask)


class UNetViTLoRA(nn.Module):
    """Stem(24) -> [Conv -> WindowViT(LoRA) -> Down] x2 (24->48->96) -> Global ViT bottleneck(96)+FiLM ->
       [Up -> WindowViT(LoRA) -> Concat skip -> Conv] x2 (48,24) -> 1x1 conv -> x0 = zf + head -> Data Consistency."""
    use_amp = True
    def __init__(self, dim=2, base=24, rank=16, enable_lora=True, heads=4, mlp_ratio=4, window=8, use_cond=True, cond_stages=True,
                 use_vit=True, use_bottleneck=True, use_dc=True, deg=(0.3, 0.08, 0.2), n_mod=4, n_ana=5):
        super().__init__()
        self.dim, self.use_cond, self.cond_stages, self.use_dc = dim, use_cond, cond_stages, use_dc
        c1, c2, c3 = base, base * 2, base * 4
        C = convnd(dim); T = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        vit = lambda c: ViTBlock(c, dim, rank, enable_lora, heads, mlp_ratio, window) if use_vit else nn.Identity()
        self.stem = nn.Sequential(C(1, c1, 3, padding=1), _gn(c1), nn.SiLU())
        self.enc1_conv, self.enc1_vit = ConvBlock(dim, c1, c1), vit(c1)
        self.down1 = C(c1, c2, 2, stride=2)
        self.enc2_conv, self.enc2_vit = ConvBlock(dim, c2, c2), vit(c2)
        self.down2 = C(c2, c3, 2, stride=2)
        self.bottleneck = ViTBlock(c3, dim, rank, enable_lora, heads, mlp_ratio, None) if use_bottleneck else nn.Identity()
        self.up1, self.dec1_vit, self.dec1_conv = T(c3, c2, 2, stride=2), vit(c2), ConvBlock(dim, 2 * c2, c2)
        self.up2, self.dec2_vit, self.dec2_conv = T(c2, c1, 2, stride=2), vit(c1), ConvBlock(dim, 2 * c1, c1)
        self.head = C(c1, 1, 1)
        self.dc = DataConsistency(n_mod) if use_dc else None
        if use_cond:
            self.register_buffer("deg", torch.tensor(deg, dtype=torch.float32))
            self.cond = CondEmbedding(n_mod, n_ana, 128)
            self.films = nn.ModuleList([nn.Linear(128, 2 * c) for c in (c1, c2, c3)])      # stage1, stage2, bottleneck
            for f in self.films: nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)          # start as identity
    def film(self, x, i, cv):
        if cv is None or (i < 2 and not self.cond_stages): return x
        sc, sh = self.films[i](cv).chunk(2, 1); s = (x.shape[0], x.shape[1]) + (1,) * self.dim
        return x * (1 + sc.view(*s)) + sh.view(*s)
    def set_stage(self, stage, adapt_norms=True):
        """pretrain: everything trainable, LoRA inactive.  adapt: backbone frozen; LoRA + FiLM/cond + head + DC-lambda
        (+ tiny norm affines) trainable."""
        lm = [m for m in self.modules() if isinstance(m, LoRALinear)]
        if stage == "pretrain":
            for m in lm: m.use_lora = False
            for p in self.parameters(): p.requires_grad_(True)
        elif stage == "adapt":
            for m in lm: m.use_lora = m.enable_lora
            for n, p in self.named_parameters(): p.requires_grad_("lora_" in n or n.startswith(("films", "cond", "head", "dc")))
            if adapt_norms:
                for m in self.modules():
                    if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                        for p in m.parameters(): p.requires_grad_(True)
        else: raise ValueError(stage)
    def forward(self, zf, y=None, mask=None, mod=None, ana=None, dimt=None):
        z = torch.zeros(zf.shape[0], dtype=torch.long, device=zf.device)
        mod = z if mod is None else mod
        cv = self.cond(mod, z if ana is None else ana, z if dimt is None else dimt, self.deg) if self.use_cond else None
        s = self.stem(zf)
        e1 = self.enc1_vit(self.film(self.enc1_conv(s), 0, cv))
        e2 = self.enc2_vit(self.film(self.enc2_conv(self.down1(e1)), 1, cv))
        b = self.film(self.bottleneck(self.down2(e2)), 2, cv)
        d1 = self.dec1_conv(torch.cat([self.dec1_vit(self.up1(b)), e2], 1))
        d2 = self.dec2_conv(torch.cat([self.dec2_vit(self.up2(d1)), e1], 1))
        x0 = zf + self.head(d2)
        if self.dc is not None and y is not None and mask is not None: return self.dc(x0, y, mask, mod)
        return x0


# =============================================================== factory / summary
def build_model(name, dim, size, **kw):
    d3 = dim == 3
    if name == "unet":      return UNetBL(dim, base=16 if d3 else 32, depth=4)
    if name == "dncnn":     return DnCNN(dim, ch=32 if d3 else 64, layers=10 if d3 else 15)
    if name == "swinunetr": return SwinUNETRBL(dim, size, 24)
    if name == "varnet":    return VarNet(dim, ch=16 if d3 else 32, T=4 if d3 else 6)
    if name == "admmnet":   return ADMMNet(dim, ch=16 if d3 else 32, K=4 if d3 else 6)
    if name in ("ours", "ours_scratch"):
        return UNetViTLoRA(dim=dim, base=24, window=4 if d3 else 8, rank=kw.get("rank", 16),
                           enable_lora=kw.get("enable_lora", name == "ours"), use_cond=kw.get("use_cond", True),
                           use_vit=kw.get("use_vit", True), use_bottleneck=kw.get("use_bottleneck", True), use_dc=kw.get("use_dc", True))
    raise ValueError(name)

def count_params(m): return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)

def model_summary_text(model, header=""):
    L = [header, "%-28s %14s %14s" % ("module", "params", "trainable"), "-" * 58]
    for n, c in model.named_children():
        L.append("%-28s %14s %14s" % (n, format(sum(p.numel() for p in c.parameters()), ","), format(sum(p.numel() for p in c.parameters() if p.requires_grad), ",")))
    t, tr = count_params(model); L += ["-" * 58, "TOTAL %s   TRAINABLE %s" % (format(t, ","), format(tr, ","))]
    return "\n".join(L)
