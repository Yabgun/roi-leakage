"""FoveaHE odaklı temsil: ROI merkezli, sabit boyutlu, çok çözünürlüklü görüntü özeti (tamamı şifrelenir).

İstemci (hastane) ROI maskesiyle üç katman çıkarır (Π_ROI ile aynı varsayım: istemci ROI'yi bilir):
- odak F×F: ROI sınırlayıcı kutusunun kare penceresi; kenar = max(w, h) × (1 + MARGIN), en az MIN_SIDE × görüntü kenarı
- yakın çevre P×P: aynı merkezli, k kat büyük pencere (P = 0 ise katman yok)
- genel bakış G×G: tüm görüntü
Pencereler alan ortalamasıyla örneklenir (roi_align, uyarlanabilir örnek ızgarası); görüntü dışı 0 ile doldurulur.
Katmanlar ve pencere geometrisi (merkez x, merkez y, kenar) tek vektörde birleşir. Uzunluk her görüntüde aynıdır;
ROI konumu ve boyutu yalnızca şifreli içerikte kalır.

`render`, ResNet-18 bilgi üst sınırı için temsili 224×224 odaklı görüntüye çevirir: her piksel onu kapsayan en ince
katmandan (en küçük örnek aralığı) bilineer olarak doldurulur. Çıktı temsilin deterministik bir fonksiyonudur.

Öz sınama: .venv\\Scripts\\python -m foveahe.representation
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

MARGIN = 0.25      # odak penceresi payı
MIN_SIDE = 0.10    # odak penceresinin en küçük kenarı, görüntü kenarına oranla
GEOM_VALUES = 3    # şifreli vektördeki geometri: merkez x, merkez y, odak kenarı (normalize)
SLOTS = 8192       # CKKS N = 16384: ciphertext başına slot


@dataclass(frozen=True)
class FoveaSpec:
    """Katman boyutları; 0 katmanın olmadığını gösterir. Odak ve yakın çevre yoksa eş örnekli küçültmedir."""
    focus: int = 0
    periphery: int = 0
    k: float = 2.0
    glob: int = 0

    @property
    def focus_key(self) -> str:
        return f"f{self.focus}"

    @property
    def periphery_key(self) -> str:
        return f"p{self.periphery}k{self.k:g}"

    @property
    def glob_key(self) -> str:
        return f"g{self.glob}"

    @property
    def uses_geometry(self) -> bool:
        return bool(self.focus or self.periphery)

    @property
    def layer_keys(self) -> list[str]:
        keys = [self.focus_key] if self.focus else []
        if self.periphery:
            keys.append(self.periphery_key)
        if self.glob:
            keys.append(self.glob_key)
        return keys

    @property
    def n_values(self) -> int:
        n = self.focus ** 2 + self.periphery ** 2 + self.glob ** 2
        return n + (GEOM_VALUES if self.uses_geometry else 0)

    @property
    def n_ciphertexts(self) -> int:
        return -(-self.n_values // SLOTS)

    @property
    def name(self) -> str:
        if not self.layer_keys:
            return "sabit"
        if not self.uses_geometry:
            return "tam" if self.glob == 224 else f"U{self.glob}"
        parts = [f"F{self.focus}"] if self.focus else []
        if self.periphery:
            parts.append(f"P{self.periphery}k{self.k:g}")
        if self.glob:
            parts.append(f"G{self.glob}")
        return "_".join(parts)

    @staticmethod
    def parse(name: str) -> "FoveaSpec":
        if name == "sabit":
            return FoveaSpec()
        if name == "tam":
            return FoveaSpec(glob=224)
        if name.startswith("U"):
            return FoveaSpec(glob=int(name[1:]))
        kw = {}
        for part in name.split("_"):
            if part[0] == "F":
                kw["focus"] = int(part[1:])
            elif part[0] == "G":
                kw["glob"] = int(part[1:])
            elif part[0] == "P":
                p, k = part[1:].split("k")
                kw["periphery"], kw["k"] = int(p), float(k)
            else:
                raise ValueError(name)
        return FoveaSpec(**kw)


def parse_layer_key(key: str) -> tuple[str, int, float]:
    """'f64' -> ('f', 64, 1.0), 'p32k2' -> ('p', 32, 2.0), 'g32' -> ('g', 32, 1.0)."""
    kind = key[0]
    if kind == "p":
        size, scale = key[1:].split("k")
        return kind, int(size), float(scale)
    if kind in "fg":
        return kind, int(key[1:]), 1.0
    raise ValueError(key)


def roi_geometry(masks: torch.Tensor, margin: float = MARGIN, min_side: float = MIN_SIDE) -> torch.Tensor:
    """masks: (N,H,W) bool, kare görüntü. Döner (N,3): odak penceresi merkez x, merkez y, kenar (görüntü kenarına göre).

    Piksel i, [i, i+1) aralığını kaplar. Boş maskede pencere tüm görüntüdür.
    """
    n, h, w = masks.shape
    if h != w:
        raise ValueError(f"kare görüntü bekleniyor: {h}x{w}")
    ar = torch.arange(h, device=masks.device)
    rows, cols = masks.any(dim=2), masks.any(dim=1)
    r0 = torch.where(rows, ar, h).amin(1)
    r1 = torch.where(rows, ar, -1).amax(1)
    c0 = torch.where(cols, ar, w).amin(1)
    c1 = torch.where(cols, ar, -1).amax(1)
    empty = r1 < 0
    cx = (c0 + c1 + 1).float() / 2
    cy = (r0 + r1 + 1).float() / 2
    side = torch.maximum(r1 - r0 + 1, c1 - c0 + 1).float() * (1 + margin)
    side = side.clamp(min=min_side * h)
    cx = torch.where(empty, torch.full_like(cx, w / 2), cx)
    cy = torch.where(empty, torch.full_like(cy, h / 2), cy)
    side = torch.where(empty, torch.full_like(side, float(h)), side)
    return torch.stack([cx / w, cy / h, side / h], dim=1)


def extract(img: torch.Tensor, geom: torch.Tensor, key: str) -> torch.Tensor:
    """img: (N,1,H,W) float; geom: (N,3). Döner (N,1,S,S): pencerenin alan ortalamalı örneği, görüntü dışı 0."""
    kind, size, scale = parse_layer_key(key)
    n, _, h, w = img.shape
    idx = torch.arange(n, device=img.device, dtype=img.dtype)
    if kind == "g":
        zero = torch.zeros(n, device=img.device, dtype=img.dtype)
        boxes = torch.stack([idx, zero, zero, zero + w, zero + h], dim=1)
    else:
        g = geom.to(img.dtype)
        cx, cy, half = g[:, 0] * w, g[:, 1] * h, g[:, 2] * h * scale / 2
        boxes = torch.stack([idx, cx - half, cy - half, cx + half, cy + half], dim=1)
    return roi_align(img, boxes, output_size=(size, size), spatial_scale=1.0, sampling_ratio=-1, aligned=True)


def vectorize(layers: dict[str, torch.Tensor], geom: torch.Tensor, spec: FoveaSpec) -> torch.Tensor:
    """Şifrelenecek sabit uzunluklu vektör: [odak, yakın çevre, genel bakış, geometri] -> (N, spec.n_values)."""
    parts = [layers[k].flatten(1) for k in spec.layer_keys]
    if spec.uses_geometry:
        parts.append(geom.to(parts[0].dtype if parts else geom.dtype))
    return torch.cat(parts, dim=1)


def _window_canvas(patch: torch.Tensor, geom: torch.Tensor, scale: float, out: int):
    """Pencere katmanını out×out tuvale örnekler. Döner (değer, pencere içi maskesi), ikisi de (N,1,out,out)."""
    n = patch.shape[0]
    u = (torch.arange(out, device=patch.device, dtype=patch.dtype) + 0.5) / out
    g = geom.to(patch.dtype)
    side = g[:, 2:3] * scale
    px = (u[None, :] - (g[:, 0:1] - side / 2)) / side
    py = (u[None, :] - (g[:, 1:2] - side / 2)) / side
    grid = torch.stack([(2 * px - 1)[:, None, :].expand(n, out, out),
                        (2 * py - 1)[:, :, None].expand(n, out, out)], dim=-1)
    values = F.grid_sample(patch, grid, mode="bilinear", padding_mode="border", align_corners=False)
    inside = ((px >= 0) & (px <= 1))[:, None, None, :] & ((py >= 0) & (py <= 1))[:, None, :, None]
    return values, inside


def render(layers: dict[str, torch.Tensor], geom: torch.Tensor, spec: FoveaSpec, out: int = 224) -> torch.Tensor:
    """Temsil -> (N,1,out,out) odaklı görüntü. Her piksel onu kapsayan en ince katmandan alınır; kapsanmayan piksel 0."""
    n = geom.shape[0]
    inf = torch.tensor(float("inf"), device=geom.device)
    values, steps = [], []
    for key, size, scale in ((spec.focus_key, spec.focus, 1.0), (spec.periphery_key, spec.periphery, spec.k)):
        if size:
            v, inside = _window_canvas(layers[key], geom, scale, out)
            step = (geom[:, 2] * scale / size).view(n, 1, 1, 1)
            values.append(v)
            steps.append(torch.where(inside, step, inf))
    if spec.glob:
        gl = layers[spec.glob_key]
        values.append(gl if spec.glob == out else
                      F.interpolate(gl, size=(out, out), mode="bilinear", align_corners=False))
        steps.append(torch.full((n, 1, out, out), 1.0 / spec.glob, device=geom.device))
    if not values:
        return torch.zeros(n, 1, out, out, device=geom.device)
    stack = torch.cat(values, dim=1)
    step = torch.cat([s.expand(n, 1, out, out) for s in steps], dim=1)
    best, choice = step.min(dim=1, keepdim=True)
    return torch.where(torch.isfinite(best), stack.gather(1, choice), torch.zeros_like(best))


def _selftest():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h = 256
    # 1) Geometri: iç ROI, köşede küçük ROI (en küçük kenar devrede), büyük ROI, boş maske
    masks = torch.zeros(4, h, h, dtype=torch.bool, device=dev)
    masks[0, 100:140, 50:70] = True
    masks[1, 0:10, 0:10] = True
    masks[2, 20:236, 30:226] = True
    geom = roi_geometry(masks)
    expect = torch.tensor([[60 / h, 120 / h, 50 / h], [5 / h, 5 / h, MIN_SIDE], [128 / h, 128 / h, 270 / h],
                           [0.5, 0.5, 1.0]], device=dev)
    assert torch.allclose(geom, expect), geom

    # 2) Sabit uzunluk: farklı ROI konum ve boyutlarında katman şekilleri ve vektör uzunluğu aynı
    torch.manual_seed(0)
    img = torch.rand(4, 1, h, h, device=dev)
    spec = FoveaSpec(focus=64, periphery=32, k=2, glob=32)
    layers = {k: extract(img, geom, k) for k in spec.layer_keys}
    for key, lay in layers.items():
        assert lay.shape == (4, 1, parse_layer_key(key)[1], parse_layer_key(key)[1]), (key, lay.shape)
    assert vectorize(layers, geom, spec).shape == (4, spec.n_values) and spec.n_values == 64 ** 2 + 32 ** 2 * 2 + 3

    # 3) Dolgu: köşedeki ROI'nin yakın çevre penceresi [-20.6, 30.6] px; kutucuk 1.6 px.
    #    Tamamen görüntü dışındaki satır ve sütunlar (0..11) tam 0, tamamen içeridekiler (13..) sabit değer.
    const = torch.full((4, 1, h, h), 0.7, device=dev)
    corner = extract(const, geom, "p32k2")[1, 0]
    assert corner[:12, :].abs().max() == 0 and corner[:, :12].abs().max() == 0
    assert torch.allclose(corner[13:, 13:], torch.full_like(corner[13:, 13:], 0.7), atol=1e-6)

    # 4) Genel bakış tam sayı katsayıda blok ortalamasıdır
    small = torch.rand(2, 1, 64, 64, device=dev)
    assert torch.allclose(extract(small, geom[:2], "g16"), F.avg_pool2d(small, 4), atol=1e-6)

    # 5) Render: G = 224 kimlik; odak doğru yere ve yalnızca penceresine yapıştırılır
    g224 = torch.rand(4, 1, 224, 224, device=dev)
    assert torch.equal(render({"g224": g224}, geom, FoveaSpec(glob=224)), g224)
    spec = FoveaSpec(focus=64, glob=16)
    canvas = render({"f64": torch.ones(4, 1, 64, 64, device=dev), "g16": torch.zeros(4, 1, 16, 16, device=dev)},
                    geom, spec)
    u = (torch.arange(224, device=dev) + 0.5) / 224
    for i in range(4):
        x0, y0, s = geom[i, 0] - geom[i, 2] / 2, geom[i, 1] - geom[i, 2] / 2, geom[i, 2]
        inside = (((u >= y0) & (u <= y0 + s))[:, None] & ((u >= x0) & (u <= x0 + s))[None, :])
        assert torch.equal(canvas[i, 0] > 0.5, inside), i

    # 6) Doğrusal eğim görüntüsü: odak içinde yeniden oluşturma eğimi korur
    ramp = ((torch.arange(h, device=dev) + 0.5) / h).view(1, 1, 1, h).expand(4, 1, h, h).contiguous()
    layers = {k: extract(ramp, geom, k) for k in spec.layer_keys}
    canvas = render(layers, geom, spec)
    x0, s = float(geom[0, 0] - geom[0, 2] / 2), float(geom[0, 2])
    cols = ((u >= x0 + 0.05 * s) & (u <= x0 + 0.95 * s)).nonzero().flatten()
    rows = ((u >= float(geom[0, 1] - geom[0, 2] / 2) + 0.05 * s) &
            (u <= float(geom[0, 1] + geom[0, 2] / 2) - 0.05 * s)).nonzero().flatten()
    err = (canvas[0, 0][rows][:, cols] - u[cols][None, :]).abs().max()
    assert err < 2e-3, err

    # 7) Adlar ve ayrıştırma
    for name in ["tam", "sabit", "U64", "F64", "F32_G16", "F64_P32k2_G32", "F64_P16k3_G16"]:
        assert FoveaSpec.parse(name).name == name, name
    print("foveahe.representation öz sınama: tamam", f"(cihaz={dev})")


if __name__ == "__main__":
    _selftest()
