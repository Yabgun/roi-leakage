"""Model C (küçük şifreli CNN): katman başına tek evrişim (adım = çekirdek), BN, kare aktivasyon, tam bağlantılı.

Temsilin her katmanı (odak, yakın çevre, genel bakış; tam görüntüde tek katman) ayrı bir görüntüdür. Çekirdek k, S'yi
tam bölen ve çıktı haritasını ~16×16 yapan değerdir (U512: 32, U256: 16, F64: 4, G32: 2). Pencereler örtüşmediği için
im2col düzeni piksellerin satır öncelikli pencere sırasıdır. Eğitimde katman başına ölçek standardizasyonu ve BN
kullanılır; aktarımda evrişime katlanır. Geometri değerleri bu modelde kullanılmaz (genel bakış katmanı konumu taşır).

CKKS (TenSEAL 0.3.16): katman başına pencere matrisi `enc_matmul_encoding` ile ciphertext'lere bölünür (ciphertext başına
en çok 8.192 değer). Sunucu kanal başına `enc_matmul_plain` (evrişim) + yanlılık + kare, sonra sınıf başına `dot` ile tam
bağlantılı katmanı hesaplar. `pack_vectors` kullanılmaz: bu sürümde birleştirilmiş vektörde sonraki çarpma "scale out of
bounds" veriyor (15 Eyl deneyi). Çarpma derinliği 3; Π_ROI parametreleri yeterli.

Öz sınama: .venv\\Scripts\\python -m foveahe.he_cnn
"""
from __future__ import annotations

import time

import numpy as np
import tenseal as ts
import torch
import torch.nn as nn

import config
from foveahe.representation import FoveaSpec, parse_layer_key
from he.piroi import Cost

CHANNELS = 4
TARGET_MAP = 16


def kernel_for(size: int, target: int = TARGET_MAP) -> int:
    """size'ı tam bölen ve çıktı haritası kenarını target'a en yakın yapan çekirdek."""
    divisors = [k for k in range(1, size + 1) if size % k == 0]
    return min(divisors, key=lambda k: (abs(size // k - target), k))


def layer_plan(spec: FoveaSpec) -> list[tuple[str, int, int]]:
    """Vektördeki sırayla (anahtar, boyut S, çekirdek k)."""
    return [(key, parse_layer_key(key)[1], kernel_for(parse_layer_key(key)[1])) for key in spec.layer_keys]


def windows(img: np.ndarray, k: int) -> np.ndarray:
    """(S,S) -> (pencere sayısı, k*k): satır öncelikli örtüşmeyen pencereler, pencere içi satır öncelikli."""
    s = img.shape[0] // k
    return img[:s * k, :s * k].reshape(s, k, s, k).transpose(0, 2, 1, 3).reshape(s * s, k * k)


class ModelC(nn.Module):
    """n_geom > 0 (yalnızca geometrili eş örnekli referans, inceleme S2): pencere geometrisi standardize edilip tam
    bağlantılı katmana doğrudan eklenir (doğrusal; şifreli çıkarımda ek çarpma derinliği gerektirmez)."""

    def __init__(self, plan, n_classes: int, stats, channels: int = CHANNELS, n_geom: int = 0, geom_stats=None):
        super().__init__()
        self.plan = list(plan)
        self.n_geom = n_geom
        self.convs = nn.ModuleList([nn.Conv2d(1, channels, k, stride=k) for _, _, k in self.plan])
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for _ in self.plan])
        self.fc = nn.Linear(sum(channels * (s // k) ** 2 for _, s, k in self.plan) + n_geom, n_classes)
        self.register_buffer("mean", torch.tensor([m for m, _ in stats], dtype=torch.float32))
        self.register_buffer("std", torch.tensor([s for _, s in stats], dtype=torch.float32))
        gm, gs = geom_stats if geom_stats is not None else (torch.zeros(n_geom), torch.ones(n_geom))
        self.register_buffer("geom_mean", torch.as_tensor(gm, dtype=torch.float32).reshape(n_geom))
        self.register_buffer("geom_std", torch.as_tensor(gs, dtype=torch.float32).reshape(n_geom))
        self.offsets = np.cumsum([0] + [s * s for _, s, _ in self.plan]).tolist()

    def forward(self, x):
        feats = []
        for i, (_, s, _) in enumerate(self.plan):
            img = x[:, self.offsets[i]:self.offsets[i + 1]].reshape(-1, 1, s, s)
            z = self.bns[i](self.convs[i]((img - self.mean[i]) / self.std[i]))
            feats.append((z * z).flatten(1))
        if self.n_geom:
            end = self.offsets[-1]
            feats.append((x[:, end:end + self.n_geom] - self.geom_mean) / self.geom_std)
        return self.fc(torch.cat(feats, dim=1))


@torch.no_grad()
def export_c(model: ModelC) -> dict:
    """Standardizasyon ve BN evrişime (geometri standardizasyonu tam bağlantılı katmana) katlanmış float64 ağırlıklar."""
    w = {"kind": "C", "n_layers": len(model.plan), "n_geom": int(model.n_geom)}
    for i, (_, s, k) in enumerate(model.plan):
        conv, bn = model.convs[i], model.bns[i]
        mean, std = model.mean[i].double(), model.std[i].double()
        kern = conv.weight[:, 0].double()
        bias = conv.bias.double() - kern.sum((1, 2)) * mean / std
        kern = kern / std
        g = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
        w[f"L{i}_size"], w[f"L{i}_k"] = s, k
        w[f"L{i}_kernels"] = (kern * g[:, None, None]).cpu().numpy()
        w[f"L{i}_bias"] = ((bias - bn.running_mean.double()) * g + bn.bias.double()).cpu().numpy()
    W2, b2 = model.fc.weight.double().clone(), model.fc.bias.double().clone()
    if model.n_geom:
        gm, gs = model.geom_mean.double(), model.geom_std.double()
        b2 -= (W2[:, -model.n_geom:] * (gm / gs)).sum(1)
        W2[:, -model.n_geom:] = W2[:, -model.n_geom:] / gs
    w["W2"], w["b2"] = W2.cpu().numpy(), b2.cpu().numpy()
    return w


def forward_numpy_c(w: dict, x: np.ndarray) -> np.ndarray:
    """Aktarılan ağırlıklarla şifresiz başvuru çıkarımı: (N, girdi) -> (N, sınıf) logit."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    feats, off = [], 0
    for i in range(int(w["n_layers"])):
        s, k = int(w[f"L{i}_size"]), int(w[f"L{i}_k"])
        imgs = x[:, off:off + s * s].reshape(-1, s, s)
        off += s * s
        win = np.stack([windows(im, k) for im in imgs])                      # (N, pencere, k*k)
        kern = w[f"L{i}_kernels"].reshape(len(w[f"L{i}_bias"]), -1)          # (kanal, k*k)
        z = np.einsum("npq,cq->ncp", win, kern) + w[f"L{i}_bias"][None, :, None]
        feats.append((z * z).reshape(len(x), -1))
    n_geom = int(w.get("n_geom", 0))
    if n_geom:
        feats.append(x[:, off:off + n_geom])
    return np.concatenate(feats, axis=1) @ w["W2"].T + w["b2"]


def intermediate_max(w: dict, x: np.ndarray) -> float:
    """CKKS başlık payı için en büyük ara değer: evrişim çıktısı ve karesi arasında en büyük mutlak değer."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    best, off = 0.0, 0
    for i in range(int(w["n_layers"])):
        s, k = int(w[f"L{i}_size"]), int(w[f"L{i}_k"])
        win = np.stack([windows(im, k) for im in x[:, off:off + s * s].reshape(-1, s, s)])
        off += s * s
        z = np.einsum("npq,cq->ncp", win, w[f"L{i}_kernels"].reshape(len(w[f"L{i}_bias"]), -1))
        z = z + w[f"L{i}_bias"][None, :, None]
        best = max(best, float(np.abs(z).max()), float((z * z).max()))
    return best


class CNNInference:
    """Model C'nin CKKS ile şifreli çıkarımı; `FoveaHEInference` ile aynı arayüz (encrypt, server, decrypt, run)."""

    def __init__(self, ctx: ts.Context, weights: dict, poly_modulus: int = config.PIROI_POLY_MODULUS):
        if int(weights.get("n_geom", 0)):
            raise NotImplementedError("geometrili Model C referansı yalnızca şifresiz doğruluk için eğitildi")
        self.ctx = ctx
        slots = poly_modulus // 2
        self.layers, fc_off = [], 0
        for i in range(int(weights["n_layers"])):
            s, k = int(weights[f"L{i}_size"]), int(weights[f"L{i}_k"])
            kern = np.asarray(weights[f"L{i}_kernels"], dtype=np.float64).reshape(len(weights[f"L{i}_bias"]), -1)
            nw = (s // k) ** 2
            per = max(1, slots // (k * k))
            self.layers.append({"size": s, "k": k, "kern": kern, "bias": np.asarray(weights[f"L{i}_bias"]),
                                "nw": nw, "chunks": [(a, min(nw, a + per)) for a in range(0, nw, per)],
                                "fc_off": fc_off})
            fc_off += kern.shape[0] * nw
        self.w2 = np.asarray(weights["W2"], dtype=np.float64)
        self.b2 = np.asarray(weights["b2"], dtype=np.float64)

    @property
    def n_ciphertexts(self) -> int:
        return sum(len(L["chunks"]) for L in self.layers)

    def encrypt(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).ravel()
        out, off = [], 0
        for L in self.layers:
            s = L["size"]
            win = windows(x[off:off + s * s].reshape(s, s), L["k"])
            off += s * s
            out.append([ts.enc_matmul_encoding(self.ctx, win[a:b].tolist()) for a, b in L["chunks"]])
        return out

    def server(self, enc):
        acc = [None] * self.w2.shape[0]
        for L, chunks in zip(self.layers, enc):
            for j in range(L["kern"].shape[0]):
                kern, base = L["kern"][j].tolist(), L["fc_off"] + j * L["nw"]
                for (a, b), e in zip(L["chunks"], chunks):
                    h = (e.enc_matmul_plain(kern, b - a) + float(L["bias"][j])).square()
                    for i in range(len(acc)):
                        d = h.dot(self.w2[i, base + a:base + b].tolist())
                        acc[i] = d if acc[i] is None else acc[i] + d
        return [a + float(b) for a, b in zip(acc, self.b2)]

    @staticmethod
    def decrypt(outs) -> np.ndarray:
        return np.array([ct.decrypt()[0] for ct in outs])

    def run(self, x: np.ndarray, measure_bytes: bool = True):
        cost = Cost()
        t0 = time.perf_counter()
        enc = self.encrypt(x)
        cost.enc_s = time.perf_counter() - t0
        cost.n_ciphertexts = self.n_ciphertexts
        cost.extra["sifreli_slot"] = int(sum(L["nw"] * L["k"] ** 2 for L in self.layers))
        if measure_bytes:
            cost.upload_bytes = sum(len(v.serialize()) for chunks in enc for v in chunks)
        t0 = time.perf_counter()
        outs = self.server(enc)
        cost.server_s = time.perf_counter() - t0
        if measure_bytes:
            cost.extra["indirme_bayt"] = sum(len(o.serialize()) for o in outs)
        t0 = time.perf_counter()
        logits = self.decrypt(outs)
        cost.dec_s = time.perf_counter() - t0
        return logits, cost


def _selftest():
    assert [kernel_for(s) for s in (512, 256, 224, 128, 90, 64, 48, 32, 16)] == [32, 16, 14, 8, 6, 4, 3, 2, 1]
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # 1) Katlanmış aktarım PyTorch modelini birebir verir (BN çalışma istatistikleri rastgele)
    spec = FoveaSpec.parse("F64_G32")
    plan = layer_plan(spec)
    model = ModelC(plan, 3, [(0.4, 0.2), (0.3, 0.25)]).double()
    for bn in model.bns:
        bn.running_mean.uniform_(-1, 1)
        bn.running_var.uniform_(0.5, 2)
        bn.weight.data.uniform_(0.5, 1.5)
        bn.bias.data.uniform_(-0.5, 0.5)
    model.eval()
    x = rng.random((5, spec.n_values))
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    w = export_c(model)
    assert np.abs(forward_numpy_c(w, x) - ref).max() < 1e-9

    # 2) Şifreli çıkarım: tek parçalı katmanlar (F64 k=4: 4.096 değer, G32 k=2: 1.024 değer)
    from he.piroi import make_context
    ctx = make_context()
    infer = CNNInference(ctx, w)
    logits, cost = infer.run(x[0])
    err1 = np.abs(logits - ref[0]).max()
    assert infer.n_ciphertexts == 2 and err1 < 1e-3, (infer.n_ciphertexts, err1)

    # 3) Parçalı katman: U256, k=16 (256 pencere × 256 = 65.536 değer -> 8 ciphertext)
    spec2 = FoveaSpec.parse("U256")
    model2 = ModelC(layer_plan(spec2), 3, [(0.5, 0.25)]).double().eval()
    w2 = export_c(model2)
    x2 = rng.random((1, spec2.n_values))
    infer2 = CNNInference(ctx, w2)
    logits2, cost2 = infer2.run(x2[0])
    err2 = np.abs(logits2 - forward_numpy_c(w2, x2)[0]).max()
    assert infer2.n_ciphertexts == 8 and err2 < 1e-3, (infer2.n_ciphertexts, err2)
    print(f"foveahe.he_cnn öz sınama: tamam (F64_G32 hata {err1:.1e}, {cost.total_s:.2f} s; "
          f"U256 8 ciphertext hata {err2:.1e}, {cost2.total_s:.2f} s)")


if __name__ == "__main__":
    _selftest()
