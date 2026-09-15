"""Π_ROI (Bakas & Schoinianakis, IACR ePrint 2026/103) hesaplama çekirdeğinin TenSEAL ile yeniden üretimi.

Protokol özeti:
- İstemci (analist) yalnızca ROI piksellerini CKKS ile şifreler, gerisini açık gönderir.
- Sunucu ağırlık matrisini ROI ve ROI-dışı sütunlara böler:
  c_out = W_ROI · Enc(ROI) + W_nonROI · X_nonROI
- Makaledeki gibi iki lineer tur uygulanır: M1 ∈ R^(20×n), M2 ∈ R^(10×20).
- Şifreli ve açık katkılar tur boyunca ayrı tutulur (lineerlik sayesinde), istemci sonunda birleştirir.
- Tam HE durumu, ROI maskesinin tüm görüntüyü kapsadığı özel durumdur.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import tenseal as ts

import config


def make_context(poly_modulus: int = config.PIROI_POLY_MODULUS,
                 coeff_bits=tuple(config.PIROI_COEFF_MOD_BITS),
                 scale_bits: int = config.PIROI_SCALE_BITS) -> ts.Context:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus, coeff_mod_bit_sizes=list(coeff_bits))
    ctx.global_scale = 2 ** scale_bits
    ctx.generate_galois_keys()
    return ctx


@dataclass
class Cost:
    enc_s: float = 0.0        # istemci: şifreleme
    server_s: float = 0.0     # sunucu: iki tur hesaplama
    dec_s: float = 0.0        # istemci: çözme ve birleştirme
    upload_bytes: int = 0     # istemciden sunucuya giden ciphertext boyutu
    n_ciphertexts: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def total_s(self) -> float:
        return self.enc_s + self.server_s + self.dec_s


class PiROI:
    def __init__(self, ctx: ts.Context, m1: np.ndarray, m2: np.ndarray,
                 poly_modulus: int = config.PIROI_POLY_MODULUS):
        self.ctx = ctx
        self.m1 = np.asarray(m1, dtype=np.float64)
        self.m2 = np.asarray(m2, dtype=np.float64)
        self.slots = poly_modulus // 2

    def run(self, x: np.ndarray, roi_mask: np.ndarray, measure_bytes: bool = True):
        """x: düzleştirilmiş görüntü (n,), roi_mask: bool (n,). Döner: (çıktı (10,), Cost)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        roi_mask = np.asarray(roi_mask, dtype=bool).ravel()
        idx_roi = np.flatnonzero(roi_mask)
        idx_non = np.flatnonzero(~roi_mask)
        cost = Cost()

        # İstemci: ROI'yi dilimler halinde şifrele (bir ciphertext en fazla `slots` değer taşır).
        # TenSEAL'in toplama işlemi 2'nin kuvveti olmayan uzunluklarda belirgin yavaşladığı için her dilim
        # sıfırla en yakın 2'nin kuvvetine tamamlanır; sıfırlar dot çarpımın sonucunu değiştirmez.
        t0 = time.perf_counter()
        chunks = [idx_roi[i:i + self.slots] for i in range(0, len(idx_roi), self.slots)]
        pads = [min(self.slots, 1 << (len(c) - 1).bit_length()) - len(c) for c in chunks]
        enc = [ts.ckks_vector(self.ctx, np.pad(x[c], (0, p)).tolist()) for c, p in zip(chunks, pads)]
        cost.enc_s = time.perf_counter() - t0
        cost.n_ciphertexts = len(enc)
        cost.extra["sifreli_slot"] = int(sum(len(c) + p for c, p in zip(chunks, pads)))
        if measure_bytes:
            cost.upload_bytes = sum(len(v.serialize()) for v in enc)

        # Sunucu, tur 1: şifreli kısım satır başına dilim dot çarpımlarının toplamı; açık kısım düz çarpım
        t0 = time.perf_counter()
        enc_r1 = []
        if enc:
            for r in range(self.m1.shape[0]):
                acc = None
                for c, p, v in zip(chunks, pads, enc):
                    d = v.dot(np.pad(self.m1[r, c], (0, p)).tolist())
                    acc = d if acc is None else acc + d
                enc_r1.append(acc)
        plain_r1 = self.m1[:, idx_non] @ x[idx_non] if len(idx_non) else np.zeros(self.m1.shape[0])

        # Sunucu, tur 2: M2 ile şifreli ara değerlerin ağırlıklı toplamı (skaler çarpım + toplama)
        enc_r2 = []
        if enc_r1:
            for i in range(self.m2.shape[0]):
                acc = None
                for j in range(self.m2.shape[1]):
                    term = enc_r1[j] * float(self.m2[i, j])
                    acc = term if acc is None else acc + term
                enc_r2.append(acc)
        plain_r2 = self.m2 @ plain_r1
        cost.server_s = time.perf_counter() - t0

        # İstemci: çöz ve açık katkıyla birleştir
        t0 = time.perf_counter()
        out = np.array([ct.decrypt()[0] for ct in enc_r2]) + plain_r2 if enc_r2 else plain_r2
        cost.dec_s = time.perf_counter() - t0
        return out, cost

    def plaintext(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        return self.m2 @ (self.m1 @ x)


def random_weights(n: int, seed: int = 0):
    """Makaledeki boyutlarda rastgele ağırlıklar (M1: 20×n, M2: 10×20)."""
    rng = np.random.default_rng(seed)
    m1 = rng.normal(0, 1 / np.sqrt(n), size=(20, n))
    m2 = rng.normal(0, 1 / np.sqrt(20), size=(10, 20))
    return m1, m2
