"""FoveaHE şifreli çıkarım (TenSEAL CKKS): odaklı temsil vektörünün tamamı şifrelenir, sunucu D veya D2 modelini
çalıştırır, istemci çıktıyı çözer.

Parametreler ve işlem düzeni Π_ROI yeniden üretimiyle aynıdır (`he/piroi.py`: N = 16384, [60, 40, 40, 40, 60],
ölçek 2^40; dilim başına 2'nin kuvvetine sıfır dolgusu; gizli birim başına `dot`, sonra skaler çarpım + toplama).
Farklar: açık kısım yoktur ve yanlılıklar sunucuda şifreli sonuca açık metin olarak eklenir. Çarpma derinliği D'de 2,
D2'de 3 seviyedir (üç ara asal yeterli). Çözülen sonuç sunucuya geri gönderilmez (IND-CPA-D anahtar kurtarma kuralı).
"""
from __future__ import annotations

import time

import numpy as np
import tenseal as ts

import config
from he.piroi import Cost


def public_context_bytes(ctx: ts.Context) -> int:
    """Sunucuya bir kez gönderilen genel bağlam: açık, yeniden doğrusallaştırma ve Galois anahtarları (gizli anahtar yok)."""
    return len(ctx.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=True,
                             save_relin_keys=True))


class FoveaHEInference:
    def __init__(self, ctx: ts.Context, weights: dict, poly_modulus: int = config.PIROI_POLY_MODULUS):
        self.ctx = ctx
        self.kind = str(weights["kind"])
        self.w1 = np.asarray(weights["W1"], dtype=np.float64)
        self.b1 = np.asarray(weights["b1"], dtype=np.float64)
        self.w2 = np.asarray(weights["W2"], dtype=np.float64)
        self.b2 = np.asarray(weights["b2"], dtype=np.float64)
        self.slots = poly_modulus // 2
        n = self.w1.shape[1]
        self.chunks = [np.arange(i, min(n, i + self.slots)) for i in range(0, n, self.slots)]
        self.pads = [min(self.slots, 1 << (len(c) - 1).bit_length()) - len(c) for c in self.chunks]
        # Sunucunun gizli birim başına kullandığı dolgulu ağırlık dilimleri (şifreli işlemden önce, bir kez)
        self.w1_rows = [[np.pad(self.w1[r, c], (0, p)).tolist() for c, p in zip(self.chunks, self.pads)]
                        for r in range(self.w1.shape[0])]

    def encrypt(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).ravel()
        return [ts.ckks_vector(self.ctx, np.pad(x[c], (0, p)).tolist()) for c, p in zip(self.chunks, self.pads)]

    def server(self, enc):
        hidden = []
        for r, rows in enumerate(self.w1_rows):
            acc = None
            for v, w in zip(enc, rows):
                d = v.dot(w)
                acc = d if acc is None else acc + d
            acc = acc + float(self.b1[r])
            hidden.append(acc.square() if self.kind == "D2" else acc)
        outs = []
        for i in range(self.w2.shape[0]):
            acc = None
            for j, h in enumerate(hidden):
                term = h * float(self.w2[i, j])
                acc = term if acc is None else acc + term
            outs.append(acc + float(self.b2[i]))
        return outs

    @staticmethod
    def decrypt(outs) -> np.ndarray:
        return np.array([ct.decrypt()[0] for ct in outs])

    def run(self, x: np.ndarray, measure_bytes: bool = True):
        """Döner: (logitler, Cost). Cost.extra: şifreli slot sayısı, indirme baytı."""
        cost = Cost()
        t0 = time.perf_counter()
        enc = self.encrypt(x)
        cost.enc_s = time.perf_counter() - t0
        cost.n_ciphertexts = len(enc)
        cost.extra["sifreli_slot"] = int(sum(len(c) + p for c, p in zip(self.chunks, self.pads)))
        if measure_bytes:
            cost.upload_bytes = sum(len(v.serialize()) for v in enc)
        t0 = time.perf_counter()
        outs = self.server(enc)
        cost.server_s = time.perf_counter() - t0
        if measure_bytes:
            cost.extra["indirme_bayt"] = sum(len(o.serialize()) for o in outs)
        t0 = time.perf_counter()
        logits = self.decrypt(outs)
        cost.dec_s = time.perf_counter() - t0
        return logits, cost

    def plaintext(self, x: np.ndarray) -> np.ndarray:
        h = self.w1 @ np.asarray(x, dtype=np.float64).ravel() + self.b1
        if self.kind == "D2":
            h = h * h
        return self.w2 @ h + self.b2
