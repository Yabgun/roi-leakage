"""Pilot: TenSEAL CKKS ile tam vs secici LR cikarim maliyeti (N=8192, [60,40,40,60], 2^40).
(a) Sutun paketleme (toplu, 1022 hasta), (b) satir paketleme (tek hasta)."""
import time, numpy as np, tenseal as ts

def ctx(galois=False):
    c = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    c.global_scale = 2 ** 40
    if galois:
        c.generate_galois_keys()
    return c

rng = np.random.default_rng(0)
n, d = 1022, 20          # test kumesi, one-hot sonrasi oznitelik sayisi
X = rng.normal(size=(n, d)); w = rng.normal(size=d); b = -0.3
REP = 5

def column_packed(k):
    """k sutun sifreli, d-k sutun acik; tum hastalar tek seferde."""
    c = ctx()
    t_enc = t_srv = t_dec = 0.0; size = 0
    for _ in range(REP):
        t0 = time.perf_counter()
        cts = [ts.ckks_vector(c, X[:, j].tolist()) for j in range(k)]
        t1 = time.perf_counter()
        acc = cts[0] * float(w[0]) if k > 0 else None
        for j in range(1, k):
            acc += cts[j] * float(w[j])
        plain_part = (X[:, k:] @ w[k:] + b).tolist()
        acc = acc + plain_part if acc is not None else None
        t2 = time.perf_counter()
        z = np.array(acc.decrypt()) if acc is not None else np.array(plain_part)
        t3 = time.perf_counter()
        t_enc += t1 - t0; t_srv += t2 - t1; t_dec += t3 - t2
        size = sum(len(ct.serialize()) for ct in cts)
    err = np.max(np.abs(z - (X @ w + b)))
    return t_enc / REP * 1e3, t_srv / REP * 1e3, t_dec / REP * 1e3, size / 1e6, err

def row_packed(k):
    """Tek hasta: k oznitelik tek ciphertext'te, dot icin rotasyon (galois) gerekir."""
    c = ctx(galois=True)
    x = X[0]
    ts_ = []
    for _ in range(REP * 4):
        t0 = time.perf_counter()
        ct = ts.ckks_vector(c, x[:k].tolist())
        z = ct.dot(w[:k].tolist()) + float(x[k:] @ w[k:] + b)
        zz = z.decrypt()[0]
        ts_.append(time.perf_counter() - t0)
    return np.median(ts_) * 1e3, len(ct.serialize()) / 1e3, abs(zz - (x @ w + b))

print("(a) SUTUN PAKETLEME, %d hasta, d=%d" % (n, d))
print("%4s %10s %10s %10s %10s %12s" % ("k", "sifr(ms)", "sunucu(ms)", "coz(ms)", "trafik(MB)", "maks hata"))
for k in [4, 7, 10, 14, 20]:
    e, s, dd, mb, err = column_packed(k)
    print("%4d %10.1f %10.1f %10.1f %10.2f %12.2e" % (k, e, s, dd, mb, err))

print("\n(b) SATIR PAKETLEME, tek hasta (sifrele+dot+coz toplam, medyan)")
for k in [4, 7, 20]:
    ms, kb, err = row_packed(k)
    print("k=%2d  toplam=%6.1f ms  ciphertext=%6.0f KB  hata=%.1e" % (k, ms, kb, err))

# anahtar uretimi (tek seferlik)
t0 = time.perf_counter(); ctx(galois=True); print("\nContext + galois anahtar uretimi: %.0f ms" % ((time.perf_counter() - t0) * 1e3))
