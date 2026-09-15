# Feasibility check (toy key, local). High-level TenSEAL API, batched scoring: the hospital packs one
# feature column of 4096 patients per ciphertext (as in efficient SIMD LR scoring), the server returns
# the encrypted weighted column, the hospital decrypts with CKKSVector.decrypt() (all 4096 values)
# and shares the scores. Attacker (server) recovers the secret key from 2 such results.
# Then repeats with scores rounded to 4 decimals before sharing.
import os, time, tempfile
import numpy as np
import tenseal as ts
import tenseal.sealapi as s

N, slots = 8192, 4096
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N, coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.global_scale = 2 ** 40
rng = np.random.default_rng(7)
TMP = tempfile.mkdtemp(prefix="indcpad_")  # geçici .bin dosyaları proje klasörüne yazılmasın

# ---- attacker-side public objects (same public parameters) ----
parms = s.EncryptionParameters(s.SCHEME_TYPE.CKKS); parms.set_poly_modulus_degree(N)
parms.set_coeff_modulus(s.CoeffModulus.Create(N, [60, 40, 40, 60]))
actx = s.SEALContext(parms, True, s.SEC_LEVEL_TYPE.TC128); cod = s.CKKSEncoder(actx)

def arr(d): return [d[i] for i in range(d.size())]

def query_and_share(round_to=None):
    col = rng.normal(size=slots)                     # one feature for 4096 patients
    enc = ts.ckks_vector(ctx, col.tolist())          # client
    out = enc * float(rng.normal())                  # server: w_j * column (TenSEAL rescales)
    out = out + 0.1
    raw = out.ciphertext()[0]; raw.save(os.path.join(TMP, "res.bin"))   # server keeps its own result ciphertext
    ct = s.Ciphertext(actx); ct.load(actx, os.path.join(TMP, "res.bin"))
    z = out.decrypt()                                # client decrypts: 4096 real values
    if round_to is not None: z = [round(v, round_to) for v in z]
    return ct, z

def attack(pairs):
    ct0 = pairs[0][0]; L = ct0.coeff_modulus_size()
    qs = [m.value() for m in actx.get_context_data(ct0.parms_id()).parms().coeff_modulus()]
    v = rng.normal(size=slots) + 1j * rng.normal(size=slots)
    pa, pb = s.Plaintext(), s.Plaintext()
    cod.encode(v.tolist(), ct0.parms_id(), 2.0 ** 40, pa); cod.encode(np.conj(v).tolist(), ct0.parms_id(), 2.0 ** 40, pb)
    A, B = arr(pa.dyn_array()), arr(pb.dyn_array())
    perm = [{A[j * N + i]: i for i in range(N)} for j in range(L)]
    perm = [[perm[j][B[j * N + i]] for i in range(N)] for j in range(L)]
    P, C0, C1 = [], [], []
    for ct, z in pairs:
        pp = s.Plaintext(); cod.encode((2 * np.array(z)).tolist(), ct.parms_id(), ct.scale, pp)
        P.append(arr(pp.dyn_array())); d = arr(ct.dyn_array()); C0.append(d[:L * N]); C1.append(d[L * N:2 * L * N])
    q, j = qs[0], 0                                  # one RNS prime suffices (s is ternary)
    sh = [None] * N
    for i in range(N):
        k = perm[j][i]
        if sh[i] is not None: continue
        (a1, b1, r1), (a2, b2, r2) = [((C1[t][i]) % q, (C1[t][k]) % q, (P[t][i] - C0[t][i] - C0[t][k]) % q) for t in range(2)]
        if i == k: sh[i] = r1 * pow(2 * a1, -1, q) % q; continue
        inv = pow((a1 * b2 - a2 * b1) % q, -1, q)
        sh[i] = (r1 * b2 - r2 * b1) * inv % q; sh[k] = (a1 * r2 - a2 * r1) * inv % q
    return sh, q

# ground truth secret key (for verification only)
ctx.secret_key().data.save(os.path.join(TMP, "sk.bin"))
sk_obj = s.SecretKey(); sk_obj.load(actx, os.path.join(TMP, "sk.bin"))
SK = arr(sk_obj.data().dyn_array())

for rnd in (None, 4):
    t0 = time.perf_counter()
    pairs = [query_and_share(rnd) for _ in range(2)]
    sh, q = attack(pairs)
    ok = all(sh[i] == SK[i] % q for i in range(N))
    print(f"scores shared {'at full precision' if rnd is None else f'rounded to {rnd} decimals'}: key recovered = {ok}  ({time.perf_counter()-t0:.2f}s)")
