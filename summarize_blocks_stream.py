#!/usr/bin/env python3
"""
summarize_blocks_stream.py
Memory‑safe computation of
  • Bmax_k = max_{t∈[-1/2,1/2]} |B_k(t)|
  • R_k    = max_{t∈[-1/4,1/4]} |B_k(t)| / rho(t)
for k = 0..15 without loading any JSON files.

Streams over the evaluation grid in chunks (default 1e6 points) so peak
RAM stays < 200 MB even for block k=15.

Run:
    python summarize_blocks_stream.py
"""
import numpy as np, json, hashlib, sys

L = 16
CHUNK = 1_000_000  # points per chunk


def Bk(t, Mk):
    t = np.asarray(t, dtype=np.float64)
    s = np.sin(np.pi * t)
    kernel = (np.sin(np.pi * Mk * t) / (Mk * np.where(s == 0, 1.0, s))) ** 2
    return kernel * (np.cos(np.pi * L * t) - (1 - L / Mk) / (2 * Mk))


def rho(t):
    return np.sinc(t) ** 2


def max_stream(Mk, half=False, grid=256):
    a, b = (-0.25, 0.25) if half else (-0.5, 0.5)
    h = 1.0 / (grid * Mk)
    max_val = 0.0
    start = a
    while start < b:
        npts = int(min(CHUNK, (b - start) / h))
        xs = start + h * np.arange(npts, dtype=np.float64)
        max_val = max(max_val, float(np.abs(Bk(xs, Mk)).max()))
        start += npts * h
    return max_val


def max_ratio_stream(Mk):
    a, b = -0.25, 0.25
    h = 1.0 / (512 * Mk)
    max_val = 0.0
    start = a
    while start < b:
        npts = int(min(CHUNK, (b - start) / h))
        xs = start + h * np.arange(npts, dtype=np.float64)
        max_val = max(max_val, float((np.abs(Bk(xs, Mk)) / rho(xs)).max()))
        start += npts * h
    return max_val


def main():
    rows = []
    for k in range(16):
        Mk = 1 << (2 * k + 1)
        print(f"processing k={k} ...", file=sys.stderr)
        Bmax = max_stream(Mk)
        Rk = 0.0 if k < 5 else max_ratio_stream(Mk)
        rows.append(dict(k=k, M=Mk, Bmax=Bmax, Rk=Rk))

    print("k,M,Bmax,Rk")
    for r in rows:
        print(f"{r['k']},{r['M']},{r['Bmax']:.6g},{r['Rk']:.6g}")

    digest = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    with open("summary.json", "w") as f:
        json.dump({"rows": rows, "sha256": digest}, f, indent=2)
    print("\nsummary.json written, sha256 =", digest)


if __name__ == "__main__":
    main()
