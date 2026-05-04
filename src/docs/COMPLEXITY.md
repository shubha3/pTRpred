# pTRPred: Computational Complexity

This document profiles the **time and space complexity** of the main pTRPred algorithms, with emphasis on the **real-time (streaming) update path**.

---

## Notation

| Symbol | Meaning |
|--------|--------|
| **n** | Length of the time series (number of observations) |
| **p** | Number of features (for rolling SVD / Kernel PCA) |
| **w** | Window length (single scale) |
| **W** | Number of window lengths (scales) in offline voting, e.g. `highwl − lowwl + 1`; with `highwl="auto"` often Θ(n) |
| **S** | Number of scales in the **incremental** detector (bounded by `n_scales`, default 25, or by W if `grid="all"`) |
| **nW** | Number of windows (rolling: ≈ ⌊(n−w+1)/step⌋; expanding: ≈ ⌊(n−w)/step⌋+1) |
| **H** | `hist_len` in incremental mode (default 200) |
| **k** | Number of components in rolling SVD (e.g. `k=1` for first eigenvalue) |

---

## 1. Offline adaptive slope-vote detector: `detect_asvotes`

**Location:** `detection.detect_asvotes`, `offline.detect_asvotes`

**Algorithm:** For each window length `w` in `[lowwl, highwl]`, partition the signal into non-overlapping blocks of length `w`, compute least-squares slope per block, robust z-score (median/MAD), and accumulate votes.

| Cost | Complexity |
|------|------------|
| **Time** | **O(W · n)** |
| **Space** | O(n) |

**Breakdown:**

- Loop over **W** window lengths.
- For each `w`:
  - Partition and reshape: **O(n)**.
  - Slopes: `t_centered @ y_block` over Θ(n/w) blocks of size w → **O(n)**.
  - Robust z (median, MAD) on Θ(n/w) slopes → **O(n/w)**.
  - Vote accumulation over blocks: worst-case **O(n)**.
- With `highwl="auto"` (e.g. `highwl = max(5, n//3)`), **W = Θ(n)**, so total time is **O(n²)**.

---

## 2. Batch “real-time” simulation: `detect_realtime`

**Location:** `detection.detect_realtime`

**Algorithm:** For each time index `i` from `burn_in` to `n`, run `detect_asvotes` on the prefix `y[:i+1]`, then optionally smooth and threshold.

| Cost | Complexity |
|------|------------|
| **Time** | **O(n³)** |
| **Space** | O(n) |

**Breakdown:**

- One full `detect_asvotes` for offline: **O(W·n) = O(n²)**.
- For each `i ∈ [burn_in, n]`, `detect_asvotes(y[:i+1])` with scale count Θ(i): **O(i²)** per step.
- Sum over i: **Σ O(i²) = O(n³)**.
- Smoothing over n points with window `smooth_k`: **O(n · smooth_k)** (dominated by n³).

So this path is **cubic in n** and intended for short series or benchmarking; for long streams use the incremental path below.

---

## 3. Real-time (streaming) update: incremental path

### 3.1 Single-scale slope state: `_SlidingSlopeState.update`

**Location:** `incremental._SlidingSlopeState`

**Algorithm:** Maintain running sums **S0 = Σ y_i** and **S1 = Σ i·y_i** over the last `w` points via a deque; slope = (S1 − (w+1)/2·S0) / denom with denom = w(w²−1)/12.

| Cost | Complexity |
|------|------------|
| **Time per update** | **O(1)** |
| **Space** | O(w) |

**Breakdown:**

- Deque append and drop oldest: **O(1)**.
- Update S0, S1 with one add/subtract: **O(1)**.
- Slope and denominator are closed-form: **O(1)**.

So each new sample updates the slope for that scale in **constant time**.

### 3.2 Vote from slope history: `_SlidingSlopeState.vote`

**Location:** `incremental._SlidingSlopeState.vote`

**Algorithm:** Append slope to a fixed-length history, then compute robust z (median, MAD) on that history and compare to threshold.

| Cost | Complexity |
|------|------------|
| **Time per call** | **O(H)** |
| **Space** | O(H) |

**Breakdown:**

- `slopes_hist` has length ≤ **H** (`hist_len`, default 200).
- Two medians (median and MAD) on an array of size H: **O(H)**.

So voting is **O(1) in stream length n**, but **O(H) in history length** per scale.

### 3.3 Multi-scale online detector: `OnlineASVotesSliding.update`

**Location:** `incremental.OnlineASVotesSliding`

**Algorithm:** For each scale in the current grid, run `_SlidingSlopeState.update(y_new)` and `vote(...)`; average the votes.

| Cost | Complexity |
|------|------------|
| **Time per update** | **O(S · H)** |
| **Space** | **O(S · (w_max + H))** |

**Breakdown:**

- **S** scales (bounded by `n_scales` with `grid="log"`, or by W with `grid="all"`).
- Per scale: **O(1)** slope update + **O(H)** vote → **O(S · (1 + H)) = O(S·H)**.
- With default **S = 25**, **H = 200**: **O(5000)** per update, i.e. **O(1) in n**.

So the **per-sample cost is constant in stream length n** and linear in the number of scales and history length.

### 3.4 Full real-time detector: `OnlineRealtimeDetector.update`

**Location:** `incremental.OnlineRealtimeDetector`

**Algorithm:** Call `OnlineASVotesSliding.update(y_new)`, then optional trailing mean over last `smooth_k` votes, then threshold check.

| Cost | Complexity |
|------|------------|
| **Time per update** | **O(S · H + smooth_k)** |
| **Space** | O(S·(w_max + H) + smooth_k) |

**Breakdown:**

- Sliding detector: **O(S·H)**.
- Smoothing: deque of size `smooth_k` + mean: **O(smooth_k)**.

Again **O(1) in n** per sample; total over n samples is **O(n)** for the online path.

### 3.5 Batch wrapper: `detect_realtime_incremental`

**Location:** `incremental.detect_realtime_incremental`

**Algorithm:** Optionally run offline `detect_asvotes` once; then simulate n steps, each calling `OnlineRealtimeDetector.update`.

| Cost | Complexity |
|------|------------|
| **Time** | **O(n²)** if `compute_offline=True` (offline dominates), **O(n)** if `compute_offline=False` |
| **Space** | O(n + S·(w_max + H)) |

**Breakdown:**

- If `compute_offline=True`: one `detect_asvotes` **O(n²)**.
- n updates × **O(S·H)** = **O(n)** (with S, H fixed).
- So total **O(n² + n) = O(n²)** when offline is used, and **O(n)** when only the incremental path is used.

---

## 4. Rolling-window building: `roll_windows`

**Location:** `windows.roll_windows`

**Algorithm:** Compute start/end indices and representative time for each rolling or expanding window.

| Cost | Complexity |
|------|------------|
| **Time** | **O(n + nW)** |
| **Space** | O(nW · w) for storing index lists (each window has w indices) |

**Breakdown:**

- Build starts/ends: **O(nW)** with nW = Θ(n/step) or Θ(n).
- Build `idx` list: **O(nW · w)** for rolling (each window w indices); for expanding the last windows are large.

---

## 5. Rolling SVD: `roll_svd`

**Location:** `rollsvd.roll_svd`

**Algorithm:** For each window, extract rows, (optionally) center/scale, compute covariance, then eigendecomposition (full or partial via SciPy `eigsh`).

| Cost | Complexity |
|------|------------|
| **Time** | **O(nW · (w·p² + p³))** (full eigh) or **O(nW · (w·p² + k·p²))** (eigsh with k components) |
| **Space** | O(nW · p) for D (and V/U if not values_only) |

**Breakdown:**

- **nW** windows.
- Per window:
  - Extract and preprocess: **O(w·p)**.
  - Covariance: **O(w·p²)**.
  - Full eigh: **O(p³)**; partial eigsh: **O(k·p²)**.
- So total **O(nW · (w·p² + p³))** or **O(nW · (w·p² + k·p²))**.
- With nW ≈ n/step and fixed w, p: **O(n)** in n.

---

## 6. Rolling Kernel PCA: `roll_kernel_pca`

**Location:** `kernelpca.roll_kernel_pca`

**Algorithm:** For each window, (optionally) center/scale, then fit `KernelPCA` (kernel matrix and eigendecomposition).

| Cost | Complexity |
|------|------------|
| **Time** | **O(nW · (w²·p + w³))** |
| **Space** | O(nW · w) for scores and eigenvalues |

**Breakdown:**

- **nW** windows.
- Per window:
  - StandardScaler: **O(w·p)**.
  - Kernel matrix (e.g. RBF): **O(w²·p)**.
  - Eigenvalue computation on w×w matrix: **O(w³)**.
- Total **O(nW · (w²·p + w³))**.
- With nW ≈ n/step: **O(n · (w²·p + w³))** in n.

---

## 7. Signal builders: `build_signal_svd`, `build_signal_kernel_pca`

**Location:** `signals.build_signal_svd`, `signals.build_signal_kernel_pca`

**Algorithm:** Call the corresponding rolling routine with `n_components=1` (or k=1), then form a DataFrame from the first eigenvalue per window.

| Cost | Complexity |
|------|------------|
| **Time** | Same as underlying `roll_svd` / `roll_kernel_pca` plus **O(nW)** to build the DataFrame. |
| **Space** | O(nW) for the output DataFrame. |

---

## 8. Summary table

| Component | Time (per call or per update) | Space | Notes |
|-----------|-------------------------------|--------|--------|
| **detect_asvotes** | O(W·n), often O(n²) | O(n) | Offline; W = Θ(n) with highwl="auto" |
| **detect_realtime** (batch) | O(n³) | O(n) | Simulates RT by repeated prefix calls; avoid for long n |
| **_SlidingSlopeState.update** | **O(1)** | O(w) | Single-scale slope update |
| **_SlidingSlopeState.vote** | O(H) | O(H) | Robust z on history of length H |
| **OnlineASVotesSliding.update** | **O(S·H)** = O(1) in n | O(S·(w_max+H)) | **Real-time update: constant in n** |
| **OnlineRealtimeDetector.update** | **O(S·H + smooth_k)** = O(1) in n | O(S·(w_max+H)+smooth_k) | **Real-time update: constant in n** |
| **detect_realtime_incremental** | O(n²) or O(n) | O(n) | O(n²) if compute_offline=True; O(n) if False |
| **roll_windows** | O(n + nW) | O(nW·w) | nW ≈ n/step |
| **roll_svd** | O(nW·(w·p² + p³)) | O(nW·p) | nW ≈ n/step |
| **roll_kernel_pca** | O(nW·(w²·p + w³)) | O(nW·w) | nW ≈ n/step |

---

## 9. Practical recommendations

1. **Real-time streaming:** Use **`OnlineRealtimeDetector`** (or **`detect_realtime_incremental`** with `compute_offline=False`). Per-sample cost is **O(S·H)**, independent of how much data has been seen; total cost over n samples is **O(n)**.
2. **Avoid long-series batch RT simulation:** **`detect_realtime`** (without incremental) is **O(n³)**; use it only for short n or validation.
3. **Offline voting:** **`detect_asvotes`** is **O(n²)** with default scale range; acceptable for moderate n.
4. **Rolling SVD/KPCA:** Cost is linear in the number of windows **nW**; use larger **step** to reduce nW and runtime when high time resolution is not needed.
