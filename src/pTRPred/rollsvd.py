from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .preprocessing import extract_X
from .windows import roll_windows


def _cov_window(
    Xc: pd.DataFrame,
    na_action: Literal["omit_rows", "impute_mean", "pairwise_complete"],
    cov_on_pairwise: bool,
) -> np.ndarray:
    """
    Compute covariance for a single window
    """
    if na_action == "pairwise_complete" and cov_on_pairwise:
        S = Xc.cov(min_periods=2)  # pairwise complete
        return S.to_numpy()
    Xc2 = Xc.dropna(axis=0, how="any")
    if Xc2.shape[0] < 2:
        return np.full((Xc.shape[1], Xc.shape[1]), np.nan, dtype=float)
    return np.cov(Xc2.to_numpy(), rowvar=False, ddof=1)


def roll_svd(
    data: pd.DataFrame,
    time: str,
    x_cols: Sequence[str] | str,
    window: int,
    step: int = 1,
    align: Literal["end", "center", "start"] = "end",
    type: Literal["rolling", "expanding"] = "rolling",
    center: bool = True,
    scale_: bool = False,
    k: Optional[int] = None,
    fast: bool = True,
    na_action: Literal["omit_rows", "impute_mean", "pairwise_complete"] = "omit_rows",
    cov_on_pairwise: bool = True,
    values_only: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Rolling SVD (eigendecomposition) of per-window covariance.

    Notes
    -----
    - If `fast=True` and SciPy is available, uses `scipy.sparse.linalg.eigsh` for
      partial eigendecomposition (when kk < p).
    - Windows with fewer than 2 complete rows produce `None` entries in output lists.
    """
    if time not in data.columns:
        raise ValueError("`time` must be a column in `data`.")
    if seed is not None:
        np.random.seed(int(seed))
    windows = roll_windows(data[time].to_numpy(), window, step, align, type)
    X_all = extract_X(data, x_cols)
    colnames = list(X_all.columns)
    p = int(X_all.shape[1])
    nW = int(windows.shape[0])
    D_list: List[Optional[np.ndarray]] = [None] * nW
    V_list: Optional[List[Optional[np.ndarray]]] = None if values_only else [None] * nW
    U_list: Optional[List[Optional[np.ndarray]]] = None if values_only else [None] * nW
    have_scipy = False
    if fast:
        try:
            from scipy.sparse.linalg import eigsh  # noqa: F401
            have_scipy = True
        except Exception:
            have_scipy = False
    for w in range(nW):
        idx_1b = windows.iloc[w]["idx"]  # list of 1-based indices
        idx_0b = [int(i) - 1 for i in idx_1b]
        Xw = X_all.iloc[idx_0b, :].copy()
        # NA handling (per window)
        if na_action == "omit_rows":
            Xw = Xw.dropna(axis=0, how="any")
        elif na_action == "impute_mean":
            for c in Xw.columns:
                arr = Xw[c].to_numpy(dtype=float)
                nan_mask = np.isnan(arr)
                if nan_mask.any():
                    n_non_na = int((~nan_mask).sum())
                    if n_non_na > 1:
                        Xw.loc[nan_mask, c] = float(np.nanmean(arr))
        elif na_action == "pairwise_complete":
            pass
        else:
            raise ValueError("`na_action` must be one of {'omit_rows','impute_mean','pairwise_complete'}")
        n_w = int(Xw.shape[0])
        if n_w < 2:
            continue
        # Center/scale per window
        if center:
            mu = Xw.mean(axis=0, skipna=True)
        else:
            mu = pd.Series(np.zeros(p), index=Xw.columns)
        if scale_:
            sc = Xw.std(axis=0, ddof=1, skipna=True).replace(0, np.nan)
        else:
            sc = pd.Series(np.ones(p), index=Xw.columns)
        sc = sc.fillna(1.0)
        Xc = (Xw - mu) / sc
        kk = min(int(k) if k is not None else min(n_w, p), n_w, p)
        S = _cov_window(Xc, na_action, cov_on_pairwise)
        if not np.isfinite(S).all():
            continue
        if values_only:
            vals = np.linalg.eigvalsh(S)[::-1]
            D_list[w] = vals[:kk].copy()
            continue
        # Full results
        if have_scipy and kk < p:
            from scipy.sparse.linalg import eigsh  # type: ignore
            vals, vecs = eigsh(S, k=kk, which="LA")
            order = np.argsort(-vals)
            vals = vals[order]
            vecs = vecs[:, order]
        else:
            vals_full, vecs_full = np.linalg.eigh(S)
            order = np.argsort(-vals_full)
            vals = vals_full[order][:kk]
            vecs = vecs_full[:, order][:, :kk]
        D_list[w] = np.asarray(vals, dtype=float).copy()
        if V_list is not None:
            V_list[w] = np.asarray(vecs, dtype=float).copy()
        if U_list is not None:
            U_list[w] = Xc.to_numpy(dtype=float) @ np.asarray(vecs, dtype=float)
    return {
        "windows": windows,
        "D": D_list,
        "V": V_list,
        "U_scores": U_list,
        "k": (None if k is None else int(k)),
        "colnames": colnames,
        "preproc": {"center": bool(center), "scale.": bool(scale_)},
        "values_only": bool(values_only),
    }


def as_tibble_rollsvd(
    x: Dict[str, Any],
    what: Literal["loadings", "singular_values"] = "singular_values",
) -> Optional[pd.DataFrame]:
    """Tidy a rollsvd output dict into long format."""
    if not isinstance(x, dict) or "D" not in x:
        raise ValueError("`x` must be a dict returned by roll_svd (with key 'D').")

    V_list: Optional[List[Optional[Any]]] = x.get("V", None)
    D_list: List[Optional[Any]] = x.get("D", [])
    colnames: List[str] = x.get("colnames", [])

    W = len(V_list) if V_list is not None else len(D_list)
    if W == 0:
        return None

    out_frames: List[pd.DataFrame] = []

    if what == "loadings":
        if V_list is None:
            raise ValueError("Loadings not available: roll_svd was run with values_only=True.")
        for w in range(W):
            V = V_list[w]
            if V is None:
                continue
            p, kk = V.shape
            dfw = pd.DataFrame(
                {
                    "window_id": (w + 1),
                    "variable": colnames * kk,
                    "factor": sum(([i + 1] * p for i in range(kk)), []),
                    "loading": np.asarray(V).reshape(-1, order="F"),
                }
            )
            out_frames.append(dfw)

    elif what == "singular_values":
        for w in range(W):
            D = D_list[w]
            if D is None:
                continue
            dfw = pd.DataFrame(
                {"window_id": (w + 1), "factor": list(range(1, len(D) + 1)), "d": list(map(float, D))}
            )
            out_frames.append(dfw)

    else:
        raise ValueError("`what` must be 'loadings' or 'singular_values'.")

    if not out_frames:
        return None
    return pd.concat(out_frames, axis=0, ignore_index=True)
