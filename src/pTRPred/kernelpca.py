"""Rolling Kernel PCA: first eigenvalue per window (analogous to roll_svd s1)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .preprocessing import extract_X
from .windows import roll_windows

def roll_kernel_pca(
    data: pd.DataFrame,
    time: str,
    x_cols: Sequence[str] | str,
    window: int,
    step: int = 1,
    align: Literal["end", "center", "start"] = "end",
    type: Literal["rolling", "expanding"] = "rolling",
    center: bool = True,
    scale_: bool = True,
    n_components: int = 1,
    kernel: Literal["rbf", "poly", "linear", "cosine"] = "rbf",
    gamma: Optional[float] = None,
    coef0: float = 1.0,
    degree: int = 3,
    na_action: Literal["omit_rows", "impute_mean", "pairwise_complete"] = "omit_rows",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Rolling Kernel PCA: first component eigenvalue per window (analogous to roll_svd s1).

    Uses sklearn.decomposition.KernelPCA per window to extract the eigenvalues and eigenvcetors.
    Return a dictionary including:
      - windows : DataFrame from roll_windows (one row per window)
      - D       : List of 1D arrays (eigenvalues of centered kernel matrix, first component)
      - scores  : List of (n_w, n_components) arrays (projected data) or None
      - colnames: List of feature names
      - preproc : {'center': bool, 'scale.': bool}
      - kernel  : kernel name - can be 'rbf', 'poly', 'linear' or 'cosine'

    Parameters
    ----------
    kernel : 'rbf' (default), 'poly', 'linear', 'cosine'
    gamma  : kernel coefficient for rbf/poly; if None, use 1/n_features

    """
    try:
        from sklearn.decomposition import KernelPCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "roll_kernel_pca requires scikit-learn. Install with: pip install scikit-learn"
        )
    if time not in data.columns:
        raise ValueError("`time` must be a column in `data`.")
    if seed is not None:
        np.random.seed(seed)
    #extract the list of the running windows.
    windows = roll_windows(data[time].to_numpy(), window, step, align, type)
    X_all = extract_X(data, x_cols)
    colnames = list(X_all.columns)
    p = X_all.shape[1]
    if gamma is None:
        gamma = 1.0 / max(p, 1)
    nW = windows.shape[0]
    D_list: List[Optional[np.ndarray]] = [None] * nW
    scores_list: List[Optional[np.ndarray]] = [None] * nW
    for w in range(nW):
        idx_1b = windows.iloc[w]["idx"]
        idx_0b = [i - 1 for i in idx_1b]
        Xw = X_all.iloc[idx_0b, :].copy()
        if na_action == "omit_rows":
            Xw = Xw.dropna(axis=0, how="any")
        elif na_action == "impute_mean":
            Xw = Xw.copy()
            for c in Xw.columns:
                arr = Xw[c].to_numpy()
                nan_mask = np.isnan(arr)
                if nan_mask.any():
                    n_non_na = (~nan_mask).sum()
                    if n_non_na > 1:
                        Xw.loc[nan_mask, c] = np.nanmean(arr)
        elif na_action != "pairwise_complete":
            raise ValueError(
                "`na_action` must be one of {'omit_rows','impute_mean','pairwise_complete'}"
            )
        n_w = Xw.shape[0]
        if n_w < 2:
            continue
        #Do the standard scaler:
        X_mat = Xw.to_numpy(dtype=float)
        if center or scale_:
            scaler = StandardScaler(with_mean=center, with_std=scale_)
            X_mat = scaler.fit_transform(X_mat)
        #Conduct the Kernel PCA procedure for the running window:
        try:
            kpca = KernelPCA(
                n_components=n_components,
                kernel=kernel,
                gamma=gamma,
                coef0=coef0,
                degree=degree,
                fit_inverse_transform=False,
            )
            X_proj = kpca.fit_transform(X_mat)
            lam = getattr(kpca, "eigenvalues_", None)
            if lam is not None and len(lam) >= 1:
                D_list[w] = np.asarray(lam[:n_components], dtype=float)
            else:
                D_list[w] = np.array(
                    [
                        np.nanvar(X_proj[:, 0])
                        if X_proj.shape[1] >= 1
                        else np.nan
                    ]
                )
            scores_list[w] = X_proj
        except Exception:
            continue
    return {
        "windows": windows,
        "D": D_list,
        "scores": scores_list,
        "colnames": colnames,
        "preproc": {"center": center, "scale.": scale_},
        "kernel": kernel,
    }
