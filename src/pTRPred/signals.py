"""Signal construction utilities (raw, SVD, kernel PCA, ARIMAX residuals)."""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .arimax import arimax_residuals_df, fit_arimax_vec
from .kernelpca import roll_kernel_pca
from .preprocessing import extract_X
from .rollsvd import roll_svd


def build_signal_raw(data: pd.DataFrame, time: str, col: str) -> pd.DataFrame:
    """Raw single-column series → (time_ind, signal)."""
    if time not in data.columns:
        raise ValueError("`time` must be a column in `data`.")
    if col not in data.columns:
        raise ValueError("`col` must be a column in `data`.")
    return pd.DataFrame(
        {
            "time_ind": data[time].to_numpy(),
            "signal": pd.to_numeric(data[col], errors="coerce").to_numpy(),
        }
    )


def build_signal_svd(
    data: pd.DataFrame,
    time: str,
    x_cols: Sequence[str] | str,
    window: int,
    step: int = 1,
    align: Literal["end", "center", "start"] = "end",
    type: Literal["rolling", "expanding"] = "rolling",
    center: bool = True,
    scale_: bool = True,
    fast: bool = True,
    na_action: Literal["omit_rows", "impute_mean", "pairwise_complete"] = "omit_rows",
    cov_on_pairwise: bool = True,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build the signal of the SVD rolling window procedure -> (t_rep, first_singular_value)
    """
    fit = roll_svd(
        data=data,
        time=time,
        x_cols=x_cols,
        window=window,
        step=step,
        align=align,
        type=type,
        center=center,
        scale_=scale_,
        k=1,
        fast=fast,
        na_action=na_action,
        cov_on_pairwise=cov_on_pairwise,
        values_only=True,
        seed=seed,
    )
    s1 = [(np.nan if (d is None or len(d) < 1) else float(d[0])) for d in fit["D"]]
    return pd.DataFrame({"time_ind": fit["windows"]["t_rep"].to_numpy(), "signal": np.asarray(s1, dtype=float)})


def build_signal_kernel_pca(
    data: pd.DataFrame,
    time: str,
    x_cols: Sequence[str] | str,
    window: int,
    step: int = 1,
    align: Literal["end", "center", "start"] = "end",
    type: Literal["rolling", "expanding"] = "rolling",
    center: bool = True,
    scale_: bool = True,
    kernel: Literal["rbf", "poly", "linear", "cosine"] = "rbf",
    gamma: Optional[float] = None,
    coef0: float = 1.0,
    degree: int = 3,
    na_action: Literal["omit_rows", "impute_mean", "pairwise_complete"] = "omit_rows",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rolling Kernel PCA first eigenvalue per window → (t_rep, signal).
    Same interface as build_signal_svd; signal = first component eigenvalue of kernel matrix.
    """
    fit = roll_kernel_pca(
        data=data,
        time=time,
        x_cols=x_cols,
        window=window,
        step=step,
        align=align,
        type=type,
        center=center,
        scale_=scale_,
        n_components=1,
        kernel=kernel,
        gamma=gamma,
        coef0=coef0,
        degree=degree,
        na_action=na_action,
        seed=seed,
    )
    s1 = [(np.nan if (d is None or len(d) < 1) else float(d[0])) for d in fit["D"]]
    return pd.DataFrame(
        {
            "time_ind": fit["windows"]["t_rep"].to_numpy(),
            "signal": np.asarray(s1, dtype=float),
        }
    )


def build_signal_arimax_resid(
    data: pd.DataFrame,
    time: str,
    y_col: Sequence[str] | str,
    xreg_cols: Sequence[str] | str,
    seasonal: bool = True,
    stepwise: bool = True,
    approximation: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """ARIMAX residuals (single y + predictors) → (time_ind, residual).
    So that to run the arimax procedure
    """
    if time not in data.columns:
        raise ValueError("`time` must be a column in `data`.")
    y_df = extract_X(data, y_col)
    if y_df.shape[1] != 1:
        raise ValueError("`y_col` must select exactly one column.")
    X = extract_X(data, xreg_cols)

    t = data[time].to_numpy()
    fit = fit_arimax_vec(
        y=y_df.iloc[:, 0].to_numpy(),
        xreg=X.to_numpy(),
        seasonal=seasonal,
        stepwise=stepwise,
        approximation=approximation,
        **kwargs,
    )
    return pd.DataFrame({"time_ind": t[fit["mask"]], "signal": fit["residuals"]})


def build_signal_arimax_svd(
    data: pd.DataFrame,
    time: str,
    y_cols: Sequence[str] | str,
    xreg_cols: Sequence[str] | str,
    window: int,
    step: int = 1,
    align: Literal["end", "center", "start"] = "end",
    type: Literal["rolling", "expanding"] = "rolling",
    center: bool = True,
    scale_: bool = True,
    fast: bool = True,
    na_action: Literal["omit_rows", "impute_mean", "pairwise_complete"] = "omit_rows",
    cov_on_pairwise: bool = True,
    seasonal: bool = True,
    stepwise: bool = True,
    approximation: bool = False,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    ARIMAX residuals for multiple series → rolling SVD s1 → (t_rep, s1).
    so that to run the SVD procedure on the running windows
    """
    arx = arimax_residuals_df(
        data=data,
        time=time,
        y_cols=y_cols,
        xreg_cols=xreg_cols,
        seasonal=seasonal,
        stepwise=stepwise,
        approximation=approximation,
        **kwargs,
    )
    residuals_df = arx["residuals_df"]
    res_cols = [c for c in residuals_df.columns if c != "time"]
    #SVD procedure on the rolling window
    fit = roll_svd(
        data=residuals_df,
        time="time",
        x_cols=res_cols,
        window=window,
        step=step,
        align=align,
        type=type,
        center=center,
        scale_=scale_,
        k=1,
        fast=fast,
        na_action=na_action,
        cov_on_pairwise=cov_on_pairwise,
        values_only=True,
        seed=seed,
    )
    s1 = [(np.nan if (d is None or len(d) < 1) else float(d[0])) for d in fit["D"]]
    return pd.DataFrame({"time_ind": fit["windows"]["t_rep"].to_numpy(), "signal": np.asarray(s1, dtype=float)})
