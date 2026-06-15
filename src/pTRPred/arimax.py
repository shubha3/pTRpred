"""ARIMAX: Automatic AXIMAX model fitting with optional regressor covariates."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import warnings

from .preprocessing import extract_X
from .rollsvd import roll_svd

def fit_arimax_vec(
    y,
    xreg: Optional[np.ndarray] = None,
    seasonal: bool = True,
    stepwise: bool = True,
    approximation: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fit a single ARIMAX model (auto-selected order) and return aligned residuals/fitted.
    
    The function incorporate the pmdarima.auto_arima to fit an ARIMAX model
    (ARIMA with exogenous regressors). It returns the residuals with the fitted 
    values, along with mask indicating whether the observations are used 

    Parameters
    ----------
    y:             pd.Series or np.array
                   Input for the response.
    xreg:          pd.Series or np.array
                   Input for the regressor columns.
    seasonal:      bool, default = True
                   If True, incorporate the seasonal trend into this.
    stepwise:      bool, default = True
                   If True, conduct the stepwise procedure in the ARIMA model fitting.
    approximation: bool, default = False
                   If True, conduct the approximation procedure in the ARIMAX model fitting.
    """
    try:
        import pmdarima as pm  # type: ignore
    except Exception as e:
        raise ImportError(
            "Package 'pmdarima' is required for ARIMAX functionality. "
            "Install it with: pip install rollsvd-tools[arima]"
        ) from e
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    n = int(y_arr.shape[0])
    if xreg is not None:
        X = np.asarray(xreg, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != n:
            raise ValueError("`xreg` must have the same number of rows as `y`.")
        ok = np.isfinite(y_arr) & np.isfinite(X).all(axis=1)
        X_used = X[ok, :]
    else:
        ok = np.isfinite(y_arr)
        X_used = None
    if not np.any(ok):
        raise ValueError("No complete cases available for ARIMAX fit.")
    y_used = y_arr[ok]
    #Conduct the arima fitting:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(
            y=y_used,
            X=X_used,
            seasonal=seasonal,
            stepwise=stepwise,
            approximation=approximation,
            error_action="warn",
            suppress_warnings=True,
            **kwargs,
        )
    try:
        fitted_used = model.predict_in_sample(X=X_used)
    except TypeError:
        fitted_used = model.predict_in_sample(exogenous=X_used)
    try:
        resid_used = model.resid()
    except Exception:
        resid_used = y_used - np.asarray(fitted_used, dtype=float)
    return {
        "model": model,
        "residuals": np.asarray(resid_used, dtype=float).reshape(-1),
        "fitted": np.asarray(fitted_used, dtype=float).reshape(-1),
        "mask": ok,
    }

"""Batch ARIMAX residuals for multiple series with a common regressor set."""
def arimax_residuals_df(
    data: pd.DataFrame,
    time: str,
    y_cols: Sequence[str] | str,
    xreg_cols: Sequence[str] | str,
    seasonal: bool = True,
    stepwise: bool = True,
    approximation: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Arimax Residuals after extracting the ARIMAX results:
    For each of the column in y_cols, run an ARIMAX model on the
    the regression matrix(xreg_cols). Residuals are aligned in time,
    and only complete cases(no missing in y or xreg) are used.

    Parameters
    ---------
    data:          pd.DataFrame
                   Input data containing time column, response columns and regression columns.
    time:          str
                   Name of the column containing timestamps(used only for indexing,
                   not for modeling order - the order of rows is assumed in terms of increasing time order)
    y_cols:        Sequence[str] or str
                   Column name(s) of the dependent time series.
    xreg_cols:     Sequence[str] or str
                   Column name(s) of exogenous regressors(common across all y).
    seasonal:      bool, default = True
                   Whether to consider seasonality in ARIMAX model
    stepwise:      bool, default = True
                   If True, use stepwise algorithm to select ARIMA orders.
    approximation: bool, default = False
                   If True, use approximation for faster model selection.
    **kwargs:      dict
                   Additional arguments passed to 'fit_arimax_vec'(e.g. `order`,  `seasonal_order`)

    Returns
    ---------
    dict:          A dictionary with keys:
                   - 'residuals_df': pd.DataFrame
                      Columns:  `time`(the timestamps for the complete cases) and each y_col.
                      Each cell contains the ARIMAX residaul(NaN where model failed).
                   - 'models': dict[str, Any]
                      Mapping from each y_col to the fitted model object returned by
                      `fit_arimax_vec`(including)
                   - 'mask': np.ndarray[bool]
                      Boolean mask of shape (len(data), ) indicating which rows had
                      complete cases(no missing in y_cols and xreg_cols).
    
    Raises
    ---------
    ValueError
                   If `time` column is not in `data`, or if there are no complete cases
                   after removing missing values.
    """
    #Validate time column exists
    if time not in data.columns:
        raise ValueError("`time` must be a column in `data`.")
    #Extract response matrix and regressor matrix as DataFrames
    Ymat = extract_X(data, y_cols)
    Xreg = extract_X(data, xreg_cols)
    tvec = data[time].to_numpy()
    #Identify rows with no missing values across all the columns of interest Xreg_cols and y
    ok = np.isfinite(Ymat.to_numpy()).all(axis=1) & np.isfinite(Xreg.to_numpy()).all(axis=1)
    if not np.any(ok):
        raise ValueError("No complete cases across y_cols and xreg_cols for ARIMAX.")
    #Subset to complete cases
    Y_ok = Ymat.loc[ok, :].reset_index(drop=True)
    X_ok = Xreg.loc[ok, :].reset_index(drop=True)
    t_ok = tvec[ok]
    n_ok, p_y = Y_ok.shape
    Resids = np.full((n_ok, p_y), np.nan, dtype=float)
    models: Dict[str, Any] = {}
    #Fit ARIMAX model for each response column individually
    for j, col in enumerate(Y_ok.columns):
        #Call external fitting function(assumed to return a dict with 'mask' and 'residuals')
        fitj = fit_arimax_vec(
            y=Y_ok[col].to_numpy(),
            xreg=X_ok.to_numpy(),
            seasonal=seasonal,
            stepwise=stepwise,
            approximation=approximation,
            **kwargs,
        )
        #Place residuals only at rows where model fitting succeeded(inner_ok)
        inner_ok = fitj["mask"]
        Resids[inner_ok, j] = fitj["residuals"]
        models[col] = fitj["model"]
    #Build output DataFrame with time column and residuals for each series
    residuals_df = pd.DataFrame({"time": t_ok})
    for j, col in enumerate(Y_ok.columns):
        residuals_df[col] = Resids[:, j]
    return {"residuals_df": residuals_df, "models": models, "mask": ok}


def arimax_then_roll_svd(
    data: pd.DataFrame,
    time: str,
    y_cols: Sequence[str] | str,
    xreg_cols: Sequence[str] | str,
    window: int,
    step: int = 1,
    align: str = "end",
    type: str = "rolling",
    center: bool = True,
    scale_: bool = False,
    k: Optional[int] = None,
    fast: bool = True,
    na_action: str = "omit_rows",
    cov_on_pairwise: bool = True,
    seasonal: bool = True,
    stepwise: bool = True,
    approximation: bool = False,
    values_only: bool = True,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Compute ARIMAX residuals for multiple series, then perform rolling SVD
    on the residual matrix over a sliding window.
    This function first extracts residuals from ARIMAX models (one per y_col,
    sharing the same regressors). Then it applies a rolling SVD
    (singular value decomposition) to the residual matrix within each time window,
    enabling dynamic monitoring of the residual covariance structure.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing time column, response columns, and regressor columns.
    time : str
        Name of the timestamp column. By default, it's increasing in time order.
    y_cols : Sequence[str] or str
        Column name(s) of the dependent time series.
    xreg_cols : Sequence[str] or str
        Column name(s) of the regressor (common across all y).
    window : int
        Size of the rolling window (number of time points per window).
    step : int, default=1
        Step size between consecutive windows.
    align : str, default="end"
        Window alignment: 'end' (right-aligned), 'center', or 'start'.
    type : str, default="rolling"
        Type of window: 'rolling', 'expanding', or 'fixed'.
    center : bool, default=True
        Whether to center the data before SVD (subtract column mean).
    scale_ : bool, default=False
        Whether to scale the data to unit variance before SVD.
    k : Optional[int], default=None
        Number of singular components to keep. If None, keep all.
    fast : bool, default=True
        If True, use faster SVD algorithm (e.g., randomized SVD) when appropriate.
    na_action : str, default="omit_rows"
        How to handle missing values: 'omit_rows' (drop rows) or 'interpolate'.
    cov_on_pairwise : bool, default=True
        If True, compute covariance matrix using pairwise comparison matrices.
    seasonal : bool, default = True
        Passed to `fit_arimax_vec`: consider seasonality in ARIMAX selection.
    stepwise : bool, default=True
        Passed to `fit_arimax_vec`: use stepwise order selection.
    approximation : bool, default=False
        Passed to `fit_arimax_vec`: use approximation for faster selection.
    values_only : bool, default=True
        If True, `roll_svd` returns only singular values/vectors; else full result.
    seed : Optional[int], default=None
        Random seed for reproducibility in randomized SVD (if used).
    **kwargs : dict
        Additional arguments passed to `fit_arimax_vec` (e.g., `order`, `seasonal_order`).
    Returns
    -------
    dict
        A dictionary with keys:
        - 'residuals_df' : pd.DataFrame
            DataFrame with columns 'time' and each y_col, containing the ARIMAX residuals
            for complete cases only(aligned in time indices)  
        - 'rollsvd' : Any
            Output from `roll_svd` function (typically a dict or object with SVD results
            per window, depending on `values_only`).
        - 'models' : dict[str, Any]
            Mapping from each y_col to the fitted ARIMAX model object.
        - 'mask' : np.ndarray[bool]
            Boolean mask indicating which rows of `data` were complete and used.
    
    Raises
    ------
    ValueError
        If `time` column is missing from `data`, or if no complete cases exist.
    """

    #Computing the ARIMAX residuals for each of the response-dimensions
    fit = arimax_residuals_df(
        data=data,
        time=time,
        y_cols=y_cols,
        xreg_cols=xreg_cols,
        seasonal=seasonal,
        stepwise=stepwise,
        approximation=approximation,
        **kwargs,
    )
    residuals_df = fit["residuals_df"]
    res_cols = [c for c in residuals_df.columns if c != "time"]
    #SVD procedure adapted on the running windowss
    svd_fit = roll_svd(
        data=residuals_df,
        time="time",
        x_cols=res_cols,
        window=window,
        step=step,
        align=align,  # type: ignore[arg-type]
        type=type,    # type: ignore[arg-type]
        center=center,
        scale_=scale_,
        k=k,
        fast=fast,
        na_action=na_action,  # type: ignore[arg-type]
        cov_on_pairwise=cov_on_pairwise,
        values_only=values_only,
        seed=seed,
    )
    return {"residuals_df": residuals_df, "rollsvd": svd_fit, "models": fit["models"], "mask": fit["mask"]}



