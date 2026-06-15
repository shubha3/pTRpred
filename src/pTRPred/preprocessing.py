from __future__ import annotations

from typing import Literal, Sequence, Union
import numpy as np
import pandas as pd


def extract_X(data: pd.DataFrame, x_cols: Union[Sequence[str], str]) -> pd.DataFrame:
    """Extract a numeric matrix (as a pandas DataFrame) from a DataFrame.

    Parameters
    ----------
    data:
        Input DataFrame.
    x_cols:
        - A list/tuple of exact column names, e.g. ["temp1","temp2"]
        - A single string: either one exact column name or a regex containing '|'
          to match multiple names, e.g. "temp|volt"
    Returns
    -------
    pd.DataFrame
        Numeric DataFrame with preserved column names. Coerces non-numeric to NaN.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")
    if isinstance(x_cols, str):
        if "|" in x_cols:
            keep = data.filter(regex=x_cols).columns.tolist()
        else:
            keep = [x_cols]
    elif isinstance(x_cols, (list, tuple)):
        keep = list(x_cols)
    else:
        raise ValueError("`x_cols` must be a list of names or a single regex string.")
    if len(keep) == 0:
        raise ValueError("No columns selected by `x_cols`.")
    X = data.loc[:, keep].copy()
    for c in keep:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def na_handle(
    M: Union[np.ndarray, pd.DataFrame],
    method: Literal["omit_rows", "impute_mean", "pairwise_complete"] = "omit_rows",
) -> np.ndarray:
    """Simple NA handling for numeric matrices.

    Parameters
    ----------
    M:
        Numpy array or pandas DataFrame.
    method:
        - "omit_rows": drop any rows containing NaN
        - "impute_mean": impute NaNs with column means (requires >=2 non-NA values)
        - "pairwise_complete": leave NaNs as-is (useful when downstream supports it)

    Returns
    -------
    np.ndarray
    """
    if isinstance(M, pd.DataFrame):
        A = M.to_numpy(dtype=float)
    else:
        A = np.asarray(M, dtype=float)
    if method == "omit_rows":
        mask = ~np.isnan(A).any(axis=1)
        return A[mask, :]
    if method == "impute_mean":
        A_copy = A.copy()
        for j in range(A_copy.shape[1]):
            col = A_copy[:, j]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue
            n_non_na = int(np.sum(~nan_mask))
            if n_non_na <= 1:
                continue
            m = float(np.nanmean(col))
            A_copy[nan_mask, j] = m
        return A_copy
    if method == "pairwise_complete":
        return A
    raise ValueError("`method` must be one of {'omit_rows','impute_mean','pairwise_complete'}")
