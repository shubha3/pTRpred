from __future__ import annotations

from typing import Literal, Union
import numpy as np


def detect_asvotes(
    signal,
    lowwl: int = 5,
    highwl: Union[str, int] = "auto",
    mad_k: float = 3.0,
    direction: Literal["positive", "both", "negative"] = "positive",
):
    """
    Adaptive slope-vote detector (block-based, offline).
    Leverage the Slope based method to calculate the anomaly score:
    Return the votes as the anomaly score(proportion of the window-sizes that
    which slopes are out of the 3(mad_k)*MAD from the median)

    - votes: A Fload range between 0 and 1, as the proportion

    Parameters:
    signal:    pd.Series or array-like
               Input time series signal.
    lowwl:     int, default = 5
               Minimal Window Size(number of points per block). 
    highwl:    Union[str, int], default = 'auto'
               Maximal value of the Window Size, if 'auto', it's set as max(5, len(signal)//3).
    mad_k:     float, default = 3.0
               Number of median absolute deviations(MAD) above/below the median slope 
               to consider a block as anomaly
    direction: {'positive', 'both', 'negative'} default = 'positive'
               Which slopes count as anomalies:
               - 'positive': only slopes > threshold
               - 'negative': only slopes < threshold
               - 'both'    : slopes in either direction(absolute value)
    Returns
    -------
    votes:     np.array
               Array of floats in [0,1] representing the real-time anomaly score for each sample.
               Higher values indicate stronger evidence of an anomalous slope pattern
               across the tested window sizes.
    """
    y = np.asarray(signal, dtype=float)
    n = y.size
    if n < 3:
        return np.zeros(n, dtype=float)
    if direction not in ("positive", "both", "negative"):
        raise ValueError("direction must be 'positive', 'both', or 'negative'")
    #Determine the range of the window size:
    if highwl == "auto":
        highwl_val = max(5, n // 3)
    else:
        highwl_val = int(highwl)
    lowwl_val = int(lowwl)
    if lowwl_val < 2:
        lowwl_val = 2
    if highwl_val < lowwl_val:
        highwl_val = lowwl_val
    wls = np.arange(lowwl_val, highwl_val + 1, dtype=int)
    votes = np.zeros(n, dtype=float)
    n_wls = wls.size
    def robust_z(x):
        """Compute robust z-score using median and MAD."""
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if not np.isfinite(mad) or mad == 0:
            mad = 1e-8
        return (x - med) / mad
    # Iterate over all window sizes
    for w in wls:
        # Determine the central contiguous part that can be divided into complete blocks.
        rem = n % w
        if rem == 0:
            start = 0
            end = n
        else:
            #Trim equally from both ends to keep only full blocks.
            pad = rem // 2
            start = pad
            end = n - (rem - pad)
        if end - start < w:
            continue
        idx = np.arange(start, end, dtype=int)
        B = idx.size // w #Number of the complete blocks
        if B == 0:
            continue
        y_block = y[idx][: w * B].reshape((w, B), order="F")
        #Calculate the slope for each of the period via the linear regression fitting.
        t = np.arange(1, w + 1, dtype=float)
        t_centered = t - (w + 1) / 2.0
        denom = np.sum(t_centered * t_centered)
        has_na = np.any(~np.isfinite(y_block), axis=0)
        num = t_centered @ y_block
        slopes = num / denom
        slopes[has_na] = np.nan
        #Robust normalization of the slopes
        z = robust_z(slopes)
        if np.all(~np.isfinite(z)):
            continue
        vote_sign = np.zeros(B, dtype=int)
        if direction == "positive":
            vote_sign[z > mad_k] = +1
        elif direction == "negative":
            vote_sign[z < -mad_k] = +1
        else:
            vote_sign[z > mad_k] = +1
            vote_sign[z < -mad_k] = -1
        # Accumulate the votes: each sample in an anomalous block
        nz_blocks = np.nonzero(vote_sign != 0)[0]
        if nz_blocks.size:
            for b in nz_blocks:
                block_slice = slice(b * w, (b + 1) * w)
                votes[idx[block_slice]] += vote_sign[b]
    #Normalize votes by number of window sizes to get proportions of the anomaly as the real-time detection score.
    if n_wls > 0:
        votes = votes / n_wls
    return votes
