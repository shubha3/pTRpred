"""Online incremental slope‑voting detector with sliding windows."""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd


def detect_asvotes(
    signal: Union[Sequence[float], np.ndarray],
    lowwl: int = 5,
    highwl: Union[str, int] = "auto",
    mad_k: float = 3.0,
    direction: Literal["positive", "both", "negative"] = "positive",
) -> np.ndarray:
    """Adaptive slope-vote detector.

    Parameters
    ----------
    signal:
        Input sequence.
    lowwl:
        Minimum block/window length.
    highwl:
        Maximum window length, or "auto" to use max(5, floor(n/3)).
    mad_k:
        Threshold in raw MAD units.
    direction:
        Which slope directions to count.

    Returns
    -------
    np.ndarray
        Vote scores in [-1, 1] (after normalization).
    """
    y = np.asarray(signal, dtype=float)
    n = int(y.size)
    if n < 3:
        return np.zeros(n, dtype=float)

    if direction not in ("positive", "both", "negative"):
        raise ValueError("direction must be 'positive', 'both', or 'negative'")

    if isinstance(highwl, str) and highwl.lower() == "auto":
        highwl_val = max(5, n // 3)
    else:
        highwl_val = int(highwl)
    lowwl_val = max(2, int(lowwl))
    if highwl_val < lowwl_val:
        highwl_val = lowwl_val

    wls = np.arange(lowwl_val, highwl_val + 1, dtype=int)
    votes = np.zeros(n, dtype=float)
    n_wls = int(wls.size)

    def robust_z(x: np.ndarray) -> np.ndarray:
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if not np.isfinite(mad) or mad == 0:
            mad = 1e-8
        return (x - med) / mad

    for w in wls:
        rem = n % w
        if rem == 0:
            start, end = 0, n
        else:
            pad = rem // 2
            start = pad
            end = n - (rem - pad)

        if end - start < w:
            continue

        idx = np.arange(start, end, dtype=int)
        B = int(idx.size // w)
        if B == 0:
            continue

        y_block = y[idx][: w * B].reshape((w, B), order="F")

        t = np.arange(1, w + 1, dtype=float)
        t_centered = t - (w + 1) / 2.0
        denom = float(np.sum(t_centered * t_centered))

        has_na = np.any(~np.isfinite(y_block), axis=0)
        num = t_centered @ y_block
        slopes = num / denom
        slopes[has_na] = np.nan
        #Robust normalization
        z = robust_z(slopes)
        if np.all(~np.isfinite(z)):
            continue
        vote_sign = np.zeros(B, dtype=int)
        if direction == "positive":
            vote_sign[z > mad_k] = +1
        elif direction == "negative":
            vote_sign[z < -mad_k] = +1
        else:  # both
            vote_sign[z > mad_k] = +1
            vote_sign[z < -mad_k] = -1
        nz_blocks = np.nonzero(vote_sign != 0)[0]
        for b in nz_blocks:
            block_slice = slice(b * w, (b + 1) * w)
            votes[idx[block_slice]] += vote_sign[b]
    if n_wls > 0:
        votes = votes / n_wls
    return votes


def detect_realtime(
    time: Sequence,
    signal: Union[Sequence[float], np.ndarray],
    lowwl: int = 5,
    highwl: Union[str, int] = "auto",
    mad_k: float = 3.0,
    direction: Literal["positive", "both", "negative"] = "positive",
    burn_in: Optional[int] = None,
    smooth_k: int = 30,
    threshold: float = 1.3,
) -> pd.DataFrame:
    """Real-time multi-scale slope voting wrapper.
    Computes:
    - Offline votes on full series
    - Real-time vote score on each prefix, optionally smoothed with trailing mean
    Returns columns:
      time_ind, signal, detected_offline, detected_value_rt, flag, first_detection_time
    """
    t = np.asarray(time)
    y = np.asarray(signal, dtype=float).reshape(-1)
    n = int(y.size)
    if t.shape[0] != n:
        raise ValueError("`time` and `signal` lengths must match.")
    if n < 3:
        return pd.DataFrame(
            {
                "time_ind": t,
                "signal": y,
                "detected_offline": np.zeros(n, dtype=float),
                "detected_value_rt": np.zeros(n, dtype=float),
                "flag": np.zeros(n, dtype=bool),
                "first_detection_time": np.array([np.nan] * n, dtype=object),
            }
        )
    lowwl_val = int(max(2, lowwl))
    if burn_in is None:
        burn_in = max(100, 5 * lowwl_val)
    burn_in = int(max(2, min(burn_in, n)))
    #Offline detected score, option.
    offline = detect_asvotes(signal=y, lowwl=lowwl_val, highwl=highwl, mad_k=mad_k, direction=direction)
    rt = np.full(n, np.nan, dtype=float)
    def _clip_highwl(cur_n: int) -> Union[str, int]:
        '''Clip the higher value of the window'''
        if isinstance(highwl, str) and highwl.lower() == "auto":
            return "auto"
        return min(int(highwl), cur_n)
    #Calculate the offline detection score in the burn-in period
    det_burn = detect_asvotes(
        signal=y[:burn_in],
        lowwl=lowwl_val,
        highwl=_clip_highwl(burn_in),
        mad_k=mad_k,
        direction=direction,
    )
    if smooth_k > 0:
        k0 = min(int(smooth_k), int(det_burn.size))
        for i in range(burn_in):
            j1 = max(0, i - k0 + 1)
            rt[i] = float(np.nanmean(det_burn[j1 : i + 1]))
    else:
        rt[:burn_in] = det_burn[:burn_in]
    #After the burn-in period, calculate the real-time detection score for each period
    for i in range(burn_in, n):
        det_i = detect_asvotes(
            signal=y[: i + 1],
            lowwl=lowwl_val,
            highwl=_clip_highwl(i + 1),
            mad_k=mad_k,
            direction=direction,
        )
        if smooth_k > 0:
            k = min(int(smooth_k), int(det_i.size))
            j1 = max(0, i - k + 1)
            rt[i] = float(np.nanmean(det_i[j1 : i + 1]))
        else:
            rt[i] = float(det_i[i])
    flag_vec = rt >= float(threshold)
    if np.any(flag_vec):
        first_time = t[int(np.argmax(flag_vec))]
    else:
        first_time = np.nan
    return pd.DataFrame(
        {
            "time_ind": t,
            "signal": y,
            "detected_offline": offline,
            "detected_value_rt": rt,
            "flag": flag_vec,
            "first_detection_time": np.repeat(first_time, n).astype(object),
        }
    )


def write_rt_csv(df: pd.DataFrame, path: str) -> str:
    """Write real-time detection output to CSV and return the path."""
    df.to_csv(path, index=False)
    return path
