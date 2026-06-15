
"""Online incremental slope‑voting detector with sliding windows.

Implements `OnlineASVotesSliding` (core voting) and `OnlineRealtimeDetector`
(burn‑in, smoothing, thresholding). Suitable for streaming data where each
point is processed once.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .offline import detect_asvotes


@dataclass
class _SlidingSlopeState:
    """
    Maintain slope for the last w points with O(1) update using (S0, S1):
      S0 = sum_{i=1..w} y_i
      S1 = sum_{i=1..w} i * y_i
    slope numerator = S1 - ((w+1)/2)*S0
    denom = sum_{i=1..w} (i-(w+1)/2)^2 = w*(w^2-1)/12
    Used for the real-time online detector.
    """
    w: int
    hist_len: int = 200
    min_hist: int = 30
    buf: Deque[float] = None
    slopes_hist: Deque[float] = None
    S0: float = 0.0
    S1: float = 0.0
    denom: float = 0.0
    def __post_init__(self) -> None:
        if self.w < 2:
            raise ValueError("w must be >= 2")
        self.buf = deque(maxlen=self.w)
        self.slopes_hist = deque(maxlen=self.hist_len)
        self.denom = self.w * (self.w**2 - 1.0) / 12.0
        if self.denom <= 0:
            self.denom = 1e-12
    def reset(self) -> None:
        self.buf.clear()
        self.slopes_hist.clear()
        self.S0 = 0.0
        self.S1 = 0.0
    def update(self, y_new: float) -> float:
        """
        Push one new point and return slope for last w points.
        If not enough points yet, returns np.nan.
        """
        y_new = float(y_new)
        if not np.isfinite(y_new):
            # simplest robust streaming behavior: reset scale on NA
            self.reset()
            return np.nan

        m = len(self.buf)
        if m < self.w:
            self.buf.append(y_new)
            self.S0 += y_new
            self.S1 += (m + 1) * y_new
            if len(self.buf) < self.w:
                return np.nan
        else:
            y_old = self.buf[0]
            S0_old = self.S0

            self.buf.append(y_new)  # deque drops oldest
            self.S0 = S0_old - y_old + y_new
            # S1_new = (S1_old - S0_old) + w*y_new
            self.S1 = (self.S1 - S0_old) + self.w * y_new

        num = self.S1 - ((self.w + 1.0) / 2.0) * self.S0
        return float(num / self.denom)

    @staticmethod
    def _robust_z(x: float, hist: np.ndarray) -> float:
        med = np.nanmedian(hist)
        mad = np.nanmedian(np.abs(hist - med))
        if not np.isfinite(mad) or mad <= 0:
            mad = 1e-8
        return (x - med) / mad

    def vote(
        self,
        slope: float,
        mad_k: float,
        direction: Literal["positive", "both", "negative"],
    ) -> int:
        if not np.isfinite(slope):
            return 0

        self.slopes_hist.append(slope)
        if len(self.slopes_hist) < self.min_hist:
            return 0

        hist = np.asarray(self.slopes_hist, dtype=float)
        z = self._robust_z(slope, hist)

        if direction == "positive":
            return 1 if z > mad_k else 0
        if direction == "negative":
            return 1 if z < -mad_k else 0

        # both
        if z > mad_k:
            return 1
        if z < -mad_k:
            return -1
        return 0


def _make_w_grid(
    lowwl: int,
    highwl: int,
    grid: Literal["all", "log"] = "log",
    n_scales: int = 25,
) -> List[int]:
    lowwl = int(max(2, lowwl))
    highwl = int(max(lowlwl, highwl))

    if grid == "all":
        return list(range(lowlwl, highwl + 1))

    if n_scales < 2:
        return [lowlwl] if lowwl == highwl else [lowlwl, highwl]

    xs = np.exp(np.linspace(np.log(lowlwl), np.log(highwl), n_scales))
    wls = sorted(set(int(round(v)) for v in xs))
    wls = [w for w in wls if lowwl <= w <= highwl]
    if lowwl not in wls:
        wls = [lowlwl] + wls
    if highwl not in wls:
        wls = wls + [highwl]
    return sorted(set(wls))


class OnlineASVotesSliding:
    """
    Online multi-scale slope voting using causal sliding windows.

    Per update:
      - for each w: slope(last w points) in O(1)
      - robust z-score vs trailing slope history
      - vote (+1/-1/0)
      - average votes across scales
    """
    def __init__(
        self,
        lowwl: int = 5,
        highwl: Union[int, str] = "auto",
        mad_k: float = 3.0,
        direction: Literal["positive", "both", "negative"] = "positive",
        hist_len: int = 200,
        min_hist: int = 30,
        highwl_cap: int = 200,
        w_grid: Literal["all", "log"] = "log",
        n_scales: int = 25,
    ) -> None:
        self.lowlw = int(lowwl)
        self.highwl = highwl
        self.mad_k = float(mad_k)
        self.direction = direction
        self.hist_len = int(hist_len)
        self.min_hist = int(min_hist)
        self.highwl_cap = int(max(self.lowlw, highwl_cap))
        self.w_grid = w_grid
        self.n_scales = int(n_scales)
        self.n_seen = 0
        self.states: Dict[int, _SlidingSlopeState] = {}
    #Estimate the current high value of the window
    def _current_highwl(self) -> int:
        if isinstance(self.highwl, str) and self.highwl.lower() == "auto":
            hw = max(5, self.n_seen // 3)
        else:
            hw = int(self.highwl)
        return int(min(hw, self.highwl_cap))
    def _ensure_states(self) -> List[int]:
        hw = self._current_highwl()
        wls = _make_w_grid(self.lowlw, hw, grid=self.w_grid, n_scales=self.n_scales)
        for w in wls:
            if w not in self.states:
                self.states[w] = _SlidingSlopeState(w=w, hist_len=self.hist_len, min_hist=self.min_hist)
        return wls
    #Real-time incremental update
    def update(self, y_new: float) -> float:
        self.n_seen += 1
        wls = self._ensure_states()
        votes = []
        for w in wls:
            st = self.states[w]
            slope = st.update(y_new)
            v = st.vote(slope=slope, mad_k=self.mad_k, direction=self.direction)
            votes.append(v)
        return float(np.mean(votes)) if votes else 0.0


class OnlineRealtimeDetector:
    """
    Streaming wrapper adding burn-in + smoothing + threshold crossing.
    Core Function: OnlineASVotesSliding 

    Parameters
    ---------
    lowwl:    int, default = 5,
              Minimal value of the window size
    highwl:   Union[int, str], default = 'auto'
              Maximal value of the window size, 'auto' means len(df)//5
    mad_k:    float, default = 3.0
              The abnormal proportion for the threshold(MAD)
    direction:Literal['positive', 'both', 'negative'], default = 'positive'
              Detection direction, only positive, both direction or only negative
    burn_in  :The starting period, default = None
              If None, set as max(100, 5 * lowwl)
    smooth_k :int, default = 30
              The aggregation window-size, if smooth_k < 0, won't do smoothing - core parameter
    threshold:float, default = 0.3 
              The threshold to trigger the alert - core parameter.
    highwl_cap: int, default = 200
               The largest possible window-size passing to `OnlineASVotesSliding` function
    w_grid:   Literal['all', 'log'], default = 'log'
              The type of grid, 'all' means using all integer windows, 'log' using the log interval
    n_scales: int, default = 25
              When w_grid = 'log', the log scale for the window-size.
    hist_len: int, default = 200
              The historical length the inner detector maintained.
    min_hist: int, default = 30
              The minimal sample-size the inner detector required.
    
    Return 
    ----------
    burn_in:              In the initial stage, won't pose triggers, to ensure the robustness in the initial stage of the algorithm.
    smoothing:            The types for averaging the votes in the running window, reduces the oscillation.
    first-detection-time: The first time indice that the real-time detection score exceeds the pre-set threshold.
    """
    def __init__(
        self,
        lowwl: int = 5,
        highwl: Union[int, str] = "auto",
        mad_k: float = 3.0,
        direction: Literal["positive", "both", "negative"] = "positive",
        burn_in: Optional[int] = None,
        smooth_k: int = 30,
        threshold: float = 1.3,
        highwl_cap: int = 200,
        w_grid: Literal["all", "log"] = "log",
        n_scales: int = 25,
        hist_len: int = 200,
        min_hist: int = 30,
    ) -> None:
        #Initialize the real-time detector
        self.det = OnlineASVotesSliding(
            lowwl=lowwl,
            highwl=highwl,
            mad_k=mad_k,
            direction=direction,
            hist_len=hist_len,
            min_hist=min_hist,
            highwl_cap=highwl_cap,
            w_grid=w_grid,
            n_scales=n_scales,
        )
        #Save the parameters
        self.lowlw = int(lowlw := lowwl)
        self.burn_in = int(burn_in) if burn_in is not None else max(100, 5 * self.lowlw)
        self.smooth_k = int(max(0, smooth_k))
        self.threshold = float(threshold)
        self._smooth_buf = deque(maxlen=self.smooth_k if self.smooth_k > 0 else 1)
        self._first_detection_time: Any = np.nan
        self._i = 0
    def update(self, t_new: Any, y_new: float) -> Dict[str, Any]:
        '''
        Update on a new observation:

        Parameters:
        ----------
        t_new: Any
               The time-indice for the new observation(can be date/int or other hashable types)
        y_new: float
               The signal value for the current sample(for detection)

        Returns:
        ----------
        A dictionary including the following keys:
            - time_ind:             Any,   the time point for the input.
            - signal:               Float, the signal value for the input.
            - detected_value_rt:    Float, the abnormal vote value after smoothing.
            - flag:                 Bool,  whether it reach the trigger condition.
            - first_detection_time: Any,   the time point for the first triggered alert(if not 
                                           triggered, record as NaN)
        '''
        self._i += 1
        vote_raw = self.det.update(y_new)
        #Smoothing via the moving average
        if self.smooth_k > 0:
            self._smooth_buf.append(vote_raw)
            vote_sm = float(np.mean(self._smooth_buf))
        else:
            vote_sm = float(vote_raw)
        #Checking whether satisfies the trigger condition
        eligible = (self._i >= self.burn_in)
        flag = bool(eligible and (vote_sm >= self.threshold))
        if flag and not np.isfinite(self._first_detection_time):
            self._first_detection_time = t_new
        return {
            "time_ind": t_new,
            "signal": float(y_new) if np.isfinite(y_new) else np.nan,
            "detected_value_rt": vote_sm,
            "flag": flag,
            "first_detection_time": self._first_detection_time,
        }


def detect_realtime_incremental(
    time: Sequence,
    signal: Union[Sequence[float], np.ndarray],
    lowwl: int = 5,
    highwl: Union[str, int] = "auto",
    mad_k: float = 3.0,
    direction: Literal["positive", "both", "negative"] = "positive",
    burn_in: Optional[int] = None,
    smooth_k: int = 30,
    threshold: float = 1.3,
    highwl_cap: int = 200,
    w_grid: Literal["all", "log"] = "log",
    n_scales: int = 25,
    hist_len: int = 200,
    min_hist: int = 30,
    compute_offline: bool = True,
) -> pd.DataFrame:
    """
    Streaming detection on a batch of data.

    Returns a DataFrame with columns:
        time_ind, signal, detected_offline, detected_value_rt, flag,
        first_detection_time.

    Parameters
    ----------
    time : sequence
        Time stamps (must match length of `signal`).
    signal : sequence of float
        Input signal.
    compute_offline : bool, default=True
        If True, adds an `detected_offline` column using `detect_asvotes`.
    lowwl: lower bound for the window size
    highwl: higher bound for the window size
    Returns
    -------
    pd.DataFrame
        One row per time point, with detection results.

    Examples
    --------
    >>> t = np.arange(100)
    >>> y = np.random.randn(100)
    >>> df = detect_realtime_incremental(t, y)
    >>> df[df['flag']].head()
    """
    t = np.asarray(time)
    y = np.asarray(signal, dtype=float).reshape(-1)
    if t.shape[0] != y.shape[0]:
        raise ValueError("`time` and `signal` lengths must match.")
    #Offline computation for the detection score
    if compute_offline:
        offline = detect_asvotes(
            signal=y,
            lowwl=lowwl,
            highwl=highwl,
            mad_k=mad_k,
            direction=direction,
        )
    else:
        offline = np.zeros_like(y, dtype=float)
    #Online Real-Tiem detector
    rt = OnlineRealtimeDetector(
        lowwl=lowwl,
        highwl=highwl,
        mad_k=mad_k,
        direction=direction,
        burn_in=burn_in,
        smooth_k=smooth_k,
        threshold=threshold,
        highwl_cap=highwl_cap,
        w_grid=w_grid,
        n_scales=n_scales,
        hist_len=hist_len,
        min_hist=min_hist,
    )
    rows = [rt.update(t_new=t[i], y_new=y[i]) for i in range(y.size)]
    out = pd.DataFrame(rows)
    out.insert(2, "detected_offline", offline)
    return out
