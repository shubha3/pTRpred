# pTRPred: Python Version of TRPred
## File structure

```text
pTRPred/
│
├── README.md
├── requirements.txt
├── tests/
│   └── __pycache__
|   └── test_kernelpca.py
|   └── test_rollsvd.py
|   └── test_windows.py
|   └── test_preprocessing.py
|   └── test_plotting.py
|   └── test_detection.py
|   └── test_arimax.py
|   └── conftest.py
│
├── Reproducible_Plots/
│   ├── NREL_Data_Visualizations
│   ├── NREL_Plots
│   └── NREL_Plots_SVD
│   └── NREL_Dataset
│   └── Simulated_Datasets
│   └── Simulated_Datasets_Plots
│   └── SVD_result
│   └── T_results
│   └── Reproduce_Plots.ipynb
│
|
├── src/
│   ├── pTRPred/
│       └── signals.py
│       └── offline.py
│       └── arimax.py
│       └── detection.py
│       └── plotting.py
│       └── preprocessing.py
│       └── rollsvd.py
│       └── windows.py
│       └── incremental.py
│       └── signals.py
│      
├── datasets/
│   ├── Simulated
│   |   └── data_generating_process
│   |   └── generated_data   
│   ├── NREL   
│       └── Plots
│       └── Battery_cells
│       └── raw_data
│       └── SVD_result
│       └── mean_load_prop_analysis
│       └── processed_data
│       └── Sensitivity_Analysis
│       └── readme.md
│   
│   
├── pyproject.toml
├── uv.lock

```

## Install 

``` bash
pip install pTRPred
```

## Load Package 
``` bash
import pTRPred
```

## What's inside:

-   Rolling/expanding window index construction (`roll_windows`)
-   Rolling SVD (eigendecomposition) of per-window covariance (`roll_svd`)
-   ARIMAX fitting and batch residual extraction (`fit_arimax_vec`, `arimax_residuals_df`)
-   Pipelines that combine ARIMAX residualization + rolling SVD (`arimax_then_roll_svd`)
-   Signal builders and multi-scale slope-vote real-time change detectors (`detect_asvotes`, `detect_realtime`)
-   Simple plotting utilities (`plot_detection_overlay`)

## Documentation

-   **[Computational complexity](docs/COMPLEXITY.md)** — Time/space complexity of all main routines, with emphasis on the **real-time (streaming) update** path (`OnlineRealtimeDetector`, `OnlineASVotesSliding`).

## Run tests

1. Raw Signal -> Real-Time Detector

``` bash
# Reproducibility (R's set.seed(1))
np.random.seed(1)

n = 1200
df = pd.DataFrame({
    "time": np.arange(1, n + 1),
    "temp": np.cumsum(np.random.normal(loc=0.0, scale=0.02, size=n)),
})

sig_raw = build_signal_raw(df, time="time", col="temp")

det_raw = detect_realtime(
    time=sig_raw["time_ind"].to_numpy(),
    signal=sig_raw["signal"].to_numpy(),
    lowwl=5,
    highwl="auto",
    mad_k=3,
    direction="positive",
    burn_in=200,
    smooth_k=30,
    threshold=0.25,
)

# Plot (left axis = detector, right axis = original signal)
fig, ax1, ax2 = plot_detection_overlay(
    det_raw,
    title="Raw temp vs. real-time detector",
    score_label="Detected value (votes)",
    signal_label="Temperature",
    x_label="Index",
    threshold=0.25,
)
```

![](docs/figures/pTRPred_unittest1.png)

2. ARIMAX residuals -> detector(single dependent + predictors)
```         
np.random.seed(2)

n = 1200
df = pd.DataFrame({
    "time": np.arange(1, n + 1),
    "temp": np.cumsum(np.random.normal(loc=0.0, scale=0.03, size=n)),
    "voltage": np.random.normal(size=n),
    "force": np.random.normal(size=n),
})

# NOTE:
# This requires pmdarima installed:
#   pip install "pTRPred[arima]"
sig_res = build_signal_arimax_resid(
    data=df,
    time="time",
    y_col="temp",
    xreg_cols="voltage|force",
    seasonal=False,
)

det_res = detect_realtime(
    time=sig_res["time_ind"].to_numpy(),
    signal=sig_res["signal"].to_numpy(),
    burn_in=200,
    smooth_k=30,
    threshold=0.25,
)

fig, ax1, ax2 = plot_detection_overlay(
    det_res,
    title="ARIMAX residuals vs. detector",
    score_label="Detected value (votes)",
    signal_label="ARIMAX residuals",
    x_label="Index",
    threshold=0.25,
)
```

![](docs/figures/pTRPred_unittest2.png)

3. Rolling SVD -> Detector
```         
np.random.seed(3)

n = 2000
df = pd.DataFrame({
    "time": np.arange(1, n + 1),
    "temperature": np.cumsum(np.random.normal(loc=0.0, scale=0.02, size=n)),
    "load": np.random.normal(loc=-200.0, scale=5.0, size=n),
    "voltage": 4.0 + np.random.normal(loc=0.0, scale=0.01, size=n),
    "force": np.random.normal(loc=-900.0, scale=6.0, size=n),
})

sig_s1 = build_signal_svd(
    data=df,
    time="time",
    x_cols="temperature|load|voltage|force",
    window=125,
    step=1,
    align="end",
    type="rolling",
    center=True,
    scale_=True,                 # R's scale. = TRUE
    na_action="pairwise_complete"
)

det_s1 = detect_realtime(
    time=sig_s1["time_ind"].to_numpy(),
    signal=sig_s1["signal"].to_numpy(),
    burn_in=200,
    smooth_k=30,
    threshold=1.3
)

fig, ax1, ax2 = plot_detection_overlay(
    det_s1,
    title="SVD s1 vs. detector",
    score_label="Detected value (votes)",
    signal_label="Top eigenvalue (s1)",
    x_label="Index",
    threshold=1.3
)
```
![](docs/figures/pTRPred_unittest3.png)

### The reproducible plot script.
https://colab.research.google.com/drive/1EXFeAl4LKNc5g1uMUyUgaCJPGTfzSDm2#scrollTo=ZFTesc-XlR7T

### project-name/
│
├── README.md
├── requirements.txt
├── config/
│   └── default_config.yaml
│
├── datasets/
│   ├── NREL/
│   ├── processed/
│   └── README.md
│   └── README.md  



