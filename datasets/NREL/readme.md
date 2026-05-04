# NREL Dataset Structure and Plots

This document describes the layout of the **NREL** (National Renewable Energy Laboratory) thermal-runaway detection dataset under `datasets/NREL/`, and how each type of plot in `Plots/` is produced.

---

## 1. Directory Overview

```
datasets/NREL/
├── Battery_cells/          # Core data: raw, processed, SVD, and T (temperature) detection results
├── Plots/                  # All generated figures (see Section 3)
├── Sensitivity_Analysis/   # Threshold/window sensitivity and accuracy summaries
└── readme2.md              # This file
```

---

## 2. Battery_cells — Data Pipeline

### 2.1 `raw_data/`

- **Format:** Excel (`.xlsx`) per battery test.
- **Naming:** `{dataset_name}.xlsx`, e.g. `1500mAh1-40S0C.xlsx`, `OE-LFP10Ah-80SOC-cell3.xlsx`.
- **Content:** Time-series from thermal runaway tests. Typical columns (from R preprocessing) include:
  - Time (two time columns, e.g. columns 1 and 20)
  - Load, voltage (e.g. columns 2, 3)
  - Temperature (e.g. column 21)
- **Role:** Source data for preprocessing and detection.

### 2.2 `processed_data/`

- **Format:** CSV.
- **Naming:** `{dataset_name}.xlsx_reform_severity_level{N}.csv`  
  Example: `1500mAh1-40S0C.xlsx_reform_severity_level2.csv`
- **Content:** Reformatted and interpolated time-series on a regular time grid (e.g. 0.1 s), with columns such as:
  - `time_point`, `temperature`, `load`, `voltage` (and sometimes `force`).
- **Severity level:** Each file is tied to a severity level (1–7) that maps the same physical test to different labeling/analysis groups (see `data_preprocessing_NREL.R` and any dataset correspondence table used elsewhere).

### 2.3 `SVD_result/`

- **Format:** CSV.
- **Naming:** `{dataset_name}_30_100_SVD_detects.csv`  
  Example: `1500mAh1-100S0C_30_100_SVD_detects.csv`
- **Content:** Detection results using an SVD-based (rolling SVD) procedure. Typical columns:
  - `time_ind`: time index
  - `signal`: first singular value (σ₁)
  - `detected_offline`, `detected_value_rt`: detection scores
  - `flag`, `first_detection_time` (optional)
- **Role:** Input for **plot_svd_result** (SVD overlay and SVD detection-only plots).

### 2.4 `T_result/`

- **Format:** CSV.
- **Naming:** `{dataset_name}.xlsx_reform_severity_level{N}.csv_rt_detect_result_window100.csv`  
  Example: `1500mAh1-20S0C.xlsx_reform_severity_level2.csv_rt_detect_result_window100.csv`
- **Content:** Real-time detection results using **temperature** as the signal (window size 100). Typical columns:
  - `temperature`, `detected_value_rt`, `detected_offline`  
  (A time index is usually derived as row index or from the corresponding processed file.)
- **Role:** Input for **plot_T_result** (temperature overlay and detection-only plots) and for **plot_multi_cells** and **plot_across_severity_level** when those use T-based detection.

---

## 3. Plots — Structure and How to Reproduce

All figures live under `Plots/`. The following subfolders and script define what each plot type is and how to regenerate it.

### 3.1 `Plots/plot_T_result/`

- **Description:** One figure per temperature-based detection run: overlay (temperature + detection on dual axes) and detection-only (single axis: detection + threshold 0.3).
- **Naming:**
  - `{base_id}_overlay.png`
  - `{base_id}_detection.png`  
  where `base_id` is the dataset identifier with hyphens replaced by underscores (e.g. `1500mAh1-20S0C` → `1500mAh1_20S0C`).
- **Data source:** `Battery_cells/T_result/*_rt_detect_result_window100.csv`.
- **Reproduction:** For each T_result CSV, add a time index column, then:
  - **Overlay:** plot temperature (e.g. left axis) and detection (e.g. right axis); add threshold 0.3 and optional vertical line at first exceedance.
  - **Detection-only:** plot `detected_value_rt` vs time index; add horizontal line at 0.3 and vertical line at first time ≥ 0.3.
- **X-axis:** Use label **"time indices(0.1 second)"** (one index = 0.1 s).

### 3.2 `Plots/plot_svd_result/`

- **Description:** Same idea as plot_T_result but for SVD-based detection: overlay (first singular value σ₁ + detection) and detection-only.
- **Naming:**
  - `{base_id}_SVD_overlay.png`
  - `{base_id}_SVD_detection.png`  
  (again `base_id` with hyphens → underscores, e.g. `1500mAh2-20S0C` → `1500mAh2_20S0C`).
- **Data source:** `Battery_cells/SVD_result/*_30_100_SVD_detects.csv`.
- **Reproduction:** Use columns `time_ind`, `signal` (σ₁), `detected_value_rt`; same overlay vs detection-only logic as T_result, with signal label “First singular value (σ₁)” and title/annotation for first ≥ 0.3 (value and index). X-axis label: **"time indices(0.1 second)"**.

### 3.3 `Plots/plot_multi_cells/`

- **Description:** Two families:
  - **Same cell, multiple SOCs (multi-SOC):** one panel per SOC; overlay (temperature + detection) or detection-only; threshold 0.3; window from start to “first ≥ 0.3” + 200 points.
  - **Same SOC, multiple cells (multi-cell):** one panel per cell; same overlay/detection-only and threshold.
- **Naming examples:**
  - `{cell_id_safe}_multiSOC_overlay.png`, `{cell_id_safe}_multiSOC_detection_only.png`
  - `SOC{soc_val}_multiCell_overlay.png`, `SOC{soc_val}_multiCell_detection_only.png`  
  (`cell_id_safe`: alphanumeric + underscores, e.g. `OE_10Ahr_NMC_cell1`.)
- **Data source:** Same T_result `*_rt_detect_result_window100.csv` files; grouping by cell (strip SOC from dataset name) or by SOC (integer parsed from name, e.g. 80SOC → 80).
- **Reproduction:** Implemented by `Plots/NREL_multiSOC_plots.R` (R + ggplot2). To reproduce in Python: discover all T_result files, parse dataset/cell/SOC from filenames, build overlay and detection-only multi-panel figures (2×2 or 2×3) as in the R script.

### 3.4 `Plots/plot_across_severity_level/`

- **Description:** Paginated grids of detection (and optionally overlay) panels, grouped by **severity level** (1–7). Each page contains multiple small panels (e.g. 2 per row); each panel is one dataset at that severity level (detection-only or overlay, threshold 0.3, optional vertical line at first ≥ 0.3).
- **Naming:**
  - `severity_level_{N}_detection_page{K}.png`
  - `severity_level_{N}_overlay_page{K}.png`
- **Data source:** T_result files whose filename includes `severity_level{N}`; dataset/species/cell/SOC can be parsed from the same filename for titles (e.g. “OE-LCO-6340mAh-0SOC 0% SOC T@≥0.3: 22.8°C”).
- **Reproduction:** Group T_result CSVs by severity level (from filename), split into pages (e.g. 2 or 4 panels per page), then for each panel draw detection-only or overlay with threshold and first-crossing marker.

### 3.5 Script: `Plots/NREL_multiSOC_plots.R`

- **Purpose:** Generates the **plot_multi_cells** family (multi-SOC and multi-cell overlay + detection-only). Expects:
  - Working directory such that `*_reform_severity_level{N}.csv_rt_detect_result_window100.csv` are found (e.g. copy/link T_result files or run from a directory that contains them).
  - Optional: `dataset_correspondence.xlsx` for metadata (script may still run with only the rt_detect filenames).
- **Output:** Writes to `NREL_cursor/` by default (script creates it); filenames as in 3.3. To match existing `plot_multi_cells/` layout, either change the script’s output directory to `Plots/plot_multi_cells/` or copy outputs there.

---

## 4. Plot Types Summary Table

| Folder / script              | Plot type        | Data source              | Main columns / content                          |
|----------------------------|------------------|---------------------------|--------------------------------------------------|
| plot_T_result/              | overlay, detection | T_result/*.csv            | temperature, detected_value_rt, time index       |
| plot_svd_result/            | SVD overlay, SVD detection | SVD_result/*.csv       | time_ind, signal (σ₁), detected_value_rt         |
| plot_multi_cells/           | multi-SOC / multi-cell overlay & detection | T_result/*.csv | Same as T_result; grouped by cell or SOC         |
| plot_across_severity_level/ | detection / overlay pages | T_result/*.csv (by severity) | Same as T_result; grouped by severity level      |
| NREL_multiSOC_plots.R       | —                | rt_detect CSVs            | Implements plot_multi_cells logic                 |

---

## 5. Reproducing Every Plot in `Plots/`

To reproduce all figures under `datasets/NREL/Plots`:

1. **plot_T_result:** For every `Battery_cells/T_result/*_rt_detect_result_window100.csv`, derive `base_id` (dataset name, hyphens → underscores), then generate `{base_id}_overlay.png` and `{base_id}_detection.png` in `Plots/plot_T_result/`.
2. **plot_svd_result:** For every `Battery_cells/SVD_result/*_30_100_SVD_detects.csv`, derive `base_id`, then generate `{base_id}_SVD_overlay.png` and `{base_id}_SVD_detection.png` in `Plots/plot_svd_result/`.
3. **plot_multi_cells:** Run the logic in `NREL_multiSOC_plots.R` (or the Python equivalent in `reproduce_nrel_plots.py`) so that multi-SOC and multi-cell overlay/detection-only figures are written to `Plots/plot_multi_cells/` (or copy from `NREL_cursor/`).
4. **plot_across_severity_level:** Group T_result files by severity level (parse `severity_level{N}` from filename), paginate, and for each page generate detection (and overlay) panels; save as `severity_level_{N}_detection_page{K}.png` and `severity_level_{N}_overlay_page{K}.png` in `Plots/plot_across_severity_level/`.

Use **threshold = 0.3** for detection in all plots, and where applicable mark the first time index (and temperature or σ₁ at that time) where `detected_value_rt >= 0.3`.

The Python script **`reproduce_nrel_plots.py`** in this directory automates (1)–(4) and is described in the next section.

---

## 6. Python Reproduction Script: `reproduce_nrel_plots.py`

A single script **`reproduce_nrel_plots.py`** is provided next to this readme. It:

- **plot_T_result:** Scans `Battery_cells/T_result/`, builds overlay and detection-only plots with threshold 0.3 and first-crossing annotation; saves to `Plots/plot_T_result/` with the naming above.
- **plot_svd_result:** Scans `Battery_cells/SVD_result/`, builds SVD overlay and SVD detection-only plots; saves to `Plots/plot_svd_result/`.
- **plot_multi_cells:** Discovers cells with multiple SOCs and SOCs with multiple cells from T_result filenames, builds multi-panel overlay and detection-only figures; saves to `Plots/plot_multi_cells/`.
- **plot_across_severity_level:** Groups T_result by severity level, paginates (e.g. 2×2 or 4 panels per page), generates detection and overlay pages; saves to `Plots/plot_across_severity_level/`.

Run from the repo root or from `datasets/NREL/`. Ensure project dependencies are installed (e.g. `pip install -e .` from the repo root for `numpy`, `pandas`, `matplotlib`):

```bash
cd datasets/NREL
python reproduce_nrel_plots.py
```

Optional: `--t`, `--svd`, `--multi`, `--severity` to run only specific plot groups; `--out-dir` to override the default `Plots/` base path.

This yields the same structure and naming as the existing `Plots/` content so that outputs can replace or compare with the current PNGs.
