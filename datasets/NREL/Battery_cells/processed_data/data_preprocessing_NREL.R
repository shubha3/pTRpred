#Data Manipulations for the battery datasets:
library(readxl)
library(dplyr)
library(stringr)

# ---- helpers ----

# Linear interpolation onto a fixed grid.
# Returns numeric vector length = length(grid), with NAs outside range.
interp_linear_on_grid <- function(time, value, grid) {
  ok <- is.finite(time) & is.finite(value)
  time <- time[ok]
  value <- value[ok]
  if (length(time) < 2) return(rep(NA_real_, length(grid)))

  # Ensure strictly increasing x for approx()
  ord <- order(time)
  time <- time[ord]
  value <- value[ord]

  approx(x = time, y = value, xout = grid, method = "linear", rule = 1)$y
}

# Process one xlsx file -> reformatted data frame
reform_one_file <- function(filename, dt = 0.1) {
  df_tr <- read_xlsx(filename, sheet = "Sheet1") |> as.data.frame()

  # Select the same columns you used: 1,2,3,20,21
  df_sel <- df_tr[, c(1, 2, 3, 20, 21)]
  names(df_sel) <- c("time1", "load", "voltage", "time2", "temperature")

  # Determine common end time (your original logic)
  t1 <- df_sel$time1[!is.na(df_sel$time1)]
  t2 <- df_sel$time2[!is.na(df_sel$time2)]
  time_end <- min(max(t1), max(t2))

  # Build grid from 0 to floor(time_end/dt)*dt, inclusive
  time_end_scale <- floor(time_end / dt) * dt
  grid <- seq(0, time_end_scale, by = dt)

  # Interpolate load/voltage on time1, temp on time2, truncated to < time_end
  df_t1 <- df_sel |> filter(is.finite(load), is.finite(voltage), is.finite(time1), time1 < time_end)
  df_t2 <- df_sel |> filter(is.finite(temperature), is.finite(time2), time2 < time_end)

  load_itp    <- interp_linear_on_grid(df_t1$time1, df_t1$load, grid)
  voltage_itp <- interp_linear_on_grid(df_t1$time1, df_t1$voltage, grid)
  temp_itp    <- interp_linear_on_grid(df_t2$time2, df_t2$temperature, grid)

  out <- data.frame(
    time_point  = grid,
    temperature = temp_itp,
    load        = load_itp,
    voltage     = voltage_itp
  )

  out <- out |> filter(!is.na(load), !is.na(temperature), load != 0, temperature > 0)

  out
}

dataset_description <- read_xlsx("dataset_correspondence_information.xlsx") |>
  as.data.frame() |>
  arrange(severity_level)

print(head(dataset_description))

for (level in 1:7) {
  rows <- which(dataset_description$severity_level == level)
  if (length(rows) == 0) next

  for (k in seq_along(rows)) {
    ds_name  <- as.character(dataset_description$dataset_name[rows[k]])
    filename <- paste0(ds_name, ".xlsx")

    df_reform <- reform_one_file(filename, dt = 0.1)

    out_csv <- paste0(ds_name, "_reform_severity_level", level, ".csv")
    write.csv(df_reform, out_csv, row.names = FALSE)

    message("level=", level, "  file=", k, "/", length(rows), "  -> ", out_csv)
  }
}