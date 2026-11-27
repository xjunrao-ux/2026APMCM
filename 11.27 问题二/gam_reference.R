# Code2 fits a Gaussian GAM on all valid coefficients, using distance in metres.
# ggplot2 uses REML and a pointwise normal interval on the identity-link scale.
args <- commandArgs(trailingOnly = TRUE)
library(mgcv)
raw <- read.csv(args[1], fileEncoding = "UTF-8-BOM")

# In the full pipeline the working directory is the original data directory.
# Identify files by their acoustic label (independent of Python's DataNN mapping).
if (length(args) == 4L) {
  call_stem <- c(flat = "flat", mod = "modulated", tv = "twovoice")
  metric_stem <- c(spectrogram = "spectrogram", spectrum = "amplitude_spectrum",
                   envelope = "envelope")
  distances <- c(4, 8, 16, 32, 64, 128, 256)
  files <- list.files(pattern = "^Data[0-9]{2}_")
  refs <- list()
  for (call in names(call_stem)) {
    for (metric in names(metric_stem)) {
      suffix <- sprintf("_correlation1m_%s_call_%s.csv", call_stem[call], metric_stem[metric])
      path <- files[endsWith(files, suffix)]
      stopifnot(length(path) == 1L)
      wide <- read.csv(path, check.names = FALSE)
      stopifnot(identical(names(wide), paste0(distances, "m")))
      series <- raw[raw$call == call & raw$metric == metric, ]
      series <- series[order(series$d, series$source_row), ]
      if (!identical(as.integer(series$d), as.integer(rep(distances, each = nrow(wide)))) ||
          !identical(series$source_row, rep(seq_len(nrow(wide)), times = ncol(wide))) ||
          !identical(as.numeric(series$correlation), as.numeric(as.matrix(wide)))) {
        stop("Long-file values, NA locations or row keys disagree with ", path)
      }
      refs[[length(refs) + 1L]] <- data.frame(
        call, metric, d = distances, source_file = path,
        wide_n = colSums(!is.na(wide)), wide_mean = colMeans(wide, na.rm = TRUE),
        wide_sd = vapply(wide, sd, numeric(1), na.rm = TRUE), row.names = NULL
      )
    }
  }
  keys <- c("call", "metric", "d")
  audit <- read.csv(args[3], fileEncoding = "UTF-8-BOM")
  reference <- do.call(rbind, refs)
  check <- merge(audit[c(keys, "valid", "mean", "sd")], reference, by = keys, all = TRUE)
  if (nrow(check) != nrow(reference) || anyNA(check) ||
      any(check$valid != check$wide_n) ||
      any(abs(check$mean - check$wide_mean) > 1e-9) ||
      any(abs(check$sd - check$wide_sd) > 1e-9)) {
    stop("Code2 wide-file summaries disagree with the long-file audit.")
  }
  write.csv(check[order(check$call, check$metric, check$d), ], args[4], row.names = FALSE)
}
raw <- raw[is.finite(raw$correlation), ]
all_predictions <- list()
for (dat in split(raw, list(raw$call, raw$metric), drop = TRUE)) {
  for (hold in c(0, 8, 16, 32, 64, 128)) {
    train <- dat[dat$d != hold, ]
    model <- gam(correlation ~ s(d, k = 3), data = train, method = "REML")
    x <- if (hold == 0) seq(4, 256, by = 0.1) else hold
    prediction <- predict(model, newdata = data.frame(d = x), se.fit = TRUE)
    half_width <- qnorm(0.95) * prediction$se.fit
    all_predictions[[length(all_predictions) + 1]] <- data.frame(
      call = dat$call[1], metric = dat$metric[1], hold = hold, d = x,
      estimate = as.numeric(prediction$fit),
      lower90 = as.numeric(prediction$fit - half_width),
      upper90 = as.numeric(prediction$fit + half_width)
    )
  }
}
write.csv(do.call(rbind, all_predictions), args[2], row.names = FALSE)
cat(R.version.string, "mgcv", as.character(packageVersion("mgcv")), "\n")
