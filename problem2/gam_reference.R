# Code2 fits a Gaussian GAM on all valid coefficients, using distance in metres.
# ggplot2's GAM smoother uses REML. No graphs or confidence bands are needed here.
args <- commandArgs(trailingOnly = TRUE)
library(mgcv)
raw <- read.csv(args[1], fileEncoding = "UTF-8-BOM")
raw <- raw[is.finite(raw$correlation), ]
all_predictions <- list()
for (dat in split(raw, list(raw$call, raw$metric), drop = TRUE)) {
  for (hold in c(0, 8, 16, 32, 64, 128)) {
    train <- dat[dat$d != hold, ]
    model <- gam(correlation ~ s(d, k = 3), data = train, method = "REML")
    x <- if (hold == 0) seq(4, 256, by = 0.1) else hold
    all_predictions[[length(all_predictions) + 1]] <- data.frame(
      call = dat$call[1], metric = dat$metric[1], hold = hold, d = x,
      estimate = as.numeric(predict(model, newdata = data.frame(d = x)))
    )
  }
}
write.csv(do.call(rbind, all_predictions), args[2], row.names = FALSE)
cat(R.version.string, "mgcv", as.character(packageVersion("mgcv")), "\n")
