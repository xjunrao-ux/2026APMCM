# Run the published Code1 expressions against the competition's Data02 file.
# Arguments: data.csv Code1.R output_dir [prepare|fit] [iterations] [warmup]
args <- commandArgs(trailingOnly = TRUE)
data_file <- normalizePath(args[1], winslash = "/")
code_file <- normalizePath(args[2], winslash = "/")
out <- normalizePath(args[3], winslash = "/")
mode <- match.arg(args[4], c("prepare", "fit"))
suppressPackageStartupMessages(library(dplyr))
expressions <- as.list(parse(code_file, encoding = "UTF-8"))
one <- function(matches) {
  if (sum(matches) != 1L) stop("Expected exactly one matching expression in Code1")
  expressions[[which(matches)]]
}
assignment <- function(name) {
  one(vapply(expressions, function(x) is.call(x) && identical(x[[1]], as.name("<-")) &&
               identical(x[[2]], as.name(name)), logical(1)))
}
dat <- read.csv(data_file, check.names = FALSE, stringsAsFactors = FALSE)
response_expr <- one(vapply(expressions, function(x)
  is.call(x) && identical(x[[1]], as.name("<-")) && identical(x[[2]], as.name("dat")) &&
  "case_when" %in% all.names(x[[3]]), logical(1)))
eval(response_expr)
count_expr <- one(vapply(expressions, function(x)
  is.call(x) && identical(x[[1]], as.name("data.frame")) && "Responses" %in% names(x),
  logical(1)))
write_table <- function(x, name) write.csv(x, file.path(out, paste0(name, ".csv")),
                                           row.names = FALSE, fileEncoding = "UTF-8")
if (mode == "prepare") {
  write_table(dat, "r_analysis_data")
  write_table(eval(count_expr), "r_baseline_counts")
  quit(save = "no")
}

suppressPackageStartupMessages(library(brms))
iters <- as.integer(args[5])
warm <- as.integer(args[6])
stopifnot(iters > warm, warm > 0L)
cache <- Sys.getenv("Q3_STAN_CACHE", unset = file.path(tools::R_user_dir("apmcm-q3", "cache"), "stan"))
dir.create(cache, recursive = TRUE, showWarnings = FALSE)
cache <- normalizePath(cache, winslash = "/")
setwd(tempdir())
grid <- expand.grid(CallType = sort(unique(dat$CallType)), Distance = c(50, 100, 150),
                    Context = sort(unique(dat$Context)), stringsAsFactors = FALSE)
grid$Npenguins <- median(dat$Npenguins)
fit_names <- c(HeadsUp = "H_brm", Vocal = "V_brm", Approach = "A_brm")
diagnostics <- params <- predictions <- list()
# Nodes and weights for integrating the approach model over a new experiment.
n_nodes <- 40L
jacobi <- matrix(0, n_nodes, n_nodes)
jacobi[cbind(seq_len(n_nodes-1L), 2:n_nodes)] <- sqrt(seq_len(n_nodes-1L))
gh <- eigen(jacobi + t(jacobi), symmetric = TRUE)
for (h in names(fit_names)) {
  selected <- as.character(assignment(fit_names[[h]])[[3]])
  call <- assignment(selected)[[3]]
  call$iter <- iters
  call$warmup <- warm
  call$seed <- 54984L
  call$cores <- 4L
  call$refresh <- 0L
  versions <- vapply(c("brms", "rstan", "StanHeaders"), function(p) as.character(packageVersion(p)), "")
  key <- paste(c(selected, tools::md5sum(c(code_file, data_file)), versions, iters, warm), collapse = "_")
  call$file <- file.path(cache, key)
  message("Code1 model: ", selected)
  fit <- eval(call)
  draws <- posterior::as_draws_array(fit, variable=c("^b_", "^sd_", "^shape$"), regex=TRUE)
  diag <- posterior::summarise_draws(draws)
  sampler <- do.call(rbind, rstan::get_sampler_params(fit$fit, inc_warmup = FALSE))
  diagnostics[[h]] <- data.frame(behavior=h, selected, chains=4, iterations=iters, warmup=warm,
    rhat_max=max(diag$rhat), bulk_ess_min=min(diag$ess_bulk), tail_ess_min=min(diag$ess_tail),
    divergences=sum(sampler[, "divergent__"]), treedepth_hits=sum(sampler[, "treedepth__"] >= 10),
    brms_version=versions[1], rstan_version=versions[2])
  f <- fixef(fit)
  params[[h]] <- data.frame(behavior=h, term=rownames(f), mean=f[,1], lo95=f[,3], hi95=f[,4])
  b <- as.matrix(fixef(fit, summary = FALSE))
  if (h == "Approach") {
    nd <- transform(grid, zN=(Npenguins-mean(dat$Npenguins))/sd(dat$Npenguins),
                    zD=(Distance-mean(dat$Distance))/sd(dat$Distance))
    x <- model.matrix(~ Context * CallType + zN + zD, nd)
    colnames(x)[match(c("(Intercept)", "zN", "zD"), colnames(x))] <-
      c("Intercept", "scaleNpenguins", "scaleDistance")
    eta <- b[,colnames(x),drop=FALSE] %*% t(x)
    sd_id <- as.numeric(posterior::as_draws_matrix(posterior::subset_draws(draws, variable="sd_expID__Intercept")))
    prob <- matrix(0, nrow(eta), ncol(eta))
    for (j in seq_len(n_nodes)) prob <- prob + gh$vectors[1,j]^2 * plogis(eta + sd_id * gh$values[j])
  } else {
    # Code1's hurdle is P(outcome=0); it has only a CallType fixed effect.
    x <- model.matrix(~ CallType, grid)
    colnames(x)[1] <- "Intercept"
    eta <- b[,paste0("hu_", colnames(x)),drop=FALSE] %*% t(x)
    prob <- plogis(-eta)
  }
  predictions[[h]] <- data.frame(grid, behavior=h, r_probability=colMeans(prob),
    r_lo95=apply(prob, 2, quantile, .025), r_hi95=apply(prob, 2, quantile, .975))
}
write_table(bind_rows(params), "r_baseline_parameters")
write_table(bind_rows(diagnostics), "r_baseline_diagnostics")
write_table(bind_rows(predictions), "r_baseline_predictions")
