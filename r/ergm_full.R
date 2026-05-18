#!/usr/bin/env Rscript
# =============================================================================
# Multiverse Analysis of the Karnataka Loan Nomination Network
# R script: ERGM multiverse — FULL VERSION (all specs, all villages)
# =============================================================================
#
# WARNING: This script runs ALL specifications (m1–m4 directed, m1–m3
# undirected) on ALL 33 villages. Specifications involving gwesp require
# full MCMC sampling and are extremely computationally intensive. On a
# standard single-core machine, expect runtime of 24–72 hours or more.
# For a computationally tractable version, use:
#   - ergm_mcmle.R  : gwesp specs restricted to top-10 villages (recommended)
#   - ergm_mple.R   : all specs via MPLE approximation (fast, less accurate)
#
# Usage:
#   Rscript r/ergm_full.R \
#       data/loan_nomination_edgelist.csv \
#       results/r
# =============================================================================

suppressPackageStartupMessages({
  library(network)
  library(ergm)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript ergm_full.R EDGELIST_CSV OUTDIR")
edgelist_csv <- args[[1]]
outdir       <- args[[2]]
dir.create(file.path(outdir, "tables"), recursive = TRUE, showWarnings = FALSE)

cat("Reading edge list from:", edgelist_csv, "\n")
el         <- read.csv(edgelist_csv, stringsAsFactors = FALSE)
all_nodes  <- unique(c(el$from, el$to))
village_of <- function(node) sub("_.*", "", node)
villages   <- unique(village_of(all_nodes))
cat("  Total villages:", length(villages), "\n")

make_village_net <- function(village_nodes, edges_df, directed = TRUE) {
  sub_el <- edges_df[edges_df$from %in% village_nodes &
                       edges_df$to   %in% village_nodes, ]
  if (nrow(sub_el) == 0) return(NULL)
  network(sub_el[, c("from", "to")], directed = directed,
          matrix.type = "edgelist")
}

# All specifications — directed and undirected
spec_defs_directed <- list(
  m1 = "edges",
  m2 = "edges + mutual",
  m3 = "edges + mutual + gwodegree(0.5, fixed=TRUE)",
  m4 = "edges + mutual + gwesp(0.5, fixed=TRUE)"
)
spec_defs_undirected <- list(
  m1 = "edges",
  m2 = "edges + gwdegree(0.5, fixed=TRUE)",
  m3 = "edges + gwesp(0.5, fixed=TRUE)"
)

rep_defs <- list(
  directed_weighted   = list(directed = TRUE),
  undirected_weighted = list(directed = FALSE)
)

# Standard MCMLE control — applied uniformly to all specs and all villages
ctrl <- control.ergm(
  MCMLE.maxit     = 20,
  MCMC.samplesize = 2048,
  MCMC.burnin     = 10000,
  seed            = 42
)

all_results <- list()
k <- 1

for (rep_nm in names(rep_defs)) {
  rd        <- rep_defs[[rep_nm]]
  spec_defs <- if (rd$directed) spec_defs_directed else spec_defs_undirected
  cat("\n=== Representation:", rep_nm, "===\n")

  for (spec_nm in names(spec_defs)) {
    cat("  Spec:", spec_nm, "(all", length(villages), "villages)\n")
    village_coefs <- list()

    for (vill in villages) {
      vnodes <- all_nodes[village_of(all_nodes) == vill]
      net <- tryCatch(
        make_village_net(vnodes, el, directed = rd$directed),
        error = function(e) NULL
      )
      if (is.null(net)) next
      if (network.size(net) < 5 || network.edgecount(net) < 3) next

      fml <- as.formula(paste0("net ~ ", spec_defs[[spec_nm]]))
      fit <- tryCatch(ergm(fml, control = ctrl), error = function(e) NULL)
      if (is.null(fit)) next
      s   <- tryCatch(summary(fit), error = function(e) NULL)
      if (is.null(s)) next

      coefs <- as.data.frame(s$coefficients)
      coefs$term           <- rownames(coefs)
      coefs$village        <- vill
      coefs$representation <- rep_nm
      coefs$spec           <- spec_nm
      coefs$n_nodes        <- network.size(net)
      coefs$n_edges        <- network.edgecount(net)
      coefs$AIC            <- tryCatch(AIC(fit), error = function(e) NA_real_)
      coefs$BIC            <- tryCatch(BIC(fit), error = function(e) NA_real_)
      village_coefs[[vill]] <- coefs
    }

    if (length(village_coefs) > 0) {
      combined <- do.call(rbind, village_coefs)
      rownames(combined) <- NULL
      all_results[[k]] <- combined
      k <- k + 1
      cat("    -> fitted in", length(village_coefs), "villages\n")
    }
  }
}

if (length(all_results) == 0) {
  cat("No ERGM fits succeeded.\n"); quit(status = 1)
}

coef_tbl <- do.call(rbind, all_results)
rownames(coef_tbl) <- NULL
names(coef_tbl) <- gsub("Std\\. Error",     "se",       names(coef_tbl))
names(coef_tbl) <- gsub("^Estimate$",       "estimate", names(coef_tbl))
names(coef_tbl) <- gsub("z value",          "z",        names(coef_tbl))
names(coef_tbl) <- gsub("Pr\\(>\\|z\\|\\)", "p",        names(coef_tbl))

write.csv(coef_tbl,
          file.path(outdir, "tables", "ergm_full_coefficients.csv"),
          row.names = FALSE)
cat("\nWrote", nrow(coef_tbl), "coefficient rows.\n")

# Inverse-variance meta-analytic pooling
pool_results <- list()
grp_keys <- unique(paste(coef_tbl$representation, coef_tbl$spec,
                          coef_tbl$term, sep = "|"))
for (grp in grp_keys) {
  parts <- strsplit(grp, "\\|")[[1]]
  sub   <- coef_tbl[coef_tbl$representation == parts[1] &
                      coef_tbl$spec          == parts[2] &
                      coef_tbl$term          == parts[3], ]
  sub   <- sub[!is.na(sub$estimate) & !is.na(sub$se) & sub$se > 0, ]
  if (nrow(sub) < 2) next
  w   <- 1 / sub$se^2
  est <- sum(w * sub$estimate) / sum(w)
  se  <- sqrt(1 / sum(w))
  z   <- est / se
  p   <- 2 * pnorm(-abs(z))
  pool_results[[grp]] <- data.frame(
    representation  = parts[1], spec = parts[2], term = parts[3],
    pooled_estimate = round(est, 4), pooled_se = round(se, 4),
    pooled_z        = round(z,   3), pooled_p  = round(p,  4),
    n_villages      = nrow(sub), stringsAsFactors = FALSE
  )
}
pool_tbl <- do.call(rbind, pool_results)
rownames(pool_tbl) <- NULL
write.csv(pool_tbl,
          file.path(outdir, "tables", "ergm_full_pooled.csv"),
          row.names = FALSE)
cat("Wrote", nrow(pool_tbl), "pooled rows.\n")
cat("All outputs written to:", outdir, "\n")
