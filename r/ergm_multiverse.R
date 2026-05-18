#!/usr/bin/env Rscript
# =============================================================================
# Multiverse Analysis of the Karnataka Loan Nomination Network
# R script: ERGM specification multiverse
# =============================================================================
#
# The Karnataka network consists of 33 disconnected village-level components.
# ERGMs are fitted village-by-village (within-component) to avoid the
# disconnected-graph problem, then pooled via inverse-variance meta-analysis.
#
# Multiverse dimensions:
#   1. Network representation: directed_weighted, undirected_weighted,
#                              directed_strong_ties (top-quartile weight)
#   2. ERGM specification:
#        Directed:   m1 (edges), m2 (edges+mutual),
#                    m3 (edges+mutual+gwodegree), m4 (edges+mutual+gwesp),
#                    m5 (edges+gwidegree+gwesp)
#        Undirected: m1 (edges), m2 (edges+gwdegree),
#                    m3 (edges+gwesp), m4 (edges+gwdegree+gwesp)
#
# Usage:
#   Rscript r/ergm_multiverse.R \
#       data/loan_nomination_edgelist.csv \
#       results/r
# =============================================================================

suppressPackageStartupMessages({
  library(network)
  library(ergm)
  library(sna)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript ergm_multiverse.R EDGELIST_CSV OUTDIR")
edgelist_csv <- args[[1]]
outdir       <- args[[2]]
dir.create(file.path(outdir, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "figs"),   recursive = TRUE, showWarnings = FALSE)

cat("Reading edge list from:", edgelist_csv, "\n")
el <- read.csv(edgelist_csv, stringsAsFactors = FALSE)
cat("  Edges:", nrow(el), "\n")
cat("  Nodes:", length(unique(c(el$from, el$to))), "\n")

all_nodes   <- unique(c(el$from, el$to))
village_of  <- function(node) sub("_.*", "", node)
villages    <- unique(village_of(all_nodes))
cat("  Villages:", length(villages), "\n")

# ── Helper: build network for one village ──────────────────────────────────
make_village_net <- function(village_nodes, edges_df, directed = TRUE,
                             strong_tie_q = NULL) {
  sub_el <- edges_df[edges_df$from %in% village_nodes &
                       edges_df$to %in% village_nodes, ]
  if (!is.null(strong_tie_q)) {
    thresh <- quantile(edges_df$weight, strong_tie_q, na.rm = TRUE)
    sub_el <- sub_el[sub_el$weight >= thresh, ]
  }
  if (nrow(sub_el) == 0) return(NULL)
  network(sub_el[, c("from", "to")], directed = directed,
          matrix.type = "edgelist")
}

# ── ERGM specifications (as character strings, built into formulas per net) ─
spec_terms_directed <- list(
  m1 = "edges",
  m2 = "edges + mutual",
  m3 = "edges + mutual + gwodegree(0.5, fixed=TRUE)",
  m4 = "edges + mutual + gwesp(0.5, fixed=TRUE)",
  m5 = "edges + gwidegree(0.5, fixed=TRUE) + gwesp(0.5, fixed=TRUE)"
)
spec_terms_undirected <- list(
  m1 = "edges",
  m2 = "edges + gwdegree(0.5, fixed=TRUE)",
  m3 = "edges + gwesp(0.5, fixed=TRUE)",
  m4 = "edges + gwdegree(0.5, fixed=TRUE) + gwesp(0.5, fixed=TRUE)"
)

# ── Representation definitions ─────────────────────────────────────────────
rep_defs <- list(
  directed_weighted   = list(directed = TRUE,  strong_q = NULL),
  undirected_weighted = list(directed = FALSE, strong_q = NULL),
  directed_strong     = list(directed = TRUE,  strong_q = 0.75)
)

# ── Main loop ──────────────────────────────────────────────────────────────
all_results <- list()
k <- 1

for (rep_nm in names(rep_defs)) {
  rd         <- rep_defs[[rep_nm]]
  spec_terms <- if (rd$directed) spec_terms_directed else spec_terms_undirected
  cat("\n=== Representation:", rep_nm, "===\n")

  for (spec_nm in names(spec_terms)) {
    cat("  Spec:", spec_nm, "\n")
    village_coefs <- list()

    for (vill in villages) {
      vnodes <- all_nodes[village_of(all_nodes) == vill]
      net <- tryCatch(
        make_village_net(vnodes, el,
                         directed    = rd$directed,
                         strong_tie_q = rd$strong_q),
        error = function(e) NULL
      )
      if (is.null(net)) next
      if (network.size(net) < 5 || network.edgecount(net) < 3) next

      # Build formula string with the network object name
      fml_str <- paste0("net ~ ", spec_terms[[spec_nm]])
      fml     <- as.formula(fml_str)

      fit <- tryCatch(
        withCallingHandlers(
          ergm(fml, control = control.ergm(MCMLE.maxit = 20, seed = 42)),
          warning = function(w) invokeRestart("muffleWarning")
        ),
        error = function(e) NULL
      )
      if (is.null(fit)) next

      s <- tryCatch(summary(fit), error = function(e) NULL)
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
  cat("No ERGM fits succeeded.\n")
  quit(status = 1)
}

coef_tbl <- do.call(rbind, all_results)
rownames(coef_tbl) <- NULL

# Standardise column names
names(coef_tbl) <- gsub("Std\\. Error",      "se",       names(coef_tbl))
names(coef_tbl) <- gsub("^Estimate$",        "estimate", names(coef_tbl))
names(coef_tbl) <- gsub("z value",           "z",        names(coef_tbl))
names(coef_tbl) <- gsub("Pr\\(>\\|z\\|\\)",  "p",        names(coef_tbl))
names(coef_tbl) <- gsub("MCMC %",            "mcmc_pct", names(coef_tbl))

write.csv(coef_tbl,
          file.path(outdir, "tables", "ergm_multiverse_coefficients.csv"),
          row.names = FALSE)
cat("\nWrote", nrow(coef_tbl), "coefficient rows to ergm_multiverse_coefficients.csv\n")

# ── Meta-analytic pooling ──────────────────────────────────────────────────
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
    representation  = parts[1],
    spec            = parts[2],
    term            = parts[3],
    pooled_estimate = round(est, 4),
    pooled_se       = round(se,  4),
    pooled_z        = round(z,   3),
    pooled_p        = round(p,   4),
    n_villages      = nrow(sub),
    stringsAsFactors = FALSE
  )
}
pool_tbl <- do.call(rbind, pool_results)
rownames(pool_tbl) <- NULL
write.csv(pool_tbl,
          file.path(outdir, "tables", "ergm_pooled_estimates.csv"),
          row.names = FALSE)
cat("Wrote", nrow(pool_tbl), "pooled rows to ergm_pooled_estimates.csv\n")

# ── Fit summary ────────────────────────────────────────────────────────────
fit_tbl <- aggregate(cbind(AIC, BIC) ~ representation + spec,
                     data = coef_tbl,
                     FUN  = function(x) round(mean(x, na.rm = TRUE), 2))
fit_tbl <- fit_tbl[order(fit_tbl$AIC), ]
write.csv(fit_tbl,
          file.path(outdir, "tables", "ergm_fit_summary.csv"),
          row.names = FALSE)
cat("Wrote fit summary to ergm_fit_summary.csv\n")

# ── Forest plot: pooled edges term ────────────────────────────────────────
edges_pool <- pool_tbl[pool_tbl$term == "edges", ]
if (nrow(edges_pool) > 0) {
  png(file.path(outdir, "figs", "ergm_edges_forest.png"),
      width = 900, height = 500, res = 120)
  par(mar = c(5, 14, 3, 2))
  y    <- seq_len(nrow(edges_pool))
  est  <- edges_pool$pooled_estimate
  lo   <- est - 1.96 * edges_pool$pooled_se
  hi   <- est + 1.96 * edges_pool$pooled_se
  xlim <- range(c(lo, hi), na.rm = TRUE)
  col  <- ifelse(edges_pool$pooled_p < 0.05, "steelblue", "grey60")
  plot(est, y, xlim = xlim, yaxt = "n", pch = 19,
       xlab = "Pooled ERGM Edges Coefficient (log-odds)",
       main = "ERGM Edges Term: Pooled Estimates by Representation and Specification",
       col  = col)
  arrows(lo, y, hi, y, length = 0.05, angle = 90, code = 3, col = col)
  abline(v = 0, lty = 2, col = "red")
  axis(2, at = y,
       labels = paste(edges_pool$representation, edges_pool$spec, sep = " / "),
       las = 2, cex.axis = 0.7)
  dev.off()
  cat("Wrote edges forest plot\n")
}

cat("\nAll ERGM outputs written to:", outdir, "\n")
