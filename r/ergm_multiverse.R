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
#   1. Network representation: directed_weighted, undirected_weighted
#   2. ERGM specification:
#        Directed:   m1 (edges)
#                    m2 (edges + mutual)
#                    m3 (edges + mutual + gwodegree)
#                    m4 (edges + mutual + gwesp)  -- TOP-10 VILLAGES ONLY
#        Undirected: m1 (edges)
#                    m2 (edges + gwdegree)
#                    m3 (edges + gwesp)            -- TOP-10 VILLAGES ONLY
#
# NOTE: Specifications involving gwesp (geometrically-weighted edgewise shared
# partners) require full MCMC sampling and are computationally prohibitive for
# all 33 villages. Following the pre-registered analysis plan, these
# specifications are restricted to the 10 largest villages by household count:
# v52 (375), v65 (362), v59 (353), v71 (310), v43 (308),
# v50 (298), v55 (271), v45 (269), v76 (267), v40 (262).
#
# Usage:
#   Rscript r/ergm_multiverse.R \
#       data/loan_nomination_edgelist.csv \
#       results/r
# =============================================================================

suppressPackageStartupMessages({
  library(network)
  library(ergm)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript ergm_multiverse.R EDGELIST_CSV OUTDIR")
edgelist_csv <- args[[1]]
outdir       <- args[[2]]
dir.create(file.path(outdir, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "figs"),   recursive = TRUE, showWarnings = FALSE)

cat("Reading edge list from:", edgelist_csv, "\n")
el          <- read.csv(edgelist_csv, stringsAsFactors = FALSE)
all_nodes   <- unique(c(el$from, el$to))
village_of  <- function(node) sub("_.*", "", node)
villages    <- unique(village_of(all_nodes))
cat("  Total villages:", length(villages), "\n")

# ── Top-10 villages by household count (pre-computed) ─────────────────────
top10_villages <- c("v52", "v65", "v59", "v71", "v43",
                    "v50", "v55", "v45", "v76", "v40")
cat("  Top-10 villages for gwesp specs:", paste(top10_villages, collapse=", "), "\n")

# ── Helper: build network for one village ─────────────────────────────────
make_village_net <- function(village_nodes, edges_df, directed = TRUE) {
  sub_el <- edges_df[edges_df$from %in% village_nodes &
                       edges_df$to   %in% village_nodes, ]
  if (nrow(sub_el) == 0) return(NULL)
  network(sub_el[, c("from", "to")], directed = directed,
          matrix.type = "edgelist")
}

# ── ERGM specifications ────────────────────────────────────────────────────
# gwesp_only = TRUE means this spec runs on top-10 villages only
spec_defs_directed <- list(
  m1 = list(terms = "edges",
            gwesp_only = FALSE),
  m2 = list(terms = "edges + mutual",
            gwesp_only = FALSE),
  m3 = list(terms = "edges + mutual + gwodegree(0.5, fixed=TRUE)",
            gwesp_only = FALSE),
  m4 = list(terms = "edges + mutual + gwesp(0.5, fixed=TRUE)",
            gwesp_only = TRUE)
)
spec_defs_undirected <- list(
  m1 = list(terms = "edges",
            gwesp_only = FALSE),
  m2 = list(terms = "edges + gwdegree(0.5, fixed=TRUE)",
            gwesp_only = FALSE),
  m3 = list(terms = "edges + gwesp(0.5, fixed=TRUE)",
            gwesp_only = TRUE)
)

rep_defs <- list(
  directed_weighted   = list(directed = TRUE),
  undirected_weighted = list(directed = FALSE)
)

# ── MCMC control settings ──────────────────────────────────────────────────
# For non-gwesp specs: standard convergence settings
ctrl_standard <- control.ergm(
  MCMLE.maxit      = 20,
  MCMC.samplesize  = 1024,
  seed             = 42
)
# For gwesp specs: more generous settings to handle triangle terms
ctrl_gwesp <- control.ergm(
  MCMLE.maxit      = 20,
  MCMC.samplesize  = 2048,
  MCMC.burnin      = 10000,
  seed             = 42
)

# ── Main loop ─────────────────────────────────────────────────────────────
all_results <- list()
k <- 1

for (rep_nm in names(rep_defs)) {
  rd         <- rep_defs[[rep_nm]]
  spec_defs  <- if (rd$directed) spec_defs_directed else spec_defs_undirected
  cat("\n=== Representation:", rep_nm, "===\n")

  for (spec_nm in names(spec_defs)) {
    spec       <- spec_defs[[spec_nm]]
    is_gwesp   <- spec$gwesp_only
    run_villages <- if (is_gwesp) top10_villages else villages
    ctrl       <- if (is_gwesp) ctrl_gwesp else ctrl_standard

    cat("  Spec:", spec_nm,
        if (is_gwesp) "(top-10 villages only)" else "(all villages)", "\n")

    village_coefs <- list()

    for (vill in run_villages) {
      vnodes <- all_nodes[village_of(all_nodes) == vill]
      net <- tryCatch(
        make_village_net(vnodes, el, directed = rd$directed),
        error = function(e) NULL
      )
      if (is.null(net)) next
      if (network.size(net) < 5 || network.edgecount(net) < 3) next

      fml <- as.formula(paste0("net ~ ", spec$terms))

      fit <- tryCatch(
        ergm(fml, control = ctrl),
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
      coefs$gwesp_sample   <- is_gwesp
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
names(coef_tbl) <- gsub("Std\\. Error",     "se",       names(coef_tbl))
names(coef_tbl) <- gsub("^Estimate$",       "estimate", names(coef_tbl))
names(coef_tbl) <- gsub("z value",          "z",        names(coef_tbl))
names(coef_tbl) <- gsub("Pr\\(>\\|z\\|\\)", "p",        names(coef_tbl))

write.csv(coef_tbl,
          file.path(outdir, "tables", "ergm_multiverse_coefficients.csv"),
          row.names = FALSE)
cat("\nWrote", nrow(coef_tbl), "coefficient rows to ergm_multiverse_coefficients.csv\n")

# ── Meta-analytic pooling ─────────────────────────────────────────────────
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
    gwesp_sample    = unique(sub$gwesp_sample)[1],
    stringsAsFactors = FALSE
  )
}
pool_tbl <- do.call(rbind, pool_results)
rownames(pool_tbl) <- NULL
write.csv(pool_tbl,
          file.path(outdir, "tables", "ergm_pooled_estimates.csv"),
          row.names = FALSE)
cat("Wrote", nrow(pool_tbl), "pooled rows to ergm_pooled_estimates.csv\n")

# ── Fit summary ───────────────────────────────────────────────────────────
fit_tbl <- aggregate(cbind(AIC, BIC) ~ representation + spec + gwesp_sample,
                     data = coef_tbl,
                     FUN  = function(x) round(mean(x, na.rm = TRUE), 2))
fit_tbl <- fit_tbl[order(fit_tbl$AIC), ]
write.csv(fit_tbl,
          file.path(outdir, "tables", "ergm_fit_summary.csv"),
          row.names = FALSE)
cat("Wrote fit summary to ergm_fit_summary.csv\n")

cat("\nAll ERGM outputs written to:", outdir, "\n")
