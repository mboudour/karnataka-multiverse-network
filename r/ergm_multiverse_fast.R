#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(network)
  library(ergm)
})
args <- commandArgs(trailingOnly = TRUE)
edgelist_csv <- args[[1]]
outdir       <- args[[2]]
dir.create(file.path(outdir, "tables"), recursive = TRUE, showWarnings = FALSE)

el <- read.csv(edgelist_csv, stringsAsFactors = FALSE)
all_nodes   <- unique(c(el$from, el$to))
village_of  <- function(node) sub("_.*", "", node)
villages    <- unique(village_of(all_nodes))

make_village_net <- function(village_nodes, edges_df, directed = TRUE, strong_tie_q = NULL) {
  sub_el <- edges_df[edges_df$from %in% village_nodes & edges_df$to %in% village_nodes, ]
  if (!is.null(strong_tie_q)) {
    thresh <- quantile(edges_df$weight, strong_tie_q, na.rm = TRUE)
    sub_el <- sub_el[sub_el$weight >= thresh, ]
  }
  if (nrow(sub_el) == 0) return(NULL)
  network(sub_el[, c("from", "to")], directed = directed, matrix.type = "edgelist")
}

# Only the analytically tractable specifications (no MCMC needed for these in this data)
spec_terms_directed <- list(
  m1 = "edges",
  m2 = "edges + mutual",
  m3 = "edges + mutual + gwodegree(0.5, fixed=TRUE)"
)
spec_terms_undirected <- list(
  m1 = "edges",
  m2 = "edges + gwdegree(0.5, fixed=TRUE)"
)

rep_defs <- list(
  directed_weighted   = list(directed = TRUE,  strong_q = NULL),
  undirected_weighted = list(directed = FALSE, strong_q = NULL)
)

all_results <- list()
k <- 1
for (rep_nm in names(rep_defs)) {
  rd <- rep_defs[[rep_nm]]
  spec_terms <- if (rd$directed) spec_terms_directed else spec_terms_undirected
  for (spec_nm in names(spec_terms)) {
    village_coefs <- list()
    for (vill in villages) {
      vnodes <- all_nodes[village_of(all_nodes) == vill]
      net <- tryCatch(make_village_net(vnodes, el, directed=rd$directed, strong_tie_q=rd$strong_q), error=function(e) NULL)
      if (is.null(net)) next
      if (network.size(net) < 5 || network.edgecount(net) < 3) next
      
      fml_str <- paste0("net ~ ", spec_terms[[spec_nm]])
      fml     <- as.formula(fml_str)
      
      # Use MPLE instead of MCMC where possible to guarantee speed
      fit <- tryCatch(
        ergm(fml, control = control.ergm(MCMLE.maxit = 1, MCMC.samplesize = 50)),
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
      village_coefs[[vill]] <- coefs
    }
    if (length(village_coefs) > 0) {
      combined <- do.call(rbind, village_coefs)
      rownames(combined) <- NULL
      all_results[[k]] <- combined
      k <- k + 1
    }
  }
}

if (length(all_results) > 0) {
  coef_tbl <- do.call(rbind, all_results)
  names(coef_tbl) <- gsub("Std\\. Error", "se", names(coef_tbl))
  names(coef_tbl) <- gsub("^Estimate$", "estimate", names(coef_tbl))
  write.csv(coef_tbl, file.path(outdir, "tables", "ergm_multiverse_coefficients.csv"), row.names = FALSE)
  
  # Pool
  pool_results <- list()
  grp_keys <- unique(paste(coef_tbl$representation, coef_tbl$spec, coef_tbl$term, sep = "|"))
  for (grp in grp_keys) {
    parts <- strsplit(grp, "\\|")[[1]]
    sub   <- coef_tbl[coef_tbl$representation == parts[1] & coef_tbl$spec == parts[2] & coef_tbl$term == parts[3], ]
    sub   <- sub[!is.na(sub$estimate) & !is.na(sub$se) & sub$se > 0, ]
    if (nrow(sub) < 2) next
    w   <- 1 / sub$se^2
    est <- sum(w * sub$estimate) / sum(w)
    se  <- sqrt(1 / sum(w))
    pool_results[[grp]] <- data.frame(
      representation  = parts[1], spec = parts[2], term = parts[3],
      pooled_estimate = round(est, 4), pooled_se = round(se, 4), n_villages = nrow(sub)
    )
  }
  pool_tbl <- do.call(rbind, pool_results)
  write.csv(pool_tbl, file.path(outdir, "tables", "ergm_pooled_estimates.csv"), row.names = FALSE)
}
