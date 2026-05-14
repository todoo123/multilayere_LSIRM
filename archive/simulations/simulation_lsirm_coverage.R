setwd('/Users/todoo/Desktop/학교/대학원/Research/joint_LSM')
source('my_LSIRM_cpp.R')

# =========================================================
# Coverage computation over repeated fits (STANDARD LSIRM)
# - Parameters: alpha, beta, (optional) log_gamma, distance(a,b)*gamma
# - coverage: TRUE value in 95% CI (inclusive boundaries)
# - repeats: fit -> compute -> store
# - output: coverage_lsirm.csv
# =========================================================

# ---------- helper: compute elementwise 95% CI coverage + width ----------
coverage_from_samples <- function(samples_mat, true_vec, prob = c(0.025, 0.975)) {
  stopifnot(is.matrix(samples_mat) || is.data.frame(samples_mat))
  samples_mat <- as.matrix(samples_mat)
  
  if (length(true_vec) != ncol(samples_mat)) {
    stop("true_vec length must match ncol(samples_mat).")
  }
  
  q_lo <- apply(samples_mat, 2, quantile, probs = prob[1], na.rm = TRUE)
  q_hi <- apply(samples_mat, 2, quantile, probs = prob[2], na.rm = TRUE)
  
  widths <- q_hi - q_lo
  
  cov_ind <- rep(NA_integer_, length(true_vec))
  ok <- !is.na(true_vec)
  cov_ind[ok] <- as.integer(q_lo[ok] <= true_vec[ok] & true_vec[ok] <= q_hi[ok])
  
  list(
    covered_ind = cov_ind,
    n_param     = sum(ok),
    n_covered   = sum(cov_ind[ok] == 1),
    pct_covered = if (sum(ok) == 0) NA_real_ else 100 * mean(cov_ind[ok] == 1),
    mean_width  = if (sum(ok) == 0) NA_real_ else mean(widths[ok]),
    sum_width   = if (sum(ok) == 0) 0 else sum(widths[ok])
  )
}

# =========================================================
# TRUE distance matrix helper (from TRUE positions)
# - A_true: [n, d], B_true: [P, d]
# - returns: [n, P] Euclidean distances
# =========================================================
make_D_true <- function(A_true, B_true) {
  A_true <- as.matrix(A_true)
  B_true <- as.matrix(B_true)
  n <- nrow(A_true); P <- nrow(B_true)
  d <- ncol(A_true)
  stopifnot(ncol(B_true) == d)
  
  D <- matrix(0, n, P)
  for (j in 1:P) {
    diff <- sweep(A_true, 2, B_true[j, ], "-")
    D[, j] <- sqrt(rowSums(diff^2))
  }
  D
}

# =========================================================
# Distance coverage with gamma scaling (STANDARD LSIRM)
# - a_samps: [S, n, d]
# - b_samps: [S, P, d]
# - log_gamma_samps: length S
# - D_true: [n, P]
# Coverage checks TRUE*(gamma_true) within posterior 95% CI of dist*(gamma_s)
# =========================================================
distance_coverage_lsirm <- function(a_samps, b_samps, log_gamma_samps,
                                    D_true, gamma_true = 1.0,
                                    prob = c(0.025, 0.975)) {
  stopifnot(length(dim(a_samps)) == 3, length(dim(b_samps)) == 3)
  S <- dim(a_samps)[1]
  n <- dim(a_samps)[2]
  d <- dim(a_samps)[3]
  P <- dim(b_samps)[2]
  stopifnot(dim(b_samps)[1] == S, dim(b_samps)[3] == d)
  stopifnot(all(dim(D_true) == c(n, P)))
  stopifnot(length(log_gamma_samps) == S)
  
  gamma_s <- exp(as.numeric(log_gamma_samps))  # length S
  true_scaled <- D_true * gamma_true           # n x P
  
  n_param <- n * P
  n_covered <- 0L
  sum_width <- 0.0
  
  for (j in 1:P) {
    dx <- a_samps[, , 1] - matrix(b_samps[, j, 1], nrow = S, ncol = n)
    if (d >= 2) {
      dy <- a_samps[, , 2] - matrix(b_samps[, j, 2], nrow = S, ncol = n)
      dist_s <- sqrt(dx^2 + dy^2)
    } else {
      dist_s <- abs(dx)
    }
    
    dist_s <- dist_s * matrix(gamma_s, nrow = S, ncol = n)
    
    q_lo <- apply(dist_s, 2, quantile, probs = prob[1], na.rm = TRUE)
    q_hi <- apply(dist_s, 2, quantile, probs = prob[2], na.rm = TRUE)
    
    sum_width <- sum_width + sum(q_hi - q_lo)
    
    tj <- true_scaled[, j]
    n_covered <- n_covered + sum(q_lo <= tj & tj <= q_hi, na.rm = TRUE)
  }
  
  list(
    n_param     = n_param,
    n_covered   = n_covered,
    pct_covered = 100 * (n_covered / n_param),
    mean_width  = sum_width / n_param,
    sum_width   = sum_width
  )
}

# =========================================================
# Coverage summary for one STANDARD LSIRM fit
# - fit$samples must include: alpha, beta, a, b, log_gamma (optional)
# - true_list: list(alpha, beta, A_true, B_true, gamma_true)  (A/B for distance)
# =========================================================
compute_coverage_onefit_lsirm <- function(fit, true_list, prob = c(0.025, 0.975)) {
  samps <- fit$samples
  out_rows <- list()
  
  # alpha
  if (!is.null(samps$alpha) && !is.null(true_list$alpha)) {
    tmp <- coverage_from_samples(samps$alpha, true_list$alpha, prob = prob)
    out_rows[["alpha"]] <- data.frame(
      param = "alpha", n_param = tmp$n_param, n_covered = tmp$n_covered,
      pct = tmp$pct_covered, mean_width = tmp$mean_width,
      sum_width = tmp$sum_width, stringsAsFactors = FALSE
    )
  }
  
  # beta (exclude ordinal dummy items with true value = 999)
  if (!is.null(samps$beta) && !is.null(true_list$beta)) {
    beta_true_use <- true_list$beta
    beta_true_use[beta_true_use == 999] <- NA
    tmp <- coverage_from_samples(samps$beta, beta_true_use, prob = prob)
    out_rows[["beta"]] <- data.frame(
      param = "beta", n_param = tmp$n_param, n_covered = tmp$n_covered,
      pct = tmp$pct_covered, mean_width = tmp$mean_width,
      sum_width = tmp$sum_width, stringsAsFactors = FALSE
    )
  }
  
  # (optional) log_gamma 자체 coverage가 필요하면 켜서 쓰면 됨
  if (!is.null(samps$log_gamma) && !is.null(true_list$log_gamma_true)) {
    tmp <- coverage_from_samples(matrix(samps$log_gamma, ncol = 1), true_list$log_gamma_true, prob = prob)
    out_rows[["log_gamma"]] <- data.frame(
      param = "log_gamma", n_param = tmp$n_param, n_covered = tmp$n_covered,
      pct = tmp$pct_covered, mean_width = tmp$mean_width,
      sum_width = tmp$sum_width, stringsAsFactors = FALSE
    )
  }
  
  # distance coverage: dist(a,b)*gamma
  if (!is.null(samps$a) && !is.null(samps$b) && !is.null(samps$log_gamma) &&
      !is.null(true_list$A_true) && !is.null(true_list$B_true)) {
    
    D_true <- make_D_true(true_list$A_true, true_list$B_true)
    gamma_true <- if (!is.null(true_list$gamma_true)) true_list$gamma_true else 1.0
    
    tmpd <- distance_coverage_lsirm(
      a_samps = samps$a,
      b_samps = samps$b,
      log_gamma_samps = samps$log_gamma,
      D_true = D_true,
      gamma_true = gamma_true,
      prob = prob
    )
    
    out_rows[["dist"]] <- data.frame(
      param = "dist", n_param = tmpd$n_param, n_covered = tmpd$n_covered,
      pct = tmpd$pct_covered, mean_width = tmpd$mean_width,
      sum_width = tmpd$sum_width, stringsAsFactors = FALSE
    )
  }
  
  do.call(rbind, out_rows)
}

# =========================================================
# 1) Column-wise thresholding binarizer (mean / quantile)
#    method ∈ {"mean","Q1","Q2","Q3","Q4"}  (Q2 = median)
#    strict: TRUE면 ">" , FALSE면 ">="
# =========================================================
binarize_by_colthreshold <- function(X,
                                     method = c("mean","Q1","Q2","Q3","Q4"),
                                     strict = TRUE,
                                     probs_map = c(Q1=0.25, Q2=0.50, Q3=0.75, Q4=0.90)){
  method <- match.arg(method)
  X <- as.matrix(X)
  
  # threshold vector per column
  thr <- switch(
    method,
    mean = colMeans(X, na.rm = TRUE),
    Q1   = apply(X, 2, quantile, probs = probs_map["Q1"], na.rm = TRUE, names = FALSE, type = 7),
    Q2   = apply(X, 2, quantile, probs = probs_map["Q2"], na.rm = TRUE, names = FALSE, type = 7),
    Q3   = apply(X, 2, quantile, probs = probs_map["Q3"], na.rm = TRUE, names = FALSE, type = 7),
    Q4   = apply(X, 2, quantile, probs = probs_map["Q4"], na.rm = TRUE, names = FALSE, type = 7)
  )
  
  if(strict){
    X_bin <- sweep(X, 2, thr, FUN = ">")
  } else {
    X_bin <- sweep(X, 2, thr, FUN = ">=")
  }
  
  storage.mode(X_bin) <- "integer"
  list(X_bin = X_bin, threshold = thr, method = method, strict = strict, probs_map = probs_map)
}

# =========================================================
# 2) 4-layer -> (optionally) different binarization rules per layer
#    - bin_method: Y_bin에 대해서도 재이진화하고 싶으면 설정 (보통 "none")
#    - con/cnt/ord_method: "mean" or "Q1/Q2/Q3/Q4"
#    - ord_input: ordinal을 무엇을 기준으로 이진화할지:
#        "raw"  : Y_ord (1..K) 자체를 threshold
#        "score": (Y_ord >= ord_top) 같은 방식으로 먼저 0/1 만들고 싶으면 아래 옵션 사용
# =========================================================
make_binarized_4layer_for_lsirm <- function(Y_bin, Y_con, Y_cnt, Y_ord,
                                            bin_method = c("none","mean","Q1","Q2","Q3","Q4"),
                                            con_method = c("mean","Q1","Q2","Q3","Q4"),
                                            cnt_method = c("mean","Q1","Q2","Q3","Q4"),
                                            ord_method = c("mean","Q1","Q2","Q3","Q4"),
                                            strict = TRUE,
                                            ord_input = c("raw","topbox"),
                                            ord_top = NULL,
                                            probs_map = c(Q1=0.25, Q2=0.50, Q3=0.75, Q4=0.90)){
  
  bin_method <- match.arg(bin_method)
  con_method <- match.arg(con_method)
  cnt_method <- match.arg(cnt_method)
  ord_method <- match.arg(ord_method)
  ord_input  <- match.arg(ord_input)
  
  # (A) Binary layer
  Y_bin1 <- as.matrix(Y_bin)
  storage.mode(Y_bin1) <- "integer"
  
  bin_info <- NULL
  if(bin_method != "none"){
    out_bin <- binarize_by_colthreshold(Y_bin1, method = bin_method, strict = strict, probs_map = probs_map)
    Y_bin1  <- out_bin$X_bin
    bin_info <- out_bin
  }
  
  # (B) Continuous layer
  out_con <- binarize_by_colthreshold(Y_con, method = con_method, strict = strict, probs_map = probs_map)
  Y_con_bin <- out_con$X_bin
  
  # (C) Count layer
  out_cnt <- binarize_by_colthreshold(Y_cnt, method = cnt_method, strict = strict, probs_map = probs_map)
  Y_cnt_bin <- out_cnt$X_bin
  
  # (D) Ordinal layer
  if(ord_input == "raw"){
    out_ord <- binarize_by_colthreshold(Y_ord, method = ord_method, strict = strict, probs_map = probs_map)
    Y_ord_bin <- out_ord$X_bin
  } else {
    # topbox: 예) 4~5면 1, 그 외 0 같은 방식 먼저 만든 다음,
    #         그 결과가 이미 0/1이면 thresholding 불필요. (원하면 binarize 적용도 가능)
    if(is.null(ord_top)) ord_top <- max(Y_ord, na.rm = TRUE)  # 기본: 최상위 범주만 1
    Y_top <- (as.matrix(Y_ord) >= ord_top) * 1L
    storage.mode(Y_top) <- "integer"
    
    # 이미 0/1이므로 그대로 사용 (원하면 아래 한 줄로 재-threshold 가능)
    # out_ord <- binarize_by_colthreshold(Y_top, method = ord_method, strict = strict, probs_map = probs_map)
    # Y_ord_bin <- out_ord$X_bin
    out_ord <- list(X_bin = Y_top, threshold = rep(ord_top, ncol(Y_top)), method = paste0("topbox(>=", ord_top, ")"))
    Y_ord_bin <- Y_top
  }
  
  # (E) Combine
  Y_bin_all <- cbind(Y_bin1, Y_con_bin, Y_cnt_bin, Y_ord_bin)
  storage.mode(Y_bin_all) <- "integer"
  
  list(
    Y_bin_all = Y_bin_all,
    info = list(
      bin = bin_info,
      con = out_con,
      cnt = out_cnt,
      ord = out_ord
    )
  )
}


# =========================================================
# Main loop: repeat fitting -> coverage -> append (STANDARD LSIRM)
# =========================================================

# (1) repetitions
n_rep <- 10  # 원하는 만큼 변경

# ---------------------------------------------------------
# (2) TRUE values (시뮬레이션일 때만 의미 있음)
# alpha_true: length P
# beta_true : length P
# A_true: [n,d], B_true: [P,d]
# gamma_true: scalar (fix_gamma=TRUE면 보통 1)
# ---------------------------------------------------------
alpha_true
beta4_true_dummy <- rep(999,10)
beta_true <- c(beta1_true, beta2_true, beta3_true, beta4_true_dummy)
B_true <- rbind(B1_true, B2_true, B3_true, B4_true)

true_list_lsirm <- list(
  alpha = alpha_true,        # length P
  beta  = beta_true,         # length P
  A_true = A_true,           # n x d
  B_true = B_true,           # P x d
  gamma_true = gamma_true    # usually 1.0 if you fix gamma
  # log_gamma_true = log(gamma_true)  # 필요하면 추가
)

# ---------------------------------------------------------
# (3) Fit settings (STANDARD LSIRM)
# ---------------------------------------------------------
fit_args_lsirm <- list(
  d = 2,
  n_iter = 30000,
  burnin = 10000,
  thin = 5,
  prop_sd = list(
    alpha = 0.70,
    beta  = 0.50,
    log_gamma = 0.05,
    a = 0.50,
    b = 0.30
  ),
  verbose = TRUE,
  fix_gamma = TRUE
)

# (4) Storage
coverage_all <- list()

# =========================================================
# Dichotomization loop + Replication loop + Coverage saving
# =========================================================

criterion <- c("mean", "Q1", "Q2", "Q3", "Q4")

coverage_all_crt <- list()  # criterion별 coverage_df 저장

for (i in seq_along(criterion)) {
  crt <- criterion[i]
  cat("\n=========================================\n")
  cat(sprintf("Dichotomization criterion: %s (%d/%d)\n", crt, i, length(criterion)))
  cat("=========================================\n")
  
  # ----- 1) Build dichotomized Y_bin_all_4 -----
  prep <- make_binarized_4layer_for_lsirm(
    Y_bin = Y_bin, Y_con = Y_con, Y_cnt = Y_cnt, Y_ord = Y_ord,
    bin_method = "none",
    con_method = crt,
    cnt_method = crt,
    ord_method = crt,
    strict = TRUE,
    ord_input = "raw"
  )
  Y_bin_all_4 <- prep$Y_bin_all
  storage.mode(Y_bin_all_4) <- "integer"
  
  # ----- 2) Replication loop (YOUR BLOCK, wrapped) -----
  coverage_all <- list()
  
  for (r in 1:n_rep) {
    cat("\n==============================\n")
    cat(sprintf("Replication %d / %d\n", r, n_rep))
    cat("==============================\n")
    
    # --- Fit model ---
    fit_r <- do.call(
      lsirm_basic,
      c(list(Y_bin = Y_bin_all_4), fit_args_lsirm)
    )
    
    # Ensure samples nested
    if (is.null(fit_r$samples)) fit_r$samples <- fit_r
    
    # --- Compute coverage ---
    cov_r <- compute_coverage_onefit_lsirm(fit_r, true_list_lsirm)
    cov_r$rep <- r
    coverage_all[[r]] <- cov_r
  }
  
  coverage_df <- do.call(rbind, coverage_all)
  
  # (optional) overall coverage within each rep
  overall_by_rep <- aggregate(
    cbind(n_param, n_covered, sum_width) ~ rep,
    data = coverage_df,
    FUN = sum
  )
  overall_by_rep$pct <- 100 * overall_by_rep$n_covered / overall_by_rep$n_param
  overall_by_rep$mean_width <- overall_by_rep$sum_width / overall_by_rep$n_param
  overall_by_rep$param <- "OVERALL"
  overall_by_rep <- overall_by_rep[, c("rep", "param", "n_param", "n_covered", "pct", "mean_width", "sum_width")]
  
  coverage_df <- rbind(
    coverage_df[, c("rep", "param", "n_param", "n_covered", "pct", "mean_width", "sum_width")],
    overall_by_rep
  )
  
  coverage_df <- coverage_df[order(coverage_df$rep, coverage_df$param), ]
  
  # ----- 3) add criterion label + store -----
  coverage_df$criterion <- crt
  coverage_df <- coverage_df[, c("criterion", "rep", "param", "n_param", "n_covered", "pct", "mean_width")]
  
  coverage_all_crt[[crt]] <- coverage_df
}

# ----- 4) combine all criteria results -----
coverage_df_all <- do.call(rbind, coverage_all_crt)
coverage_df_all <- coverage_df_all[order(coverage_df_all$criterion, coverage_df_all$rep, coverage_df_all$param), ]

# ----- 5) save -----
write.csv(coverage_df_all, file = "coverage_lsirm_by_dichotomization.csv", row.names = FALSE)
cat("\nSaved: coverage_lsirm_by_dichotomization.csv\n")
print(head(coverage_df_all, 50))

