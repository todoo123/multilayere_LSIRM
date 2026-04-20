library(igraph)
library(network)
library(brainGraph)
library(MASS)
library(ggplot2)

# traceplot true 값 복원하는지 확인하는 용도
# scalar version
plot_trace_scalar <- function(x, true = NA, main = "", transform = identity,
                              inv_transform = identity) {
  z <- transform(x)
  q <- quantile(z, c(.025, .975), na.rm = TRUE)
  m <- mean(z, na.rm = TRUE)
  tv <- if(!is.na(true)) transform(true) else NA
  
  ts.plot(z, main = main)
  abline(h = c(m, q, tv),
         col = c("darkgreen", "blue", "blue", "red"),
         lwd = 2,
         lty = c(1, 3, 3, 2))
  
  legend(
    "topright",
    legend = c("Posterior mean", "95% credible interval", "True value"),
    col    = c("darkgreen", "blue", "red"),
    lwd    = 2,
    lty    = c(1, 3, 2),
    bty    = "n",
    cex    = 0.8
  )
}

# vector version
plot_trace_vec <- function(samples_mat, true_vec, name = "param", mfrow = c(2,2), leg = leg){
  stopifnot(is.matrix(samples_mat) || is.data.frame(samples_mat))
  samples_mat <- as.matrix(samples_mat)
  if(length(true_vec) != ncol(samples_mat)) stop("true_vec length must match ncol(samples_mat).")
  
  par(mfrow = mfrow)
  for(i in seq_len(ncol(samples_mat))) {
    x <- samples_mat[, i]
    q <- quantile(x, c(.025, .975), na.rm = TRUE)
    
    ts.plot(x, main = sprintf("%s_%d", name, i))
    abline(h = c(mean(x, na.rm = TRUE), q, true_vec[i]),
           col = c("darkgreen", "blue", "blue", "red"),
           lwd = 2,
           lty = c(1, 3, 3, 2))
    do.call(legend, leg)
  }
}


# parameter 모아 놓고 traceplot 으로 boxplot 그리고 true 값 복원했는지 보는 용도
plot_group_box_ci <- function(samples_mat, true_vec, prefix,
                              per_page = 24, ncol = 1,
                              show_true_points = TRUE,
                              add_ci = TRUE) {
  samples_mat <- as.matrix(samples_mat)
  stopifnot(length(true_vec) == ncol(samples_mat))
  
  p  <- ncol(samples_mat)
  nm <- sprintf("%s_%d", prefix, seq_len(p))
  
  # sort by median
  med <- apply(samples_mat, 2, median, na.rm = TRUE)
  ord <- order(med)
  samples_mat <- samples_mat[, ord, drop = FALSE]
  true_vec    <- true_vec[ord]
  nm          <- nm[ord]
  
  # 95% CrI
  q95 <- NULL
  if (add_ci) {
    q95 <- apply(samples_mat, 2, quantile, probs = c(.025, .975), na.rm = TRUE)
  }
  
  # paging
  idx_pages <- split(seq_len(p), ceiling(seq_len(p) / per_page))
  
  for (pg in seq_along(idx_pages)) {
    idx <- idx_pages[[pg]]
    
    sam_list <- lapply(idx, function(k) samples_mat[, k])
    names_pg <- nm[idx]
    true_pg  <- true_vec[idx]
    q95_pg   <- if (add_ci) q95[, idx, drop = FALSE] else NULL
    
    op <- par(no.readonly = TRUE)
    par(mfrow = c(1,1), mar = c(8,4,3,1))
    
    bp <- boxplot(
      sam_list,
      names   = names_pg,
      las     = 2,
      outline = FALSE,
      main    = sprintf("%s (page %d/%d) - sorted by median", prefix, pg, length(idx_pages)),
      ylab    = "value"
    )
    
    x_pos <- seq_along(idx)
    
    # add 95% CrI as error bars (green)
    if (add_ci) {
      segments(x_pos, q95_pg[1, ], x_pos, q95_pg[2, ], lwd = 2, col = "blue")
      segments(x_pos - 0.15, q95_pg[1, ], x_pos + 0.15, q95_pg[1, ], lwd = 2, col = "blue")
      segments(x_pos - 0.15, q95_pg[2, ], x_pos + 0.15, q95_pg[2, ], lwd = 2, col = "blue")
    }
    
    # true values
    if (show_true_points) {
      points(x_pos, true_pg, pch = 19, col = "red")
    }
    
    legend(
      "topleft",
      legend = c("Posterior (boxplot)", "95% credible interval", "True value"),
      lwd    = c(NA, 2, NA),
      pch    = c(NA, NA, 19),
      col    = c("black", "blue", "red"),
      bty    = "n",
      cex    = 0.9
    )
    
    par(op)
  }
  
  invisible(list(order = ord, median = med, q95 = q95))
}





# cluster 개수 넣고 prop_matrix 로 sampling 하는 함수
sampling_prox_dist <- function(cluster_num, prob_mat, prox = 'resistance_distance', output = F){
  par(mfrow = c(1,2))
  sizes <- cluster_num
  pref.matrix <- prob_mat
  if(prox == 'resistance_distance'){
    graph_samp <- sample_sbm(sum(sizes), pref.matrix, block.sizes = sizes)
    A <- as_adjacency_matrix(graph_samp, sparse = FALSE)
    D <- diag(rowSums(A), nrow = dim(A)[1])
    L <- D - A
    L_ginv <- ginv(L)
    volG <- sum(rowSums(A))
    diagLp <- diag(L_ginv)
    proximity <- outer(diagLp, diagLp, `+`) - 2 * L_ginv
    # log 변환
    proximity <- log(proximity)
    plot(graph_samp,
         vertex.color = membership(cluster_louvain(graph_samp)),
         vertex.label = V(graph_samp)$name,
         vertex.size = 9,
         layout = layout_with_fr)
    plot(density(proximity[proximity != 0]))
    
  } else if(prox == 'communicability'){
    graph_samp <- sample_sbm(sum(sizes), pref.matrix, block.sizes = sizes)
    if(!is_connected(graph_samp)){
      while(!is_connected(graph_samp)){
        graph_samp <- sample_sbm(sum(sizes), pref.matrix, block.sizes = sizes)
      }
    }
    proximity <- log(communicability(graph_samp))
    
    plot(graph_samp,
         vertex.color = membership(cluster_louvain(graph_samp)),
         vertex.label = V(graph_samp)$name,
         vertex.size = 9,
         layout = layout_with_fr)
    plot(density(proximity[proximity != 0]))
  }
  par(mfrow = c(1,1))
  if(output == T){
    result <- list()
    result$proximity <- proximity
    result$g <- graph_samp
    return(result)
  }
}


# true - estimated result comparison
result_assess <- function(result, true_value, model, comparison_plot = FALSE){
  
  # 전처리
  diag(true_value) <- 0
  
  if(model == 'MNLPM'){
    u2_post <- colMeans(result$samples$u2, dim = 1)
    alpha2_post <- mean(result$samples$alpha2)
    theta_post <- colMeans(result$samples$theta2, dim = 1)
    kappa2_post <- mean(result$samples$kappa2) 
    sigma2_2_post <- mean(result$samples$sigma2_2)
    beta_post <- mean(result$samples$beta2)
    theta_sum <- matrix(nrow = length(theta_post), ncol = length(theta_post))
    
    for(i in 1:length(theta_post)){
      for(j in 1:length(theta_post)){
        theta_sum[i,j] <- theta_post[i] + theta_post[j]
      }
    }
    D <- dist(u2_post, upper = TRUE, diag = TRUE)
    D <- as.matrix(D)
    
    eta_post = alpha2_post + theta_sum - beta_post * D
    mu_log = eta_post - kappa2_post / 2.0
    diag(mu_log) <- 0
    
    estimation <- mu_log
    true_value <- log(true_value)
  }else if(model == 'LSM_conti'){
    u2_post <- colMeans(result$samples$z, dim = 1)
    alpha2_post <- mean(result$samples$alpha)
    kappa2_post <- mean(result$samples$kappa)
    D <- dist(u2_post, upper = TRUE, diag = TRUE)
    D <- as.matrix(D)
    
    eta_post = alpha2_post + D
    mu_log = eta_post - kappa2_post / 2.0
    estimation <- mu_log
    true_value <- log(true_value)
  }else if(model == 'LSM_conti_mixture_add_only'){
    u2_post <- colMeans(aperm(result$samples$Z, c(3,1,2)), dim = 1)
    theta_post <- colMeans(aperm(result$samples$theta, c(2,1)), dim = 1)
    D <- dist(u2_post, upper = TRUE, diag = TRUE)
    D <- as.matrix(D)
    theta_mat <- outer(theta_post, theta_post, "+")
    sig_0_post <- mean(result$samples$sigma_0_sq)
    
    eta_post = theta_mat - D
    estimation <- eta_post
    true_value <- log(true_value)
  }
  
  
  par(mfrow =c(1,2))
  
  if(comparison_plot == TRUE){
    d1 <- density(estimation)
    d2 <- density(true_value)
    
    # 두 커브가 커버하는 x 범위 합치기
    xmin <- min(d1$x, d2$x)
    xmax <- max(d1$x, d2$x)
    
    plot(d1, xlim = c(xmin, xmax), main = "estimation vs true", col = "black", lwd = 2)
    lines(d2, col = "red", lwd = 2)
    legend("topright", legend = c("estimation", "true value"), col = c("black", "red"), 
           lwd = 2)
    }
  hist(estimation - true_value, main = 'residual distribution')
  resid <- true_value - estimation
  diag(resid) <- 0
  print(paste0("residual sum: ", sum(resid)))
  par(mfrow = c(1,1))
}



# =============================================================================
# Column-wise dichotomization helpers (for unilayered LSIRM 비교용)
# -----------------------------------------------------------------------------
# binarize_by_colthreshold: 각 컬럼을 mean / Q1..Q4 기준으로 이진화
# make_binarized_multilayer_for_lsirm: bin/con/cnt/ord1/ord2 계층을 묶어 하나의
#   binary matrix로 반환 (unilayered LSIRM의 Y_bin_all로 바로 투입 가능)
# =============================================================================
binarize_by_colthreshold <- function(X,
                                     method = c("mean","Q1","Q2","Q3","Q4"),
                                     strict = TRUE,
                                     probs_map = c(Q1=0.25, Q2=0.50, Q3=0.75, Q4=0.90)) {
  method <- match.arg(method)
  X <- as.matrix(X)
  thr <- switch(
    method,
    mean = colMeans(X, na.rm = TRUE),
    Q1   = apply(X, 2, quantile, probs = probs_map["Q1"], na.rm = TRUE, names = FALSE, type = 7),
    Q2   = apply(X, 2, quantile, probs = probs_map["Q2"], na.rm = TRUE, names = FALSE, type = 7),
    Q3   = apply(X, 2, quantile, probs = probs_map["Q3"], na.rm = TRUE, names = FALSE, type = 7),
    Q4   = apply(X, 2, quantile, probs = probs_map["Q4"], na.rm = TRUE, names = FALSE, type = 7)
  )
  X_bin <- if (strict) sweep(X, 2, thr, FUN = ">") else sweep(X, 2, thr, FUN = ">=")
  storage.mode(X_bin) <- "integer"
  list(X_bin = X_bin, threshold = thr, method = method, strict = strict)
}

# Y_bin / Y_con / Y_cnt / Y_ord1 / Y_ord2 중 NULL이 아닌 계층만 묶어 이진화 후 cbind.
# 반환 list:
#   Y_bin_all    : 합쳐진 binary matrix (n × P_total)
#   layer_labels : 각 컬럼이 어느 layer에서 왔는지 (길이 P_total)
#   col_names    : 원 컬럼 이름
#   info         : 각 layer별 threshold 정보
make_binarized_multilayer_for_lsirm <- function(Y_bin  = NULL,
                                                Y_con  = NULL,
                                                Y_cnt  = NULL,
                                                Y_ord1 = NULL,
                                                Y_ord2 = NULL,
                                                bin_method  = c("none","mean","Q1","Q2","Q3","Q4"),
                                                con_method  = c("mean","Q1","Q2","Q3","Q4"),
                                                cnt_method  = c("mean","Q1","Q2","Q3","Q4"),
                                                ord1_method = c("mean","Q1","Q2","Q3","Q4"),
                                                ord2_method = c("mean","Q1","Q2","Q3","Q4"),
                                                strict = TRUE,
                                                ord_input = c("raw","topbox"),
                                                ord_top = NULL,
                                                probs_map = c(Q1=0.25, Q2=0.50, Q3=0.75, Q4=0.90)) {

  bin_method  <- match.arg(bin_method)
  con_method  <- match.arg(con_method)
  cnt_method  <- match.arg(cnt_method)
  ord1_method <- match.arg(ord1_method)
  ord2_method <- match.arg(ord2_method)
  ord_input   <- match.arg(ord_input)

  parts     <- list()
  col_parts <- list()
  info      <- list()

  # --- Binary (optional re-binarize) ---
  if (!is.null(Y_bin) && ncol(as.matrix(Y_bin)) > 0) {
    Yb <- as.matrix(Y_bin)
    storage.mode(Yb) <- "integer"
    if (bin_method != "none") {
      out <- binarize_by_colthreshold(Yb, method = bin_method, strict = strict, probs_map = probs_map)
      Yb  <- out$X_bin
      info$bin <- out
    }
    parts$bin     <- Yb
    col_parts$bin <- colnames(Yb)
  }

  # --- Continuous ---
  if (!is.null(Y_con) && ncol(as.matrix(Y_con)) > 0) {
    out <- binarize_by_colthreshold(Y_con, method = con_method, strict = strict, probs_map = probs_map)
    info$con      <- out
    parts$con     <- out$X_bin
    col_parts$con <- colnames(out$X_bin)
  }

  # --- Count ---
  if (!is.null(Y_cnt) && ncol(as.matrix(Y_cnt)) > 0) {
    out <- binarize_by_colthreshold(Y_cnt, method = cnt_method, strict = strict, probs_map = probs_map)
    info$cnt      <- out
    parts$cnt     <- out$X_bin
    col_parts$cnt <- colnames(out$X_bin)
  }

  # --- Ordinal 1 / 2 ---
  ord_specs <- list(
    list(key = "ord1", Y = Y_ord1, method = ord1_method),
    list(key = "ord2", Y = Y_ord2, method = ord2_method)
  )
  for (spec in ord_specs) {
    Y_o <- spec$Y
    if (!is.null(Y_o) && ncol(as.matrix(Y_o)) > 0) {
      if (ord_input == "raw") {
        out <- binarize_by_colthreshold(Y_o, method = spec$method,
                                        strict = strict, probs_map = probs_map)
        info[[spec$key]]      <- out
        parts[[spec$key]]     <- out$X_bin
        col_parts[[spec$key]] <- colnames(out$X_bin)
      } else {  # "topbox"
        top_cut <- if (is.null(ord_top)) max(Y_o, na.rm = TRUE) else ord_top
        Y_top   <- (as.matrix(Y_o) >= top_cut) * 1L
        storage.mode(Y_top) <- "integer"
        info[[spec$key]]      <- list(X_bin = Y_top,
                                      threshold = rep(top_cut, ncol(Y_top)),
                                      method = paste0("topbox(>=", top_cut, ")"))
        parts[[spec$key]]     <- Y_top
        col_parts[[spec$key]] <- colnames(Y_top)
      }
    }
  }

  if (length(parts) == 0) stop("모든 layer가 비어 있습니다.")

  Y_bin_all <- do.call(cbind, parts)
  storage.mode(Y_bin_all) <- "integer"
  layer_labels <- unlist(lapply(names(parts), function(k) rep(k, ncol(parts[[k]]))))
  col_names    <- unlist(col_parts)
  colnames(Y_bin_all) <- col_names

  list(
    Y_bin_all    = Y_bin_all,
    layer_labels = layer_labels,
    col_names    = col_names,
    info         = info
  )
}


# =============================================================================
# K-means clustering on item-position matrix b (rows = items, cols = latent dims)
# -----------------------------------------------------------------------------
# Input:
#   b           : (p × d) matrix; rownames = item names (없으면 자동 생성)
#   b_layer     : (optional) length-p character vector of layer labels
#                 (e.g. c("bin","con","cnt","ord1")). shape 인코딩에 사용.
#                 NULL이면 모든 item이 같은 모양으로 표시됨.
#   K_range     : 후보 K 범위. K가 직접 주어지면 무시됨.
#   K           : (optional) fix K. NULL이면 silhouette 최댓값으로 자동 선택.
#   plot_dir    : PDF 저장 경로. NULL이면 현재 device로 출력.
#   file_prefix : 저장 파일 prefix (plot_dir 지정 시에만 사용)
#   seed        : reproducibility
#   verbose     : TRUE면 console 요약 출력
#
# Returns (invisibly): list(K_best, km, cluster_summary, sil, sil_scores, wss_scores)
# =============================================================================
kmeans_cluster_b <- function(b,
                             b_layer   = NULL,
                             K_range   = 2:8,
                             K         = NULL,
                             plot_dir  = NULL,
                             file_prefix = "kmeans_b",
                             seed      = 42,
                             verbose   = TRUE) {

  if (!requireNamespace("cluster", quietly = TRUE)) {
    install.packages("cluster", repos = "https://cloud.r-project.org")
  }
  stopifnot(is.matrix(b) || is.data.frame(b))
  b <- as.matrix(b)
  p <- nrow(b)
  if (p < 3) stop("b must have at least 3 rows for clustering.")

  b_names <- rownames(b)
  if (is.null(b_names)) b_names <- paste0("item", seq_len(p))

  if (is.null(b_layer)) {
    b_layer <- rep("item", p)
  } else if (length(b_layer) != p) {
    stop("length(b_layer) must equal nrow(b).")
  }

  # ---- 1. K selection (silhouette + within-SS) -----------------------------
  set.seed(seed)
  sil_scores <- sapply(K_range, function(k) {
    km <- kmeans(b, centers = k, nstart = 25, iter.max = 100)
    if (length(unique(km$cluster)) < 2) return(NA_real_)
    mean(cluster::silhouette(km$cluster, dist(b))[, "sil_width"])
  })
  set.seed(seed)
  wss_scores <- sapply(K_range, function(k) {
    kmeans(b, centers = k, nstart = 25, iter.max = 100)$tot.withinss
  })

  if (is.null(K)) {
    K_best <- K_range[which.max(sil_scores)]
  } else {
    K_best <- as.integer(K)
    if (!(K_best %in% K_range)) K_range <- sort(unique(c(K_range, K_best)))
  }

  if (verbose) {
    cat("\n=== K-means diagnostic ===\n")
    print(data.frame(K          = K_range,
                     silhouette = round(sil_scores, 3),
                     within_SS  = round(wss_scores, 3)))
    cat(sprintf("\nSelected K = %d (silhouette = %.3f)\n",
                K_best,
                sil_scores[K_range == K_best]))
  }

  # ---- 2. Final k-means at K_best ------------------------------------------
  set.seed(seed)
  km_final   <- kmeans(b, centers = K_best, nstart = 50, iter.max = 200)
  cluster_id <- km_final$cluster
  sil_final  <- cluster::silhouette(cluster_id, dist(b))
  sil_mean   <- mean(sil_final[, "sil_width"])

  cluster_summary <- data.frame(
    item    = b_names,
    layer   = b_layer,
    cluster = cluster_id,
    sil_w   = round(sil_final[, "sil_width"], 3)
  )
  cluster_summary <- cluster_summary[order(cluster_summary$cluster,
                                           -cluster_summary$sil_w), ]

  if (verbose) {
    cat(sprintf("\n=== K-means result (K=%d, avg silhouette=%.3f) ===\n",
                K_best, sil_mean))
    print(cluster_summary, row.names = FALSE)
  }

  # ---- 3. Plotting ----------------------------------------------------------
  cluster_pal <- c("#E41A1C","#377EB8","#4DAF4A","#984EA3",
                   "#FF7F00","#FFFF33","#A65628","#F781BF")
  cluster_pal <- rep(cluster_pal, length.out = K_best)

  unique_layers <- unique(b_layer)
  shape_pool    <- c(21, 22, 24, 23, 25, 3, 4, 8)
  layer_pch     <- setNames(shape_pool[seq_along(unique_layers)],
                            unique_layers)

  # (a) diagnostic
  if (!is.null(plot_dir)) {
    if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
    pdf(file.path(plot_dir, paste0(file_prefix, "_diagnostic.pdf")),
        width = 10, height = 5)
  }
  op <- par(no.readonly = TRUE)
  par(mfrow = c(1,2), mar = c(4,4,3,1))
  plot(K_range, sil_scores, type = "b", pch = 19, col = "steelblue", lwd = 2,
       xlab = "K (clusters)", ylab = "Avg silhouette width",
       main = "Silhouette vs K")
  abline(v = K_best, col = "red", lty = 2)
  plot(K_range, wss_scores, type = "b", pch = 19, col = "darkorange", lwd = 2,
       xlab = "K (clusters)", ylab = "Total within-cluster SS",
       main = "Elbow: within-SS vs K")
  abline(v = K_best, col = "red", lty = 2)
  par(op)
  if (!is.null(plot_dir)) dev.off()

  # (b) cluster biplot (using first 2 dims of b)
  if (ncol(b) >= 2) {
    if (!is.null(plot_dir)) {
      pdf(file.path(plot_dir,
                    sprintf("%s_clusters_K%d.pdf", file_prefix, K_best)),
          width = 11, height = 9)
    }
    op <- par(no.readonly = TRUE)
    par(mfrow = c(1,1), mar = c(4,4,3,8), xpd = FALSE)
    xr <- range(b[,1]); yr <- range(b[,2])
    dx <- diff(xr); dy <- diff(yr); expand <- 0.12
    plot(b[, 1:2],
         pch = layer_pch[b_layer], bg = cluster_pal[cluster_id],
         col = "black", cex = 1.6,
         xlim = xr + c(-1,1)*expand*dx,
         ylim = yr + c(-1,1)*expand*dy,
         xlab = "Dim1", ylab = "Dim2",
         main = sprintf("K-means on b (K=%d, avg silhouette=%.3f)",
                        K_best, sil_mean))
    text(b[,1], b[,2], labels = b_names, pos = 4, cex = 0.6, offset = 0.4)
    points(km_final$centers[, 1:2], pch = 4, cex = 2.5, lwd = 3,
           col = cluster_pal[1:K_best])
    par(xpd = TRUE)
    legend("topright", inset = c(-0.22, 0),
           title = "Cluster",
           legend = paste0("C", 1:K_best),
           pch = 21, pt.bg = cluster_pal, pt.cex = 1.5, bty = "n")
    legend("right", inset = c(-0.22, 0),
           title = "Layer",
           legend = unique_layers,
           pch = layer_pch, pt.bg = "gray80", pt.cex = 1.3, bty = "n")
    par(op)
    if (!is.null(plot_dir)) dev.off()
  }

  # (c) silhouette plot
  if (!is.null(plot_dir)) {
    pdf(file.path(plot_dir,
                  sprintf("%s_silhouette_K%d.pdf", file_prefix, K_best)),
        width = 8, height = 9)
  }
  op <- par(no.readonly = TRUE)
  par(mar = c(4,4,3,1))
  plot(sil_final, col = cluster_pal[1:K_best], border = NA,
       main = sprintf("Silhouette (K=%d, avg=%.3f)", K_best, sil_mean))
  abline(v = sil_mean, lty = 2, col = "red")
  par(op)
  if (!is.null(plot_dir)) dev.off()

  # Save cluster assignment CSV
  if (!is.null(plot_dir)) {
    write.csv(cluster_summary,
              file.path(plot_dir,
                        sprintf("%s_clusters_K%d.csv", file_prefix, K_best)),
              row.names = FALSE)
    if (verbose) {
      cat(sprintf("\n-> Plots & CSV saved to: %s\n", plot_dir))
      cat(sprintf("   * %s_diagnostic.pdf\n", file_prefix))
      cat(sprintf("   * %s_clusters_K%d.pdf\n", file_prefix, K_best))
      cat(sprintf("   * %s_silhouette_K%d.pdf\n", file_prefix, K_best))
      cat(sprintf("   * %s_clusters_K%d.csv\n", file_prefix, K_best))
    }
  }

  invisible(list(
    K_best          = K_best,
    km              = km_final,
    cluster_summary = cluster_summary,
    sil             = sil_final,
    sil_mean        = sil_mean,
    sil_scores      = setNames(sil_scores, K_range),
    wss_scores      = setNames(wss_scores, K_range)
  ))
}


rank_svd_compare_refactored <- function(D_true, D_hat, k = 3,
                                        plotting = TRUE,
                                        center = TRUE,
                                        rank_mode = c("row", "col")) {
  # ------------------------------------------------------------
  # Compare two n x p distance matrices via rank-based SVD.
  #
  # Inputs:
  #   D_true, D_hat : n x p distance matrices
  #   k             : number of leading components
  #   plotting      : scree plots (TRUE/FALSE)
  #   center        : TRUE/FALSE
  #   rank_mode     : "row" (row-wise rank) or "col" (col-wise rank)
  #
  # Centering rule (as requested):
  #   - rank_mode="row" -> column-wise centering only (remove col means)
  #   - rank_mode="col" -> row-wise centering only (remove row means)
  #
  # Output:
  #   list including rank matrices, SVD results, scree info, subspace similarity,
  #   and per-respondent / per-item cosine similarities.
  # ------------------------------------------------------------
  
  # ---- checks ----
  if(!is.matrix(D_true) || !is.matrix(D_hat)) stop("D_true and D_hat must be matrices.")
  if(any(dim(D_true) != dim(D_hat))) stop("D_true and D_hat must have identical dimensions.")
  if(any(!is.finite(D_true)) || any(!is.finite(D_hat))) stop("Distance matrices must be finite (no NA/Inf).")
  
  n <- nrow(D_true); p <- ncol(D_true)
  rank_mode <- match.arg(rank_mode)
  
  if(k < 1) stop("k must be >= 1.")
  if(k > min(n, p)) stop("k cannot exceed min(n, p).")
  
  # ---- rank constructors ----
  row_rank <- function(D){
    # for each respondent i: rank items by distance
    t(apply(D, 1, function(x) rank(x, ties.method = "average")))
  }
  col_rank <- function(D){
    # for each item j: rank respondents by distance
    apply(D, 2, function(x) rank(x, ties.method = "average"))
  }
  
  # ---- centering (only one side, depending on rank_mode) ----
  center_one_side <- function(M, mode){
    if(!center) return(M)
    if(mode == "row"){
      # remove column means only
      M - matrix(colMeans(M), nrow(M), ncol(M), byrow = TRUE)
    } else {
      # mode == "col": remove row means only
      M - rowMeans(M)
    }
  }
  
  # ---- cosine for rows ----
  row_cosine <- function(A, B){
    num <- rowSums(A * B)
    den <- sqrt(rowSums(A^2) * rowSums(B^2))
    out <- num / den
    out[den == 0] <- NA_real_
    out
  }
  
  # ---- 1) rank matrices ----
  if(rank_mode == "row"){
    R_true <- row_rank(D_true)  # n x p
    R_hat  <- row_rank(D_hat)
  } else {
    R_true <- col_rank(D_true)  # n x p
    R_hat  <- col_rank(D_hat)
  }
  
  # ---- optional centering ----
  Rc_true <- center_one_side(R_true, rank_mode)
  Rc_hat  <- center_one_side(R_hat,  rank_mode)
  
  # ---- 2) SVD ----
  sv_true <- svd(Rc_true)
  sv_hat  <- svd(Rc_hat)
  
  # ---- 3) variance explained + scree plot ----
  var_explained <- function(sv){
    d2 <- sv$d^2
    prop <- d2 / sum(d2)
    cum  <- cumsum(prop)
    list(prop = prop, cum = cum)
  }
  ve_true <- var_explained(sv_true)
  ve_hat  <- var_explained(sv_hat)
  
  if(isTRUE(plotting)){
    oldpar <- par(no.readonly = TRUE)
    on.exit(par(oldpar), add = TRUE)
    par(mfrow = c(1,2))
    
    base::plot(ve_true$prop, type="b", xlab="Component", ylab="Proportion variance",
               main=paste0("Scree (", rank_mode, "-rank)"))
    lines(ve_hat$prop, type="b")
    legend("topright", legend=c("true","hat"), lty=1, bty="n")
    
    base::plot(ve_true$cum, type="b", ylim=c(0,1),
               xlab="Component", ylab="Cumulative variance",
               main="Cumulative variance")
    lines(ve_hat$cum, type="b")
    abline(h=c(0.6,0.7,0.8), lty=3)
    legend("bottomright", legend=c("true","hat","60/70/80%"),
           lty=c(1,1,3), bty="n")
  }
  
  # ---- 4) select leading k components ----
  U1 <- sv_true$u[, 1:k, drop=FALSE]
  V1 <- sv_true$v[, 1:k, drop=FALSE]
  d1 <- sv_true$d[1:k]
  
  U2 <- sv_hat$u[, 1:k, drop=FALSE]
  V2 <- sv_hat$v[, 1:k, drop=FALSE]
  d2 <- sv_hat$d[1:k]
  
  # ---- 5) structure comparison ----
  subspace_sim <- function(A, B){
    # A,B: m x k, orthonormal cols (from svd)
    sum((t(A) %*% B)^2) / ncol(A)  # in [0,1]
  }
  
  sim_U <- subspace_sim(U1, U2)  # respondent-side subspace alignment
  sim_V <- subspace_sim(V1, V2)  # item-side subspace alignment
  
  # per-respondent & per-item similarity in k-dim score space
  X1 <- U1 %*% diag(d1, k, k)
  X2 <- U2 %*% diag(d2, k, k)
  respondent_cos <- row_cosine(X1, X2)  # length n
  
  Y1 <- V1 %*% diag(d1, k, k)
  Y2 <- V2 %*% diag(d2, k, k)
  item_cos <- row_cosine(Y1, Y2)        # length p
  
  # singular value shape correlation (optional summary)
  m <- min(length(sv_true$d), length(sv_hat$d))
  sv_shape_cor <- cor(sv_true$d[1:m], sv_hat$d[1:m])
  
  list(
    dims = c(n=n, p=p),
    k = k,
    rank_mode = rank_mode,
    center = center,
    rank_matrices = list(R_true = R_true, R_hat = R_hat,
                         Rc_true = Rc_true, Rc_hat = Rc_hat),
    svd = list(true = sv_true, hat = sv_hat),
    variance = list(true = ve_true, hat = ve_hat),
    similarity = list(
      subspace_U = sim_U,
      subspace_V = sim_V,
      sv_shape_cor = sv_shape_cor,
      respondent_cosine = respondent_cos,
      item_cosine = item_cos
    )
  )
}

