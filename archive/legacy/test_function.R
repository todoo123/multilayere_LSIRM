# simulation code

# rm(list = ls())
## =========================================================
## High-order transitivity PPC for ergmm (latentnet)
## =========================================================
library(Matrix)     # sparse 연산 & forceSymmetric
library(ggplot2)    # 박스플롯
library(network)    # 관찰망을 network 객체로 갖고 있다면 사용
library(ergm)
library(intergraph)
library(igraph)
library(latentnet)
library(dplyr)
library(patchwork)

# =========================================================
# utility
# =========================================================
as_sym_adj <- function(A){
  # forcely symmetricity
  A <- Matrix((A > 0) * 1, sparse = TRUE)
  diag(A) <- 0
  A <- forceSymmetric(A, uplo = "U")
  return(A)
}

common_neighbors_mat <- function(A){
  # 각 쌍의 공동이웃 수: A^2의 (i,j) 로 이루어진 matrix
  C <- A %*% A
  diag(C) <- 0
  return(C)
}


## -------------------------- (0) Utility ------------------------
L2_dist <- function(z) {
  n <- nrow(z)
  D <- matrix(0.0, n, n)
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      # distance metric
      d <- sqrt(sum((z[i,] - z[j,])^2))
      D[i,j] <- d
      D[j,i] <- d
    }
  } 
  return(D)
}

dot_dist <- function(z) {
  n <- nrow(z)
  # 각 행(노드)의 norm 계산
  norms <- sqrt(rowSums(z^2))
  
  # dot product 행렬
  G <- z %*% t(z)
  
  # 코사인 유사도 (dot product / norm 표준화)
  sim <- G / (outer(norms, norms))
  
  # distance 로 변환 (1 - similarity)
  D <- 1 - sim
  
  return(D)
}


# =========================================================
# motif computation function
# =========================================================

# ---- 1) 기본 motif 계산 ----

count_edges <- function(A){
  # edge 개수 계산
  return(sum(A[upper.tri(A)]))   # 무방향 간선 수
}

count_kstars <- function(A, k = 2){
  # 설정된 k star 개수 계산
  d <- rowSums(A)
  return(sum(choose(d, k)))      # 각 노드별 조합
}

count_two_paths <- function(A){
  # 2 path - 2 star 개수 계산
  d <- rowSums(A)
  return(sum(choose(d, 2)))      # 각 노드 중심으로 2-경로
}

count_triangles <- function(A){
  # triangle 개수 계산
  return(sum(diag(A %*% A %*% A)) / 6)   # 무방향 삼각형 개수
}

# ---- 2) ESP / DSP 분포 (테이블) ----
esp_table <- function(A){
  # Edgewise Shared Partners: 연결된 shared partnership
  S <- common_neighbors_mat(A)
  idx <- which(upper.tri(A) & A == 1, arr.ind = TRUE)
  mvals <- S[idx]
  return(as.data.frame(table(m = mvals)))
}

dsp_table <- function(A, only_nonedges = FALSE){
  # Dyadwise Shared Partners: 모든 shared partnership
  S <- common_neighbors_mat(A)
  U <- upper.tri(A)
  if(only_nonedges){
    idx <- which(U & A == 0, arr.ind = TRUE)
  } else {
    idx <- which(U, arr.ind = TRUE)
  }
  mvals <- S[idx]
  return(as.data.frame(table(m = mvals)))
}

closure_by_common_neighbors <- function(A, S_grid = 1:5){
  # esp_table / dsp_table
  A <- as_sym_adj(A)
  C <- common_neighbors_mat(A)
  a <- A[upper.tri(A, diag = FALSE)]
  c <- C[upper.tri(C, diag = FALSE)]
  
  
  S_grid <- S_grid[S_grid >= 1]
  if(length(c)) {
    S_grid <- S_grid[S_grid <= max(c)]
    if(length(S_grid) == 0) S_grid <- 1
  } else {
    S_grid <- 1
  }
  
  res_list <- list()
  for (i in seq_along(S_grid)) {
    s <- S_grid[i]
    idx <- which(c >= s)
    den <- length(idx)
    if (den > 0) {
      num <- sum(a[idx])
      rate <- num / den
    } else {
      num <- NA_real_
      rate <- NA_real_
    }
    res_list[[i]] <- data.frame(
      metric = "CN",
      level = s,
      rate = rate,
      num = num,
      den = den
    )
  }
  
  res <- do.call(rbind, res_list)
  return(res)
}

# =========================================================
# simulation with ergm
# =========================================================
# ---- ergmm 적합 객체에서 타이 확률 행렬 추출 ----
prob_from_ergmm <- function(fit,
                            symmetrize = TRUE,
                            diag_zero  = TRUE,
                            eps = 1e-8) {
  # 작은 보조함수: [eps, 1-eps]로 클리핑
  bound01 <- function(x, e = eps) pmin(pmax(x, e), 1 - e)
  
  P <- NULL
  
  # 1) posterior mean tie prob (latentnet::predict type="post")
  P <- tryCatch(predict(fit, type = "post"),
                error = function(e) NULL)
  
  # 2) 일반적인 응답확률 (type="response")
  if (is.null(P)) {
    P <- tryCatch(predict(fit, type = "response"),
                  error = function(e) NULL)
  }
  
  # 3) 선형예측자 (type="link")를 시그모이드로 변환
  if (is.null(P)) {
    eta <- tryCatch(predict(fit, type = "link"),
                    error = function(e) NULL)
    if (!is.null(eta)) {
      P <- 1 / (1 + exp(-eta))
    }
  }
  
  if (is.null(P)) {
    stop("prob_from_ergmm(): predict()가 'post', 'response', 'link' 모두에서 실패했습니다.")
  }
  
  P <- as.matrix(P)
  
  # 무한/NA 처리
  P[!is.finite(P)] <- NA_real_
  P[is.na(P)] <- 0.5  # 보수적 대체 (원하면 0 또는 평균 등으로 변경 가능)
  
  # 무방향 네트워크일 경우 대칭화
  if (isTRUE(symmetrize)) {
    P <- 0.5 * (P + t(P))
  }
  
  # 대각선은 자기루프 제거
  if (isTRUE(diag_zero)) {
    diag(P) <- 0
  }
  
  # 수치 안정화
  P <- bound01(P, eps)
  
  return(P)
}


# ---- 확률행렬 P에서 Bernoulli 네트워크 한 장 샘플 ----
simulate_from_P <- function(P,
                            force_symmetric = TRUE,
                            seed            = NULL,
                            allow_loops     = FALSE) {
  stopifnot(is.matrix(P), nrow(P) == ncol(P))
  n <- nrow(P)

  # 무방향: 필요시 대칭 강제
  if (isTRUE(force_symmetric)) {
    P <- 0.5 * (P + t(P))
  }
  # 대각선(자기루프) 처리
  if (!allow_loops) diag(P) <- 0
  
  A <- matrix(0L, n, n)
  # 상삼각만 샘플링 후 대칭 복원
  idx <- which(upper.tri(P), arr.ind = TRUE)
  U <- runif(nrow(idx))
  A_vals <- as.integer(U < P[idx])
  A[idx] <- A_vals
  A <- A + t(A)
  storage.mode(A) <- "integer"
  return(A)
}

# ----- Posterior predictive: 한 번에 이항 네트워크 1개 샘플 -----
sample_binary_once_from_fit <- function(fit, dist_fun = L2_dist) {
  smp <- fit$samples
  S   <- dim(smp$z)[1]
  n   <- dim(smp$z)[2]
  s   <- sample.int(S, 1)                  # 사후표본 1개 선택
  
  z   <- smp$z[s,,]
  D   <- dist_fun(z)
  
  # 파라미터 이름이 모델별로 다를 수 있어 안전하게 선택
  a1  <- if (!is.null(smp$alpha1)) smp$alpha1[s] else
    if (!is.null(smp$alpha_bin)) smp$alpha_bin[s] else smp$alpha[s]
  b1  <- if (!is.null(smp$beta1))  smp$beta1[s]  else
    if (!is.null(smp$beta_bin))  smp$beta_bin[s]  else smp$beta[s]
  
  eta <- a1 - b1 * D                      # 거리 ↑ → 연결확률 ↓ (일관화)
  P   <- plogis(eta)
  
  U   <- matrix(runif(length(P)), nrow(P))
  Y   <- (U < P) * 1L
  Y[lower.tri(Y)] <- t(Y)[lower.tri(Y)]; diag(Y) <- 0L
  as_sym_adj(Y)
}

# ----- 모델 타입별로 한 장 샘플 뽑기 -----
draw_network_from_model <- function(fit, model = 'LSJM', dist_fun = L2_dist) {
  if (model == 'ergm') {
    # latentnet::ergmm 는 predict(type="post")가 "사후평균 tie prob"를 주므로
    # 여기서는 그 확률로 Bernoulli를 한 번 더 샘플 (parameter 불확실성은 미반영)
    P_hat <- prob_from_ergmm(fit)
    return(simulate_from_P(P_hat))
  } else if (model %in% c('LSJM','LSM','LSJM_3')) {
    # 사후표본 기반 posterior predictive
    return(sample_binary_once_from_fit(fit, dist_fun = dist_fun))
  } else {
    stop("Unknown model type in draw_network_from_model().")
  }
}

# =========================================================
# boxplot generation
# =========================================================
# argument: count_edges / count_kstars / count_two_paths / count_triangles / esp_table / dsp_table / closure_by_common_neighbors
# 에 대하여 argument 입력하면 각 argument 에 대하여 true value, simulation 값 기반으로 box plot 그리는 function

# count_edges / count_kstars / count_two_paths / count_triangles / esp_table / dsp_table / closure_by_common_neighbors
motif_GOF_test <- function(fit, A_obs, B = 200, k_grid = 1:5, S_grid = 1:5, 
                           arg = 'simple_motif', custom_text = NULL, model = 'ergm', dist_func){
  
  # ✔ 더 이상 P_hat 만들지 않음 (posterior predictive 샘플로 대체)
  # if(model == 'ergm'){ ... } 등 P_hat 관련 블록 삭제
  
  if(arg == 'simple_motif'){
    obs_edges     <- count_edges(A_obs)
    obs_two_paths <- count_two_paths(A_obs)
    obs_triangles <- count_triangles(A_obs)
    
    pb <- txtProgressBar(min = 0, max = B, style = 3)  # style=3: [====    ] 형식
    reps <- vector("list", B)
    for(i in 1:B){
      A_b <- draw_network_from_model(fit, model = model, dist_fun = dist_func)
      reps[[i]] <- c(count_edges(A_b), count_two_paths(A_b), count_triangles(A_b))
      setTxtProgressBar(pb, i)                          # <- 진행 업데이트
    }
    close(pb)
    
    df  <- as.data.frame(do.call(rbind, reps))
    colnames(df) <- c('edges', 'two_paths', 'triangles')
    obs <- c(edges = obs_edges, two_paths = obs_two_paths, triangles = obs_triangles)
    
    # 2) facet용 long 데이터 (base::stack)
    mets <- names(obs)
    sdf <- stack(df[mets])                 # -> columns: values, ind
    names(sdf) <- c('value', 'metric')
    
    # 3) RMSE 계산 (metric별)
    rmse_vec <- sapply(mets, function(k){
      sqrt(mean((df[[k]] - obs[[k]])^2, na.rm = TRUE))
    })
    rmse_df <- data.frame(
      metric = mets,
      x = 1,
      y = -Inf,                                  # 패널 하단
      label = sprintf('RMSE = %.3f', rmse_vec),
      stringsAsFactors = FALSE
    )
    
    # 4) 관찰치(빨간 점) 레이어용 데이터
    obs_df <- data.frame(
      metric = mets,
      x = 1,
      y = as.numeric(obs[mets]),
      stringsAsFactors = FALSE
    )
    
    # 5) 커스텀 텍스트(우상단 파란 글자) — 옵션
    if(!is.null(custom_text)){
      text_df <- data.frame(
        metric = mets,
        x = Inf, y = Inf,
        label = custom_text,
        stringsAsFactors = FALSE
      )
    }
    
    p <- ggplot(sdf, aes(x = 1, y = value)) +
      geom_boxplot(outlier.shape = NA, width = 0.25) +
      geom_point(data = obs_df, aes(x = x, y = y),
                 color = 'red', size = 2, inherit.aes = FALSE) +
      geom_text(data = rmse_df, aes(x = x, y = y, label = label),
                vjust = -0.5, inherit.aes = FALSE) +
      facet_wrap(~ metric, scales = 'free_y', nrow = 1) +
      # 하단 여백을 조금 늘려 RMSE 텍스트가 잘 보이게
      scale_y_continuous(expand = expansion(mult = c(0.15, 0.05))) +
      scale_x_continuous(breaks = NULL) +
      labs(x = NULL, y = NULL) +
      theme_minimal(base_size = 12) +
      theme(
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        strip.text = element_text(face = 'bold')
      )
    
    if(!is.null(custom_text)){
      p <- p + geom_text(
        data = text_df, aes(x = x, y = y, label = label),
        hjust = 1.1, vjust = 1.1, color = 'blue', size = 3,
        inherit.aes = FALSE
      )
    }
    
    print(p)
    
  } else if(arg == 'kstars'){
    reps <- list(); obs <- list(); index <- 1
    maxd <- max(rowSums(A_obs))
    
    total <- (if (maxd < max(k_grid)) maxd else length(k_grid)) * B
    pb <- txtProgressBar(min = 0, max = total, style = 3)
    cnt <- 0
    
    if (maxd < max(k_grid)) {
      for (i in 1:maxd) {
        lev <- if (k_grid[[i]] < maxd) k_grid[[i]] else maxd
        for (j in 1:B) {
          A_b <- draw_network_from_model(fit, model = model, dist_fun = dist_func)
          reps[[index]] <- c(lev, count_kstars(A_b, k = lev)); index <- index + 1
          cnt <- cnt + 1
          setTxtProgressBar(pb, cnt) 
        }
        obs[[i]] <- c(lev, count_kstars(A_obs, k = lev))
      }
    } else {
      for (i in seq_along(k_grid)) {
        lev <- k_grid[[i]]
        for (j in 1:B) {
          A_b <- draw_network_from_model(fit, model = model, dist_fun = dist_func)
          reps[[index]] <- c(lev, count_kstars(A_b, k = lev)); index <- index + 1
          cnt <- cnt + 1
          setTxtProgressBar(pb, cnt) 
        }
        obs[[i]] <- c(lev, count_kstars(A_obs, k = lev))
      }
    }
    close(pb)
    
    df     <- as.data.frame(do.call(rbind, reps))
    obs_df <- as.data.frame(do.call(rbind, obs))
    colnames(df) <- c('level', 'kstar_count'); colnames(obs_df) <- c('level', 'kstar_count')
    
    df_res <- left_join(df, obs_df, by = 'level')
    names(df_res)[names(df_res)=="kstar_count.x"] <- "kstar_count_rep"
    names(df_res)[names(df_res)=="kstar_count.y"] <- "kstar_count_obs"
    
    ## 관찰값(레드 점) 데이터: level당 1개
    obs_by_level <- obs_df %>%
      distinct(level, .keep_all = TRUE) %>%
      rename(obs = kstar_count)
    
    ## k-level별 RMSE
    rmse_df <- df_res %>%
      group_by(level) %>%
      summarize(
        rmse = sqrt(mean((kstar_count_rep - kstar_count_obs)^2, na.rm = TRUE)),
        .groups = "drop"
      ) %>%
      mutate(label = sprintf("RMSE = %.3f", rmse))
    
    ## x축 순서 고정(선택)
    x_levels <- sort(unique(df_res$level))
    
    ## 메인 플롯: boxplot(복제), 빨간 점(관찰), 하단 RMSE 텍스트
    p <- ggplot(df_res, aes(x = factor(level, levels = x_levels), y = kstar_count_rep)) +
      geom_boxplot(outlier.shape = NA, width = 0.6) +
      geom_point(data = obs_by_level,
                 aes(x = factor(level, levels = x_levels), y = obs),
                 color = "red", size = 2, inherit.aes = FALSE) +
      geom_text(data = rmse_df,
                aes(x = factor(level, levels = x_levels), y = -Inf, label = label),
                vjust = -0.5, size = 3.3, inherit.aes = FALSE) +
      scale_y_continuous(expand = expansion(mult = c(0.15, 0.05))) +  # 하단 여백(=RMSE 텍스트 자리)
      labs(x = "k level", y = "k-star count") +
      theme_minimal(base_size = 12) +
      theme(
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank()
      )
    
    ## (옵션) 우상단에 커스텀 텍스트 넣고 싶으면:
    if (!is.null(custom_text)) {
      p <- p + annotate("text", x = Inf, y = Inf, label = custom_text,
                        hjust = 1.1, vjust = 1.2, size = 3, color = "blue")
    }
    
    print(p)
    
  } else if(arg == 'esp'){
    reps <- vector("list", B)
    pb <- txtProgressBar(min = 0, max = B, style = 3)  # style=3: [====    ] 형식
    for (b in 1:B) {
      A_b <- draw_network_from_model(fit, model = model, dist_fun = dist_func)
      t <- esp_table(A_b); colnames(t) <- c("level", "kstar_count")
      t$level <- as.integer(as.character(t$level))
      t$kstar_count <- as.numeric(t$kstar_count); t$rep <- b
      reps[[b]] <- t
      setTxtProgressBar(pb, b)                          # <- 진행 업데이트
    }
    close(pb)
    df <- dplyr::bind_rows(reps)
    
    # 2) 관찰망 ESP 분포
    o <- esp_table(A_obs)
    colnames(o) <- c("level", "kstar_count")
    o$level <- as.integer(as.character(o$level))
    o$kstar_count <- as.numeric(o$kstar_count)
    obs_df <- o
    
    # ★ 관찰망에서 나타난 level만 사용
    obs_levels <- sort(unique(obs_df$level))
    
    # 3) 누락된 level(복제에서 등장 안 한 경우) 0으로 채우기 — 단, obs_levels만
    full_grid <- expand.grid(rep = 1:B, level = obs_levels)
    df <- full_grid %>%
      dplyr::left_join(df, by = c("rep", "level")) %>%
      dplyr::mutate(kstar_count = ifelse(is.na(kstar_count), 0, kstar_count)) %>%
      dplyr::select(level, kstar_count, rep)
    
    # 4) 복제 vs 관찰 조인 (_rep / _obs 고정)
    df_res <- dplyr::left_join(
      df %>% dplyr::select(level, kstar_count),
      obs_df %>% dplyr::select(level, kstar_count),
      by = "level",
      suffix = c("_rep", "_obs")
    )
    
    # 5) 관찰값(빨간 점): level당 1개 (관찰망에 존재하는 레벨만)
    obs_by_level <- obs_df %>%
      dplyr::distinct(level, .keep_all = TRUE) %>%
      dplyr::rename(obs = kstar_count)
    
    # 6) level별 RMSE
    rmse_df <- df_res %>%
      dplyr::group_by(level) %>%
      dplyr::summarize(
        rmse = sqrt(mean((kstar_count_rep - kstar_count_obs)^2, na.rm = TRUE)),
        .groups = "drop"
      ) %>%
      dplyr::mutate(label = sprintf("RMSE = %.3f", rmse))
    
    # 7) 플롯
    x_levels <- obs_levels
    y_lab <- "ESP count"
    
    p <- ggplot2::ggplot(df_res, ggplot2::aes(x = factor(level, levels = x_levels), y = kstar_count_rep)) +
      ggplot2::geom_boxplot(outlier.shape = NA, width = 0.6) +
      ggplot2::geom_point(data = obs_by_level,
                          ggplot2::aes(x = factor(level, levels = x_levels), y = obs),
                          color = "red", size = 2, inherit.aes = FALSE) +
      ggplot2::geom_text(data = rmse_df,
                         ggplot2::aes(x = factor(level, levels = x_levels), y = -Inf, label = label),
                         vjust = -0.5, size = 3.3, inherit.aes = FALSE) +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.15, 0.05))) +
      ggplot2::labs(x = "m (number of shared partners)", y = y_lab) +
      ggplot2::theme_minimal(base_size = 12) +
      ggplot2::theme(
        panel.grid.major.x = ggplot2::element_blank(),
        panel.grid.minor = ggplot2::element_blank()
      )
    
    if (!is.null(custom_text)) {
      p <- p + ggplot2::annotate("text", x = Inf, y = Inf, label = custom_text,
                                 hjust = 1.1, vjust = 1.2, size = 3, color = "blue")
    }
    
    print(p)
    
  } else if(arg == 'dsp'){
    if (!exists("only_nonedges")) only_nonedges <- FALSE
    reps <- vector("list", B)
    pb <- txtProgressBar(min = 0, max = B, style = 3)  # style=3: [====    ] 형식
    for (b in 1:B) {
      A_b <- draw_network_from_model(fit, model = model, dist_fun = dist_func)
      t <- dsp_table(A_b, only_nonedges = only_nonedges)
      colnames(t) <- c("level", "kstar_count")
      t$level <- as.integer(as.character(t$level))
      t$kstar_count <- as.numeric(t$kstar_count); t$rep <- b
      reps[[b]] <- t
      setTxtProgressBar(pb, b)                          # <- 진행 업데이트
    }
    close(pb)
    df <- dplyr::bind_rows(reps)
    
    # 2) 관찰망 DSP 분포
    o <- dsp_table(A_obs, only_nonedges = only_nonedges)
    colnames(o) <- c("level", "kstar_count")
    o$level <- as.integer(as.character(o$level))
    o$kstar_count <- as.numeric(o$kstar_count)
    obs_df <- o
    
    # ★ 핵심: 관찰망에서 나타난 level만 사용
    obs_levels <- sort(unique(obs_df$level))
    
    # 3) 누락된 level(복제에서 등장 안 한 경우) 0으로 채우기 — 단, obs_levels만
    full_grid <- expand.grid(rep = 1:B, level = obs_levels)
    df <- full_grid %>%
      dplyr::left_join(df, by = c("rep", "level")) %>%
      dplyr::mutate(kstar_count = ifelse(is.na(kstar_count), 0, kstar_count)) %>%
      dplyr::select(level, kstar_count, rep)
    
    # 4) 복제 vs 관찰 조인 (_rep / _obs 고정)
    df_res <- dplyr::left_join(
      df %>% dplyr::select(level, kstar_count),
      obs_df %>% dplyr::select(level, kstar_count),
      by = "level",
      suffix = c("_rep", "_obs")
    )
    
    # 5) 관찰값(빨간 점): level당 1개 (관찰망에 존재하는 레벨만)
    obs_by_level <- obs_df %>%
      dplyr::distinct(level, .keep_all = TRUE) %>%
      dplyr::rename(obs = kstar_count)
    
    # 6) level별 RMSE
    rmse_df <- df_res %>%
      dplyr::group_by(level) %>%
      dplyr::summarize(
        rmse = sqrt(mean((kstar_count_rep - kstar_count_obs)^2, na.rm = TRUE)),
        .groups = "drop"
      ) %>%
      dplyr::mutate(label = sprintf("RMSE = %.3f", rmse))
    
    # 7) 플롯
    x_levels <- obs_levels
    y_lab <- if (only_nonedges) "DSP count (non-edges)" else "DSP count"
    
    p <- ggplot2::ggplot(df_res, ggplot2::aes(x = factor(level, levels = x_levels), y = kstar_count_rep)) +
      ggplot2::geom_boxplot(outlier.shape = NA, width = 0.6) +
      ggplot2::geom_point(data = obs_by_level,
                          ggplot2::aes(x = factor(level, levels = x_levels), y = obs),
                          color = "red", size = 2, inherit.aes = FALSE) +
      ggplot2::geom_text(data = rmse_df,
                         ggplot2::aes(x = factor(level, levels = x_levels), y = -Inf, label = label),
                         vjust = -0.5, size = 3.3, inherit.aes = FALSE) +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.15, 0.05))) +
      ggplot2::labs(x = "m (number of shared partners)", y = y_lab) +
      ggplot2::theme_minimal(base_size = 12) +
      ggplot2::theme(
        panel.grid.major.x = ggplot2::element_blank(),
        panel.grid.minor = ggplot2::element_blank()
      )
    
    if (!is.null(custom_text)) {
      p <- p + ggplot2::annotate("text", x = Inf, y = Inf, label = custom_text,
                                 hjust = 1.1, vjust = 1.2, size = 3, color = "blue")
    }
    
    print(p)
    
  } else if(arg == 'closure_rate'){
    o <- closure_by_common_neighbors(A_obs, S_grid = S_grid)
    obs_df <- o %>% dplyr::select(level, rate)
    colnames(obs_df) <- c("level", "kstar_count")
    obs_levels <- sort(unique(obs_df$level))
    
    reps <- vector("list", B)
    pb <- txtProgressBar(min = 0, max = B, style = 3)  # style=3: [====    ] 형식
    for (b in 1:B) {
      A_b <- draw_network_from_model(fit, model = model, dist_fun = dist_func)
      tb  <- closure_by_common_neighbors(A_b, S_grid = obs_levels) %>% dplyr::select(level, rate)
      colnames(tb) <- c("level", "kstar_count"); tb$rep <- b
      reps[[b]] <- tb
      setTxtProgressBar(pb, b)                          # <- 진행 업데이트
    }
    close(pb)
    df_raw <- dplyr::bind_rows(reps)
    
    ## 3) 누락된 level(해당 복제망에서 den=0 등으로 결과 행 자체가 빠진 경우)을 NA로 유지
    #    — counts와 달리 rate는 "없음=0"이 아님. RMSE/boxplot에서 NA는 제외됨.
    full_grid <- expand.grid(rep = 1:B, level = obs_levels)
    df <- full_grid %>%
      dplyr::left_join(df_raw, by = c("rep", "level")) %>%
      dplyr::select(level, kstar_count, rep)
    # 여기서 kstar_count는 일부 NA일 수 있음(그 레벨에서 c>=s인 dyad가 없었음)
    
    ## 4) 복제 vs 관찰 조인 (_rep / _obs 접미사 고정)
    df_res <- dplyr::left_join(
      df %>% dplyr::select(level, kstar_count),
      obs_df %>% dplyr::select(level, kstar_count),
      by = "level",
      suffix = c("_rep", "_obs")
    )
    
    ## 5) 관찰값(빨간 점): level당 1개
    obs_by_level <- obs_df %>%
      dplyr::distinct(level, .keep_all = TRUE) %>%
      dplyr::rename(obs = kstar_count)
    
    ## 6) level별 RMSE (NA는 제외하고 평균)
    rmse_df <- df_res %>%
      dplyr::group_by(level) %>%
      dplyr::summarize(
        rmse = sqrt(mean((kstar_count_rep - kstar_count_obs)^2, na.rm = TRUE)),
        .groups = "drop"
      ) %>%
      dplyr::mutate(label = sprintf("RMSE = %.3f", rmse))
    
    ## 7) 플롯
    x_levels <- obs_levels
    y_lab <- "closure rate"
    
    p <- ggplot2::ggplot(df_res, ggplot2::aes(x = factor(level, levels = x_levels),
                                              y = kstar_count_rep)) +
      ggplot2::geom_boxplot(outlier.shape = NA, width = 0.6, na.rm = TRUE) +
      ggplot2::geom_point(data = obs_by_level,
                          ggplot2::aes(x = factor(level, levels = x_levels), y = obs),
                          color = "red", size = 2, inherit.aes = FALSE) +
      ggplot2::geom_text(data = rmse_df,
                         ggplot2::aes(x = factor(level, levels = x_levels), y = -Inf, label = label),
                         vjust = -0.5, size = 3.3, inherit.aes = FALSE) +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.15, 0.05))) +
      ggplot2::labs(x = "s (minimum number of common neighbors)", y = y_lab) +
      ggplot2::theme_minimal(base_size = 12) +
      ggplot2::theme(
        panel.grid.major.x = ggplot2::element_blank(),
        panel.grid.minor = ggplot2::element_blank()
      )
    
    if (!is.null(custom_text)) {
      p <- p + ggplot2::annotate("text", x = Inf, y = Inf, label = custom_text,
                                 hjust = 1.1, vjust = 1.2, size = 3, color = "blue")
    }
    
    print(p)
    
  } else {
    print("warning: wrong argument")
  }
}



# # ===========================================================================
# # practice
# # ===========================================================================
# # 빈 네트워크(노드 수만 지정)
# n <- 100
# g0 <- network.initialize(n, directed = FALSE)
# 
# # gwesp 계수에 따라 샘플링하는 함수
# sample_with_gwesp <- function(gwesp_coef, theta_edges = 1, decay = 0.5) {
#   sims <- simulate(
#     g0 ~ edges + gwesp(decay, fixed = TRUE),
#     coef = c(theta_edges, gwesp_coef),   # edges, gwesp 순서
#     nsim = 1,
#     output = "network",
#     control = control.simulate(
#       MCMC.burnin = 1e5,
#       MCMC.interval = 1e3
#     )
#   )
#   return(sims)
# }
# 
# # gwesp 값을 낮음 -> 높음으로 바꿔가며 샘플링
# decay_values <- seq(from = 0.1, to = 1.2, by = 0.05)
# n <- length(decay_values)
# # edges_coef <- seq(from = -4.5, to = -2.0, by = 0.5)
# length(decay_values)
# 
# nets <- list()
# for(i in 1:n){
#   nets[[i]] <- sample_with_gwesp(gwesp_coef = 1.0, theta_edges = -3.5, decay = decay_values[i])
# }
# 
# 
# # 지표 확인 (전이도 / 삼각형 수)
# for (i in seq_along(nets)) {
#   g_ig <- intergraph::asIgraph(nets[[i]])
#   cat(sprintf("decay_par = % .1f | edges = %d | transitivity = %.3f | triangles = %d\n",
#               decay_values[i],
#               gsize(g_ig),
#               transitivity(g_ig, type = "global"),
#               sum(count_triangles(as_adjacency_matrix(g_ig)))))
#   plot(g_ig,
#        main = paste("gwesp =", decay_values[i]),
#        vertex.size = 6, vertex.label = NA, edge.width = 1)
# }
# 
# 
# 
# # compute communicability
# library(brainGraph)
# for(i in 1:n){
#   comm       <- communicability(intergraph::asIgraph(nets[[i]]))
#   comm_angle <- comm / sqrt(outer(diag(comm), diag(comm)))
#   comm_dist  <- sqrt(outer(diag(comm), diag(comm), "+") - 2*comm)
#   comm_dist.scaled  <- matrix(scale(as.numeric(nets[[i]]$comm_dist)), nrow = 100, ncol = 100)
#   
#   nets[[i]]$comm <- comm
#   nets[[i]]$comm_angle <- comm_angle
#   nets[[i]]$comm_dist <- comm_dist
#   nets[[i]]$comm_dist.scaled <- comm_dist.scaled
# }
# 
# 
# mod1 <- list()
# mod2 <- list()
# 
# for(i in c(1,10,20)){
#   mod1[[i]] <- ergmm(nets[[i]] ~ euclidean(d = 2),
#                      control = ergmm.control(burnin = 2000,
#                                              sample.size = 6000,
#                                              interval = 5))
#   
#   network::set.network.attribute(nets[[i]], "comm_dist.scaled", comm_dist.scaled)
#   mod2[[i]] <- ergmm(nets[[i]] ~ edgecov("comm_dist.scaled") + euclidean(d = 2),
#                      control = ergmm.control(burnin = 2000,
#                                              sample.size = 6000,
#                                              interval = 5))
#   print(i)
# }
# 
# # sampled network has underestimated closure rate with decay parameter lower than 0.65
# # on decay parameter 0.4, 9th closure rate is overestimated than true closure rate
# # with lower decay parameter(generally under 0.4) high ordered closure rate such that over 7, has been overestimated than true closure rate
# # with decay parameter over 0.7, almost of all true rates are covered by boxplot.
# 
# # question: generalized overestimation and peculiar underestimation under decay parameter lower than 0.7
# # 1. generalized overestimation:
# # 
# # 2. underestimation of high order clusure rate on lower decay parameter(relatively sparse network):
# # first, with lower decay parameter, high order closure is very sparse so true rate is very small = 0. high order shared partnership is exist,
# # but closure doesnt exist. Or, NA. high order shared partnership doesnt exist.
# # therefore, vanilla LSM overestimate high order shared partnership closure. -> 애초에 이런 식의 shared partnership 이 존재하지 않는데 복원한거임.
# # 다른 motif 는 잘 복원하는가 보자.
# 
# # kstars
# # esp
# # dsp
# 
# 
# for(i in c(1,10,20)){
#   motif_GOF_test(mod1[[i]], 
#                  as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
#                  B = 500, custom_text = paste0("decay value: ",decay_values[i], " model 1"), arg = "simple_motif")
#   
#   motif_GOF_test(mod1[[i]], 
#                  as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
#                  B = 500, custom_text = paste0("decay value: ",decay_values[i], " model 1"), arg = "kstars")
#   
#   motif_GOF_test(mod1[[i]], 
#                  as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
#                  B = 500, custom_text = paste0("decay value: ",decay_values[i], " model 1"), arg = "esp")
#   
#   motif_GOF_test(mod1[[i]], 
#                  as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
#                  B = 500, custom_text = paste0("decay value: ",decay_values[i], " model 1"), arg = "dsp")
#   
#   motif_GOF_test(mod1[[i]], 
#                  as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
#                  B = 500, custom_text = paste0("decay value: ",decay_values[i], " model 1"), arg = "closure_rate")
#   
# }
# 
# 
# 
# for(i in c(1,10,20)){
#   motif_GOF_test(mod2[[i]], 
#                  as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
#                  B = 500, custom_text = paste0("decay value: ",decay_values[i], " model 2"), arg = "simple_motif")
#   
#   motif_GOF_test(mod2[[i]], 
#                  as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
#                  B = 500, custom_text = paste0("decay value: ",decay_values[i], " model 2"), arg = "kstars")
#   
#   motif_GOF_test(mod2[[i]], 
#                  as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
#                  B = 500, custom_text = paste0("decay value: ",decay_values[i], " model 2"), arg = "esp")
#   
#   motif_GOF_test(mod2[[i]], 
#                  as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
#                  B = 500, custom_text = paste0("decay value: ",decay_values[i], " model 2"), arg = "dsp")
#   
#   motif_GOF_test(mod2[[i]], 
#                  as_adjacency_matrix(intergraph::asIgraph(nets[[i]])), 
#                  B = 500, custom_text = paste0("decay value: ",decay_values[i], " model 2"), arg = "closure_rate")
#   
# }

