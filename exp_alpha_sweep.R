## v12 EPA alpha-sweep
##
## Hypothesis: α=1 fixed gives the EPA prior a strong K=1 preference (every
## new-cluster step pays a 1/(α+t) factor), so the chain visits over-merged
## states often -> PSM is averaged across K=2 and K=4 modes -> uniform.
## Larger α relaxes the per-step new-cluster penalty; if the diagnosis is
## right, α ∈ {5, 10} should give cleaner PSM contrast and a working
## Dahl/Binder estimator.
##
## Setting: SAME data-gen as exp_sigma_freeze.R / simulation_4_layered_v12_2.R
## (seed 20260501, P=120, K_true=4).  Free σ (M_perm = P_total = 120) to match
## the user's MIDUS production setting.  τ=1 fixed.  α-fix ∈ {1, 5, 10}.

suppressPackageStartupMessages({
  library(Rcpp)
  library(vegan)
})

proj_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
data_dir <- file.path(proj_dir, "data")
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_FMC_cpp_v12.R"))

set.seed(20260501)
n  <- 150
P1 <- 30; P2 <- 30; P3 <- 30; P4 <- 30
P_total <- P1 + P2 + P3 + P4
d <- 2L
K_true <- 4L
K1 <- 5L
gamma_true <- 1.0
sigma0_sq_true <- 1.0
kappa_true <- 1.0
nu2_true <- 5
nu2_fit <- 4
sigma_b <- 1.0

centers_meta <- rbind(c(-1,1), c(1,1), c(1,-1), c(-1,-1))
sigma_meta <- list(diag(c(0.05,0.05)), diag(c(0.07,0.07)),
                   diag(c(0.03,0.03)), diag(c(0.06,0.06)))
assign_meta <- function(P,K) rep_len(seq_len(K), P)
B1_meta <- assign_meta(P1,K_true); B2_meta <- assign_meta(P2,K_true)
B3_meta <- assign_meta(P3,K_true); B4_meta <- assign_meta(P4,K_true)
true_item_cluster <- c(B1_meta, B2_meta, B3_meta, B4_meta)

centers_resp <- centers_meta * 0.7
sd_cluster_resp <- 0.3
sample_around_centers <- function(C, ids, sd) {
  C[ids,,drop=FALSE] + matrix(rnorm(length(ids)*ncol(C),0,sd), length(ids), ncol(C))
}
sample_around_centers_sigma <- function(C, ids, S_list) {
  out <- matrix(0, length(ids), ncol(C))
  for (k in seq_along(S_list)) {
    idx <- which(ids==k); if (!length(idx)) next
    L <- chol(S_list[[k]]); z <- matrix(rnorm(length(idx)*ncol(C)), length(idx), ncol(C))
    out[idx,] <- matrix(C[k,], length(idx), ncol(C), byrow=TRUE) + z %*% L
  }; out
}
dist_mat <- function(A,B) {
  out <- matrix(0, nrow(A), nrow(B))
  for (j in 1:nrow(B)) out[,j] <- sqrt(rowSums((A - matrix(B[j,], nrow(A), ncol(A), byrow=TRUE))^2))
  out
}
invlogit <- function(x) 1/(1+exp(-x))

resp_cl <- sample.int(nrow(centers_resp), n, replace=TRUE)
A_true  <- sample_around_centers(centers_resp, resp_cl, sd_cluster_resp)
B1_true <- sample_around_centers_sigma(centers_meta, B1_meta, sigma_meta)
B2_true <- sample_around_centers_sigma(centers_meta, B2_meta, sigma_meta)
B3_true <- sample_around_centers_sigma(centers_meta, B3_meta, sigma_meta)
B4_true <- sample_around_centers_sigma(centers_meta, B4_meta, sigma_meta)

alpha1_true <- rnorm(n,0,1); alpha2_true <- rnorm(n,0,1)
alpha3_true <- rnorm(n,0,1); alpha4_true <- rnorm(n,0,1)
beta1_true <- rnorm(P1,0,1); beta2_true <- rnorm(P2,0,1); beta3_true <- rnorm(P3,0,0.5)

generate_grm_thresholds <- function(P,K) {
  Km1 <- K-1; out <- matrix(NA_real_, P, Km1)
  for (j in 1:P) out[j,] <- sort(rnorm(Km1, 0, 1.5), decreasing=TRUE); out
}
beta4_true <- generate_grm_thresholds(P4, K1)

D1 <- dist_mat(A_true,B1_true); D2 <- dist_mat(A_true,B2_true)
D3 <- dist_mat(A_true,B3_true); D4 <- dist_mat(A_true,B4_true)
ETA1 <- outer(alpha1_true, rep(1,P1)) - outer(rep(1,n), beta1_true) - gamma_true*D1
ETA2 <- outer(alpha2_true, rep(1,P2)) - outer(rep(1,n), beta2_true) - gamma_true*D2
ETA3 <- outer(alpha3_true, rep(1,P3)) - outer(rep(1,n), beta3_true) - gamma_true*D3
ETA4 <- outer(alpha4_true, rep(1,P4))                                 - gamma_true*D4

Y_bin <- matrix(rbinom(n*P1,1,as.vector(invlogit(ETA1))), n, P1)
lambda_true <- matrix(rgamma(n*P2, nu2_true/2, nu2_true/2), n, P2)
Y_con <- ETA2 + matrix(rnorm(n*P2,0,sqrt(sigma0_sq_true)), n, P2)/sqrt(lambda_true)
storage.mode(Y_con) <- "numeric"
size_nb <- 1/kappa_true
Y_cnt <- matrix(rnbinom(n*P3, size=size_nb, mu=as.vector(exp(ETA3))), n, P3)
generate_grm_data <- function(ETA, beta_thr, K_cat) {
  Y <- matrix(NA_integer_, nrow(ETA), ncol(ETA))
  for (j in 1:ncol(ETA)) for (i in 1:nrow(ETA)) {
    p_ge <- invlogit(ETA[i,j] + beta_thr[j,])
    p <- numeric(K_cat); p[1] <- 1-p_ge[1]
    for (k in 2:(K_cat-1)) p[k] <- p_ge[k-1]-p_ge[k]
    p[K_cat] <- p_ge[K_cat-1]
    p[p<0] <- 0; ps <- sum(p)
    if (ps<=0) { p <- rep(0,K_cat); p[round(K_cat/2)] <- 1 } else p <- p/ps
    Y[i,j] <- sample.int(K_cat, 1, prob=p)
  }; storage.mode(Y) <- "integer"; Y
}
Y_ord1 <- generate_grm_data(ETA4, beta4_true, K1)
Y_ord2 <- matrix(0L, n, 0)

cat(sprintf("Generated data: P=%d, K_true=%d\n", P_total, K_true))
cat(sprintf("True item cluster sizes: %s\n", paste(table(true_item_cluster), collapse=" ")))

init_b_proxy_via_pca <- function(Y_bin,Y_con,Y_cnt,Y_ord1,Y_ord2, d) {
  blocks <- list()
  if (ncol(Y_bin)>0)  blocks <- c(blocks, list(scale(Y_bin)))
  if (ncol(Y_con)>0)  blocks <- c(blocks, list(scale(Y_con)))
  if (ncol(Y_cnt)>0)  blocks <- c(blocks, list(scale(log1p(Y_cnt))))
  if (ncol(Y_ord1)>0) blocks <- c(blocks, list(scale(as.matrix(Y_ord1))))
  X <- do.call(cbind, blocks); X[is.na(X)] <- 0
  pc <- prcomp(t(X), center=TRUE, scale.=FALSE)
  rk <- min(d, ncol(pc$x))
  out <- matrix(0, ncol(X), d); out[,seq_len(rk)] <- pc$x[,seq_len(rk),drop=FALSE]; out
}
b_init_proxy <- init_b_proxy_via_pca(Y_bin,Y_con,Y_cnt,Y_ord1,Y_ord2, d)
set.seed(0502)
km <- kmeans(b_init_proxy, centers = K_true, nstart=25, iter.max=50)
random_sigma <- { set.seed(0502); sample.int(P_total, P_total) }

common_lsirm_hyper <- list(
  a_sigma=1, b_sigma=1, a_tau1=1, b_tau1=1, a_tau2=1, b_tau2=1, a_tau3=1, b_tau3=1,
  a_sigma0=1, b_sigma0=1,
  mu_log_gamma1=0, sd_log_gamma1=0.5, mu_log_gamma2=0, sd_log_gamma2=0.5,
  mu_log_gamma3=0, sd_log_gamma3=0.5, mu_log_gamma4=0, sd_log_gamma4=0.5,
  mu_log_gamma5=0, sd_log_gamma5=0.5,
  mu_log_kappa=0, sd_log_kappa=0.1,
  mu_beta4=0, sd_beta4=2, mu_beta5=0, sd_beta5=2
)
common_lsirm_prop_sd <- list(
  alpha1=1.15, alpha2=0.48, alpha3=1.15, alpha4=0.92, alpha5=0.5,
  log_gamma1=0.07, log_gamma2=0.018, log_gamma3=0.05,
  log_gamma4=0.05, log_gamma5=0.05,
  a=0.26,
  beta1=0.44, beta2=0.13, beta3=0.37, beta4=0.30, beta5=0.30,
  b1=0.33, b2=0.13, b3=0.30, b4=0.26, b5=0.20,
  log_kappa=0.20
)
common_epa_hyper   <- list(a_alpha=1.0,b_alpha=1.0,a_tau=1.0,b_tau=1.0,delta=0.0)
common_epa_prop_sd <- list(log_alpha=0.5, log_tau=0.5)

adj_rand_index <- function(a,b) {
  tab <- table(a,b); n_ <- sum(tab); if (n_<2) return(NA_real_)
  sum_c <- sum(choose(rowSums(tab),2)); sum_k <- sum(choose(colSums(tab),2))
  sum_t <- sum(choose(tab,2))
  expected <- sum_c*sum_k/choose(n_,2); max_idx <- (sum_c+sum_k)/2
  if (abs(max_idx-expected)<1e-12) return(1)
  (sum_t-expected)/(max_idx-expected)
}

run_one <- function(label, alpha_fix,
                    n_iter=20000, burnin=8000, thin=5) {
  cat(sprintf("\n========== Run %s (alpha_fix=%g, tau_fix=1, free sigma) ==========\n",
              label, alpha_fix))
  epa_init <- list(c=as.integer(km$cluster), sigma=as.integer(random_sigma),
                   alpha=alpha_fix, tau=1.0)
  set.seed(20260507)
  res <- lsirm_epa_v12_cpp(
    Y_bin=Y_bin, Y_con=Y_con, Y_cnt=Y_cnt, Y_ord1=Y_ord1, Y_ord2=Y_ord2,
    d=d, n_iter=n_iter, burnin=burnin, thin=thin, nu2=nu2_fit,
    lsirm_hyper=common_lsirm_hyper,
    epa_hyper=common_epa_hyper,
    lsirm_prop_sd=common_lsirm_prop_sd,
    epa_prop_sd=common_epa_prop_sd,
    lsirm_init=NULL,
    epa_init=epa_init,
    sigma_b=sigma_b,
    compute_co_cluster_online=TRUE,
    epa_warmup=0L,
    n_split_merge=1L,
    n_split_merge_R=5L,
    n_perm_swaps=as.integer(P_total),
    b_epa_coupling=TRUE,
    update_alpha=FALSE,
    update_tau=FALSE,
    verbose=FALSE,
    fix_gamma=FALSE,
    procrustes_target=NULL
  )

  diag <- res$epa_diagnostics
  co <- res$epa_co_cluster
  P <- nrow(co)
  ord <- order(true_item_cluster)
  co_o <- co[ord,ord]
  within_mask <- outer(true_item_cluster[ord], true_item_cluster[ord], "==")
  diag(within_mask) <- FALSE
  between_mask <- !within_mask & !diag(P)
  within_v <- co_o[within_mask]; between_v <- co_o[between_mask]

  median_K <- max(2, round(median(res$epa_K_plus)))
  hc <- hclust(as.dist(1-co), method="average")
  hc_part <- cutree(hc, k=min(median_K, P-1))
  ari_hc <- adj_rand_index(hc_part, true_item_cluster)

  c_samps <- res$epa_c
  S <- nrow(c_samps); ut <- upper.tri(co); C_ut <- co[ut]
  loss <- numeric(S)
  for (s in seq_len(S)) {
    cs_ut <- (outer(c_samps[s,], c_samps[s,], "=="))[ut]
    loss[s] <- sum((cs_ut - C_ut)^2)
  }
  s_star <- which.min(loss)
  dahl_part <- as.integer(factor(c_samps[s_star,]))
  ari_dahl <- adj_rand_index(dahl_part, true_item_cluster)

  list(label=label, alpha_fix=alpha_fix,
       sigma_swap_rate=diag$sigma_swap_rate,
       split_rate=diag$split_rate, merge_rate=diag$merge_rate,
       median_K=median(res$epa_K_plus), mean_K=mean(res$epa_K_plus),
       sd_K=sd(res$epa_K_plus),
       within_psm=mean(within_v), between_psm=mean(between_v),
       psm_ratio=mean(within_v)/mean(between_v),
       psm_range=range(co[!diag(P)]),
       ari_hc=ari_hc, K_hc=length(unique(hc_part)),
       ari_dahl=ari_dahl, K_dahl=length(unique(dahl_part)))
}

n_iter_use <- 20000; burnin_use <- 8000; thin_use <- 5

R_a1  <- run_one("alpha=1",  1.0,  n_iter_use, burnin_use, thin_use)
R_a5  <- run_one("alpha=5",  5.0,  n_iter_use, burnin_use, thin_use)
R_a10 <- run_one("alpha=10", 10.0, n_iter_use, burnin_use, thin_use)

cat("\n\n================== ALPHA SWEEP SUMMARY ==================\n")
print_one <- function(R) {
  cat(sprintf("[%s]\n", R$label))
  cat(sprintf("  sigma_swap=%.3f  split=%.3f  merge=%.3f\n",
              R$sigma_swap_rate, R$split_rate, R$merge_rate))
  cat(sprintf("  K_+: median=%g  mean=%.2f  sd=%.2f\n",
              R$median_K, R$mean_K, R$sd_K))
  cat(sprintf("  PSM within=%.3f  between=%.3f  ratio=%.2f  range=[%.3f,%.3f]\n",
              R$within_psm, R$between_psm, R$psm_ratio,
              R$psm_range[1], R$psm_range[2]))
  cat(sprintf("  ARI(hclust K=%d)=%.3f  ARI(Dahl K=%d)=%.3f\n",
              R$K_hc, R$ari_hc, R$K_dahl, R$ari_dahl))
}
print_one(R_a1)
print_one(R_a5)
print_one(R_a10)

saveRDS(list(a1=R_a1, a5=R_a5, a10=R_a10),
        file.path(proj_dir, "exp_alpha_sweep_results.rds"))
cat("\nResults saved to exp_alpha_sweep_results.rds\n")
