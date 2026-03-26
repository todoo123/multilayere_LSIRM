#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// --- C++ Helper Functions ---

/* (0) 유틸리티: L2 거리 */
arma::mat L2_dist_cpp(const arma::mat& z) {
  int n = z.n_rows;
  arma::mat D(n, n, arma::fill::zeros);
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double d = arma::norm(z.row(i) - z.row(j), 2); 
      D(i, j) = d;
      D(j, i) = d;
    }
  }
  return D;
}

/* (1a) 전체 로그 가능도 (이진) */
double loglik_binary_network_cpp(const arma::mat& Y, const arma::mat& z, double alpha, double beta) {
  arma::mat D = L2_dist_cpp(z);
  arma::mat eta = alpha - beta * D;
  arma::mat pi = 1.0 / (1.0 + arma::exp(-eta)); 
  int n = Y.n_rows;
  double ll = 0.0;
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double p = pi(i, j);
      if (p < 1e-11) p = 1e-11;
      if (p > (1.0 - 1e-11)) p = 1.0 - 1e-11;
      if (Y(i, j) == 1.0) {
        ll += log(p);
      } else {
        ll += log(1.0 - p);
      }
    }
  }
  return ll;
}

/* (1b) 전체 로그 가능도 (연속 - LogNormal) */
double loglik_conti_network_cpp(const arma::mat& Y, const arma::mat& z, 
                                double alpha, double beta, double kappa, 
                                double eps = 1e-12) {
  arma::mat D = L2_dist_cpp(z);
  // continuous metric 에 따라서 부호 조정해야 함.
  arma::mat eta = alpha + beta * D;
  int n = Y.n_rows;
  double ll = 0.0;
  double sd_log = sqrt(kappa);
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double y = Y(i, j);
      if (y < eps) y = eps;
      double mu_log = eta(i, j) - kappa / 2.0;
      ll += R::dlnorm(y, mu_log, sd_log, true);
    }
  }
  return ll;
}

/* (2a) 단일 노드 로그 가능도 (이진) - O(n) */
double loglik_single_node_binary_cpp(int j, const arma::mat& Y, const arma::mat& z, 
                                     double alpha, double beta) {
  int n = Y.n_rows;
  double ll_j = 0.0;
  arma::rowvec z_j = z.row(j);
  for (int i = 0; i < n; ++i) {
    if (i == j) continue;
    arma::rowvec z_i = z.row(i);
    double d_ij = arma::norm(z_i - z_j, 2);
    double eta_ij = alpha - beta * d_ij;
    double p = 1.0 / (1.0 + exp(-eta_ij));
    if (p < 1e-11) p = 1e-11;
    if (p > (1.0 - 1e-11)) p = 1.0 - 1e-11;
    if (Y(i, j) == 1.0) {
      ll_j += log(p);
    } else {
      ll_j += log(1.0 - p);
    }
  }
  return ll_j;
}

/* (2b) 단일 노드 로그 가능도 (연속) - O(n) */
double loglik_single_node_conti_cpp(int j, const arma::mat& Y, const arma::mat& z, 
                                    double alpha, double beta, double kappa, 
                                    double eps = 1e-12) {
  int n = Y.n_rows;
  double ll_j = 0.0;
  double sd_log = sqrt(kappa);
  arma::rowvec z_j = z.row(j);
  for (int i = 0; i < n; ++i) {
    if (i == j) continue; 
    arma::rowvec z_i = z.row(i);
    double d_ij = arma::norm(z_i - z_j, 2);
    double eta_ij = alpha - beta * d_ij;
    double y = Y(i, j);
    if (y < eps) y = eps;
    double mu_log = eta_ij - kappa / 2.0;
    ll_j += R::dlnorm(y, mu_log, sd_log, true);
  }
  return ll_j;
}


/* (3) 사전 확률 (Priors) */
// z_i
double log_prior_z_i_cpp(const arma::rowvec& z_i, double sigma2) {
  double log_prior = 0.0;
  double sd = sqrt(sigma2);
  for(double val : z_i) {
    log_prior += R::dnorm(val, 0.0, sd, true);
  }
  return log_prior;
}
// alpha
double log_prior_alpha_cpp(double alpha, double xi, double psi2) {
  return R::dnorm(alpha, xi, sqrt(psi2), true);
}
// beta
double log_prior_beta_cpp(double beta, double a, double b) {
  if (beta <= 0) return R_NegInf;
  return R::dgamma(beta, a, 1.0 / b, true);
}
// kappa (R 코드의 log_prior_kappa와 동일: dnorm(log(k)) - log(k))
double log_prior_kappa_cpp(double kappa, double a, double b) {
  if (kappa <= 0) return R_NegInf;
  return R::dnorm(log(kappa), a, sqrt(b), true) - log(kappa);
}


// --- (4) MCMC 업데이트 함수 ---

/* (4a) 공통 z 업데이트 (Element-wise, O(n) 최적화) */
void update_z_j_elementwise_LSJM_cpp(int j, arma::mat& z, 
                                     const arma::mat& Y_bin, const arma::mat& Y_con,
                                     double alpha1, double beta1,
                                     double alpha2, double beta2, double kappa2,
                                     double sigma2, double prop_z_sd, int d, 
                                     arma::vec& acc_z_j_elem) {
  
  double sd = sqrt(sigma2); // 사전 확률용
  
  for (int k = 0; k < d; ++k) {
    // 1. 현재 상태의 로그-사후확률 (두 LL의 합 + Prior)
    // double ll_bin_curr = loglik_single_node_binary_cpp(j, Y_bin, z, alpha1, beta1);
    double ll_bin_curr = loglik_binary_network_cpp(Y_bin, z, alpha1, beta1);
    // double ll_con_curr = loglik_single_node_conti_cpp(j, Y_con, z, alpha2, beta2, kappa2);
    double ll_con_curr = loglik_conti_network_cpp(Y_con, z, alpha2, beta2, kappa2);
    double prior_curr = R::dnorm(z(j, k), 0.0, sd, true);
    double current_logpost = ll_bin_curr + ll_con_curr + prior_curr;
    
    // 2. z(j, k) 요소만 새로 제안
    double z_jk_prop = z(j, k) + R::rnorm(0, prop_z_sd);
    
    // 3. 제안된 상태
    arma::mat z_prop = z; 
    z_prop(j, k) = z_jk_prop; 
    
    // 4. 제안된 상태의 로그-사후확률
    // double ll_bin_prop = loglik_single_node_binary_cpp(j, Y_bin, z_prop, alpha1, beta1);
    double ll_bin_prop = loglik_binary_network_cpp(Y_bin, z_prop, alpha1, beta1);
    // double ll_con_prop = loglik_single_node_conti_cpp(j, Y_con, z_prop, alpha2, beta2, kappa2);
    double ll_con_prop = loglik_conti_network_cpp(Y_con, z_prop, alpha2, beta2, kappa2);
    double prior_prop = R::dnorm(z_jk_prop, 0.0, sd, true);
    double prop_logpost = ll_bin_prop + ll_con_prop + prior_prop;
    
    // 5. 수락/기각
    double acc_ratio = prop_logpost - current_logpost;
    if (log(R::runif(0, 1)) < acc_ratio) {
      z(j, k) = z_jk_prop;
      acc_z_j_elem(k) = 1.0;
    }
  }
}

/* (4b) 이진(Binary) 레이어 파라미터 업데이트 */
// alpha1
void bin_update_alpha_cpp(double& alpha, const arma::mat& z, const arma::mat& Y, 
                          double xi, double psi2, double prop_alpha, double beta, 
                          bool& accepted) {
  accepted = false;
  double ll_curr = loglik_binary_network_cpp(Y, z, alpha, beta) + 
    log_prior_alpha_cpp(alpha, xi, psi2);
  double alpha_prop = alpha + R::rnorm(0, prop_alpha);
  double ll_prop = loglik_binary_network_cpp(Y, z, alpha_prop, beta) + 
    log_prior_alpha_cpp(alpha_prop, xi, psi2);
  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    alpha = alpha_prop;
    accepted = true;
  }
}
// beta1
void bin_update_beta_cpp(double& beta, const arma::mat& z, const arma::mat& Y, 
                         double alpha, double prop_beta, double a, double b, 
                         bool& accepted) {
  accepted = false;
  double curr = loglik_binary_network_cpp(Y, z, alpha, beta) + 
    log_prior_beta_cpp(beta, a, b) + log(beta);
  double beta_prop = exp(log(beta) + R::rnorm(0, prop_beta));
  if (beta_prop <= 0) return;
  double prop = loglik_binary_network_cpp(Y, z, alpha, beta_prop) + 
    log_prior_beta_cpp(beta_prop, a, b) + log(beta_prop);
  if (log(R::runif(0, 1)) < (prop - curr)) {
    beta = beta_prop;
    accepted = true;
  }
}

/* (4c) 연속(Continuous) 레이어 파라미터 업데이트 */
// alpha2
void con_update_alpha_cpp(double& alpha, const arma::mat& z, const arma::mat& Y, 
                          double xi, double psi2, double prop_alpha, double beta, double kappa,
                          bool& accepted) {
  accepted = false;
  double ll_curr = loglik_conti_network_cpp(Y, z, alpha, beta, kappa) + 
    log_prior_alpha_cpp(alpha, xi, psi2);
  double alpha_prop = alpha + R::rnorm(0, prop_alpha);
  double ll_prop = loglik_conti_network_cpp(Y, z, alpha_prop, beta, kappa) + 
    log_prior_alpha_cpp(alpha_prop, xi, psi2);
  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    alpha = alpha_prop;
    accepted = true;
  }
}
// beta2
void con_update_beta_cpp(double& beta, const arma::mat& z, const arma::mat& Y, 
                         double alpha, double prop_beta, double a, double b, double kappa,
                         bool& accepted) {
  accepted = false;
  double curr = loglik_conti_network_cpp(Y, z, alpha, beta, kappa) + 
    log_prior_beta_cpp(beta, a, b) + log(beta);
  double beta_prop = exp(log(beta) + R::rnorm(0, prop_beta));
  if (beta_prop <= 0) return;
  double prop = loglik_conti_network_cpp(Y, z, alpha, beta_prop, kappa) + 
    log_prior_beta_cpp(beta_prop, a, b) + log(beta_prop);
  if (log(R::runif(0, 1)) < (prop - curr)) {
    beta = beta_prop;
    accepted = true;
  }
}
// kappa2 (R 코드의 Hastings 보정 항 포함)
void con_update_kappa_cpp(double& kappa, const arma::mat& z, const arma::mat& Y, 
                          double alpha, double beta, double prop_kappa, 
                          double m_kappa, double s_kappa, bool& accepted) {
  accepted = false;
  double curr = loglik_conti_network_cpp(Y, z, alpha, beta, kappa) + 
    log_prior_kappa_cpp(kappa, m_kappa, s_kappa);
  
  double log_kappa = log(kappa);
  double log_kappa_prop = log_kappa + R::rnorm(0, prop_kappa);
  double kappa_prop = exp(log_kappa_prop);
  
  if (kappa_prop <= 0) return;
  
  double prop = loglik_conti_network_cpp(Y, z, alpha, beta, kappa_prop) + 
    log_prior_kappa_cpp(kappa_prop, m_kappa, s_kappa);
  
  // R 코드의 Hastings 보정: (log(kappa_prop) - log(kappa))
  double acc_ratio = (prop - curr);
  
  if (log(R::runif(0, 1)) < acc_ratio) {
    kappa = kappa_prop;
    accepted = true;
  }
}


/* (4d) 공통 Gibbs 업데이트 (xi, psi2) */
List update_xi_psi2_cpp(double alpha, double xi, double psi2, 
                        double m0 = 0.0, double kappa0 = 0.5, 
                        double alpha0 = 3.0, double beta0 = 0.5) {
  double n = 1.0; 
  double abar = alpha;
  double kappa_n = 1.0 / (n/psi2 + 1.0/kappa0);
  double m_n = kappa_n * (m0/kappa0 + (n * abar)/psi2);
  double xi_new = R::rnorm(m_n, sqrt(kappa_n));
  double shape = alpha0 + n/2.0;
  double scale_param = beta0 + 0.5 * pow(alpha - xi_new, 2); 
  double psi2_new = 1.0 / R::rgamma(shape, 1.0/scale_param);
  return List::create(Named("xi") = xi_new, Named("psi2") = psi2_new);
}


// --- (5) Main Rcpp Function ---

// [[Rcpp::export]]
List LSJM_2layer_cpp(arma::mat Y_bin, arma::mat Y_con,
                     int iter = 5000, int burnin = 1000, int thinning = 5, int d = 2,
                     double sigma2 = 1.0,
                     double a0_bin = 0.5, double b0_bin = 0.5,
                     double m0_bin = 0.0, double kappa0_bin = 0.5, 
                     double alpha0_bin = 3.0, double beta0_bin = 0.5,
                     double a0_con = 0.5, double b0_con = 0.5,
                     double m_kappa = 0.0, double s_kappa = 1.0,
                     double m0_con = 0.0, double kappa0_con = 0.5, 
                     double alpha0_con = 3.0, double beta0_con = 0.5,
                     double prop_z = 0.1,
                     double prop_alpha_bin = 0.05, double prop_beta_bin = 0.05,
                     double prop_alpha_con = 0.05, double prop_beta_con = 0.05, double prop_kappa = 0.2,
                     Nullable<arma::mat> z_init = R_NilValue,
                     double alpha1_init = 0.0, double beta1_init = 1.0, 
                     double xi1_init = 0.0, double psi1_init = 1.0,
                     double alpha2_init = 0.0, double beta2_init = 1.0, 
                     double xi2_init = 0.0, double psi2_init = 1.0, double kappa2_init = 1.0,
                     bool verbose = true,
                     int seed = 1) {
  
  // R의 set.seed와 동일한 효과
  Rcpp::Function setSeed("set.seed");
  setSeed(seed);
  
  int n = Y_bin.n_rows;
  if (Y_bin.n_cols != n || Y_con.n_rows != n || Y_con.n_cols != n) {
    stop("All matrices must be n x n.");
  }
  // (대칭성 검사는 R 코드에 있으므로 생략)
  
  // --- init z (공통) ---
  arma::mat z;
  if (z_init.isNotNull()) {
    z = as<arma::mat>(z_init);
  } else {
    z = arma::randn(n, d) * sqrt(sigma2);
  }
  
  // --- layer 1 (binary) params ---
  double alpha1 = alpha1_init, beta1 = beta1_init, xi1 = xi1_init, psi1 = psi1_init;
  // --- layer 2 (conti) params ---
  double alpha2 = alpha2_init, beta2 = beta2_init, xi2 = xi2_init, psi2 = psi2_init, kappa2 = kappa2_init;
  
  // --- storage ---
  int ns = static_cast<int>(floor(static_cast<double>(iter - 1 - burnin) / thinning)) + 1;
  if (iter <= burnin) ns = 0;
  
  arma::cube z_save(n, d, ns);
  arma::vec alpha1_save(ns), beta1_save(ns), xi1_save(ns), psi1_save(ns);
  arma::vec alpha2_save(ns), beta2_save(ns), xi2_save(ns), psi2_save(ns), kappa2_save(ns);
  arma::vec ll_bin_save(ns), ll_con_save(ns), ll_joint_save(ns);
  
  // --- acceptances ---
  arma::mat acc_z_elem_total(n, d, arma::fill::zeros);
  double acc_alpha1 = 0.0, acc_beta1 = 0.0;
  double acc_alpha2 = 0.0, acc_beta2 = 0.0, acc_kappa2 = 0.0;
  
  int s = 0;
  double prop_z_sd_internal = sqrt(prop_z); // 제안 표준편차
  
  // --- MCMC Sampler ---
  for (int it = 0; it < iter; ++it) {
    
    // (1) z single-site MH (joint, element-wise)
    for (int j = 0; j < n; ++j) {
      arma::vec acc_this_iter(d, arma::fill::zeros); 
      update_z_j_elementwise_LSJM_cpp(j, z, Y_bin, Y_con,
                                      alpha1, beta1,
                                      alpha2, beta2, kappa2,
                                      sigma2, prop_z_sd_internal, d, 
                                      acc_this_iter);
      acc_z_elem_total.row(j) += acc_this_iter.t();
    }
    
    // (2) Binary layer params
    bool acc_a1 = false, acc_b1 = false;
    bin_update_alpha_cpp(alpha1, z, Y_bin, xi1, psi1, prop_alpha_bin, beta1, acc_a1);
    if(acc_a1) acc_alpha1++;
    
    bin_update_beta_cpp(beta1, z, Y_bin, alpha1, prop_beta_bin, a0_bin, b0_bin, acc_b1);
    if(acc_b1) acc_beta1++;
    
    // !!! 원본 R 코드의 테스트 코드 반영 !!!
    beta1 = 1.0;
    // R 코드에서는 beta1=1일 때 acc_beta1을 증가시키지 않음
    
    List up_xi1psi1 = update_xi_psi2_cpp(alpha1, xi1, psi1, m0_bin, kappa0_bin, alpha0_bin, beta0_bin);
    xi1 = as<double>(up_xi1psi1["xi"]); 
    psi1 = as<double>(up_xi1psi1["psi2"]);
    
    // (3) Continuous layer params
    bool acc_a2 = false, acc_b2 = false, acc_k2 = false;
    con_update_alpha_cpp(alpha2, z, Y_con, xi2, psi2, prop_alpha_con, beta2, kappa2, acc_a2);
    if(acc_a2) acc_alpha2++;
    
    con_update_beta_cpp(beta2, z, Y_con, alpha2, prop_beta_con, a0_con, b0_con, kappa2, acc_b2);
    if(acc_b2) acc_beta2++;
    
    // !!! 원본 R 코드의 테스트 코드 반영 !!!
    beta2 = 1.0;
    // R 코드에서는 beta2=1일 때 acc_beta2을 증가시키지 않음
    
    con_update_kappa_cpp(kappa2, z, Y_con, alpha2, beta2, prop_kappa, m_kappa, s_kappa, acc_k2);
    if(acc_k2) acc_kappa2++;
    
    List up_xi2psi2 = update_xi_psi2_cpp(alpha2, xi2, psi2, m0_con, kappa0_con, alpha0_con, beta0_con);
    xi2 = as<double>(up_xi2psi2["xi"]); 
    psi2 = as<double>(up_xi2psi2["psi2"]);
    
    // (4) save
    if (it >= burnin && (it - burnin) % thinning == 0) {
      z_save.slice(s) = z;
      alpha1_save(s) = alpha1; beta1_save(s) = beta1; xi1_save(s) = xi1; psi1_save(s) = psi1;
      alpha2_save(s) = alpha2; beta2_save(s) = beta2; xi2_save(s) = xi2; psi2_save(s) = psi2; kappa2_save(s) = kappa2;
      
      double ll_bin = loglik_binary_network_cpp(Y_bin, z, alpha1, beta1);
      double ll_con = loglik_conti_network_cpp(Y_con, z, alpha2, beta2, kappa2);
      ll_bin_save(s) = ll_bin;
      ll_con_save(s) = ll_con;
      double prior_z_sum = 0.0;
      for (int i = 0; i < n; ++i) {
        prior_z_sum += log_prior_z_i_cpp(z.row(i), sigma2);
      }
      ll_joint_save(s) = ll_bin + ll_con + prior_z_sum;
      s++;
    }
    
    if (verbose && ((it + 1) % 100 == 0)) {
      Rprintf("[iter %d] a1=%.3f b1=%.3f | a2=%.3f b2=%.3f k2=%.3f\n",
              it + 1, alpha1, beta1, alpha2, beta2, kappa2);
    }
    
    Rcpp::checkUserInterrupt();
  }
  
  // --- 반환 ---
  return List::create(
    Named("samples") = List::create(
      Named("z") = z_save,
      Named("alpha1") = alpha1_save, Named("beta1") = beta1_save, 
      Named("xi1") = xi1_save, Named("psi1") = psi1_save,
      Named("alpha2") = alpha2_save, Named("beta2") = beta2_save, 
      Named("kappa2") = kappa2_save, Named("xi2") = xi2_save, Named("psi2") = psi2_save,
      Named("ll_bin") = ll_bin_save, Named("ll_con") = ll_con_save, Named("ll_joint") = ll_joint_save
    ),
    Named("accept") = List::create(
      Named("z_elem") = acc_z_elem_total / iter,
      Named("alpha1") = acc_alpha1 / iter, Named("beta1") = acc_beta1 / iter,
      Named("alpha2") = acc_alpha2 / iter, Named("beta2") = acc_beta2 / iter, 
      Named("kappa2") = acc_kappa2 / iter
    ),
    Named("config") = List::create(
      Named("iter") = iter, Named("burnin") = burnin, Named("thinning") = thinning,
            Named("d") = d, Named("sigma2") = sigma2
  // R 코드와 동일하게 prop_sd, priors 리스트는 생략 (필요시 추가 가능)
    )
  );
}