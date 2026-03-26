// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// -----------------------------------------------------------------------------
// (0) Utility (C++ 내부 헬퍼 함수)
// -----------------------------------------------------------------------------

// R의 L2_dist 함수
arma::mat L2_dist_cpp(const arma::mat& z) {
  int n = z.n_rows;
  arma::mat D(n, n, arma::fill::zeros);
  if (n == 0) {
    return D;
  }
  
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      // arma::norm은 두 벡터(행) 간의 유클리드 거리를 계산합니다.
      double d = arma::norm(z.row(i) - z.row(j), 2); 
      D(i, j) = d;
      D(j, i) = d;
    }
  }
  return D;
}

// -----------------------------------------------------------------------------
// (1) Log-Likelihood 및 Prior 함수 (C++ 내부 헬퍼)
// -----------------------------------------------------------------------------

// --- Log-Likelihoods ---

// R의 loglik_binary_network_hier
double loglik_binary_cpp(const arma::mat& Y, 
                         const arma::mat& u, 
                         double alpha, 
                         const arma::vec& theta, 
                         double beta) {
  
  int n = Y.n_rows;
  if (n == 0) return 0.0;
  
  arma::mat D = L2_dist_cpp(u);
  
  // R의 outer(theta, theta, "+")
  arma::mat theta_sum(n, n);
  for (int i = 0; i < n; ++i) {
    theta_sum.col(i) = theta + theta(i);
  }
  
  arma::mat eta = alpha + theta_sum - beta * D;
  // R::plogis(eta)와 동일 (요소별 연산)
  arma::mat pi = 1.0 / (1.0 + arma::exp(-eta)); 
  
  double ll = 0.0;
  double eps = 1e-11; 
  
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (Y(i, j) == 1.0) {
        ll += log(pi(i, j) + eps);
      } else {
        ll += log(1.0 - pi(i, j) + eps);
      }
    }
  }
  return ll;
}

// R의 loglik_conti_network_hier
double loglik_conti_cpp(const arma::mat& Y, 
                        const arma::mat& u, 
                        double alpha, 
                        const arma::vec& theta, 
                        double beta, 
                        double kappa) {
  
  int n = Y.n_rows;
  if (n == 0) return 0.0;
  
  arma::mat D = L2_dist_cpp(u);
  
  arma::mat theta_sum(n, n);
  for (int i = 0; i < n; ++i) {
    theta_sum.col(i) = theta + theta(i);
  }
  
  arma::mat eta = alpha + theta_sum - beta * D;
  
  double ll = 0.0;
  double eps = 1e-12;
  double mu_log = 0.0;
  double y_ij = 0.0;
  double sd_log = sqrt(kappa);
  
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      y_ij = Y(i, j) > eps ? Y(i, j) : eps; // pmax
      mu_log = eta(i, j) - kappa / 2.0;
      
      // R::dlnorm(x, meanlog, sdlog, give_log)
      ll += R::dlnorm(y_ij, mu_log, sd_log, true); 
    }
  }
  return ll;
}

// --- Log-Priors ---

// z_i ~ N(0, sigma2_z * I)
double log_prior_z_i(const arma::rowvec& z_i, double sigma2_z) {
  // dnorm(z_i, 0, sd, log=T)
  double log_prior = 0.0;
  double sd = sqrt(sigma2_z);
  for(double val : z_i) {
    log_prior += R::dnorm(val, 0.0, sd, true);
  }
  return log_prior;
}


// u_i^(l) | z_i, sigma2_l ~ N(z_i, sigma2_l * I)
double log_prior_u_i(const arma::rowvec& u_i, const arma::rowvec& z_i, double sigma2_l) {
  // dnorm(u_i, z_i, sd, log=T)
  double log_prior = 0.0;
  double sd = sqrt(sigma2_l);
  for (std::size_t d = 0; d < u_i.n_elem; ++d) {
    double u_val = u_i[d];
    double z_val = z_i[d];
    log_prior += R::dnorm(u_val, z_val, sd, true);
  }
  return log_prior;
}



// theta_i^(l) ~ N(0, sigma2_theta)
double log_prior_theta_i(double theta_i, double sigma2_theta) {
  return R::dnorm(theta_i, 0.0, sqrt(sigma2_theta), true);
}

// alpha ~ N(xi, psi2)
double log_prior_alpha(double alpha, double xi, double psi2) {
  return R::dnorm(alpha, xi, sqrt(psi2), true);
}

// beta ~ Gamma(a, b)
double log_prior_beta(double beta, double a, double b) {
  return R::dgamma(beta, a, 1.0/b, true); // R의 dgamma는 rate (1/scale)을 사용. R 코드는 rate (b) 사용
}

// log(kappa) ~ N(a, b)
double log_prior_kappa(double kappa, double a, double b) {
  return R::dnorm(log(kappa), a, sqrt(b), true) - log(kappa);
}

// -----------------------------------------------------------------------------
// (2) Update 함수 (C++ 내부 헬퍼)
// -----------------------------------------------------------------------------

// --- Binary Layer (l=1) ---

Rcpp::List bin_update_alpha_hier_cpp(double alpha, const arma::mat& u, const arma::vec& theta, 
                                     const arma::mat& Y, double xi, double psi2, 
                                     double prop_alpha, double beta) {
  double ll_curr = loglik_binary_cpp(Y, u, alpha, theta, beta) + 
    log_prior_alpha(alpha, xi, psi2);
  double alpha_prop = alpha + R::rnorm(0, prop_alpha);
  double ll_prop = loglik_binary_cpp(Y, u, alpha_prop, theta, beta) + 
    log_prior_alpha(alpha_prop, xi, psi2);
  
  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    return Rcpp::List::create(Named("alpha") = alpha_prop, Named("accepted") = true);
  } else {
    return Rcpp::List::create(Named("alpha") = alpha, Named("accepted") = false);
  }
}

Rcpp::List bin_update_beta_hier_cpp(double beta, const arma::mat& u, const arma::vec& theta, 
                                    const arma::mat& Y, double alpha, double prop_beta, 
                                    double a, double b){
  double curr = loglik_binary_cpp(Y, u, alpha, theta, beta) + 
    log_prior_beta(beta, a, b) + log(beta);
  double beta_prop = exp(log(beta) + R::rnorm(0, prop_beta));
  
  double prop = loglik_binary_cpp(Y, u, alpha, theta, beta_prop) + 
    log_prior_beta(beta_prop, a, b) + log(beta_prop);
  
  if (log(R::runif(0, 1)) < (prop - curr)) {
    return Rcpp::List::create(Named("beta") = beta_prop, Named("accepted") = true);
  } else {
    return Rcpp::List::create(Named("beta") = beta, Named("accepted") = false);
  }
}

Rcpp::List bin_update_theta_i_hier_cpp(int i, arma::vec theta, const arma::mat& Y, 
                                       const arma::mat& u, double alpha, double beta, 
                                       double prop_theta, double sigma2_theta) {
  double theta_i_curr = theta(i);
  double ll_curr = loglik_binary_cpp(Y, u, alpha, theta, beta) +
    log_prior_theta_i(theta_i_curr, sigma2_theta);
  
  arma::vec theta_prop = theta;
  double theta_i_prop = theta_i_curr + R::rnorm(0, prop_theta);
  theta_prop(i) = theta_i_prop;
  
  double ll_prop = loglik_binary_cpp(Y, u, alpha, theta_prop, beta) +
    log_prior_theta_i(theta_i_prop, sigma2_theta);
  
  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    return Rcpp::List::create(Named("theta") = theta_prop, Named("accepted") = true);
  } else {
    return Rcpp::List::create(Named("theta") = theta, Named("accepted") = false);
  }
}

Rcpp::List bin_update_u_i_hier_cpp(int i, arma::mat u, const arma::mat& Y, double alpha, 
                                   const arma::vec& theta, double beta, const arma::rowvec& z_i, 
                                   double sigma2_l, double prop_u, int d) {
  arma::rowvec u_i_curr = u.row(i);
  double ll_curr = loglik_binary_cpp(Y, u, alpha, theta, beta) +
    log_prior_u_i(u_i_curr, z_i, sigma2_l);
  
  arma::mat u_prop = u;
  arma::rowvec u_i_prop = u_i_curr + as<arma::rowvec>(Rcpp::rnorm(d, 0, prop_u));
  u_prop.row(i) = u_i_prop;
  
  double ll_prop = loglik_binary_cpp(Y, u_prop, alpha, theta, beta) +
    log_prior_u_i(u_i_prop, z_i, sigma2_l);
  
  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    return Rcpp::List::create(Named("u") = u_prop, Named("accepted") = true);
  } else {
    return Rcpp::List::create(Named("u") = u, Named("accepted") = false);
  }
}

// --- Continuous Layer (l=2) ---

Rcpp::List con_update_alpha_hier_cpp(double alpha, const arma::mat& u, const arma::vec& theta, 
                                     const arma::mat& Y, double xi, double psi2, 
                                     double prop_alpha, double beta, double kappa) {
  double ll_curr = loglik_conti_cpp(Y, u, alpha, theta, beta, kappa) + 
    log_prior_alpha(alpha, xi, psi2);
  double alpha_prop = alpha + R::rnorm(0, prop_alpha);
  double ll_prop = loglik_conti_cpp(Y, u, alpha_prop, theta, beta, kappa) + 
    log_prior_alpha(alpha_prop, xi, psi2);
  
  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    return Rcpp::List::create(Named("alpha") = alpha_prop, Named("accepted") = true);
  } else {
    return Rcpp::List::create(Named("alpha") = alpha, Named("accepted") = false);
  }
}

Rcpp::List con_update_beta_hier_cpp(double beta, const arma::mat& u, const arma::vec& theta, 
                                    const arma::mat& Y, double alpha, double prop_beta, 
                                    double a, double b, double kappa){
  double curr = loglik_conti_cpp(Y, u, alpha, theta, beta, kappa) + 
    log_prior_beta(beta, a, b) + log(beta);
  double beta_prop = exp(log(beta) + R::rnorm(0, prop_beta));
  
  double prop = loglik_conti_cpp(Y, u, alpha, theta, beta_prop, kappa) + 
    log_prior_beta(beta_prop, a, b) + log(beta_prop);
  
  if (log(R::runif(0, 1)) < (prop - curr)) {
    return Rcpp::List::create(Named("beta") = beta_prop, Named("accepted") = true);
  } else {
    return Rcpp::List::create(Named("beta") = beta, Named("accepted") = false);
  }
}

Rcpp::List con_update_kappa_hier_cpp(double kappa, const arma::mat& u, const arma::vec& theta, 
                                     const arma::mat& Y, double alpha, double beta, 
                                     double prop_kappa, double m_kappa, double s_kappa){
  double curr = loglik_conti_cpp(Y, u, alpha, theta, beta, kappa) + 
    log_prior_kappa(kappa, m_kappa, s_kappa);
  double kappa_prop = exp(log(kappa) + R::rnorm(0, prop_kappa));
  
  double prop = loglik_conti_cpp(Y, u, alpha, theta, beta, kappa_prop) + 
    log_prior_kappa(kappa_prop, m_kappa, s_kappa);
  double acc = (prop - curr) + (log(kappa_prop) - log(kappa)); // 보정 항
  
  if (log(R::runif(0, 1)) < acc) {
    return Rcpp::List::create(Named("kappa") = kappa_prop, Named("accepted") = true);
  } else {
    return Rcpp::List::create(Named("kappa") = kappa, Named("accepted") = false);
  }
}

Rcpp::List con_update_theta_i_hier_cpp(int i, arma::vec theta, const arma::mat& Y, 
                                       const arma::mat& u, double alpha, double beta, double kappa,
                                       double prop_theta, double sigma2_theta) {
  double theta_i_curr = theta(i);
  double ll_curr = loglik_conti_cpp(Y, u, alpha, theta, beta, kappa) +
    log_prior_theta_i(theta_i_curr, sigma2_theta);
  
  arma::vec theta_prop = theta;
  double theta_i_prop = theta_i_curr + R::rnorm(0, prop_theta);
  theta_prop(i) = theta_i_prop;
  
  double ll_prop = loglik_conti_cpp(Y, u, alpha, theta_prop, beta, kappa) +
    log_prior_theta_i(theta_i_prop, sigma2_theta);
  
  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    return Rcpp::List::create(Named("theta") = theta_prop, Named("accepted") = true);
  } else {
    return Rcpp::List::create(Named("theta") = theta, Named("accepted") = false);
  }
}

Rcpp::List con_update_u_i_hier_cpp(int i, arma::mat u, const arma::mat& Y, double alpha, 
                                   const arma::vec& theta, double beta, double kappa, 
                                   const arma::rowvec& z_i, double sigma2_l, 
                                   double prop_u, int d) {
  arma::rowvec u_i_curr = u.row(i);
  double ll_curr = loglik_conti_cpp(Y, u, alpha, theta, beta, kappa) +
    log_prior_u_i(u_i_curr, z_i, sigma2_l);
  
  arma::mat u_prop = u;
  arma::rowvec u_i_prop = u_i_curr + as<arma::rowvec>(Rcpp::rnorm(d, 0, prop_u));
  u_prop.row(i) = u_i_prop;
  
  double ll_prop = loglik_conti_cpp(Y, u_prop, alpha, theta, beta, kappa) +
    log_prior_u_i(u_i_prop, z_i, sigma2_l);
  
  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    return Rcpp::List::create(Named("u") = u_prop, Named("accepted") = true);
  } else {
    return Rcpp::List::create(Named("u") = u, Named("accepted") = false);
  }
}

// --- Gibbs Samplers ---

// (NEW) sigma_l^2 | u^(l), z ~ IG(a_post, b_post)
double update_sigma2_l_gibbs_cpp(const arma::mat& u_l, const arma::mat& z, 
                                 double a1, double b1) {
  int n = u_l.n_rows;
  int d = u_l.n_cols;
  
  double a_post = a1 + (static_cast<double>(n) * d) / 2.0;
  
  double ss = arma::accu(arma::pow(u_l - z, 2));
  double b_post = b1 + ss / 2.0;
  
  // IG(a, b) = 1 / Gamma(a, rate=b)
  return 1.0 / R::rgamma(a_post, 1.0/b_post); // R::rgamma는 rate (1/scale) 사용
}

// --- Global Parameter (z_i) ---

// (NEW) z_i | u_i^(1), u_i^(2), sigma2_1, sigma2_2
Rcpp::List update_z_i_hier_cpp(int i, arma::mat z, const arma::mat& u1, const arma::mat& u2,
                               double sigma2_1, double sigma2_2, double prop_z, int d, 
                               double sigma2_z) {
  arma::rowvec z_i_curr = z.row(i);
  
  double ll_curr = log_prior_u_i(u1.row(i), z_i_curr, sigma2_1) +
    log_prior_u_i(u2.row(i), z_i_curr, sigma2_2) +
    log_prior_z_i(z_i_curr, sigma2_z);
  
  arma::mat z_prop = z;
  arma::rowvec z_i_prop = z_i_curr + as<arma::rowvec>(Rcpp::rnorm(d, 0, prop_z));
  z_prop.row(i) = z_i_prop;
  
  double ll_prop = log_prior_u_i(u1.row(i), z_i_prop, sigma2_1) +
    log_prior_u_i(u2.row(i), z_i_prop, sigma2_2) +
    log_prior_z_i(z_i_prop, sigma2_z);
  
  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    return Rcpp::List::create(Named("z") = z_prop, Named("accepted") = true);
  } else {
    return Rcpp::List::create(Named("z") = z, Named("accepted") = false);
  }
}


// -----------------------------------------------------------------------------
// (3) 메인 드라이버 (Rcpp)
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List MNLPM_2layer_cpp(
    const arma::mat& Y_bin, const arma::mat& Y_con,
    int iter = 5000, int burnin = 1000, int thinning = 5,
    int d = 2,
    double sigma2_z = 1.0,
    double sigma2_theta_bin = 1.0, double sigma2_theta_con = 1.0,
    double a1 = 2.0, double b1 = 1.0,
    double a0_bin = 0.5, double b0_bin = 0.5,
    double a0_con = 0.5, double b0_con = 0.5,
    double xi1_prior = 0, double psi1_prior = 1.0,
    double xi2_prior = 0, double psi2_prior = 1.0,
    double m_kappa = 0, double s_kappa = 1,
    double prop_z = 0.1,
    double prop_u_bin = 0.1, double prop_u_con = 0.1,
    double prop_theta_bin = 0.1, double prop_theta_con = 0.1,
    double prop_alpha_bin = 0.05, double prop_beta_bin = 0.05,
    double prop_alpha_con = 0.05, double prop_beta_con = 0.05, 
    double prop_kappa = 0.2,
    Rcpp::Nullable<arma::mat> z_init = R_NilValue,
    double alpha1_init = 0, double beta1_init = 1,
    double alpha2_init = 0, double beta2_init = 1, double kappa2_init = 1,
    bool verbose = true,
    int seed = 1) {
  
  // R의 set.seed()는 Rcpp가 R::rnorm 등을 호출할 때 영향을 줍니다.
  // Rcpp::Function set_seed("set.seed");
  // set_seed(seed); // Rcpp 내에서 시드 설정 (필요시)
  
  int n = Y_bin.n_rows;
  
  // --- init z (global) ---
  arma::mat z;
  if (z_init.isNotNull()) {
    z = Rcpp::as<arma::mat>(z_init);
  } else {
    z = arma::randn(n, d) * std::sqrt(sigma2_z);
  }
  
  // --- init layer-specific params ---
  // l=1 (binary)
  double sigma2_1 = 1.0;
  arma::mat u1 = z + arma::randn(n, d) * std::sqrt(sigma2_1);
  arma::vec theta1 = arma::randn(n) * std::sqrt(sigma2_theta_bin);
  double alpha1 = alpha1_init; 
  double beta1 = beta1_init;
  
  // l=2 (continuous)
  double sigma2_2 = 1.0;
  arma::mat u2 = z + arma::randn(n, d) * std::sqrt(sigma2_2);
  arma::vec theta2 = arma::randn(n) * std::sqrt(sigma2_theta_con);
  double alpha2 = alpha2_init;
  double beta2 = beta2_init; 
  double kappa2 = kappa2_init;
  
  // hyperparams (고정)
  double xi1 = xi1_prior; double psi1 = psi1_prior;
  double xi2 = xi2_prior; double psi2 = psi2_prior;
  
  // --- storage ---
  int ns = 0;
  for (int it = burnin + 1; it <= iter; ++it) {
    if ((it - (burnin + 1)) % thinning == 0) {
      ns++;
    }
  }
  
  arma::cube z_save(n, d, ns);
  // binary layer (l=1)
  arma::cube u1_save(n, d, ns);
  arma::mat theta1_save(ns, n);
  arma::vec alpha1_save(ns);
  arma::vec beta1_save(ns);
  arma::vec sigma2_1_save(ns);
  // conti layer (l=2)
  arma::cube u2_save(n, d, ns);
  arma::mat theta2_save(ns, n);
  arma::vec alpha2_save(ns);
  arma::vec beta2_save(ns);
  arma::vec kappa2_save(ns);
  arma::vec sigma2_2_save(ns);
  
  // traces
  arma::vec ll_bin_save(ns);
  arma::vec ll_con_save(ns);
  
  // acceptances
  arma::vec acc_z(n, arma::fill::zeros);
  arma::vec acc_u1(n, arma::fill::zeros);
  arma::vec acc_theta1(n, arma::fill::zeros);
  double acc_alpha1 = 0.0; 
  double acc_beta1 = 0.0;
  arma::vec acc_u2(n, arma::fill::zeros);
  arma::vec acc_theta2(n, arma::fill::zeros);
  double acc_alpha2 = 0.0; double acc_beta2 = 0.0; double acc_kappa2 = 0.0;
  
  int s = 0; // 저장 인덱스
  
  // --- sampler ---
  Rcpp::List up;
  
  for (int it = 1; it <= iter; ++it) {
    
    // (1) Update global z (node-by-node)
    for (int j = 0; j < n; ++j) {
      up = update_z_i_hier_cpp(j, z, u1, u2, sigma2_1, sigma2_2, prop_z, d, sigma2_z);
      z = Rcpp::as<arma::mat>(up["z"]);
      if (Rcpp::as<bool>(up["accepted"])) acc_z(j)++;
    }
    
    // (2) Update Binary layer (l=1) params
    // u_i^(1)
    for (int j = 0; j < n; ++j) {
      up = bin_update_u_i_hier_cpp(j, u1, Y_bin, alpha1, theta1, beta1, 
                                   z.row(j), sigma2_1, prop_u_bin, d);
      u1 = Rcpp::as<arma::mat>(up["u"]);
      if (Rcpp::as<bool>(up["accepted"])) acc_u1(j)++;
    }
    // theta_i^(1)
    for (int j = 0; j < n; ++j) {
      up = bin_update_theta_i_hier_cpp(j, theta1, Y_bin, u1, alpha1, beta1,
                                       prop_theta_bin, sigma2_theta_bin);
      theta1 = Rcpp::as<arma::vec>(up["theta"]);
      if (Rcpp::as<bool>(up["accepted"])) acc_theta1(j)++;
    }
    // alpha^(1)
    up = bin_update_alpha_hier_cpp(alpha1, u1, theta1, Y_bin, xi1, psi1, prop_alpha_bin, beta1);
    alpha1 = Rcpp::as<double>(up["alpha"]);
    if (Rcpp::as<bool>(up["accepted"])) acc_alpha1++;
    // beta^(1)
    up = bin_update_beta_hier_cpp(beta1, u1, theta1, Y_bin, alpha1, prop_beta_bin, a0_bin, b0_bin);
    beta1 = Rcpp::as<double>(up["beta"]);
    // beta1 를 1로 고정
    // beta1 = 1;
    if (Rcpp::as<bool>(up["accepted"])) acc_beta1++;
    // sigma2^(1)
    sigma2_1 = update_sigma2_l_gibbs_cpp(u1, z, a1, b1);
    
    
    // (3) Update Continuous layer (l=2) params
    // u_i^(2)
    for (int j = 0; j < n; ++j) {
      up = con_update_u_i_hier_cpp(j, u2, Y_con, alpha2, theta2, beta2, kappa2,
                                   z.row(j), sigma2_2, prop_u_con, d);
      u2 = Rcpp::as<arma::mat>(up["u"]);
      if (Rcpp::as<bool>(up["accepted"])) acc_u2(j)++;
    }
    // theta_i^(2)
    for (int j = 0; j < n; ++j) {
      up = con_update_theta_i_hier_cpp(j, theta2, Y_con, u2, alpha2, beta2, kappa2,
                                       prop_theta_con, sigma2_theta_con);
      theta2 = Rcpp::as<arma::vec>(up["theta"]);
      if (Rcpp::as<bool>(up["accepted"])) acc_theta2(j)++;
    }
    // alpha^(2)
    up = con_update_alpha_hier_cpp(alpha2, u2, theta2, Y_con, xi2, psi2, prop_alpha_con, beta2, kappa2);
    alpha2 = Rcpp::as<double>(up["alpha"]);
    if (Rcpp::as<bool>(up["accepted"])) acc_alpha2++;
    // beta^(2)
    up = con_update_beta_hier_cpp(beta2, u2, theta2, Y_con, alpha2, prop_beta_con, a0_con, b0_con, kappa2);
    beta2 = Rcpp::as<double>(up["beta"]);
    // beta2 를 1로 고정
    // beta2 = 1;
    if (Rcpp::as<bool>(up["accepted"])) acc_beta2++;
    // kappa^(2)
    up = con_update_kappa_hier_cpp(kappa2, u2, theta2, Y_con, alpha2, beta2, prop_kappa, m_kappa, s_kappa);
    kappa2 = Rcpp::as<double>(up["kappa"]);
    if (Rcpp::as<bool>(up["accepted"])) acc_kappa2++;
    // sigma2^(2)
    sigma2_2 = update_sigma2_l_gibbs_cpp(u2, z, a1, b1);
    
    
    // (4) save
    if (it > burnin && (it - (burnin + 1)) % thinning == 0) {
      z_save.slice(s) = z;
      
      u1_save.slice(s) = u1;
      theta1_save.row(s) = theta1.t();
      alpha1_save(s) = alpha1;
      beta1_save(s) = beta1;
      sigma2_1_save(s) = sigma2_1;
      
      u2_save.slice(s) = u2;
      theta2_save.row(s) = theta2.t();
      alpha2_save(s) = alpha2;
      beta2_save(s) = beta2;
      kappa2_save(s) = kappa2;
      sigma2_2_save(s) = sigma2_2;
      
      ll_bin_save(s) = loglik_binary_cpp(Y_bin, u1, alpha1, theta1, beta1);
      ll_con_save(s) = loglik_conti_cpp(Y_con, u2, alpha2, theta2, beta2, kappa2);
      
      s++; // 저장 인덱스 증가
    }
    
    if (verbose && it % 100 == 0) {
      // Rprintf는 R의 콘솔에 출력합니다.
      Rprintf("[iter %d] a1=%.2f b1=%.2f | a2=%.2f b2=%.2f k2=%.2f | s2_1=%.2f s2_2=%.2f | ll_bin=%.1f ll_con=%.1f\n",
              it, alpha1, beta1, alpha2, beta2, kappa2, sigma2_1, sigma2_2,
              loglik_binary_cpp(Y_bin, u1, alpha1, theta1, beta1),
              loglik_conti_cpp(Y_con, u2, alpha2, theta2, beta2, kappa2));
    }
  } // end MCMC loop
  // Rcpp::List로 결과 반환
  // Procrustes 정렬은 R에서 수행합니다.
  return List::create(
    Named("samples") = List::create(
      Named("z") = z_save,
      Named("u1") = u1_save,
      Named("theta1") = theta1_save,
      Named("alpha1") = alpha1_save, 
      Named("beta1") = beta1_save, 
      Named("sigma2_1") = sigma2_1_save,
      Named("u2") = u2_save,
      Named("theta2") = theta2_save,
      Named("alpha2") = alpha2_save, 
      Named("beta2") = beta2_save, 
      Named("kappa2") = kappa2_save, 
      Named("sigma2_2") = sigma2_2_save,
      Named("ll_bin") = ll_bin_save, 
      Named("ll_con") = ll_con_save
    ),
    Named("accept") = List::create(
      Named("z") = acc_z / iter,
      Named("u1") = acc_u1 / iter, 
      Named("theta1") = acc_theta1 / iter,
      Named("alpha1") = acc_alpha1 / iter, 
      Named("beta1") = acc_beta1 / iter,
      Named("u2") = acc_u2 / iter,
      Named("theta2") = acc_theta2 / iter,
      Named("alpha2") = acc_alpha2 / iter, 
      Named("beta2") = acc_beta2 / iter, 
      Named("kappa2") = acc_kappa2 / iter
    )
  );
}








