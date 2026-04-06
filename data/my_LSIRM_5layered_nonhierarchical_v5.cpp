// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// =========================================================
// Helpers
// =========================================================

// Stable log(1 + exp(x))
inline double log1pexp(double x) {
  if (x > 0) return x + log1p(exp(-x));
  else return log1p(exp(x));
}

// Inverse Logit (stable)
static inline double inv_logit(double x) {
  if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
  const double ex = std::exp(x);
  return ex / (1.0 + ex);
}

// Euclidean distance
inline double calc_dist(const rowvec& a, const rowvec& b) {
  return sqrt(sum(square(a - b)));
}

// GRM log probability (0-based y, descending thresholds)
// P(Y >= k) = logistic(eta + beta_{j,k}), beta in descending order
template <typename RowLike>
static inline double log_grm_prob_y0based(int y, const RowLike& beta_j, double eta) {
  const int Km1 = (int)beta_j.n_elem;
  const int K = Km1 + 1;
  const double eps = 1e-10;

  auto p_ge = [&](int k) -> double {
    return inv_logit(eta + beta_j((arma::uword)(k - 1)));
  };

  double p = 0.0;
  if (y <= 0) {
    p = 1.0 - p_ge(1);
  } else if (y >= K - 1) {
    p = p_ge(K - 1);
  } else {
    p = p_ge(y) - p_ge(y + 1);
  }

  if (!std::isfinite(p) || p <= eps) p = eps;
  if (p >= 1.0 - eps) p = 1.0 - eps;
  return std::log(p);
}

// =========================================================
// Main MCMC Function (5-layered: bin, con(robust), cnt, ord1-GRM, ord2-GRM)
// =========================================================
// [[Rcpp::export]]
List run_lsirm_v5_cpp(
    mat Y_bin, mat Y_con, mat Y_cnt, IntegerMatrix Y_ord1, IntegerMatrix Y_ord2,
    int d, int n_iter, int burnin, int thin,
    List hyper, List prop_sd, List init,
    bool verbose, bool fix_gamma,
    double nu2
) {
  // Dimensions
  int n = Y_bin.n_rows;
  if (n == 0) n = Y_con.n_rows;
  if (n == 0) n = Y_cnt.n_rows;
  if (n == 0) n = Y_ord1.nrow();
  if (n == 0) n = Y_ord2.nrow();

  int P1 = Y_bin.n_cols;
  int P2 = Y_con.n_cols;
  int P3 = Y_cnt.n_cols;
  int P4 = Y_ord1.ncol();
  int P5 = Y_ord2.ncol();

  // Find K1 (max of Y_ord1) and K2 (max of Y_ord2)
  int K1 = 0;
  for(int i=0; i<n; ++i)
    for(int j=0; j<P4; ++j)
      if(Y_ord1(i,j) > K1) K1 = Y_ord1(i,j);

  int K2 = 0;
  for(int i=0; i<n; ++i)
    for(int j=0; j<P5; ++j)
      if(Y_ord2(i,j) > K2) K2 = Y_ord2(i,j);

  int K1m1 = (K1 > 0) ? K1 - 1 : 0;
  int K2m1 = (K2 > 0) ? K2 - 1 : 0;

  // Convert Y_ord to 0-based for GRM
  imat Y_ord1_0(n, P4);
  for(int i=0; i<n; ++i)
    for(int j=0; j<P4; ++j)
      Y_ord1_0(i,j) = Y_ord1(i,j) - 1;

  imat Y_ord2_0(n, P5);
  for(int i=0; i<n; ++i)
    for(int j=0; j<P5; ++j)
      Y_ord2_0(i,j) = Y_ord2(i,j) - 1;

  if (verbose) {
    Rcout << "n=" << n << " P1=" << P1 << " P2=" << P2
          << " P3=" << P3 << " P4=" << P4 << " P5=" << P5
          << " K1=" << K1 << " K2=" << K2 << " nu2=" << nu2
          << " [GRM ordinal]\n";
  }

  // Unpack Hyperparameters
  double a_sigma = hyper["a_sigma"]; double b_sigma = hyper["b_sigma"];
  double a_tau1  = hyper["a_tau1"];  double b_tau1  = hyper["b_tau1"];
  double a_tau2  = hyper["a_tau2"];  double b_tau2  = hyper["b_tau2"];
  double a_tau3  = hyper["a_tau3"];  double b_tau3  = hyper["b_tau3"];
  double a_sigma0= hyper["a_sigma0"];double b_sigma0= hyper["b_sigma0"];

  double mu_log_gamma1 = hyper["mu_log_gamma1"]; double sd_log_gamma1 = hyper["sd_log_gamma1"];
  double mu_log_gamma2 = hyper["mu_log_gamma2"]; double sd_log_gamma2 = hyper["sd_log_gamma2"];
  double mu_log_gamma3 = hyper["mu_log_gamma3"]; double sd_log_gamma3 = hyper["sd_log_gamma3"];
  double mu_log_gamma4 = hyper["mu_log_gamma4"]; double sd_log_gamma4 = hyper["sd_log_gamma4"];
  double mu_log_gamma5 = hyper["mu_log_gamma5"]; double sd_log_gamma5 = hyper["sd_log_gamma5"];
  double mu_log_kappa = hyper["mu_log_kappa"]; double sd_log_kappa = hyper["sd_log_kappa"];

  // GRM threshold priors
  double mu_beta4 = hyper["mu_beta4"]; double sd_beta4_hyp = hyper["sd_beta4"];
  double mu_beta5 = hyper["mu_beta5"]; double sd_beta5_hyp = hyper["sd_beta5"];

  // Unpack Proposal SDs
  double sd_alpha1 = prop_sd["alpha1"];
  double sd_alpha2 = prop_sd["alpha2"];
  double sd_alpha3 = prop_sd["alpha3"];
  double sd_alpha4 = prop_sd["alpha4"];
  double sd_alpha5 = prop_sd["alpha5"];
  double sd_beta1 = prop_sd["beta1"];
  double sd_beta2 = prop_sd["beta2"];
  double sd_beta3 = prop_sd["beta3"];
  double sd_beta4_prop = prop_sd["beta4"];
  double sd_beta5_prop = prop_sd["beta5"];
  double sd_log_gamma1_prop = prop_sd["log_gamma1"];
  double sd_log_gamma2_prop = prop_sd["log_gamma2"];
  double sd_log_gamma3_prop = prop_sd["log_gamma3"];
  double sd_log_gamma4_prop = prop_sd["log_gamma4"];
  double sd_log_gamma5_prop = prop_sd["log_gamma5"];
  double sd_log_kappa_prop = prop_sd["log_kappa"];
  double sd_a = prop_sd["a"];
  double sd_b1 = prop_sd["b1"];
  double sd_b2 = prop_sd["b2"];
  double sd_b3 = prop_sd["b3"];
  double sd_b4 = prop_sd["b4"];
  double sd_b5 = prop_sd["b5"];

  // Initialize Parameters
  vec alpha1 = as<vec>(init["alpha1"]);
  vec alpha2 = as<vec>(init["alpha2"]);
  vec alpha3 = as<vec>(init["alpha3"]);
  vec alpha4 = as<vec>(init["alpha4"]);
  vec alpha5 = as<vec>(init["alpha5"]);
  vec beta1 = as<vec>(init["beta1"]);
  vec beta2 = as<vec>(init["beta2"]);
  vec beta3 = as<vec>(init["beta3"]);

  mat a = as<mat>(init["a"]);
  mat b1 = as<mat>(init["b1"]);
  mat b2 = as<mat>(init["b2"]);
  mat b3 = as<mat>(init["b3"]);
  mat b4 = as<mat>(init["b4"]);
  mat b5 = as<mat>(init["b5"]);

  double log_gamma1 = init["log_gamma1"];  double gamma_val1 = exp(log_gamma1);
  double log_gamma2 = init["log_gamma2"];  double gamma_val2 = exp(log_gamma2);
  double log_gamma3 = init["log_gamma3"];  double gamma_val3 = exp(log_gamma3);
  double log_gamma4 = init["log_gamma4"];  double gamma_val4 = exp(log_gamma4);
  double log_gamma5 = init["log_gamma5"];  double gamma_val5 = exp(log_gamma5);
  double log_kappa = init["log_kappa"];
  double kappa = exp(log_kappa);

  double sigma_alpha1_sq = init["sigma_alpha1_sq"];
  double sigma_alpha2_sq = init["sigma_alpha2_sq"];
  double sigma_alpha3_sq = init["sigma_alpha3_sq"];
  double sigma_alpha4_sq = init["sigma_alpha4_sq"];
  double sigma_alpha5_sq = init["sigma_alpha5_sq"];
  double tau_beta1_sq = init["tau_beta1_sq"];
  double tau_beta2_sq = init["tau_beta2_sq"];
  double tau_beta3_sq = init["tau_beta3_sq"];
  double sigma0_sq = init["sigma0_sq"];

  // GRM thresholds: P4 x K1m1 (descending per item) and P5 x K2m1
  mat beta4_thr;
  if (P4 > 0 && K1m1 > 0) beta4_thr = as<mat>(init["beta4"]);

  mat beta5_thr;
  if (P5 > 0 && K2m1 > 0) beta5_thr = as<mat>(init["beta5"]);

  // --- Lambda (latent scale for robust Layer 2) ---
  mat lambda2(n, P2, fill::ones);

  // Storage setup
  int n_save = (n_iter - burnin) / thin;

  mat store_alpha1(n_save, n);
  mat store_alpha2(n_save, n);
  mat store_alpha3(n_save, n);
  mat store_alpha4(n_save, n);
  mat store_alpha5(n_save, n);
  mat store_beta1(n_save, P1);
  mat store_beta2(n_save, P2);
  mat store_beta3(n_save, P3);

  vec store_log_gamma1(n_save);
  vec store_log_gamma2(n_save);
  vec store_log_gamma3(n_save);
  vec store_log_gamma4(n_save);
  vec store_log_gamma5(n_save);
  vec store_log_kappa(n_save);

  vec store_sigma_alpha1_sq(n_save);
  vec store_sigma_alpha2_sq(n_save);
  vec store_sigma_alpha3_sq(n_save);
  vec store_sigma_alpha4_sq(n_save);
  vec store_sigma_alpha5_sq(n_save);
  vec store_tau_beta1_sq(n_save);
  vec store_tau_beta2_sq(n_save);
  vec store_tau_beta3_sq(n_save);
  vec store_sigma0_sq(n_save);

  cube store_a(n, d, n_save);
  cube store_b1(P1, d, n_save);
  cube store_b2(P2, d, n_save);
  cube store_b3(P3, d, n_save);
  cube store_b4(P4, d, n_save);
  cube store_b5(P5, d, n_save);

  // GRM threshold storage
  cube store_beta4_thr;
  if (P4 > 0 && K1m1 > 0) store_beta4_thr.set_size(P4, K1m1, n_save);

  cube store_beta5_thr;
  if (P5 > 0 && K2m1 > 0) store_beta5_thr.set_size(P5, K2m1, n_save);

  // Lambda2 storage
  vec store_lambda2_mean(n_save);
  cube store_lambda2(n, P2, n_save);
  mat  sum_lambda2(n, P2, fill::zeros);

  // Acceptance counters
  vec acc_alpha1 = zeros<vec>(n);
  vec acc_alpha2 = zeros<vec>(n);
  vec acc_alpha3 = zeros<vec>(n);
  vec acc_alpha4 = zeros<vec>(n);
  vec acc_alpha5 = zeros<vec>(n);
  vec acc_beta1 = zeros<vec>(P1);
  vec acc_beta2 = zeros<vec>(P2);
  vec acc_beta3 = zeros<vec>(P3);
  double acc_log_gamma1 = 0;
  double acc_log_gamma2 = 0;
  double acc_log_gamma3 = 0;
  double acc_log_gamma4 = 0;
  double acc_log_gamma5 = 0;
  double acc_log_kappa = 0;
  vec acc_a = zeros<vec>(n);
  vec acc_b1 = zeros<vec>(P1);
  vec acc_b2 = zeros<vec>(P2);
  vec acc_b3 = zeros<vec>(P3);
  vec acc_b4 = zeros<vec>(P4);
  vec acc_b5 = zeros<vec>(P5);

  // GRM threshold acceptance (per item x per threshold)
  mat acc_beta4_thr;
  if (P4 > 0 && K1m1 > 0) acc_beta4_thr = zeros<mat>(P4, K1m1);
  mat acc_beta5_thr;
  if (P5 > 0 && K2m1 > 0) acc_beta5_thr = zeros<mat>(P5, K2m1);

  int save_idx = 0;

  // MCMC Loop
  for(int iter = 0; iter < n_iter; ++iter) {

    if (verbose && (iter + 1) % 500 == 0) Rcout << "Iter: " << iter + 1 << " / " << n_iter << "\n";

    // --- 0. Update latent scales lambda2_{ij} (Gibbs) ---
    for(int i=0; i<n; ++i) {
      for(int j=0; j<P2; ++j) {
        double dist = calc_dist(a.row(i), b2.row(j));
        double mu = alpha2(i) - beta2(j) - gamma_val2 * dist;
        double e2 = pow(Y_con(i,j) - mu, 2);
        double shape = (nu2 + 1.0) / 2.0;
        double rate  = (nu2 + e2 / sigma0_sq) / 2.0;
        lambda2(i,j) = R::rgamma(shape, 1.0 / rate);
      }
    }

    // --- 1. Update alpha_l_i (Layer-specific) ---
    // Alpha 1 (Bin)
    for(int i=0; i<n; ++i) {
      double a_old = alpha1(i);
      double a_prop = R::rnorm(a_old, sd_alpha1);
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for(int j=0; j<P1; ++j) {
          double dist = calc_dist(a.row(i), b1.row(j));
          double eta = val - beta1(j) - gamma_val1 * dist;
          ll += Y_bin(i,j) * eta - log1pexp(eta);
        }
        return ll;
      };
      double lp_old = -0.5 * pow(a_old, 2) / sigma_alpha1_sq;
      double lp_new = -0.5 * pow(a_prop, 2) / sigma_alpha1_sq;
      if (log(R::runif(0,1)) < (ll_f(a_prop) + lp_new - ll_f(a_old) - lp_old)) {
        alpha1(i) = a_prop; acc_alpha1(i)++;
      }
    }
    // Alpha 2 (Con - robust)
    for(int i=0; i<n; ++i) {
      double a_old = alpha2(i);
      double a_prop = R::rnorm(a_old, sd_alpha2);
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for(int j=0; j<P2; ++j) {
          double dist = calc_dist(a.row(i), b2.row(j));
          double mu = val - beta2(j) - gamma_val2 * dist;
          double lam = lambda2(i,j);
          double sd_eff = sqrt(sigma0_sq / lam);
          ll += R::dnorm(Y_con(i,j), mu, sd_eff, 1);
        }
        return ll;
      };
      double lp_old = -0.5 * pow(a_old, 2) / sigma_alpha2_sq;
      double lp_new = -0.5 * pow(a_prop, 2) / sigma_alpha2_sq;
      if (log(R::runif(0,1)) < (ll_f(a_prop) + lp_new - ll_f(a_old) - lp_old)) {
        alpha2(i) = a_prop; acc_alpha2(i)++;
      }
    }
    // Alpha 3 (Cnt)
    for(int i=0; i<n; ++i) {
      double a_old = alpha3(i);
      double a_prop = R::rnorm(a_old, sd_alpha3);
      double size = 1.0/kappa;
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for(int j=0; j<P3; ++j) {
          double dist = calc_dist(a.row(i), b3.row(j));
          double mu = exp(val - beta3(j) - gamma_val3 * dist);
          double prob = size / (size + mu);
          ll += R::dnbinom(Y_cnt(i,j), size, prob, 1);
        }
        return ll;
      };
      double lp_old = -0.5 * pow(a_old, 2) / sigma_alpha3_sq;
      double lp_new = -0.5 * pow(a_prop, 2) / sigma_alpha3_sq;
      if (log(R::runif(0,1)) < (ll_f(a_prop) + lp_new - ll_f(a_old) - lp_old)) {
        alpha3(i) = a_prop; acc_alpha3(i)++;
      }
    }
    // Alpha 4 (Ord1 - GRM)
    for(int i=0; i<n; ++i) {
      double a_old = alpha4(i);
      double a_prop = R::rnorm(a_old, sd_alpha4);
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for(int j=0; j<P4; ++j) {
          double dist = calc_dist(a.row(i), b4.row(j));
          double eta = val - gamma_val4 * dist;
          ll += log_grm_prob_y0based(Y_ord1_0(i,j), beta4_thr.row(j), eta);
        }
        return ll;
      };
      double lp_old = -0.5 * pow(a_old, 2) / sigma_alpha4_sq;
      double lp_new = -0.5 * pow(a_prop, 2) / sigma_alpha4_sq;
      if (log(R::runif(0,1)) < (ll_f(a_prop) + lp_new - ll_f(a_old) - lp_old)) {
        alpha4(i) = a_prop; acc_alpha4(i)++;
      }
    }
    // Alpha 5 (Ord2 - GRM)
    for(int i=0; i<n; ++i) {
      double a_old = alpha5(i);
      double a_prop = R::rnorm(a_old, sd_alpha5);
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for(int j=0; j<P5; ++j) {
          double dist = calc_dist(a.row(i), b5.row(j));
          double eta = val - gamma_val5 * dist;
          ll += log_grm_prob_y0based(Y_ord2_0(i,j), beta5_thr.row(j), eta);
        }
        return ll;
      };
      double lp_old = -0.5 * pow(a_old, 2) / sigma_alpha5_sq;
      double lp_new = -0.5 * pow(a_prop, 2) / sigma_alpha5_sq;
      if (log(R::runif(0,1)) < (ll_f(a_prop) + lp_new - ll_f(a_old) - lp_old)) {
        alpha5(i) = a_prop; acc_alpha5(i)++;
      }
    }

    // --- 2. Update Beta (Layer 1, 2, 3) ---
    // Beta 1
    for(int j=0; j<P1; ++j) {
      double b_old = beta1(j);
      double b_prop = R::rnorm(b_old, sd_beta1);

      auto ll_func = [&](double b_val) {
        double ll = 0;
        for(int i=0; i<n; ++i) {
          double dist = calc_dist(a.row(i), b1.row(j));
          double eta = alpha1(i) - b_val - gamma_val1 * dist;
          ll += Y_bin(i,j) * eta - log1pexp(eta);
        }
        return ll;
      };

      double ll_curr = ll_func(b_old);
      double ll_next = ll_func(b_prop);
      double lp_curr = -0.5 * pow(b_old, 2) / tau_beta1_sq;
      double lp_next = -0.5 * pow(b_prop, 2) / tau_beta1_sq;

      if (log(R::runif(0,1)) < (ll_next + lp_next - ll_curr - lp_curr)) {
        beta1(j) = b_prop;
        acc_beta1(j)++;
      }
    }

    // Beta 2 (robust)
    for(int j=0; j<P2; ++j) {
      double b_old = beta2(j);
      double b_prop = R::rnorm(b_old, sd_beta2);

      auto ll_func = [&](double b_val) {
        double ll = 0;
        for(int i=0; i<n; ++i) {
          double dist = calc_dist(a.row(i), b2.row(j));
          double mu = alpha2(i) - b_val - gamma_val2 * dist;
          double lam = lambda2(i,j);
          double sd_eff = sqrt(sigma0_sq / lam);
          ll += R::dnorm(Y_con(i,j), mu, sd_eff, 1);
        }
        return ll;
      };
      if (log(R::runif(0,1)) < (ll_func(b_prop) - 0.5*pow(b_prop,2)/tau_beta2_sq - (ll_func(b_old) - 0.5*pow(b_old,2)/tau_beta2_sq))) {
        beta2(j) = b_prop;
        acc_beta2(j)++;
      }
    }

    // Beta 3
    for(int j=0; j<P3; ++j) {
      double b_old = beta3(j);
      double b_prop = R::rnorm(b_old, sd_beta3);

      auto ll_func = [&](double b_val) {
        double ll = 0;
        double size = 1.0/kappa;
        for(int i=0; i<n; ++i) {
          double dist = calc_dist(a.row(i), b3.row(j));
          double mu = exp(alpha3(i) - b_val - gamma_val3 * dist);
          double prob = size / (size + mu);
          ll += R::dnbinom(Y_cnt(i,j), size, prob, 1);
        }
        return ll;
      };
      if (log(R::runif(0,1)) < (ll_func(b_prop) - 0.5*pow(b_prop,2)/tau_beta3_sq - (ll_func(b_old) - 0.5*pow(b_old,2)/tau_beta3_sq))) {
        beta3(j) = b_prop;
        acc_beta3(j)++;
      }
    }

    // --- 3. Update Log Gamma (independent per layer) ---
    if (fix_gamma == true){
      acc_log_gamma1++; acc_log_gamma2++; acc_log_gamma3++; acc_log_gamma4++; acc_log_gamma5++;
    } else {
      // --- 3a. Update Gamma1 (Bin) ---
      {
        double lg_old = log_gamma1;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma1_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        auto ll_g1 = [&](double g) {
          double ll = 0;
          for(int i=0; i<n; ++i)
            for(int j=0; j<P1; ++j) {
              double dist = calc_dist(a.row(i), b1.row(j));
              double eta = alpha1(i) - beta1(j) - g * dist;
              ll += Y_bin(i,j)*eta - log1pexp(eta);
            }
          return ll;
        };
        double ll_old = ll_g1(g_old);
        double ll_new = ll_g1(g_prop);
        double lp_old = R::dnorm(lg_old, mu_log_gamma1, sd_log_gamma1, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma1, sd_log_gamma1, 1);
        if(log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)){
          log_gamma1 = lg_prop; gamma_val1 = g_prop; acc_log_gamma1++;
        }
      }
      // --- 3b. Update Gamma2 (Con - robust) ---
      {
        double lg_old = log_gamma2;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma2_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        auto ll_g2 = [&](double g) {
          double ll = 0;
          for(int i=0; i<n; ++i)
            for(int j=0; j<P2; ++j) {
              double dist = calc_dist(a.row(i), b2.row(j));
              double mu = alpha2(i) - beta2(j) - g * dist;
              double lam = lambda2(i,j);
              double sd_eff = sqrt(sigma0_sq / lam);
              ll += R::dnorm(Y_con(i,j), mu, sd_eff, 1);
            }
          return ll;
        };
        double ll_old = ll_g2(g_old);
        double ll_new = ll_g2(g_prop);
        double lp_old = R::dnorm(lg_old, mu_log_gamma2, sd_log_gamma2, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma2, sd_log_gamma2, 1);
        if(log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)){
          log_gamma2 = lg_prop; gamma_val2 = g_prop; acc_log_gamma2++;
        }
      }
      // --- 3c. Update Gamma3 (Cnt) ---
      {
        double lg_old = log_gamma3;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma3_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        double size = 1.0/kappa;
        auto ll_g3 = [&](double g) {
          double ll = 0;
          for(int i=0; i<n; ++i)
            for(int j=0; j<P3; ++j) {
              double dist = calc_dist(a.row(i), b3.row(j));
              double mu = exp(alpha3(i) - beta3(j) - g * dist);
              double prob = size/(size+mu);
              ll += R::dnbinom(Y_cnt(i,j), size, prob, 1);
            }
          return ll;
        };
        double ll_old = ll_g3(g_old);
        double ll_new = ll_g3(g_prop);
        double lp_old = R::dnorm(lg_old, mu_log_gamma3, sd_log_gamma3, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma3, sd_log_gamma3, 1);
        if(log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)){
          log_gamma3 = lg_prop; gamma_val3 = g_prop; acc_log_gamma3++;
        }
      }
      // --- 3d. Update Gamma4 (Ord1 - GRM) ---
      {
        double lg_old = log_gamma4;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma4_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        auto ll_g4 = [&](double g) {
          double ll = 0;
          for(int i=0; i<n; ++i)
            for(int j=0; j<P4; ++j) {
              double dist = calc_dist(a.row(i), b4.row(j));
              double eta = alpha4(i) - g * dist;
              ll += log_grm_prob_y0based(Y_ord1_0(i,j), beta4_thr.row(j), eta);
            }
          return ll;
        };
        double ll_old = ll_g4(g_old);
        double ll_new = ll_g4(g_prop);
        double lp_old = R::dnorm(lg_old, mu_log_gamma4, sd_log_gamma4, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma4, sd_log_gamma4, 1);
        if(log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)){
          log_gamma4 = lg_prop; gamma_val4 = g_prop; acc_log_gamma4++;
        }
      }
      // --- 3e. Update Gamma5 (Ord2 - GRM) ---
      {
        double lg_old = log_gamma5;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma5_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        auto ll_g5 = [&](double g) {
          double ll = 0;
          for(int i=0; i<n; ++i)
            for(int j=0; j<P5; ++j) {
              double dist = calc_dist(a.row(i), b5.row(j));
              double eta = alpha5(i) - g * dist;
              ll += log_grm_prob_y0based(Y_ord2_0(i,j), beta5_thr.row(j), eta);
            }
          return ll;
        };
        double ll_old = ll_g5(g_old);
        double ll_new = ll_g5(g_prop);
        double lp_old = R::dnorm(lg_old, mu_log_gamma5, sd_log_gamma5, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma5, sd_log_gamma5, 1);
        if(log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)){
          log_gamma5 = lg_prop; gamma_val5 = g_prop; acc_log_gamma5++;
        }
      }
    }


    // --- 4. Update Shared Position a_i ---
    for(int i=0; i<n; ++i) {
      rowvec a_old = a.row(i);
      rowvec a_prop = a_old + randn<rowvec>(d) * sd_a;

      auto pos_ll = [&](const rowvec& pos) {
        double ll = 0;
        double size = 1.0/kappa;

        // L1
        for(int j=0; j<P1; ++j) {
          double dist = calc_dist(pos, b1.row(j));
          double eta = alpha1(i) - beta1(j) - gamma_val1 * dist;
          ll += Y_bin(i,j)*eta - log1pexp(eta);
        }
        // L2 (robust)
        for(int j=0; j<P2; ++j) {
          double dist = calc_dist(pos, b2.row(j));
          double mu = alpha2(i) - beta2(j) - gamma_val2 * dist;
          double lam = lambda2(i,j);
          double sd_eff = sqrt(sigma0_sq / lam);
          ll += R::dnorm(Y_con(i,j), mu, sd_eff, 1);
        }
        // L3
        for(int j=0; j<P3; ++j) {
          double dist = calc_dist(pos, b3.row(j));
          double mu = exp(alpha3(i) - beta3(j) - gamma_val3 * dist);
          double prob = size/(size+mu);
          ll += R::dnbinom(Y_cnt(i,j), size, prob, 1);
        }
        // L4 (GRM)
        for(int j=0; j<P4; ++j) {
          double dist = calc_dist(pos, b4.row(j));
          double eta = alpha4(i) - gamma_val4 * dist;
          ll += log_grm_prob_y0based(Y_ord1_0(i,j), beta4_thr.row(j), eta);
        }
        // L5 (GRM)
        for(int j=0; j<P5; ++j) {
          double dist = calc_dist(pos, b5.row(j));
          double eta = alpha5(i) - gamma_val5 * dist;
          ll += log_grm_prob_y0based(Y_ord2_0(i,j), beta5_thr.row(j), eta);
        }
        return ll;
      };

      double ll_curr = pos_ll(a_old);
      double ll_next = pos_ll(a_prop);

      double lp_curr = -0.5 * dot(a_old, a_old);
      double lp_next = -0.5 * dot(a_prop, a_prop);

      if(log(R::runif(0,1)) < (ll_next + lp_next - ll_curr - lp_curr)) {
        a.row(i) = a_prop;
        acc_a(i)++;
      }
    }

    // --- 5. Update Item Positions b^(l)_j ---
    // b1
    for(int j=0; j<P1; ++j) {
      rowvec b_old = b1.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b1;
      auto ll_f = [&](const rowvec& b_val) {
        double ll=0;
        for(int i=0; i<n; ++i){
          double dist = calc_dist(a.row(i), b_val);
          double eta = alpha1(i) - beta1(j) - gamma_val1 * dist;
          ll += Y_bin(i,j)*eta - log1pexp(eta);
        }
        return ll;
      };
      if(log(R::runif(0,1)) < (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) - (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b1.row(j) = b_prop; acc_b1(j)++;
      }
    }
    // b2 (robust)
    for(int j=0; j<P2; ++j) {
      rowvec b_old = b2.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b2;
      auto ll_f = [&](const rowvec& b_val) {
        double ll=0;
        for(int i=0; i<n; ++i){
          double dist = calc_dist(a.row(i), b_val);
          double mu = alpha2(i) - beta2(j) - gamma_val2 * dist;
          double lam = lambda2(i,j);
          double sd_eff = sqrt(sigma0_sq / lam);
          ll += R::dnorm(Y_con(i,j), mu, sd_eff, 1);
        }
        return ll;
      };
      if(log(R::runif(0,1)) < (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) - (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b2.row(j) = b_prop; acc_b2(j)++;
      }
    }
    // b3
    for(int j=0; j<P3; ++j) {
      rowvec b_old = b3.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b3;
      double size = 1.0/kappa;
      auto ll_f = [&](const rowvec& b_val) {
        double ll=0;
        for(int i=0; i<n; ++i){
          double dist = calc_dist(a.row(i), b_val);
          double mu = exp(alpha3(i) - beta3(j) - gamma_val3 * dist);
          ll += R::dnbinom(Y_cnt(i,j), size, size/(size+mu), 1);
        }
        return ll;
      };
      if(log(R::runif(0,1)) < (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) - (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b3.row(j) = b_prop; acc_b3(j)++;
      }
    }
    // b4 (Ord1 - GRM)
    for(int j=0; j<P4; ++j) {
      rowvec b_old = b4.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b4;
      auto ll_f = [&](const rowvec& b_val) {
        double ll=0;
        for(int i=0; i<n; ++i){
          double dist = calc_dist(a.row(i), b_val);
          double eta = alpha4(i) - gamma_val4 * dist;
          ll += log_grm_prob_y0based(Y_ord1_0(i,j), beta4_thr.row(j), eta);
        }
        return ll;
      };
      if(log(R::runif(0,1)) < (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) - (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b4.row(j) = b_prop; acc_b4(j)++;
      }
    }
    // b5 (Ord2 - GRM)
    for(int j=0; j<P5; ++j) {
      rowvec b_old = b5.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b5;
      auto ll_f = [&](const rowvec& b_val) {
        double ll=0;
        for(int i=0; i<n; ++i){
          double dist = calc_dist(a.row(i), b_val);
          double eta = alpha5(i) - gamma_val5 * dist;
          ll += log_grm_prob_y0based(Y_ord2_0(i,j), beta5_thr.row(j), eta);
        }
        return ll;
      };
      if(log(R::runif(0,1)) < (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) - (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b5.row(j) = b_prop; acc_b5(j)++;
      }
    }

    // --- 6a. Update GRM Thresholds L4 (individual MH with descending order constraint) ---
    for(int j=0; j<P4; ++j) {
      rowvec beta_old_j = beta4_thr.row(j);

      for(int c=0; c<K1m1; ++c) {
        double proposal = R::rnorm(beta_old_j(c), sd_beta4_prop);

        // Ordering constraint (descending: beta[c-1] > beta[c] > beta[c+1])
        bool valid = true;
        if (c > 0 && !(beta_old_j(c-1) > proposal)) valid = false;
        if (c < K1m1-1 && !(proposal > beta_old_j(c+1))) valid = false;
        if (!valid) continue;

        rowvec beta_new_j = beta_old_j;
        beta_new_j(c) = proposal;

        double ll_old = 0.0, ll_new = 0.0;
        for(int i=0; i<n; ++i) {
          double dist = calc_dist(a.row(i), b4.row(j));
          double eta = alpha4(i) - gamma_val4 * dist;
          ll_old += log_grm_prob_y0based(Y_ord1_0(i,j), beta_old_j, eta);
          ll_new += log_grm_prob_y0based(Y_ord1_0(i,j), beta_new_j, eta);
        }

        double lp_old = R::dnorm(beta_old_j(c), mu_beta4, sd_beta4_hyp, 1);
        double lp_new = R::dnorm(proposal, mu_beta4, sd_beta4_hyp, 1);

        if(log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
          beta_old_j(c) = proposal;
          acc_beta4_thr(j, c)++;
        }
      }
      beta4_thr.row(j) = beta_old_j;
    }

    // --- 6b. Update GRM Thresholds L5 (individual MH with descending order constraint) ---
    for(int j=0; j<P5; ++j) {
      rowvec beta_old_j = beta5_thr.row(j);

      for(int c=0; c<K2m1; ++c) {
        double proposal = R::rnorm(beta_old_j(c), sd_beta5_prop);

        // Ordering constraint (descending)
        bool valid = true;
        if (c > 0 && !(beta_old_j(c-1) > proposal)) valid = false;
        if (c < K2m1-1 && !(proposal > beta_old_j(c+1))) valid = false;
        if (!valid) continue;

        rowvec beta_new_j = beta_old_j;
        beta_new_j(c) = proposal;

        double ll_old = 0.0, ll_new = 0.0;
        for(int i=0; i<n; ++i) {
          double dist = calc_dist(a.row(i), b5.row(j));
          double eta = alpha5(i) - gamma_val5 * dist;
          ll_old += log_grm_prob_y0based(Y_ord2_0(i,j), beta_old_j, eta);
          ll_new += log_grm_prob_y0based(Y_ord2_0(i,j), beta_new_j, eta);
        }

        double lp_old = R::dnorm(beta_old_j(c), mu_beta5, sd_beta5_hyp, 1);
        double lp_new = R::dnorm(proposal, mu_beta5, sd_beta5_hyp, 1);

        if(log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
          beta_old_j(c) = proposal;
          acc_beta5_thr(j, c)++;
        }
      }
      beta5_thr.row(j) = beta_old_j;
    }


    // --- 7. Log Kappa ---
    {
      double lk_old = log_kappa;
      double lk_prop = R::rnorm(lk_old, sd_log_kappa_prop);
      double k_old = exp(lk_old);
      double k_prop = exp(lk_prop);

      auto kap_ll = [&](double k_val) {
        double ll = 0;
        double size = 1.0/k_val;
        for(int i=0; i<n; ++i) {
          for(int j=0; j<P3; ++j) {
            double dist = calc_dist(a.row(i), b3.row(j));
            double mu = exp(alpha3(i) - beta3(j) - gamma_val3 * dist);
            ll += R::dnbinom(Y_cnt(i,j), size, size/(size+mu), 1);
          }
        }
        return ll;
      };

      double ll_c = kap_ll(k_old);
      double ll_p = kap_ll(k_prop);
      double lp_c = R::dnorm(lk_old, mu_log_kappa, sd_log_kappa, 1);
      double lp_p = R::dnorm(lk_prop, mu_log_kappa, sd_log_kappa, 1);

      if(log(R::runif(0,1)) < (ll_p + lp_p - ll_c - lp_c)) {
        log_kappa = lk_prop;
        kappa = k_prop;
        acc_log_kappa++;
      }
    }

    // --- 8. Gibbs Updates ---
    // sigma0_sq (robust: weighted residual sum of squares)
    {
      double wSSE = 0;
      for(int i=0; i<n; ++i) {
        for(int j=0; j<P2; ++j) {
          double dist = calc_dist(a.row(i), b2.row(j));
          double mu = alpha2(i) - beta2(j) - gamma_val2 * dist;
          wSSE += lambda2(i,j) * pow(Y_con(i,j) - mu, 2);
        }
      }
      double shape = a_sigma0 + (n*P2)/2.0;
      double rate = b_sigma0 + 0.5 * wSSE;
      sigma0_sq = 1.0 / R::rgamma(shape, 1.0/rate);
    }
    sigma_alpha1_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha1, alpha1)));
    sigma_alpha2_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha2, alpha2)));
    sigma_alpha3_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha3, alpha3)));
    sigma_alpha4_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha4, alpha4)));
    sigma_alpha5_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha5, alpha5)));
    tau_beta1_sq = 1.0;
    tau_beta2_sq = 1.0;
    tau_beta3_sq = 1.0;

    // --- 9. Save Samples ---
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      store_alpha1.row(save_idx) = alpha1.t();
      store_alpha2.row(save_idx) = alpha2.t();
      store_alpha3.row(save_idx) = alpha3.t();
      store_alpha4.row(save_idx) = alpha4.t();
      store_alpha5.row(save_idx) = alpha5.t();
      store_beta1.row(save_idx) = beta1.t();
      store_beta2.row(save_idx) = beta2.t();
      store_beta3.row(save_idx) = beta3.t();
      store_log_gamma1(save_idx) = log_gamma1;
      store_log_gamma2(save_idx) = log_gamma2;
      store_log_gamma3(save_idx) = log_gamma3;
      store_log_gamma4(save_idx) = log_gamma4;
      store_log_gamma5(save_idx) = log_gamma5;
      store_log_kappa(save_idx) = log_kappa;

      store_sigma_alpha1_sq(save_idx) = sigma_alpha1_sq;
      store_sigma_alpha2_sq(save_idx) = sigma_alpha2_sq;
      store_sigma_alpha3_sq(save_idx) = sigma_alpha3_sq;
      store_sigma_alpha4_sq(save_idx) = sigma_alpha4_sq;
      store_sigma_alpha5_sq(save_idx) = sigma_alpha5_sq;
      store_tau_beta1_sq(save_idx) = tau_beta1_sq;
      store_tau_beta2_sq(save_idx) = tau_beta2_sq;
      store_tau_beta3_sq(save_idx) = tau_beta3_sq;
      store_sigma0_sq(save_idx) = sigma0_sq;

      store_a.slice(save_idx) = a;
      store_b1.slice(save_idx) = b1;
      store_b2.slice(save_idx) = b2;
      store_b3.slice(save_idx) = b3;
      store_b4.slice(save_idx) = b4;
      store_b5.slice(save_idx) = b5;

      // GRM thresholds
      if (P4 > 0 && K1m1 > 0) store_beta4_thr.slice(save_idx) = beta4_thr;
      if (P5 > 0 && K2m1 > 0) store_beta5_thr.slice(save_idx) = beta5_thr;

      // Lambda2 storage
      store_lambda2_mean(save_idx) = accu(lambda2) / (n * P2);
      store_lambda2.slice(save_idx) = lambda2;
      sum_lambda2 += lambda2;

      save_idx++;
    }
  }

  return List::create(
    Named("alpha1") = store_alpha1,
    Named("alpha2") = store_alpha2,
    Named("alpha3") = store_alpha3,
    Named("alpha4") = store_alpha4,
    Named("alpha5") = store_alpha5,
    Named("beta1") = store_beta1,
    Named("beta2") = store_beta2,
    Named("beta3") = store_beta3,
    Named("log_gamma1") = store_log_gamma1,
    Named("log_gamma2") = store_log_gamma2,
    Named("log_gamma3") = store_log_gamma3,
    Named("log_gamma4") = store_log_gamma4,
    Named("log_gamma5") = store_log_gamma5,
    Named("log_kappa") = store_log_kappa,
    Named("sigma0_sq")      = store_sigma0_sq,
    Named("sigma_alpha1_sq") = store_sigma_alpha1_sq,
    Named("sigma_alpha2_sq") = store_sigma_alpha2_sq,
    Named("sigma_alpha3_sq") = store_sigma_alpha3_sq,
    Named("sigma_alpha4_sq") = store_sigma_alpha4_sq,
    Named("sigma_alpha5_sq") = store_sigma_alpha5_sq,
    Named("tau_beta1_sq")   = store_tau_beta1_sq,
    Named("tau_beta2_sq")   = store_tau_beta2_sq,
    Named("tau_beta3_sq")   = store_tau_beta3_sq,
    Named("a") = store_a,
    Named("b1") = store_b1,
    Named("b2") = store_b2,
    Named("b3") = store_b3,
    Named("b4") = store_b4,
    Named("b5") = store_b5,
    Named("beta4") = (P4>0 && K1m1>0) ? wrap(store_beta4_thr) : R_NilValue,
    Named("beta5") = (P5>0 && K2m1>0) ? wrap(store_beta5_thr) : R_NilValue,
    Named("lambda2_mean") = store_lambda2_mean,
    Named("lambda2") = store_lambda2,
    Named("lambda2_postmean") = sum_lambda2 / n_save,
    Named("accept") = List::create(
      Named("alpha1")    = acc_alpha1 / n_iter,
      Named("alpha2")    = acc_alpha2 / n_iter,
      Named("alpha3")    = acc_alpha3 / n_iter,
      Named("alpha4")    = acc_alpha4 / n_iter,
      Named("alpha5")    = acc_alpha5 / n_iter,
      Named("beta1")     = acc_beta1 / n_iter,
      Named("beta2")     = acc_beta2 / n_iter,
      Named("beta3")     = acc_beta3 / n_iter,
      Named("log_gamma1") = acc_log_gamma1 / n_iter,
      Named("log_gamma2") = acc_log_gamma2 / n_iter,
      Named("log_gamma3") = acc_log_gamma3 / n_iter,
      Named("log_gamma4") = acc_log_gamma4 / n_iter,
      Named("log_gamma5") = acc_log_gamma5 / n_iter,
      Named("log_kappa") = acc_log_kappa / n_iter,
      Named("a")         = acc_a / n_iter,
      Named("b1")        = acc_b1 / n_iter,
      Named("b2")        = acc_b2 / n_iter,
      Named("b3")        = acc_b3 / n_iter,
      Named("b4")        = acc_b4 / n_iter,
      Named("b5")        = acc_b5 / n_iter,
      Named("beta4_thr") = (P4>0 && K1m1>0) ? wrap(acc_beta4_thr / n_iter) : R_NilValue,
      Named("beta5_thr") = (P5>0 && K2m1>0) ? wrap(acc_beta5_thr / n_iter) : R_NilValue
    )
  );
}
