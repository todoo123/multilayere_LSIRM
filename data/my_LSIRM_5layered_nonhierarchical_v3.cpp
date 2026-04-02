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

// softplus(x) = log(1 + exp(x)) (stable)
inline double softplus(double x) {
  return log1pexp(x);
}

// log(sigmoid(x)) stable
inline double log_sigmoid(double x) {
  if (x >= 0) return -log1p(exp(-x));
  else        return x - log1p(exp(x));
}

// Inverse Logit
inline double inv_logit(double x) {
  return 1.0 / (1.0 + exp(-x));
}

// Build Thresholds vector from u and delta
// Returns a vec of length K-1
vec build_thresholds_cpp(double u, const rowvec& delta_row) {
  int Kminus2 = delta_row.n_elem;
  int Kminus1 = Kminus2 + 1;

  vec b(Kminus1);
  b(0) = u;
  if (Kminus1 >= 2) {
    for (int k = 1; k < Kminus1; ++k) {
      b(k) = b(k - 1) + softplus(delta_row(k - 1));
    }
  }
  return b;
}

// Log Probability for Ordinal Item (Single Observation)
// y_val: 1-based integer (1..K)
double log_p_ordinal_single_cpp(int y_val, double eta, const vec& b_vec) {
  int Kminus1 = b_vec.n_elem;
  int K = Kminus1 + 1;

  vec C(Kminus1);
  for(int k=0; k<Kminus1; ++k){
    C(k) = inv_logit(eta - b_vec(k));
  }

  double p = 0.0;
  if (y_val == 1) {
    p = 1.0 - C(0);
  } else if (y_val == K) {
    p = C(Kminus1 - 1);
  } else {
    p = C(y_val - 2) - C(y_val - 1);
  }

  if (p <= 1e-16) return -1e16;
  return log(p);
}

// Helper: Calculate Euclidean distance between vector a and vector b
inline double calc_dist(const rowvec& a, const rowvec& b) {
  return sqrt(sum(square(a - b)));
}

// =========================================================
// Main MCMC Function (5-layered: bin, con, cnt, ord1, ord2)
// =========================================================
// [[Rcpp::export]]
List run_lsirm_cpp(
    mat Y_bin, mat Y_con, mat Y_cnt, IntegerMatrix Y_ord1, IntegerMatrix Y_ord2,
    int d, int n_iter, int burnin, int thin,
    List hyper, List prop_sd, List init,
    bool verbose, bool fix_gamma
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

  if (verbose) {
    Rcout << "n=" << n << " P1=" << P1 << " P2=" << P2
          << " P3=" << P3 << " P4=" << P4 << " P5=" << P5
          << " K1=" << K1 << " K2=" << K2 << "\n";
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
  double mu_u = hyper["mu_u"]; double sd_u = hyper["sd_u"];
  double mu_delta = hyper["mu_delta"]; double sd_delta = hyper["sd_delta"];

  // Unpack Proposal SDs
  double sd_alpha = prop_sd["alpha"];
  double sd_beta1 = prop_sd["beta1"];
  double sd_beta2 = prop_sd["beta2"];
  double sd_beta3 = prop_sd["beta3"];
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
  double sd_u_prop = prop_sd["u"];
  double sd_delta_prop  = prop_sd["delta"];
  double sd_delta2_prop = prop_sd["delta2"];

  // Initialize Parameters
  vec alpha = as<vec>(init["alpha"]);
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

  double sigma_alpha_sq = init["sigma_alpha_sq"];
  double tau_beta1_sq = init["tau_beta1_sq"];
  double tau_beta2_sq = init["tau_beta2_sq"];
  double tau_beta3_sq = init["tau_beta3_sq"];
  double sigma0_sq = init["sigma0_sq"];

  vec u(P4, fill::zeros);   // u fixed at 0 for L4
  vec u2(P5, fill::zeros);  // u2 fixed at 0 for L5

  mat delta;
  if (K1 > 2) delta = as<mat>(init["delta"]);

  mat delta2;
  if (K2 > 2) delta2 = as<mat>(init["delta2"]);

  // Storage setup
  int n_save = (n_iter - burnin) / thin;

  mat store_alpha(n_save, n);
  mat store_beta1(n_save, P1);
  mat store_beta2(n_save, P2);
  mat store_beta3(n_save, P3);

  vec store_log_gamma1(n_save);
  vec store_log_gamma2(n_save);
  vec store_log_gamma3(n_save);
  vec store_log_gamma4(n_save);
  vec store_log_gamma5(n_save);
  vec store_log_kappa(n_save);

  vec store_sigma_alpha_sq(n_save);
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

  // L4 ordinal storage
  mat store_u(n_save, P4);
  cube store_delta;
  if (K1 > 2) store_delta.set_size(P4, K1-2, n_save);
  cube store_thr;
  if (K1 > 2) store_thr.set_size(P4, K1-1, n_save);

  // L5 ordinal storage
  mat store_u2(n_save, P5);
  cube store_delta2;
  if (K2 > 2) store_delta2.set_size(P5, K2-2, n_save);
  cube store_thr2;
  if (K2 > 2) store_thr2.set_size(P5, K2-1, n_save);

  // Acceptance counters
  vec acc_alpha = zeros<vec>(n);
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
  vec acc_thr  = zeros<vec>(P4);
  vec acc_thr2 = zeros<vec>(P5);

  int save_idx = 0;

  // MCMC Loop
  for(int iter = 0; iter < n_iter; ++iter) {

    if (verbose && (iter + 1) % 500 == 0) Rcout << "Iter: " << iter + 1 << " / " << n_iter << "\n";

    // --- 1. Update alpha_i (Shared) ---
    for(int i=0; i<n; ++i) {
      double alpha_old = alpha(i);
      double alpha_prop = R::rnorm(alpha_old, sd_alpha);

      auto loglik_i = [&](double val_alpha) {
        double ll = 0.0;

        // Layer 1 (Bin)
        for(int j=0; j<P1; ++j) {
          double dist = calc_dist(a.row(i), b1.row(j));
          double eta = val_alpha - beta1(j) - gamma_val1 * dist;
          ll += Y_bin(i,j) * eta - log1pexp(eta);
        }
        // Layer 2 (Con)
        for(int j=0; j<P2; ++j) {
          double dist = calc_dist(a.row(i), b2.row(j));
          double mu = val_alpha - beta2(j) - gamma_val2 * dist;
          ll += R::dnorm(Y_con(i,j), mu, sqrt(sigma0_sq), 1);
        }
        // Layer 3 (Cnt)
        for(int j=0; j<P3; ++j) {
          double dist = calc_dist(a.row(i), b3.row(j));
          double mu = exp(val_alpha - beta3(j) - gamma_val3 * dist);
          double size = 1.0/kappa;
          double prob = size / (size + mu);
          ll += R::dnbinom(Y_cnt(i,j), size, prob, 1);
        }
        // Layer 4 (Ord1)
        for(int j=0; j<P4; ++j) {
          double dist = calc_dist(a.row(i), b4.row(j));
          double eta = val_alpha - gamma_val4 * dist;
          vec b_thresh = build_thresholds_cpp(0.0, (K1>2) ? delta.row(j) : rowvec());
          ll += log_p_ordinal_single_cpp(Y_ord1(i,j), eta, b_thresh);
        }
        // Layer 5 (Ord2)
        for(int j=0; j<P5; ++j) {
          double dist = calc_dist(a.row(i), b5.row(j));
          double eta = val_alpha - gamma_val5 * dist;
          vec b_thresh = build_thresholds_cpp(0.0, (K2>2) ? delta2.row(j) : rowvec());
          ll += log_p_ordinal_single_cpp(Y_ord2(i,j), eta, b_thresh);
        }
        return ll;
      };

      double ll_old = loglik_i(alpha_old);
      double ll_new = loglik_i(alpha_prop);

      double lp_old = -0.5 * pow(alpha_old, 2) / sigma_alpha_sq;
      double lp_new = -0.5 * pow(alpha_prop, 2) / sigma_alpha_sq;

      if (log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
        alpha(i) = alpha_prop;
        acc_alpha(i)++;
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
          double eta = alpha(i) - b_val - gamma_val1 * dist;
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

    // Beta 2
    for(int j=0; j<P2; ++j) {
      double b_old = beta2(j);
      double b_prop = R::rnorm(b_old, sd_beta2);

      auto ll_func = [&](double b_val) {
        double ll = 0;
        double sd0 = sqrt(sigma0_sq);
        for(int i=0; i<n; ++i) {
          double dist = calc_dist(a.row(i), b2.row(j));
          double mu = alpha(i) - b_val - gamma_val2 * dist;
          ll += R::dnorm(Y_con(i,j), mu, sd0, 1);
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
          double mu = exp(alpha(i) - b_val - gamma_val3 * dist);
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
              double eta = alpha(i) - beta1(j) - g * dist;
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
      // --- 3b. Update Gamma2 (Con) ---
      {
        double lg_old = log_gamma2;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma2_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        double sd0 = sqrt(sigma0_sq);
        auto ll_g2 = [&](double g) {
          double ll = 0;
          for(int i=0; i<n; ++i)
            for(int j=0; j<P2; ++j) {
              double dist = calc_dist(a.row(i), b2.row(j));
              double mu = alpha(i) - beta2(j) - g * dist;
              ll += R::dnorm(Y_con(i,j), mu, sd0, 1);
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
              double mu = exp(alpha(i) - beta3(j) - g * dist);
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
      // --- 3d. Update Gamma4 (Ord1) ---
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
              double eta = alpha(i) - g * dist;
              vec b_th = build_thresholds_cpp(0.0, (K1>2)? delta.row(j) : rowvec());
              ll += log_p_ordinal_single_cpp(Y_ord1(i,j), eta, b_th);
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
      // --- 3e. Update Gamma5 (Ord2) ---
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
              double eta = alpha(i) - g * dist;
              vec b_th = build_thresholds_cpp(0.0, (K2>2)? delta2.row(j) : rowvec());
              ll += log_p_ordinal_single_cpp(Y_ord2(i,j), eta, b_th);
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
        double sd0 = sqrt(sigma0_sq);
        double size = 1.0/kappa;

        // L1
        for(int j=0; j<P1; ++j) {
          double dist = calc_dist(pos, b1.row(j));
          double eta = alpha(i) - beta1(j) - gamma_val1 * dist;
          ll += Y_bin(i,j)*eta - log1pexp(eta);
        }
        // L2
        for(int j=0; j<P2; ++j) {
          double dist = calc_dist(pos, b2.row(j));
          double mu = alpha(i) - beta2(j) - gamma_val2 * dist;
          ll += R::dnorm(Y_con(i,j), mu, sd0, 1);
        }
        // L3
        for(int j=0; j<P3; ++j) {
          double dist = calc_dist(pos, b3.row(j));
          double mu = exp(alpha(i) - beta3(j) - gamma_val3 * dist);
          double prob = size/(size+mu);
          ll += R::dnbinom(Y_cnt(i,j), size, prob, 1);
        }
        // L4
        for(int j=0; j<P4; ++j) {
          double dist = calc_dist(pos, b4.row(j));
          double eta = alpha(i) - gamma_val4 * dist;
          vec b_th = build_thresholds_cpp(0.0, (K1>2)? delta.row(j) : rowvec());
          ll += log_p_ordinal_single_cpp(Y_ord1(i,j), eta, b_th);
        }
        // L5
        for(int j=0; j<P5; ++j) {
          double dist = calc_dist(pos, b5.row(j));
          double eta = alpha(i) - gamma_val5 * dist;
          vec b_th = build_thresholds_cpp(0.0, (K2>2)? delta2.row(j) : rowvec());
          ll += log_p_ordinal_single_cpp(Y_ord2(i,j), eta, b_th);
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
          double eta = alpha(i) - beta1(j) - gamma_val1 * dist;
          ll += Y_bin(i,j)*eta - log1pexp(eta);
        }
        return ll;
      };
      if(log(R::runif(0,1)) < (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) - (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b1.row(j) = b_prop; acc_b1(j)++;
      }
    }
    // b2
    for(int j=0; j<P2; ++j) {
      rowvec b_old = b2.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b2;
      double sd0 = sqrt(sigma0_sq);
      auto ll_f = [&](const rowvec& b_val) {
        double ll=0;
        for(int i=0; i<n; ++i){
          double dist = calc_dist(a.row(i), b_val);
          double mu = alpha(i) - beta2(j) - gamma_val2 * dist;
          ll += R::dnorm(Y_con(i,j), mu, sd0, 1);
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
          double mu = exp(alpha(i) - beta3(j) - gamma_val3 * dist);
          ll += R::dnbinom(Y_cnt(i,j), size, size/(size+mu), 1);
        }
        return ll;
      };
      if(log(R::runif(0,1)) < (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) - (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b3.row(j) = b_prop; acc_b3(j)++;
      }
    }
    // b4 (Ord1)
    for(int j=0; j<P4; ++j) {
      rowvec b_old = b4.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b4;
      vec b_th = build_thresholds_cpp(0.0, (K1>2)? delta.row(j) : rowvec());
      auto ll_f = [&](const rowvec& b_val) {
        double ll=0;
        for(int i=0; i<n; ++i){
          double dist = calc_dist(a.row(i), b_val);
          double eta = alpha(i) - gamma_val4 * dist;
          ll += log_p_ordinal_single_cpp(Y_ord1(i,j), eta, b_th);
        }
        return ll;
      };
      if(log(R::runif(0,1)) < (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) - (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b4.row(j) = b_prop; acc_b4(j)++;
      }
    }
    // b5 (Ord2)
    for(int j=0; j<P5; ++j) {
      rowvec b_old = b5.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b5;
      vec b_th = build_thresholds_cpp(0.0, (K2>2)? delta2.row(j) : rowvec());
      auto ll_f = [&](const rowvec& b_val) {
        double ll=0;
        for(int i=0; i<n; ++i){
          double dist = calc_dist(a.row(i), b_val);
          double eta = alpha(i) - gamma_val5 * dist;
          ll += log_p_ordinal_single_cpp(Y_ord2(i,j), eta, b_th);
        }
        return ll;
      };
      if(log(R::runif(0,1)) < (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) - (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b5.row(j) = b_prop; acc_b5(j)++;
      }
    }

    // --- 6a. Update Ordinal Thresholds L4 (u fixed at 0, update delta only) ---
    for(int j=0; j<P4; ++j) {
      rowvec delta_curr;
      if (K1>2) delta_curr = delta.row(j);

      rowvec delta_prop_val;
      if (K1>2) delta_prop_val = delta_curr + randn<rowvec>(K1-2) * sd_delta_prop;

      vec thr_curr = build_thresholds_cpp(0.0, delta_curr);
      vec thr_prop = build_thresholds_cpp(0.0, delta_prop_val);

      auto ord_ll = [&](const vec& th) {
        double ll = 0;
        for(int i=0; i<n; ++i) {
          double dist = calc_dist(a.row(i), b4.row(j));
          double eta = alpha(i) - gamma_val4 * dist;
          ll += log_p_ordinal_single_cpp(Y_ord1(i,j), eta, th);
        }
        return ll;
      };

      double ll_c = ord_ll(thr_curr);
      double ll_p = ord_ll(thr_prop);

      double lp_c = 0.0, lp_p = 0.0;
      if (K1>2) {
        for(int k=0; k<K1-2; ++k) {
          lp_c += R::dnorm(delta_curr(k), mu_delta, sd_delta, 1);
          lp_p += R::dnorm(delta_prop_val(k), mu_delta, sd_delta, 1);
        }
      }

      double jac_c = 0.0, jac_p = 0.0;
      if (K1 > 2) {
        for (int k = 0; k < K1-2; ++k) {
          jac_c += log_sigmoid(delta_curr(k));
          jac_p += log_sigmoid(delta_prop_val(k));
        }
      }

      if(log(R::runif(0,1)) < (ll_p + lp_p + jac_p - (ll_c + lp_c + jac_c))) {
        if(K1>2) delta.row(j) = delta_prop_val;
        acc_thr(j)++;
      }
    }

    // --- 6b. Update Ordinal Thresholds L5 (u2 fixed at 0, update delta2 only) ---
    for(int j=0; j<P5; ++j) {
      rowvec delta_curr;
      if (K2>2) delta_curr = delta2.row(j);

      rowvec delta_prop_val;
      if (K2>2) delta_prop_val = delta_curr + randn<rowvec>(K2-2) * sd_delta2_prop;

      vec thr_curr = build_thresholds_cpp(0.0, delta_curr);
      vec thr_prop = build_thresholds_cpp(0.0, delta_prop_val);

      auto ord_ll = [&](const vec& th) {
        double ll = 0;
        for(int i=0; i<n; ++i) {
          double dist = calc_dist(a.row(i), b5.row(j));
          double eta = alpha(i) - gamma_val5 * dist;
          ll += log_p_ordinal_single_cpp(Y_ord2(i,j), eta, th);
        }
        return ll;
      };

      double ll_c = ord_ll(thr_curr);
      double ll_p = ord_ll(thr_prop);

      double lp_c = 0.0, lp_p = 0.0;
      if (K2>2) {
        for(int k=0; k<K2-2; ++k) {
          lp_c += R::dnorm(delta_curr(k), mu_delta, sd_delta, 1);
          lp_p += R::dnorm(delta_prop_val(k), mu_delta, sd_delta, 1);
        }
      }

      double jac_c = 0.0, jac_p = 0.0;
      if (K2 > 2) {
        for (int k = 0; k < K2-2; ++k) {
          jac_c += log_sigmoid(delta_curr(k));
          jac_p += log_sigmoid(delta_prop_val(k));
        }
      }

      if(log(R::runif(0,1)) < (ll_p + lp_p + jac_p - (ll_c + lp_c + jac_c))) {
        if(K2>2) delta2.row(j) = delta_prop_val;
        acc_thr2(j)++;
      }
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
            double mu = exp(alpha(i) - beta3(j) - gamma_val3 * dist);
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
    // sigma0_sq
    {
      double SSE = 0;
      for(int i=0; i<n; ++i) {
        for(int j=0; j<P2; ++j) {
          double dist = calc_dist(a.row(i), b2.row(j));
          double mu = alpha(i) - beta2(j) - gamma_val2 * dist;
          SSE += pow(Y_con(i,j) - mu, 2);
        }
      }
      double shape = a_sigma0 + (n*P2)/2.0;
      double rate = b_sigma0 + 0.5 * SSE;
      sigma0_sq = 1.0 / R::rgamma(shape, 1.0/rate);
    }
    sigma_alpha_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha, alpha)));
    tau_beta1_sq = 1.0;
    tau_beta2_sq = 1.0;
    tau_beta3_sq = 1.0;

    // --- 9. Save Samples ---
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      store_alpha.row(save_idx) = alpha.t();
      store_beta1.row(save_idx) = beta1.t();
      store_beta2.row(save_idx) = beta2.t();
      store_beta3.row(save_idx) = beta3.t();
      store_log_gamma1(save_idx) = log_gamma1;
      store_log_gamma2(save_idx) = log_gamma2;
      store_log_gamma3(save_idx) = log_gamma3;
      store_log_gamma4(save_idx) = log_gamma4;
      store_log_gamma5(save_idx) = log_gamma5;
      store_log_kappa(save_idx) = log_kappa;

      store_sigma_alpha_sq(save_idx) = sigma_alpha_sq;
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

      // L4 ordinal
      store_u.row(save_idx) = u.t();
      if(K1>2) store_delta.slice(save_idx) = delta;
      for(int j=0; j<P4; ++j) {
        rowvec d_r;
        if(K1 > 2) d_r = delta.row(j);
        store_thr.slice(save_idx).row(j) = build_thresholds_cpp(0.0, d_r).t();
      }

      // L5 ordinal
      store_u2.row(save_idx) = u2.t();
      if(K2>2) store_delta2.slice(save_idx) = delta2;
      for(int j=0; j<P5; ++j) {
        rowvec d_r;
        if(K2 > 2) d_r = delta2.row(j);
        store_thr2.slice(save_idx).row(j) = build_thresholds_cpp(0.0, d_r).t();
      }

      save_idx++;
    }
  }

  return List::create(
    Named("alpha") = store_alpha,
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
    Named("sigma_alpha_sq") = store_sigma_alpha_sq,
    Named("tau_beta1_sq")   = store_tau_beta1_sq,
    Named("tau_beta2_sq")   = store_tau_beta2_sq,
    Named("tau_beta3_sq")   = store_tau_beta3_sq,
    Named("a") = store_a,
    Named("b1") = store_b1,
    Named("b2") = store_b2,
    Named("b3") = store_b3,
    Named("b4") = store_b4,
    Named("b5") = store_b5,
    Named("u")  = store_u,
    Named("u2") = store_u2,
    Named("delta")  = (K1>2) ? wrap(store_delta) : R_NilValue,
    Named("delta2") = (K2>2) ? wrap(store_delta2) : R_NilValue,
    Named("thr")  = store_thr,
    Named("thr2") = store_thr2,
    Named("accept") = List::create(
      Named("alpha")     = acc_alpha / n_iter,
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
      Named("thr")       = acc_thr / n_iter,
      Named("thr2")      = acc_thr2 / n_iter
    )
  );
}
