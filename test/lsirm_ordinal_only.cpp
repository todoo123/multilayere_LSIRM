// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// =========================================================
// Helpers
// =========================================================

inline double log1pexp(double x) {
  if (x > 0) return x + log1p(exp(-x));
  else return log1p(exp(x));
}

inline double softplus(double x) {
  return log1pexp(x);
}

inline double log_sigmoid(double x) {
  if (x >= 0) return -log1p(exp(-x));
  else        return x - log1p(exp(x));
}

inline double inv_logit(double x) {
  return 1.0 / (1.0 + exp(-x));
}

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

inline double calc_dist(const rowvec& a, const rowvec& b) {
  return sqrt(sum(square(a - b)));
}

// =========================================================
// Main MCMC Function (Ordinal-only, single layer)
// =========================================================
// [[Rcpp::export]]
List run_lsirm_ordinal_only_cpp(
    IntegerMatrix Y_ord,
    int d, int n_iter, int burnin, int thin,
    List hyper, List prop_sd, List init,
    bool verbose, bool fix_gamma
) {
  int n = Y_ord.nrow();
  int P = Y_ord.ncol();

  // Find K (max of Y_ord)
  int K = 0;
  for(int i=0; i<n; ++i)
    for(int j=0; j<P; ++j)
      if(Y_ord(i,j) > K) K = Y_ord(i,j);

  if (verbose) {
    Rcout << "n=" << n << " P=" << P << " K=" << K << "\n";
  }

  // Unpack Hyperparameters
  double a_sigma = hyper["a_sigma"]; double b_sigma = hyper["b_sigma"];
  double mu_log_gamma = hyper["mu_log_gamma"]; double sd_log_gamma = hyper["sd_log_gamma"];
  double mu_delta = hyper["mu_delta"]; double sd_delta = hyper["sd_delta"];

  // Unpack Proposal SDs
  double sd_alpha = prop_sd["alpha"];
  double sd_log_gamma_prop = prop_sd["log_gamma"];
  double sd_a = prop_sd["a"];
  double sd_b = prop_sd["b"];
  double sd_delta_prop = prop_sd["delta"];

  // Initialize Parameters
  vec alpha = as<vec>(init["alpha"]);
  mat a = as<mat>(init["a"]);
  mat b = as<mat>(init["b"]);

  double log_gamma = init["log_gamma"];
  double gamma_val = exp(log_gamma);
  double sigma_alpha_sq = init["sigma_alpha_sq"];

  mat delta;
  if (K > 2) delta = as<mat>(init["delta"]);

  // Storage setup
  int n_save = (n_iter - burnin) / thin;

  mat store_alpha(n_save, n);
  vec store_log_gamma(n_save);
  vec store_sigma_alpha_sq(n_save);

  cube store_a(n, d, n_save);
  cube store_b(P, d, n_save);

  cube store_delta;
  if (K > 2) store_delta.set_size(P, K-2, n_save);
  cube store_thr;
  if (K > 2) store_thr.set_size(P, K-1, n_save);

  // Acceptance counters
  vec acc_alpha = zeros<vec>(n);
  double acc_log_gamma = 0;
  vec acc_a = zeros<vec>(n);
  vec acc_b = zeros<vec>(P);
  vec acc_thr = zeros<vec>(P);

  int save_idx = 0;

  // MCMC Loop
  for(int iter = 0; iter < n_iter; ++iter) {

    if (verbose && (iter + 1) % 500 == 0) Rcout << "Iter: " << iter + 1 << " / " << n_iter << "\n";

    // --- 1. Update alpha_i ---
    for(int i=0; i<n; ++i) {
      double a_old = alpha(i);
      double a_prop = R::rnorm(a_old, sd_alpha);
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for(int j=0; j<P; ++j) {
          double dist = calc_dist(a.row(i), b.row(j));
          double eta = val - gamma_val * dist;
          vec b_thresh = build_thresholds_cpp(0.0, (K>2) ? delta.row(j) : rowvec());
          ll += log_p_ordinal_single_cpp(Y_ord(i,j), eta, b_thresh);
        }
        return ll;
      };
      double lp_old = -0.5 * pow(a_old, 2) / sigma_alpha_sq;
      double lp_new = -0.5 * pow(a_prop, 2) / sigma_alpha_sq;
      if (log(R::runif(0,1)) < (ll_f(a_prop) + lp_new - ll_f(a_old) - lp_old)) {
        alpha(i) = a_prop; acc_alpha(i)++;
      }
    }

    // --- 2. Update Log Gamma ---
    if (!fix_gamma) {
      double lg_old = log_gamma;
      double lg_prop = R::rnorm(lg_old, sd_log_gamma_prop);
      double g_old = exp(lg_old);
      double g_prop = exp(lg_prop);
      auto ll_g = [&](double g) {
        double ll = 0;
        for(int i=0; i<n; ++i)
          for(int j=0; j<P; ++j) {
            double dist = calc_dist(a.row(i), b.row(j));
            double eta = alpha(i) - g * dist;
            vec b_th = build_thresholds_cpp(0.0, (K>2)? delta.row(j) : rowvec());
            ll += log_p_ordinal_single_cpp(Y_ord(i,j), eta, b_th);
          }
        return ll;
      };
      double ll_old = ll_g(g_old);
      double ll_new = ll_g(g_prop);
      double lp_old = R::dnorm(lg_old, mu_log_gamma, sd_log_gamma, 1);
      double lp_new = R::dnorm(lg_prop, mu_log_gamma, sd_log_gamma, 1);
      if(log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)){
        log_gamma = lg_prop; gamma_val = g_prop; acc_log_gamma++;
      }
    } else {
      acc_log_gamma++;
    }

    // --- 3. Update Shared Position a_i ---
    for(int i=0; i<n; ++i) {
      rowvec a_old = a.row(i);
      rowvec a_prop = a_old + randn<rowvec>(d) * sd_a;

      auto pos_ll = [&](const rowvec& pos) {
        double ll = 0;
        for(int j=0; j<P; ++j) {
          double dist = calc_dist(pos, b.row(j));
          double eta = alpha(i) - gamma_val * dist;
          vec b_th = build_thresholds_cpp(0.0, (K>2)? delta.row(j) : rowvec());
          ll += log_p_ordinal_single_cpp(Y_ord(i,j), eta, b_th);
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

    // --- 4. Update Item Positions b_j ---
    for(int j=0; j<P; ++j) {
      rowvec b_old = b.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b;
      vec b_th = build_thresholds_cpp(0.0, (K>2)? delta.row(j) : rowvec());
      auto ll_f = [&](const rowvec& b_val) {
        double ll=0;
        for(int i=0; i<n; ++i){
          double dist = calc_dist(a.row(i), b_val);
          double eta = alpha(i) - gamma_val * dist;
          ll += log_p_ordinal_single_cpp(Y_ord(i,j), eta, b_th);
        }
        return ll;
      };
      if(log(R::runif(0,1)) < (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) - (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b.row(j) = b_prop; acc_b(j)++;
      }
    }

    // --- 5. Update Ordinal Thresholds (u fixed at 0, update delta only) ---
    for(int j=0; j<P; ++j) {
      rowvec delta_curr;
      if (K>2) delta_curr = delta.row(j);

      rowvec delta_prop_val;
      if (K>2) delta_prop_val = delta_curr + randn<rowvec>(K-2) * sd_delta_prop;

      vec thr_curr = build_thresholds_cpp(0.0, delta_curr);
      vec thr_prop = build_thresholds_cpp(0.0, delta_prop_val);

      auto ord_ll = [&](const vec& th) {
        double ll = 0;
        for(int i=0; i<n; ++i) {
          double dist = calc_dist(a.row(i), b.row(j));
          double eta = alpha(i) - gamma_val * dist;
          ll += log_p_ordinal_single_cpp(Y_ord(i,j), eta, th);
        }
        return ll;
      };

      double ll_c = ord_ll(thr_curr);
      double ll_p = ord_ll(thr_prop);

      double lp_c = 0.0, lp_p = 0.0;
      if (K>2) {
        for(int k=0; k<K-2; ++k) {
          lp_c += R::dnorm(delta_curr(k), mu_delta, sd_delta, 1);
          lp_p += R::dnorm(delta_prop_val(k), mu_delta, sd_delta, 1);
        }
      }

      double jac_c = 0.0, jac_p = 0.0;
      if (K > 2) {
        for (int k = 0; k < K-2; ++k) {
          jac_c += log_sigmoid(delta_curr(k));
          jac_p += log_sigmoid(delta_prop_val(k));
        }
      }

      if(log(R::runif(0,1)) < (ll_p + lp_p + jac_p - (ll_c + lp_c + jac_c))) {
        if(K>2) delta.row(j) = delta_prop_val;
        acc_thr(j)++;
      }
    }

    // --- 6. Gibbs Updates ---
    sigma_alpha_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha, alpha)));

    // --- 7. Save Samples ---
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      store_alpha.row(save_idx) = alpha.t();
      store_log_gamma(save_idx) = log_gamma;
      store_sigma_alpha_sq(save_idx) = sigma_alpha_sq;

      store_a.slice(save_idx) = a;
      store_b.slice(save_idx) = b;

      if(K>2) store_delta.slice(save_idx) = delta;
      for(int j=0; j<P; ++j) {
        rowvec d_r;
        if(K > 2) d_r = delta.row(j);
        store_thr.slice(save_idx).row(j) = build_thresholds_cpp(0.0, d_r).t();
      }

      save_idx++;
    }
  }

  return List::create(
    Named("alpha") = store_alpha,
    Named("log_gamma") = store_log_gamma,
    Named("sigma_alpha_sq") = store_sigma_alpha_sq,
    Named("a") = store_a,
    Named("b") = store_b,
    Named("delta") = (K>2) ? wrap(store_delta) : R_NilValue,
    Named("thr") = store_thr,
    Named("accept") = List::create(
      Named("alpha")     = acc_alpha / n_iter,
      Named("log_gamma") = acc_log_gamma / n_iter,
      Named("a")         = acc_a / n_iter,
      Named("b")         = acc_b / n_iter,
      Named("thr")       = acc_thr / n_iter
    )
  );
}
