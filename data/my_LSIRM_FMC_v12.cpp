// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector>
#include <algorithm>

using namespace Rcpp;
using namespace arma;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =========================================================
// v12: Joint multilayered LSIRM (5-layer max, per-item kappa)
//      + EPA pairwise partition prior on items
//      + Single-item EPA Gibbs c update
//      + Random-swap MH on allocation permutation sigma
//      + Log-scale MH on EPA hypers (alpha, tau)
//      + Jain-Neal NONCONJUGATE split-merge (EPA-decoupled)
//
//   Differences from v11:
//     - REMOVED: NIW mixture (mu_l, Sigma_l, S0, S0 hyperprior),
//                NIW-collapsed c Gibbs, NIW-conjugate split-merge.
//     - ADDED  : EPA pmf (sequential allocation with attraction
//                kernel lambda_qr = exp(-tau ||z_q-z_r||^2)),
//                permutation sigma in Perm(P), EPA hypers
//                (alpha mass, tau temperature), nonconjugate
//                split-merge that uses log-EPA pmf differences
//                (LSIRM likelihood cancels by decoupling).
//     - CHANGED: b_j prior is N_d(0, sigma_b^2 I_d); the b MH
//                ratio includes the EPA prior contribution
//                because moving z_q rescales pairwise similarities.
// =========================================================

// Fisher-Yates shuffle using R's RNG.
static inline void rcpp_shuffle(std::vector<int>& v) {
  int n = (int) v.size();
  for (int i = n - 1; i > 0; --i) {
    int j = (int) std::floor(R::runif(0.0, (double)(i + 1)));
    if (j > i) j = i;
    std::swap(v[i], v[j]);
  }
}

// ===== LSIRM helpers =====
inline double log1pexp(double x) {
  if (x > 0) return x + log1p(exp(-x));
  else return log1p(exp(x));
}

static inline double inv_logit(double x) {
  if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
  const double ex = std::exp(x);
  return ex / (1.0 + ex);
}

inline double calc_dist(const rowvec& a, const rowvec& b) {
  return sqrt(sum(square(a - b)));
}

template <typename RowLike>
static inline double log_grm_prob_y0based(int y, const RowLike& beta_j, double eta) {
  const int Km1 = (int)beta_j.n_elem;
  const int K = Km1 + 1;
  const double eps = 1e-10;

  auto p_ge = [&](int k) -> double {
    return inv_logit(eta + beta_j((arma::uword)(k - 1)));
  };

  double p = 0.0;
  if (y <= 0) p = 1.0 - p_ge(1);
  else if (y >= K - 1) p = p_ge(K - 1);
  else p = p_ge(y) - p_ge(y + 1);

  if (!std::isfinite(p) || p <= eps) p = eps;
  if (p >= 1.0 - eps) p = 1.0 - eps;
  return std::log(p);
}

// Sample categorical from log-weights (returns 0-based index).
static inline int sample_log_weights(const vec& log_w) {
  double M = log_w.max();
  vec p = arma::exp(log_w - M);
  double s = arma::sum(p);
  double u = R::runif(0.0, 1.0) * s;
  double cum = 0.0;
  int K = (int)p.n_elem;
  for (int k = 0; k < K; ++k) {
    cum += p(k);
    if (u <= cum) return k;
  }
  return K - 1;
}

// Stack b1..b5 into z_all (canonical order: bin->con->cnt->ord1->ord2).
static inline void stack_b_layers(
    const mat& b1, const mat& b2, const mat& b3, const mat& b4, const mat& b5,
    int P1, int P2, int P3, int P4, int P5,
    mat& z_all
) {
  int off = 0;
  if (P1 > 0) { z_all.rows(off, off + P1 - 1) = b1; off += P1; }
  if (P2 > 0) { z_all.rows(off, off + P2 - 1) = b2; off += P2; }
  if (P3 > 0) { z_all.rows(off, off + P3 - 1) = b3; off += P3; }
  if (P4 > 0) { z_all.rows(off, off + P4 - 1) = b4; off += P4; }
  if (P5 > 0) { z_all.rows(off, off + P5 - 1) = b5; off += P5; }
}

// =========================================================
// EPA pmf helpers
// =========================================================

// Sequential-allocation log EPA pmf, O(P^2).
//   c_label    : P-vec of 0-based cluster labels
//   sigma_perm : P-vec of 0-based item indices (the allocation permutation)
//   z_all      : P x d positions (lambda_qr = exp(-tau ||z_q-z_r||^2))
//   alpha, delta, tau : EPA parameters
// Returns log p_EPA(P | sigma, z, alpha, delta, tau).
// Cached version: lambda_mat is the precomputed pairwise similarity matrix
//   lambda_mat(q, r) = exp(-tau * ||z_q - z_r||^2),
// maintained as state across MCMC iterations so the inner loop is a pure
// memory lookup (no exp / no distance recomputation).  When tau or any z_q
// changes, the caller is responsible for refreshing the affected entries
// of lambda_mat *before* calling this function.
static double log_epa_pmf(
    const uvec& c_label,
    const std::vector<int>& sigma_perm,
    const mat& lambda_mat,
    double alpha, double delta
) {
  int P = (int) sigma_perm.size();
  if (P <= 0) return 0.0;

  // cluster_seen[l] = 1 if cluster label l has appeared in the prefix.
  // Labels are guaranteed to be in [0, P), so size = P is always sufficient.
  std::vector<int> cluster_seen(P, 0);

  double log_p = 0.0;
  int q_prev = 0;       // q_{t-1}: number of distinct clusters in prefix.

  for (int t = 0; t < P; ++t) {
    int item_t = sigma_perm[t];
    int cl_t   = (int) c_label((uword) item_t);
    double denom_log = std::log(alpha + (double) t);

    if (cluster_seen[cl_t] == 0) {
      // New cluster at step (t+1): factor (alpha + delta q_{t-1}) / (alpha + t).
      log_p += std::log(alpha + delta * (double) q_prev) - denom_log;
      cluster_seen[cl_t] = 1;
      q_prev += 1;
    } else {
      // Existing cluster: factor (t - delta q_{t-1}) / (alpha + t)
      //                          * (sum_{r in S, r in prefix} lambda_{t,r})
      //                          / (sum_{r in prefix}        lambda_{t,r}).
      double sum_S = 0.0, sum_all = 0.0;
      for (int s = 0; s < t; ++s) {
        int item_s = sigma_perm[s];
        double lam = lambda_mat((uword) item_t, (uword) item_s);
        sum_all += lam;
        if ((int) c_label((uword) item_s) == cl_t) sum_S += lam;
      }
      const double EPS = 1e-300;
      log_p += std::log((double) t - delta * (double) q_prev) - denom_log
             + std::log(sum_S + EPS) - std::log(sum_all + EPS);
    }
  }

  return log_p;
}

// Find the lowest unused cluster label in c_label (range 0..P-1).
// If exclude_q >= 0, c_label[exclude_q] is treated as not occupying its label.
static int find_free_label(const uvec& c_label, int P, int exclude_q) {
  std::vector<int> used(P, 0);
  for (int q = 0; q < P; ++q) {
    if (q == exclude_q) continue;
    int cl = (int) c_label((uword) q);
    if (cl >= 0 && cl < P) used[cl] = 1;
  }
  for (int l = 0; l < P; ++l) if (used[l] == 0) return l;
  return -1;
}

// One restricted Gibbs scan over S_star with two cluster-label slots
// (cl_u, cl_v).  In sampling mode, draws each L_{q*} from the EPA conditional
// and accumulates log path probability.  In scoring mode, forces each
// L_{q*} to target_labels[idx] and records its conditional log-probability.
static double restricted_gibbs_scan_epa(
    uvec& c_label,
    const std::vector<int>& S_star,
    int cl_u, int cl_v,
    const std::vector<int>& sigma_perm,
    const mat& lambda_mat,
    double alpha_epa, double delta_epa,
    int mode,                                  // 0 = sampling, 1 = scoring
    const std::vector<int>& target_labels
) {
  double log_q = 0.0;
  for (int idx = 0; idx < (int) S_star.size(); ++idx) {
    int q_star = S_star[idx];

    c_label((uword) q_star) = (uword) cl_u;
    double log_pu = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                alpha_epa, delta_epa);
    c_label((uword) q_star) = (uword) cl_v;
    double log_pv = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                alpha_epa, delta_epa);

    double M = std::max(log_pu, log_pv);
    double pu = std::exp(log_pu - M);
    double pv = std::exp(log_pv - M);
    double s = pu + pv;
    if (s <= 0.0) { pu = 0.5; pv = 0.5; } else { pu /= s; pv /= s; }

    int chosen;
    if (mode == 0) {
      double u_rand = R::runif(0.0, 1.0);
      chosen = (u_rand < pu) ? cl_u : cl_v;
    } else {
      chosen = target_labels[idx];
    }
    double pc = (chosen == cl_u) ? pu : pv;
    if (pc <= 0.0) pc = 1e-300;
    log_q += std::log(pc);

    c_label((uword) q_star) = (uword) chosen;
  }
  return log_q;
}

// =========================================================
// Main MCMC
// =========================================================
// [[Rcpp::export]]
List run_lsirm_epa_v12_cpp(
    mat Y_bin, mat Y_con, mat Y_cnt, IntegerMatrix Y_ord1, IntegerMatrix Y_ord2,
    int d, int n_iter, int burnin, int thin,
    List lsirm_hyper, List epa_hyper,
    List lsirm_prop_sd, List epa_prop_sd,
    List lsirm_init, List epa_init,
    bool verbose, bool fix_gamma, double nu2,
    bool compute_co_cluster_online,
    int  epa_warmup,
    int  n_split_merge,
    int  n_split_merge_R,
    int  n_perm_swaps,
    double sigma_b,
    bool b_epa_coupling,
    bool update_alpha_epa,
    bool update_tau_epa
) {
  // ===== Dimensions =====
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
  int P_total = P1 + P2 + P3 + P4 + P5;

  if (sigma_b <= 0.0) Rcpp::stop("sigma_b must be > 0");
  double inv_sigma_b_sq = 1.0 / (sigma_b * sigma_b);

  // GRM categories
  int K1 = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < P4; ++j)
      if (Y_ord1(i, j) > K1) K1 = Y_ord1(i, j);
  int K2 = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < P5; ++j)
      if (Y_ord2(i, j) > K2) K2 = Y_ord2(i, j);
  int K1m1 = (K1 > 0) ? K1 - 1 : 0;
  int K2m1 = (K2 > 0) ? K2 - 1 : 0;

  imat Y_ord1_0(n, P4);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < P4; ++j)
      Y_ord1_0(i, j) = Y_ord1(i, j) - 1;
  imat Y_ord2_0(n, P5);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < P5; ++j)
      Y_ord2_0(i, j) = Y_ord2(i, j) - 1;

  if (verbose) {
    Rcout << "[v12 EPA] n=" << n
          << " P1=" << P1 << " P2=" << P2 << " P3=" << P3
          << " P4=" << P4 << " P5=" << P5
          << " P_total=" << P_total
          << " K1=" << K1 << " K2=" << K2 << " nu2=" << nu2
          << " d=" << d
          << " sigma_b=" << sigma_b
          << " M_SM=" << n_split_merge
          << " R_SM=" << n_split_merge_R
          << " M_perm=" << n_perm_swaps
          << " b_epa_coupling=" << (b_epa_coupling ? "TRUE" : "FALSE")
          << " update_alpha=" << (update_alpha_epa ? "TRUE" : "FALSE")
          << " update_tau="   << (update_tau_epa   ? "TRUE" : "FALSE")
          << "\n";
    if (!b_epa_coupling) {
      Rcout << "[v12 EPA]   note: b_epa_coupling=FALSE -> b update uses LSIRM"
            << " + N(0, sigma_b^2) prior only (1-way coupling).\n";
    }
    if (!update_alpha_epa) {
      Rcout << "[v12 EPA]   note: alpha_epa fixed at " << "(initial value)\n";
    }
    if (!update_tau_epa) {
      Rcout << "[v12 EPA]   note: tau_epa fixed at "   << "(initial value)\n";
    }
  }

  int off_L1 = 0;
  int off_L2 = off_L1 + P1;
  int off_L3 = off_L2 + P2;
  int off_L4 = off_L3 + P3;
  int off_L5 = off_L4 + P4;

  // ===== LSIRM hyperparameters =====
  double a_sigma = lsirm_hyper["a_sigma"];   double b_sigma = lsirm_hyper["b_sigma"];
  double a_tau1  = lsirm_hyper["a_tau1"];    double b_tau1  = lsirm_hyper["b_tau1"];
  double a_tau2  = lsirm_hyper["a_tau2"];    double b_tau2  = lsirm_hyper["b_tau2"];
  double a_tau3  = lsirm_hyper["a_tau3"];    double b_tau3  = lsirm_hyper["b_tau3"];
  double a_sigma0= lsirm_hyper["a_sigma0"];  double b_sigma0= lsirm_hyper["b_sigma0"];
  (void)a_tau1; (void)b_tau1; (void)a_tau2; (void)b_tau2; (void)a_tau3; (void)b_tau3;

  double mu_log_gamma1 = lsirm_hyper["mu_log_gamma1"]; double sd_log_gamma1 = lsirm_hyper["sd_log_gamma1"];
  double mu_log_gamma2 = lsirm_hyper["mu_log_gamma2"]; double sd_log_gamma2 = lsirm_hyper["sd_log_gamma2"];
  double mu_log_gamma3 = lsirm_hyper["mu_log_gamma3"]; double sd_log_gamma3 = lsirm_hyper["sd_log_gamma3"];
  double mu_log_gamma4 = lsirm_hyper["mu_log_gamma4"]; double sd_log_gamma4 = lsirm_hyper["sd_log_gamma4"];
  double mu_log_gamma5 = lsirm_hyper["mu_log_gamma5"]; double sd_log_gamma5 = lsirm_hyper["sd_log_gamma5"];
  double mu_log_kappa  = lsirm_hyper["mu_log_kappa"];  double sd_log_kappa  = lsirm_hyper["sd_log_kappa"];
  double mu_beta4 = lsirm_hyper["mu_beta4"]; double sd_beta4_hyp = lsirm_hyper["sd_beta4"];
  double mu_beta5 = lsirm_hyper["mu_beta5"]; double sd_beta5_hyp = lsirm_hyper["sd_beta5"];

  // ===== LSIRM proposal SDs =====
  double sd_alpha1 = lsirm_prop_sd["alpha1"];
  double sd_alpha2 = lsirm_prop_sd["alpha2"];
  double sd_alpha3 = lsirm_prop_sd["alpha3"];
  double sd_alpha4 = lsirm_prop_sd["alpha4"];
  double sd_alpha5 = lsirm_prop_sd["alpha5"];
  double sd_beta1 = lsirm_prop_sd["beta1"];
  double sd_beta2 = lsirm_prop_sd["beta2"];
  double sd_beta3 = lsirm_prop_sd["beta3"];
  double sd_beta4_prop = lsirm_prop_sd["beta4"];
  double sd_beta5_prop = lsirm_prop_sd["beta5"];
  double sd_log_gamma1_prop = lsirm_prop_sd["log_gamma1"];
  double sd_log_gamma2_prop = lsirm_prop_sd["log_gamma2"];
  double sd_log_gamma3_prop = lsirm_prop_sd["log_gamma3"];
  double sd_log_gamma4_prop = lsirm_prop_sd["log_gamma4"];
  double sd_log_gamma5_prop = lsirm_prop_sd["log_gamma5"];
  double sd_log_kappa_prop = lsirm_prop_sd["log_kappa"];
  double sd_a  = lsirm_prop_sd["a"];
  double sd_b1 = lsirm_prop_sd["b1"];
  double sd_b2 = lsirm_prop_sd["b2"];
  double sd_b3 = lsirm_prop_sd["b3"];
  double sd_b4 = lsirm_prop_sd["b4"];
  double sd_b5 = lsirm_prop_sd["b5"];

  // ===== EPA hyperparameters =====
  double a_alpha_epa = epa_hyper["a_alpha"];
  double b_alpha_epa = epa_hyper["b_alpha"];
  double a_tau_epa   = epa_hyper["a_tau"];
  double b_tau_epa   = epa_hyper["b_tau"];
  double delta_epa   = 0.0;
  if (epa_hyper.containsElementNamed("delta")) delta_epa = as<double>(epa_hyper["delta"]);
  if (delta_epa < 0.0 || delta_epa >= 1.0)
    Rcpp::stop("epa_hyper$delta must be in [0, 1)");

  double sd_log_alpha_prop = epa_prop_sd["log_alpha"];
  double sd_log_tau_prop   = epa_prop_sd["log_tau"];

  // ===== LSIRM init =====
  vec alpha1 = as<vec>(lsirm_init["alpha1"]);
  vec alpha2 = as<vec>(lsirm_init["alpha2"]);
  vec alpha3 = as<vec>(lsirm_init["alpha3"]);
  vec alpha4 = as<vec>(lsirm_init["alpha4"]);
  vec alpha5 = as<vec>(lsirm_init["alpha5"]);
  vec beta1 = as<vec>(lsirm_init["beta1"]);
  vec beta2 = as<vec>(lsirm_init["beta2"]);
  vec beta3 = as<vec>(lsirm_init["beta3"]);

  mat a  = as<mat>(lsirm_init["a"]);
  mat b1 = as<mat>(lsirm_init["b1"]);
  mat b2 = as<mat>(lsirm_init["b2"]);
  mat b3 = as<mat>(lsirm_init["b3"]);
  mat b4 = as<mat>(lsirm_init["b4"]);
  mat b5 = as<mat>(lsirm_init["b5"]);

  double log_gamma1 = lsirm_init["log_gamma1"];  double gamma_val1 = exp(log_gamma1);
  double log_gamma2 = lsirm_init["log_gamma2"];  double gamma_val2 = exp(log_gamma2);
  double log_gamma3 = lsirm_init["log_gamma3"];  double gamma_val3 = exp(log_gamma3);
  double log_gamma4 = lsirm_init["log_gamma4"];  double gamma_val4 = exp(log_gamma4);
  double log_gamma5 = lsirm_init["log_gamma5"];  double gamma_val5 = exp(log_gamma5);

  vec log_kappa = as<vec>(lsirm_init["log_kappa"]);
  vec kappa = exp(log_kappa);

  double sigma_alpha1_sq = lsirm_init["sigma_alpha1_sq"];
  double sigma_alpha2_sq = lsirm_init["sigma_alpha2_sq"];
  double sigma_alpha3_sq = lsirm_init["sigma_alpha3_sq"];
  double sigma_alpha4_sq = lsirm_init["sigma_alpha4_sq"];
  double sigma_alpha5_sq = lsirm_init["sigma_alpha5_sq"];
  double tau_beta1_sq = lsirm_init["tau_beta1_sq"];
  double tau_beta2_sq = lsirm_init["tau_beta2_sq"];
  double tau_beta3_sq = lsirm_init["tau_beta3_sq"];
  double sigma0_sq = lsirm_init["sigma0_sq"];

  mat beta4_thr;
  if (P4 > 0 && K1m1 > 0) beta4_thr = as<mat>(lsirm_init["beta4"]);
  mat beta5_thr;
  if (P5 > 0 && K2m1 > 0) beta5_thr = as<mat>(lsirm_init["beta5"]);

  mat lambda2(n, P2, fill::ones);

  // ===== EPA init =====
  uvec c_label = as<uvec>(epa_init["c"]);            // P_total-vec, 0-based
  std::vector<int> sigma_perm;                       // P_total-vec, 0-based
  {
    IntegerVector sp = epa_init["sigma"];
    int n_sp = sp.size();
    sigma_perm.resize(n_sp);
    for (int i = 0; i < n_sp; ++i) sigma_perm[i] = sp[i];
  }
  double alpha_epa = as<double>(epa_init["alpha"]);
  double tau_epa   = as<double>(epa_init["tau"]);

  if ((int) c_label.n_elem != P_total)
    Rcpp::stop("epa_init$c length must equal P_total");
  if ((int) sigma_perm.size() != P_total)
    Rcpp::stop("epa_init$sigma length must equal P_total");

  // Sanity-check sigma is a permutation of {0,...,P_total-1}.
  {
    std::vector<int> seen(P_total, 0);
    for (int t = 0; t < P_total; ++t) {
      int s = sigma_perm[t];
      if (s < 0 || s >= P_total) Rcpp::stop("epa_init$sigma out of range");
      if (seen[s]) Rcpp::stop("epa_init$sigma has duplicates");
      seen[s] = 1;
    }
  }
  // Sanity-check c labels in [0, P_total).
  for (int q = 0; q < P_total; ++q) {
    int cl = (int) c_label((uword) q);
    if (cl < 0 || cl >= P_total)
      Rcpp::stop("epa_init$c labels must be in [0, P_total)");
  }

  // ===== Storage =====
  int n_save = (n_iter - burnin) / thin;
  if (n_save < 0) n_save = 0;

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
  mat store_log_kappa(n_save, P3);

  vec store_sigma_alpha1_sq(n_save);
  vec store_sigma_alpha2_sq(n_save);
  vec store_sigma_alpha3_sq(n_save);
  vec store_sigma_alpha4_sq(n_save);
  vec store_sigma_alpha5_sq(n_save);
  vec store_tau_beta1_sq(n_save);
  vec store_tau_beta2_sq(n_save);
  vec store_tau_beta3_sq(n_save);
  vec store_sigma0_sq(n_save);

  cube store_a(n,  d, n_save);
  cube store_b1(P1, d, n_save);
  cube store_b2(P2, d, n_save);
  cube store_b3(P3, d, n_save);
  cube store_b4(P4, d, n_save);
  cube store_b5(P5, d, n_save);

  cube store_beta4_thr;
  if (P4 > 0 && K1m1 > 0) store_beta4_thr.set_size(P4, K1m1, n_save);
  cube store_beta5_thr;
  if (P5 > 0 && K2m1 > 0) store_beta5_thr.set_size(P5, K2m1, n_save);

  vec store_lambda2_mean(n_save);
  cube store_lambda2(n, P2, n_save);
  mat sum_lambda2(n, P2, fill::zeros);

  // EPA storage
  umat store_c    (n_save, P_total, fill::zeros);
  umat store_sigma(n_save, P_total, fill::zeros);
  vec  store_alpha_epa(n_save, fill::zeros);
  vec  store_tau_epa  (n_save, fill::zeros);
  vec  store_K_plus   (n_save, fill::zeros);
  vec  store_log_epa_pmf(n_save, fill::zeros);

  umat co_count;
  if (compute_co_cluster_online) {
    co_count.set_size(P_total, P_total);
    co_count.zeros();
  }

  // Split-merge / permutation counters
  int sm_split_attempts = 0,  sm_split_accepts  = 0;
  int sm_merge_attempts = 0,  sm_merge_accepts  = 0;
  long sigma_swap_attempts = 0, sigma_swap_accepts = 0;
  int alpha_epa_attempts = 0, alpha_epa_accepts = 0;
  int tau_epa_attempts   = 0, tau_epa_accepts   = 0;

  // ===== LSIRM acceptance counters =====
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
  vec acc_log_kappa = zeros<vec>(P3);
  vec acc_a = zeros<vec>(n);
  vec acc_b1 = zeros<vec>(P1);
  vec acc_b2 = zeros<vec>(P2);
  vec acc_b3 = zeros<vec>(P3);
  vec acc_b4 = zeros<vec>(P4);
  vec acc_b5 = zeros<vec>(P5);
  mat acc_beta4_thr; if (P4 > 0 && K1m1 > 0) acc_beta4_thr = zeros<mat>(P4, K1m1);
  mat acc_beta5_thr; if (P5 > 0 && K2m1 > 0) acc_beta5_thr = zeros<mat>(P5, K2m1);

  int save_idx = 0;

  // Reusable workspace
  mat z_all(P_total, d, fill::zeros);
  stack_b_layers(b1, b2, b3, b4, b5, P1, P2, P3, P4, P5, z_all);

  // ----- Cached pairwise similarity matrix lambda_mat -----
  //   lambda_mat(q, r) = exp(-tau_epa * ||z_q - z_r||^2),   diag = 1
  // Maintained as MCMC state so log_epa_pmf is a pure lookup loop.
  // Refreshed:
  //   - on accepted b update (one row/col, O(P) work),
  //   - on accepted tau MH (full rebuild, O(P^2) work, once per sweep).
  // Not affected by partition / sigma / alpha updates.
  mat lambda_mat(P_total, P_total, fill::zeros);
  auto rebuild_lambda_full = [&](double tau_use) {
    for (int i = 0; i < P_total; ++i) {
      lambda_mat((uword) i, (uword) i) = 1.0;
      for (int j = 0; j < i; ++j) {
        double d2 = 0.0;
        for (int k = 0; k < d; ++k) {
          double dk = z_all((uword) i, (uword) k) - z_all((uword) j, (uword) k);
          d2 += dk * dk;
        }
        double v = std::exp(-tau_use * d2);
        lambda_mat((uword) i, (uword) j) = v;
        lambda_mat((uword) j, (uword) i) = v;
      }
    }
  };
  // Helper: replace row & col q of lambda_mat using new_pos as z_q (other
  // items take their coordinates from z_all).  Returns the old column for
  // restoration on reject.
  auto update_lambda_row = [&](int q, const rowvec& new_pos, double tau_use) {
    vec lambda_save = lambda_mat.col((uword) q);
    for (int r = 0; r < P_total; ++r) {
      if (r == q) {
        lambda_mat((uword) q, (uword) q) = 1.0;
      } else {
        double d2 = 0.0;
        for (int k = 0; k < d; ++k) {
          double dk = new_pos((uword) k) - z_all((uword) r, (uword) k);
          d2 += dk * dk;
        }
        double v = std::exp(-tau_use * d2);
        lambda_mat((uword) q, (uword) r) = v;
        lambda_mat((uword) r, (uword) q) = v;
      }
    }
    return lambda_save;
  };
  auto restore_lambda_row = [&](int q, const vec& lambda_save) {
    lambda_mat.col((uword) q) = lambda_save;
    lambda_mat.row((uword) q) = lambda_save.t();
  };

  rebuild_lambda_full(tau_epa);

  // Cached log EPA pmf at the current state.
  double log_epa_curr = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                    alpha_epa, delta_epa);

  // ===== MCMC LOOP =====
  for (int iter = 0; iter < n_iter; ++iter) {
    if (verbose && (iter + 1) % 500 == 0)
      Rcout << "[v12] Iter: " << iter + 1 << " / " << n_iter
            << "  log p_EPA = " << log_epa_curr
            << "  alpha_epa = " << alpha_epa
            << "  tau_epa = "   << tau_epa << "\n";

    // ===================== LSIRM block =====================
    // 0. lambda2
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < P2; ++j) {
        double dist = calc_dist(a.row(i), b2.row(j));
        double mu = alpha2(i) - beta2(j) - gamma_val2 * dist;
        double e2 = pow(Y_con(i,j) - mu, 2);
        double shape = (nu2 + 1.0) / 2.0;
        double rate  = (nu2 + e2 / sigma0_sq) / 2.0;
        lambda2(i,j) = R::rgamma(shape, 1.0 / rate);
      }
    }

    // 1a. alpha1 (Bin)
    for (int i = 0; i < n; ++i) {
      double a_old = alpha1(i);
      double a_prop = R::rnorm(a_old, sd_alpha1);
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for (int j = 0; j < P1; ++j) {
          double dist = calc_dist(a.row(i), b1.row(j));
          double eta_v = val - beta1(j) - gamma_val1 * dist;
          ll += Y_bin(i,j) * eta_v - log1pexp(eta_v);
        }
        return ll;
      };
      double lp_old = -0.5 * pow(a_old, 2) / sigma_alpha1_sq;
      double lp_new = -0.5 * pow(a_prop, 2) / sigma_alpha1_sq;
      if (log(R::runif(0,1)) < (ll_f(a_prop) + lp_new - ll_f(a_old) - lp_old)) {
        alpha1(i) = a_prop; acc_alpha1(i)++;
      }
    }
    // 1b. alpha2 (Con - robust)
    for (int i = 0; i < n; ++i) {
      double a_old = alpha2(i);
      double a_prop = R::rnorm(a_old, sd_alpha2);
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for (int j = 0; j < P2; ++j) {
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
    // 1c. alpha3 (Cnt)
    for (int i = 0; i < n; ++i) {
      double a_old = alpha3(i);
      double a_prop = R::rnorm(a_old, sd_alpha3);
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for (int j = 0; j < P3; ++j) {
          double size_j = 1.0 / kappa(j);
          double dist = calc_dist(a.row(i), b3.row(j));
          double mu = exp(val - beta3(j) - gamma_val3 * dist);
          double prob = size_j / (size_j + mu);
          ll += R::dnbinom(Y_cnt(i,j), size_j, prob, 1);
        }
        return ll;
      };
      double lp_old = -0.5 * pow(a_old, 2) / sigma_alpha3_sq;
      double lp_new = -0.5 * pow(a_prop, 2) / sigma_alpha3_sq;
      if (log(R::runif(0,1)) < (ll_f(a_prop) + lp_new - ll_f(a_old) - lp_old)) {
        alpha3(i) = a_prop; acc_alpha3(i)++;
      }
    }
    // 1d. alpha4 (Ord1)
    for (int i = 0; i < n; ++i) {
      double a_old = alpha4(i);
      double a_prop = R::rnorm(a_old, sd_alpha4);
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for (int j = 0; j < P4; ++j) {
          double dist = calc_dist(a.row(i), b4.row(j));
          double eta_v = val - gamma_val4 * dist;
          ll += log_grm_prob_y0based(Y_ord1_0(i,j), beta4_thr.row(j), eta_v);
        }
        return ll;
      };
      double lp_old = -0.5 * pow(a_old, 2) / sigma_alpha4_sq;
      double lp_new = -0.5 * pow(a_prop, 2) / sigma_alpha4_sq;
      if (log(R::runif(0,1)) < (ll_f(a_prop) + lp_new - ll_f(a_old) - lp_old)) {
        alpha4(i) = a_prop; acc_alpha4(i)++;
      }
    }
    // 1e. alpha5 (Ord2)
    for (int i = 0; i < n; ++i) {
      double a_old = alpha5(i);
      double a_prop = R::rnorm(a_old, sd_alpha5);
      auto ll_f = [&](double val) {
        double ll = 0.0;
        for (int j = 0; j < P5; ++j) {
          double dist = calc_dist(a.row(i), b5.row(j));
          double eta_v = val - gamma_val5 * dist;
          ll += log_grm_prob_y0based(Y_ord2_0(i,j), beta5_thr.row(j), eta_v);
        }
        return ll;
      };
      double lp_old = -0.5 * pow(a_old, 2) / sigma_alpha5_sq;
      double lp_new = -0.5 * pow(a_prop, 2) / sigma_alpha5_sq;
      if (log(R::runif(0,1)) < (ll_f(a_prop) + lp_new - ll_f(a_old) - lp_old)) {
        alpha5(i) = a_prop; acc_alpha5(i)++;
      }
    }

    // 2a. beta1
    for (int j = 0; j < P1; ++j) {
      double b_old = beta1(j);
      double b_prop = R::rnorm(b_old, sd_beta1);
      auto ll_func = [&](double b_val) {
        double ll = 0;
        for (int i = 0; i < n; ++i) {
          double dist = calc_dist(a.row(i), b1.row(j));
          double eta_v = alpha1(i) - b_val - gamma_val1 * dist;
          ll += Y_bin(i,j) * eta_v - log1pexp(eta_v);
        }
        return ll;
      };
      double ll_curr = ll_func(b_old);
      double ll_next = ll_func(b_prop);
      double lp_curr = -0.5 * pow(b_old, 2) / tau_beta1_sq;
      double lp_next = -0.5 * pow(b_prop, 2) / tau_beta1_sq;
      if (log(R::runif(0,1)) < (ll_next + lp_next - ll_curr - lp_curr)) {
        beta1(j) = b_prop; acc_beta1(j)++;
      }
    }
    // 2b. beta2
    for (int j = 0; j < P2; ++j) {
      double b_old = beta2(j);
      double b_prop = R::rnorm(b_old, sd_beta2);
      auto ll_func = [&](double b_val) {
        double ll = 0;
        for (int i = 0; i < n; ++i) {
          double dist = calc_dist(a.row(i), b2.row(j));
          double mu = alpha2(i) - b_val - gamma_val2 * dist;
          double lam = lambda2(i,j);
          double sd_eff = sqrt(sigma0_sq / lam);
          ll += R::dnorm(Y_con(i,j), mu, sd_eff, 1);
        }
        return ll;
      };
      if (log(R::runif(0,1)) <
          (ll_func(b_prop) - 0.5*pow(b_prop,2)/tau_beta2_sq -
           (ll_func(b_old) - 0.5*pow(b_old,2)/tau_beta2_sq))) {
        beta2(j) = b_prop; acc_beta2(j)++;
      }
    }
    // 2c. beta3
    for (int j = 0; j < P3; ++j) {
      double b_old = beta3(j);
      double b_prop = R::rnorm(b_old, sd_beta3);
      double size_j = 1.0 / kappa(j);
      auto ll_func = [&](double b_val) {
        double ll = 0;
        for (int i = 0; i < n; ++i) {
          double dist = calc_dist(a.row(i), b3.row(j));
          double mu = exp(alpha3(i) - b_val - gamma_val3 * dist);
          double prob = size_j / (size_j + mu);
          ll += R::dnbinom(Y_cnt(i,j), size_j, prob, 1);
        }
        return ll;
      };
      if (log(R::runif(0,1)) <
          (ll_func(b_prop) - 0.5*pow(b_prop,2)/tau_beta3_sq -
           (ll_func(b_old) - 0.5*pow(b_old,2)/tau_beta3_sq))) {
        beta3(j) = b_prop; acc_beta3(j)++;
      }
    }

    // 3. log_gamma1..5
    if (fix_gamma == true) {
      acc_log_gamma1++; acc_log_gamma2++; acc_log_gamma3++;
      acc_log_gamma4++; acc_log_gamma5++;
    } else {
      // 3a. gamma1
      {
        double lg_old = log_gamma1;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma1_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        auto ll_g = [&](double g) {
          double ll = 0;
          for (int i = 0; i < n; ++i)
            for (int j = 0; j < P1; ++j) {
              double dist = calc_dist(a.row(i), b1.row(j));
              double eta_v = alpha1(i) - beta1(j) - g * dist;
              ll += Y_bin(i,j) * eta_v - log1pexp(eta_v);
            }
          return ll;
        };
        double ll_old = ll_g(g_old), ll_new = ll_g(g_prop);
        double lp_old = R::dnorm(lg_old,  mu_log_gamma1, sd_log_gamma1, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma1, sd_log_gamma1, 1);
        if (log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
          log_gamma1 = lg_prop; gamma_val1 = g_prop; acc_log_gamma1++;
        }
      }
      // 3b. gamma2
      {
        double lg_old = log_gamma2;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma2_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        auto ll_g = [&](double g) {
          double ll = 0;
          for (int i = 0; i < n; ++i)
            for (int j = 0; j < P2; ++j) {
              double dist = calc_dist(a.row(i), b2.row(j));
              double mu = alpha2(i) - beta2(j) - g * dist;
              double lam = lambda2(i,j);
              double sd_eff = sqrt(sigma0_sq / lam);
              ll += R::dnorm(Y_con(i,j), mu, sd_eff, 1);
            }
          return ll;
        };
        double ll_old = ll_g(g_old), ll_new = ll_g(g_prop);
        double lp_old = R::dnorm(lg_old,  mu_log_gamma2, sd_log_gamma2, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma2, sd_log_gamma2, 1);
        if (log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
          log_gamma2 = lg_prop; gamma_val2 = g_prop; acc_log_gamma2++;
        }
      }
      // 3c. gamma3
      {
        double lg_old = log_gamma3;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma3_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        auto ll_g = [&](double g) {
          double ll = 0;
          for (int i = 0; i < n; ++i)
            for (int j = 0; j < P3; ++j) {
              double size_j = 1.0 / kappa(j);
              double dist = calc_dist(a.row(i), b3.row(j));
              double mu = exp(alpha3(i) - beta3(j) - g * dist);
              double prob = size_j / (size_j + mu);
              ll += R::dnbinom(Y_cnt(i,j), size_j, prob, 1);
            }
          return ll;
        };
        double ll_old = ll_g(g_old), ll_new = ll_g(g_prop);
        double lp_old = R::dnorm(lg_old,  mu_log_gamma3, sd_log_gamma3, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma3, sd_log_gamma3, 1);
        if (log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
          log_gamma3 = lg_prop; gamma_val3 = g_prop; acc_log_gamma3++;
        }
      }
      // 3d. gamma4
      {
        double lg_old = log_gamma4;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma4_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        auto ll_g = [&](double g) {
          double ll = 0;
          for (int i = 0; i < n; ++i)
            for (int j = 0; j < P4; ++j) {
              double dist = calc_dist(a.row(i), b4.row(j));
              double eta_v = alpha4(i) - g * dist;
              ll += log_grm_prob_y0based(Y_ord1_0(i,j), beta4_thr.row(j), eta_v);
            }
          return ll;
        };
        double ll_old = ll_g(g_old), ll_new = ll_g(g_prop);
        double lp_old = R::dnorm(lg_old,  mu_log_gamma4, sd_log_gamma4, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma4, sd_log_gamma4, 1);
        if (log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
          log_gamma4 = lg_prop; gamma_val4 = g_prop; acc_log_gamma4++;
        }
      }
      // 3e. gamma5
      {
        double lg_old = log_gamma5;
        double lg_prop = R::rnorm(lg_old, sd_log_gamma5_prop);
        double g_old = exp(lg_old);
        double g_prop = exp(lg_prop);
        auto ll_g = [&](double g) {
          double ll = 0;
          for (int i = 0; i < n; ++i)
            for (int j = 0; j < P5; ++j) {
              double dist = calc_dist(a.row(i), b5.row(j));
              double eta_v = alpha5(i) - g * dist;
              ll += log_grm_prob_y0based(Y_ord2_0(i,j), beta5_thr.row(j), eta_v);
            }
          return ll;
        };
        double ll_old = ll_g(g_old), ll_new = ll_g(g_prop);
        double lp_old = R::dnorm(lg_old,  mu_log_gamma5, sd_log_gamma5, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma5, sd_log_gamma5, 1);
        if (log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
          log_gamma5 = lg_prop; gamma_val5 = g_prop; acc_log_gamma5++;
        }
      }
    }

    // 4. shared a (aggregates likelihood across all 5 layers; prior N(0, I_d))
    for (int i = 0; i < n; ++i) {
      rowvec a_old = a.row(i);
      rowvec a_prop = a_old + randn<rowvec>(d) * sd_a;
      auto pos_ll = [&](const rowvec& pos) {
        double ll = 0;
        for (int j = 0; j < P1; ++j) {
          double dist = calc_dist(pos, b1.row(j));
          double eta_v = alpha1(i) - beta1(j) - gamma_val1 * dist;
          ll += Y_bin(i,j) * eta_v - log1pexp(eta_v);
        }
        for (int j = 0; j < P2; ++j) {
          double dist = calc_dist(pos, b2.row(j));
          double mu = alpha2(i) - beta2(j) - gamma_val2 * dist;
          double lam = lambda2(i,j);
          double sd_eff = sqrt(sigma0_sq / lam);
          ll += R::dnorm(Y_con(i,j), mu, sd_eff, 1);
        }
        for (int j = 0; j < P3; ++j) {
          double size_j = 1.0 / kappa(j);
          double dist = calc_dist(pos, b3.row(j));
          double mu = exp(alpha3(i) - beta3(j) - gamma_val3 * dist);
          double prob = size_j / (size_j + mu);
          ll += R::dnbinom(Y_cnt(i,j), size_j, prob, 1);
        }
        for (int j = 0; j < P4; ++j) {
          double dist = calc_dist(pos, b4.row(j));
          double eta_v = alpha4(i) - gamma_val4 * dist;
          ll += log_grm_prob_y0based(Y_ord1_0(i,j), beta4_thr.row(j), eta_v);
        }
        for (int j = 0; j < P5; ++j) {
          double dist = calc_dist(pos, b5.row(j));
          double eta_v = alpha5(i) - gamma_val5 * dist;
          ll += log_grm_prob_y0based(Y_ord2_0(i,j), beta5_thr.row(j), eta_v);
        }
        return ll;
      };
      double ll_curr = pos_ll(a_old);
      double ll_next = pos_ll(a_prop);
      double lp_curr = -0.5 * dot(a_old, a_old);
      double lp_next = -0.5 * dot(a_prop, a_prop);
      if (log(R::runif(0,1)) < (ll_next + lp_next - ll_curr - lp_curr)) {
        a.row(i) = a_prop; acc_a(i)++;
      }
    }

    // ---- z_all and log_epa_curr are kept up-to-date incrementally:
    //      every accepted b/c/sigma/alpha/tau move refreshes them.  The
    //      defensive recompute that used to live here was removed for speed.

    // ---- b updates: prior N(0, sigma_b^2 I_d) + EPA prior contribution ----
    // 5a. b1
    for (int j = 0; j < P1; ++j) {
      int q = off_L1 + j;
      rowvec b_old = b1.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b1;
      auto ll_f = [&](const rowvec& b_val) {
        double ll = 0;
        for (int i = 0; i < n; ++i) {
          double dist = calc_dist(a.row(i), b_val);
          double eta_v = alpha1(i) - beta1(j) - gamma_val1 * dist;
          ll += Y_bin(i,j) * eta_v - log1pexp(eta_v);
        }
        return ll;
      };
      double ll_old = ll_f(b_old);
      double ll_new = ll_f(b_prop);
      double prior_diff = -0.5 * inv_sigma_b_sq *
        (arma::dot(b_prop, b_prop) - arma::dot(b_old, b_old));

      rowvec z_save = z_all.row((uword) q);
      vec lambda_save = update_lambda_row(q, b_prop, tau_epa);
      z_all.row((uword) q) = b_prop;
      double log_R, log_epa_new = 0.0;
      if (b_epa_coupling) {
        log_epa_new = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                  alpha_epa, delta_epa);
        log_R = (ll_new - ll_old) + prior_diff
                + (log_epa_new - log_epa_curr);
      } else {
        log_R = (ll_new - ll_old) + prior_diff;
      }
      if (log(R::runif(0,1)) < log_R) {
        b1.row(j) = b_prop;
        if (b_epa_coupling) log_epa_curr = log_epa_new;
        acc_b1(j)++;
      } else {
        z_all.row((uword) q) = z_save;
        restore_lambda_row(q, lambda_save);
      }
    }
    // 5b. b2
    for (int j = 0; j < P2; ++j) {
      int q = off_L2 + j;
      rowvec b_old = b2.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b2;
      auto ll_f = [&](const rowvec& b_val) {
        double ll = 0;
        for (int i = 0; i < n; ++i) {
          double dist = calc_dist(a.row(i), b_val);
          double mu = alpha2(i) - beta2(j) - gamma_val2 * dist;
          double lam = lambda2(i,j);
          double sd_eff = sqrt(sigma0_sq / lam);
          ll += R::dnorm(Y_con(i,j), mu, sd_eff, 1);
        }
        return ll;
      };
      double ll_old = ll_f(b_old);
      double ll_new = ll_f(b_prop);
      double prior_diff = -0.5 * inv_sigma_b_sq *
        (arma::dot(b_prop, b_prop) - arma::dot(b_old, b_old));

      rowvec z_save = z_all.row((uword) q);
      vec lambda_save = update_lambda_row(q, b_prop, tau_epa);
      z_all.row((uword) q) = b_prop;
      double log_R, log_epa_new = 0.0;
      if (b_epa_coupling) {
        log_epa_new = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                  alpha_epa, delta_epa);
        log_R = (ll_new - ll_old) + prior_diff
                + (log_epa_new - log_epa_curr);
      } else {
        log_R = (ll_new - ll_old) + prior_diff;
      }
      if (log(R::runif(0,1)) < log_R) {
        b2.row(j) = b_prop;
        if (b_epa_coupling) log_epa_curr = log_epa_new;
        acc_b2(j)++;
      } else {
        z_all.row((uword) q) = z_save;
        restore_lambda_row(q, lambda_save);
      }
    }
    // 5c. b3
    for (int j = 0; j < P3; ++j) {
      int q = off_L3 + j;
      rowvec b_old = b3.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b3;
      double size_j = 1.0 / kappa(j);
      auto ll_f = [&](const rowvec& b_val) {
        double ll = 0;
        for (int i = 0; i < n; ++i) {
          double dist = calc_dist(a.row(i), b_val);
          double mu = exp(alpha3(i) - beta3(j) - gamma_val3 * dist);
          ll += R::dnbinom(Y_cnt(i,j), size_j, size_j / (size_j + mu), 1);
        }
        return ll;
      };
      double ll_old = ll_f(b_old);
      double ll_new = ll_f(b_prop);
      double prior_diff = -0.5 * inv_sigma_b_sq *
        (arma::dot(b_prop, b_prop) - arma::dot(b_old, b_old));

      rowvec z_save = z_all.row((uword) q);
      vec lambda_save = update_lambda_row(q, b_prop, tau_epa);
      z_all.row((uword) q) = b_prop;
      double log_R, log_epa_new = 0.0;
      if (b_epa_coupling) {
        log_epa_new = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                  alpha_epa, delta_epa);
        log_R = (ll_new - ll_old) + prior_diff
                + (log_epa_new - log_epa_curr);
      } else {
        log_R = (ll_new - ll_old) + prior_diff;
      }
      if (log(R::runif(0,1)) < log_R) {
        b3.row(j) = b_prop;
        if (b_epa_coupling) log_epa_curr = log_epa_new;
        acc_b3(j)++;
      } else {
        z_all.row((uword) q) = z_save;
        restore_lambda_row(q, lambda_save);
      }
    }
    // 5d. b4
    for (int j = 0; j < P4; ++j) {
      int q = off_L4 + j;
      rowvec b_old = b4.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b4;
      auto ll_f = [&](const rowvec& b_val) {
        double ll = 0;
        for (int i = 0; i < n; ++i) {
          double dist = calc_dist(a.row(i), b_val);
          double eta_v = alpha4(i) - gamma_val4 * dist;
          ll += log_grm_prob_y0based(Y_ord1_0(i,j), beta4_thr.row(j), eta_v);
        }
        return ll;
      };
      double ll_old = ll_f(b_old);
      double ll_new = ll_f(b_prop);
      double prior_diff = -0.5 * inv_sigma_b_sq *
        (arma::dot(b_prop, b_prop) - arma::dot(b_old, b_old));

      rowvec z_save = z_all.row((uword) q);
      vec lambda_save = update_lambda_row(q, b_prop, tau_epa);
      z_all.row((uword) q) = b_prop;
      double log_R, log_epa_new = 0.0;
      if (b_epa_coupling) {
        log_epa_new = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                  alpha_epa, delta_epa);
        log_R = (ll_new - ll_old) + prior_diff
                + (log_epa_new - log_epa_curr);
      } else {
        log_R = (ll_new - ll_old) + prior_diff;
      }
      if (log(R::runif(0,1)) < log_R) {
        b4.row(j) = b_prop;
        if (b_epa_coupling) log_epa_curr = log_epa_new;
        acc_b4(j)++;
      } else {
        z_all.row((uword) q) = z_save;
        restore_lambda_row(q, lambda_save);
      }
    }
    // 5e. b5
    for (int j = 0; j < P5; ++j) {
      int q = off_L5 + j;
      rowvec b_old = b5.row(j);
      rowvec b_prop = b_old + randn<rowvec>(d) * sd_b5;
      auto ll_f = [&](const rowvec& b_val) {
        double ll = 0;
        for (int i = 0; i < n; ++i) {
          double dist = calc_dist(a.row(i), b_val);
          double eta_v = alpha5(i) - gamma_val5 * dist;
          ll += log_grm_prob_y0based(Y_ord2_0(i,j), beta5_thr.row(j), eta_v);
        }
        return ll;
      };
      double ll_old = ll_f(b_old);
      double ll_new = ll_f(b_prop);
      double prior_diff = -0.5 * inv_sigma_b_sq *
        (arma::dot(b_prop, b_prop) - arma::dot(b_old, b_old));

      rowvec z_save = z_all.row((uword) q);
      vec lambda_save = update_lambda_row(q, b_prop, tau_epa);
      z_all.row((uword) q) = b_prop;
      double log_R, log_epa_new = 0.0;
      if (b_epa_coupling) {
        log_epa_new = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                  alpha_epa, delta_epa);
        log_R = (ll_new - ll_old) + prior_diff
                + (log_epa_new - log_epa_curr);
      } else {
        log_R = (ll_new - ll_old) + prior_diff;
      }
      if (log(R::runif(0,1)) < log_R) {
        b5.row(j) = b_prop;
        if (b_epa_coupling) log_epa_curr = log_epa_new;
        acc_b5(j)++;
      } else {
        z_all.row((uword) q) = z_save;
        restore_lambda_row(q, lambda_save);
      }
    }

    // 6a. GRM thresholds L4 (descending order convention, +beta in linear pred)
    for (int j = 0; j < P4; ++j) {
      rowvec beta_old_j = beta4_thr.row(j);
      for (int c = 0; c < K1m1; ++c) {
        double proposal = R::rnorm(beta_old_j(c), sd_beta4_prop);
        bool valid = true;
        if (c > 0 && !(beta_old_j(c-1) > proposal)) valid = false;
        if (c < K1m1 - 1 && !(proposal > beta_old_j(c+1))) valid = false;
        if (!valid) continue;
        rowvec beta_new_j = beta_old_j;
        beta_new_j(c) = proposal;
        double ll_old = 0.0, ll_new = 0.0;
        for (int i = 0; i < n; ++i) {
          double dist = calc_dist(a.row(i), b4.row(j));
          double eta_v = alpha4(i) - gamma_val4 * dist;
          ll_old += log_grm_prob_y0based(Y_ord1_0(i,j), beta_old_j, eta_v);
          ll_new += log_grm_prob_y0based(Y_ord1_0(i,j), beta_new_j, eta_v);
        }
        double lp_old = R::dnorm(beta_old_j(c), mu_beta4, sd_beta4_hyp, 1);
        double lp_new = R::dnorm(proposal,      mu_beta4, sd_beta4_hyp, 1);
        if (log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
          beta_old_j(c) = proposal;
          acc_beta4_thr(j, c)++;
        }
      }
      beta4_thr.row(j) = beta_old_j;
    }
    // 6b. GRM thresholds L5
    for (int j = 0; j < P5; ++j) {
      rowvec beta_old_j = beta5_thr.row(j);
      for (int c = 0; c < K2m1; ++c) {
        double proposal = R::rnorm(beta_old_j(c), sd_beta5_prop);
        bool valid = true;
        if (c > 0 && !(beta_old_j(c-1) > proposal)) valid = false;
        if (c < K2m1 - 1 && !(proposal > beta_old_j(c+1))) valid = false;
        if (!valid) continue;
        rowvec beta_new_j = beta_old_j;
        beta_new_j(c) = proposal;
        double ll_old = 0.0, ll_new = 0.0;
        for (int i = 0; i < n; ++i) {
          double dist = calc_dist(a.row(i), b5.row(j));
          double eta_v = alpha5(i) - gamma_val5 * dist;
          ll_old += log_grm_prob_y0based(Y_ord2_0(i,j), beta_old_j, eta_v);
          ll_new += log_grm_prob_y0based(Y_ord2_0(i,j), beta_new_j, eta_v);
        }
        double lp_old = R::dnorm(beta_old_j(c), mu_beta5, sd_beta5_hyp, 1);
        double lp_new = R::dnorm(proposal,      mu_beta5, sd_beta5_hyp, 1);
        if (log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
          beta_old_j(c) = proposal;
          acc_beta5_thr(j, c)++;
        }
      }
      beta5_thr.row(j) = beta_old_j;
    }

    // 7. log_kappa per-item
    for (int j = 0; j < P3; ++j) {
      double lk_old = log_kappa(j);
      double lk_prop = R::rnorm(lk_old, sd_log_kappa_prop);
      double k_old = exp(lk_old);
      double k_prop = exp(lk_prop);
      double size_old  = 1.0 / k_old;
      double size_prop = 1.0 / k_prop;
      double ll_old = 0.0, ll_new = 0.0;
      for (int i = 0; i < n; ++i) {
        double dist = calc_dist(a.row(i), b3.row(j));
        double mu = exp(alpha3(i) - beta3(j) - gamma_val3 * dist);
        ll_old += R::dnbinom(Y_cnt(i,j), size_old,  size_old  / (size_old  + mu), 1);
        ll_new += R::dnbinom(Y_cnt(i,j), size_prop, size_prop / (size_prop + mu), 1);
      }
      double lp_old = R::dnorm(lk_old,  mu_log_kappa, sd_log_kappa, 1);
      double lp_new = R::dnorm(lk_prop, mu_log_kappa, sd_log_kappa, 1);
      if (log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
        log_kappa(j) = lk_prop;
        kappa(j)     = k_prop;
        acc_log_kappa(j)++;
      }
    }

    // 8. Gibbs scalars
    {
      double wSSE = 0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P2; ++j) {
          double dist = calc_dist(a.row(i), b2.row(j));
          double mu = alpha2(i) - beta2(j) - gamma_val2 * dist;
          wSSE += lambda2(i,j) * pow(Y_con(i,j) - mu, 2);
        }
      }
      double shape = a_sigma0 + (n*P2)/2.0;
      double rate = b_sigma0 + 0.5 * wSSE;
      sigma0_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
    }
    sigma_alpha1_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha1, alpha1)));
    sigma_alpha2_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha2, alpha2)));
    sigma_alpha3_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha3, alpha3)));
    sigma_alpha4_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha4, alpha4)));
    sigma_alpha5_sq = 1.0 / R::rgamma(a_sigma + n/2.0, 1.0 / (b_sigma + 0.5*dot(alpha5, alpha5)));
    tau_beta1_sq = 4.0;
    tau_beta2_sq = 4.0;
    tau_beta3_sq = 4.0;

    // ============================================================
    // (B) EPA partition block.  Requires up-to-date z_all and
    //     log_epa_curr.  z_all and lambda_mat are always maintained
    //     by the b update loop; log_epa_curr is maintained only when
    //     b_epa_coupling = TRUE.  When FALSE we refresh it here
    //     (one extra O(P^2) eval per sweep, dominated by what
    //     follows).
    // ============================================================
    if (iter >= epa_warmup) {
      if (!b_epa_coupling) {
        log_epa_curr = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                   alpha_epa, delta_epa);
      }

      // ---- C1. Single-item EPA Gibbs sweep ----
      // For each q in random order, enumerate candidate cluster labels:
      // every distinct label currently used by some other item, plus a
      // free label (a "new singleton").  Sample from the categorical
      // proportional to log p_EPA(P^{(S)}).
      std::vector<int> q_order(P_total);
      for (int q = 0; q < P_total; ++q) q_order[q] = q;
      rcpp_shuffle(q_order);

      for (int q : q_order) {
        // Existing distinct labels (excluding q)
        std::vector<int> seen(P_total, 0);
        for (int q2 = 0; q2 < P_total; ++q2) {
          if (q2 == q) continue;
          int cl = (int) c_label((uword) q2);
          if (cl >= 0 && cl < P_total) seen[cl] = 1;
        }
        std::vector<int> existing_clusters;
        for (int l = 0; l < P_total; ++l) if (seen[l]) existing_clusters.push_back(l);
        int new_label = -1;
        for (int l = 0; l < P_total; ++l) if (!seen[l]) { new_label = l; break; }
        if (new_label < 0) new_label = (int) c_label((uword) q); // safety

        int K_existing = (int) existing_clusters.size();
        int K_total = K_existing + 1;
        vec lw(K_total);
        uword orig_cl = c_label((uword) q);

        for (int k = 0; k < K_total; ++k) {
          int cand_cl;
          if (k < K_existing) cand_cl = existing_clusters[k];
          else                cand_cl = new_label;
          c_label((uword) q) = (uword) cand_cl;
          lw(k) = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                              alpha_epa, delta_epa);
        }
        int chosen_idx = sample_log_weights(lw);
        int chosen_cl = (chosen_idx < K_existing) ? existing_clusters[chosen_idx]
                                                  : new_label;
        c_label((uword) q) = (uword) chosen_cl;
        log_epa_curr = lw(chosen_idx);
        (void)orig_cl;
      }

      // ---- C2. Jain-Neal NONCONJUGATE split-merge ----
      // EPA-decoupled: LSIRM likelihood cancels.  Acceptance is
      // governed entirely by the EPA pmf ratio and the restricted
      // Gibbs proposal correction.
      for (int sm = 0; sm < n_split_merge; ++sm) {
        if (P_total < 2) break;

        int u = (int) std::floor(R::runif(0.0, (double) P_total));
        if (u >= P_total) u = P_total - 1;
        int v = u;
        while (v == u) {
          v = (int) std::floor(R::runif(0.0, (double) P_total));
          if (v >= P_total) v = P_total - 1;
        }

        int cl_u = (int) c_label((uword) u);
        int cl_v = (int) c_label((uword) v);
        uvec c_save = c_label;

        if (cl_u == cl_v) {
          // ---- SPLIT ----
          sm_split_attempts++;
          int cl_v_new = find_free_label(c_label, P_total, -1);
          if (cl_v_new < 0) continue;

          std::vector<int> S_star;
          for (int q = 0; q < P_total; ++q) {
            if (q == u || q == v) continue;
            if ((int) c_label((uword) q) == cl_u) S_star.push_back(q);
          }

          // Launch state: random 1/2 assignment to {cl_u, cl_v_new}.
          c_label((uword) v) = (uword) cl_v_new;
          for (int q_star : S_star) {
            c_label((uword) q_star) =
              (uword) ((R::runif(0.0, 1.0) < 0.5) ? cl_u : cl_v_new);
          }
          // R launch scans (sampling, no log-q recorded)
          for (int r = 0; r < n_split_merge_R; ++r) {
            restricted_gibbs_scan_epa(c_label, S_star, cl_u, cl_v_new,
                                      sigma_perm, lambda_mat,
                                      alpha_epa, delta_epa,
                                      0, std::vector<int>());
          }
          // Final scan (sampling, recording log_q_fwd)
          double log_q_fwd =
            restricted_gibbs_scan_epa(c_label, S_star, cl_u, cl_v_new,
                                      sigma_perm, lambda_mat,
                                      alpha_epa, delta_epa,
                                      0, std::vector<int>());
          double log_epa_split = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                             alpha_epa, delta_epa);
          double log_R = log_epa_split - log_epa_curr - log_q_fwd;
          if (log(R::runif(0.0, 1.0)) < log_R) {
            log_epa_curr = log_epa_split;
            sm_split_accepts++;
          } else {
            c_label = c_save;
          }
        } else {
          // ---- MERGE ----
          sm_merge_attempts++;
          std::vector<int> S_star;
          std::vector<int> orig_labels_in_S_star;
          for (int q = 0; q < P_total; ++q) {
            if (q == u || q == v) continue;
            int cl = (int) c_label((uword) q);
            if (cl == cl_u || cl == cl_v) {
              S_star.push_back(q);
              orig_labels_in_S_star.push_back(cl);
            }
          }
          // Forward (deterministic merge): set everyone in S to cl_u.
          c_label((uword) v) = (uword) cl_u;
          for (int q_star : S_star) c_label((uword) q_star) = (uword) cl_u;
          double log_epa_mrg = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                           alpha_epa, delta_epa);

          // Reverse: start from {S_u, S_v} (= original c_label), run R launch
          // scans, then a scoring scan that forces the original labels.
          c_label = c_save;
          for (int r = 0; r < n_split_merge_R; ++r) {
            restricted_gibbs_scan_epa(c_label, S_star, cl_u, cl_v,
                                      sigma_perm, lambda_mat,
                                      alpha_epa, delta_epa,
                                      0, std::vector<int>());
          }
          double log_q_rev =
            restricted_gibbs_scan_epa(c_label, S_star, cl_u, cl_v,
                                      sigma_perm, lambda_mat,
                                      alpha_epa, delta_epa,
                                      1, orig_labels_in_S_star);
          double log_R = log_epa_mrg - log_epa_curr + log_q_rev;
          if (log(R::runif(0.0, 1.0)) < log_R) {
            // Accept: redo the merge.
            c_label((uword) v) = (uword) cl_u;
            for (int q_star : S_star) c_label((uword) q_star) = (uword) cl_u;
            log_epa_curr = log_epa_mrg;
            sm_merge_accepts++;
          } else {
            c_label = c_save;
          }
        }
      }

      // ---- C3. Permutation update: random-swap MH ----
      for (int sw = 0; sw < n_perm_swaps; ++sw) {
        if (P_total < 2) break;
        int i_pos = (int) std::floor(R::runif(0.0, (double) P_total));
        if (i_pos >= P_total) i_pos = P_total - 1;
        int j_pos = i_pos;
        while (j_pos == i_pos) {
          j_pos = (int) std::floor(R::runif(0.0, (double) P_total));
          if (j_pos >= P_total) j_pos = P_total - 1;
        }
        std::swap(sigma_perm[i_pos], sigma_perm[j_pos]);
        sigma_swap_attempts++;
        double log_epa_new = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                         alpha_epa, delta_epa);
        if (log(R::runif(0.0, 1.0)) < log_epa_new - log_epa_curr) {
          log_epa_curr = log_epa_new;
          sigma_swap_accepts++;
        } else {
          std::swap(sigma_perm[i_pos], sigma_perm[j_pos]); // undo
        }
      }

      // ---- C4. EPA hyperparameter updates ----
      // alpha_epa: log-scale random walk; prior Gamma(a_alpha, b_alpha).
      // alpha does not enter lambda_mat, so no rebuild needed.
      // Skipped entirely if update_alpha_epa = FALSE (fixed-value mode).
      if (update_alpha_epa) {
        alpha_epa_attempts++;
        double log_a_old  = std::log(alpha_epa);
        double log_a_prop = R::rnorm(log_a_old, sd_log_alpha_prop);
        double a_prop     = std::exp(log_a_prop);
        double log_epa_new = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                         a_prop, delta_epa);
        // log Gamma prior + log-Jacobian (= log a)
        double lp_old = (a_alpha_epa - 1.0) * log_a_old - b_alpha_epa * alpha_epa
                        + log_a_old;
        double lp_new = (a_alpha_epa - 1.0) * log_a_prop - b_alpha_epa * a_prop
                        + log_a_prop;
        double log_R = (log_epa_new - log_epa_curr) + (lp_new - lp_old);
        if (log(R::runif(0.0, 1.0)) < log_R) {
          alpha_epa = a_prop;
          log_epa_curr = log_epa_new;
          alpha_epa_accepts++;
        }
      }
      // tau_epa: log-scale random walk; prior Gamma(a_tau, b_tau).
      // tau enters lambda_mat through every entry, so we rebuild lambda_mat
      // for the proposal and restore on reject.
      // Skipped entirely if update_tau_epa = FALSE (fixed-value mode);
      // in that case lambda_mat is initialized once at the chosen tau value
      // and never recomputed.
      if (update_tau_epa) {
        tau_epa_attempts++;
        double log_t_old  = std::log(tau_epa);
        double log_t_prop = R::rnorm(log_t_old, sd_log_tau_prop);
        double t_prop     = std::exp(log_t_prop);
        mat lambda_save = lambda_mat;
        rebuild_lambda_full(t_prop);
        double log_epa_new = log_epa_pmf(c_label, sigma_perm, lambda_mat,
                                         alpha_epa, delta_epa);
        double lp_old = (a_tau_epa - 1.0) * log_t_old - b_tau_epa * tau_epa
                        + log_t_old;
        double lp_new = (a_tau_epa - 1.0) * log_t_prop - b_tau_epa * t_prop
                        + log_t_prop;
        double log_R = (log_epa_new - log_epa_curr) + (lp_new - lp_old);
        if (log(R::runif(0.0, 1.0)) < log_R) {
          tau_epa = t_prop;
          log_epa_curr = log_epa_new;
          tau_epa_accepts++;
        } else {
          lambda_mat = lambda_save;  // restore old lambda_mat on reject
        }
      }
    } // end if (iter >= epa_warmup)

    // ===================== Save =====================
    if (iter >= burnin && (iter - burnin) % thin == 0 && save_idx < n_save) {
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
      if (P3 > 0) store_log_kappa.row(save_idx) = log_kappa.t();

      store_sigma_alpha1_sq(save_idx) = sigma_alpha1_sq;
      store_sigma_alpha2_sq(save_idx) = sigma_alpha2_sq;
      store_sigma_alpha3_sq(save_idx) = sigma_alpha3_sq;
      store_sigma_alpha4_sq(save_idx) = sigma_alpha4_sq;
      store_sigma_alpha5_sq(save_idx) = sigma_alpha5_sq;
      store_tau_beta1_sq(save_idx) = tau_beta1_sq;
      store_tau_beta2_sq(save_idx) = tau_beta2_sq;
      store_tau_beta3_sq(save_idx) = tau_beta3_sq;
      store_sigma0_sq(save_idx) = sigma0_sq;

      store_a.slice(save_idx)  = a;
      store_b1.slice(save_idx) = b1;
      store_b2.slice(save_idx) = b2;
      store_b3.slice(save_idx) = b3;
      store_b4.slice(save_idx) = b4;
      store_b5.slice(save_idx) = b5;
      if (P4 > 0 && K1m1 > 0) store_beta4_thr.slice(save_idx) = beta4_thr;
      if (P5 > 0 && K2m1 > 0) store_beta5_thr.slice(save_idx) = beta5_thr;

      store_lambda2_mean(save_idx) = (n*P2 > 0) ? accu(lambda2) / (n * P2) : 1.0;
      store_lambda2.slice(save_idx) = lambda2;
      sum_lambda2 += lambda2;

      // EPA draws
      for (int q = 0; q < P_total; ++q) store_c(save_idx, q) = c_label((uword) q);
      for (int t = 0; t < P_total; ++t) store_sigma(save_idx, t) = (uword) sigma_perm[t];
      store_alpha_epa(save_idx) = alpha_epa;
      store_tau_epa(save_idx)   = tau_epa;
      store_log_epa_pmf(save_idx) = log_epa_curr;

      // K_+: number of distinct labels currently used
      std::vector<int> seen_save(P_total, 0);
      for (int q = 0; q < P_total; ++q) {
        int cl = (int) c_label((uword) q);
        if (cl >= 0 && cl < P_total) seen_save[cl] = 1;
      }
      int Kp = 0;
      for (int l = 0; l < P_total; ++l) if (seen_save[l]) Kp++;
      store_K_plus(save_idx) = (double) Kp;

      if (compute_co_cluster_online) {
        for (int q = 0; q < P_total; ++q) {
          uword cq = c_label((uword) q);
          co_count(q, q) += 1u;
          for (int qp = q + 1; qp < P_total; ++qp) {
            if (c_label((uword) qp) == cq) {
              co_count(q,  qp) += 1u;
              co_count(qp, q ) += 1u;
            }
          }
        }
      }

      save_idx++;
    }
  }

  // ===================== Wrap up =====================
  mat co_cluster_pp;
  if (compute_co_cluster_online) {
    co_cluster_pp.set_size(P_total, P_total);
    if (n_save > 0) co_cluster_pp = arma::conv_to<mat>::from(co_count) / (double)n_save;
    else co_cluster_pp.zeros();
  }

  List out;

  out["alpha1"] = store_alpha1;
  out["alpha2"] = store_alpha2;
  out["alpha3"] = store_alpha3;
  out["alpha4"] = store_alpha4;
  out["alpha5"] = store_alpha5;
  out["beta1"]  = store_beta1;
  out["beta2"]  = store_beta2;
  out["beta3"]  = store_beta3;
  out["log_gamma1"] = store_log_gamma1;
  out["log_gamma2"] = store_log_gamma2;
  out["log_gamma3"] = store_log_gamma3;
  out["log_gamma4"] = store_log_gamma4;
  out["log_gamma5"] = store_log_gamma5;
  out["log_kappa"]  = store_log_kappa;
  out["sigma0_sq"]  = store_sigma0_sq;
  out["sigma_alpha1_sq"] = store_sigma_alpha1_sq;
  out["sigma_alpha2_sq"] = store_sigma_alpha2_sq;
  out["sigma_alpha3_sq"] = store_sigma_alpha3_sq;
  out["sigma_alpha4_sq"] = store_sigma_alpha4_sq;
  out["sigma_alpha5_sq"] = store_sigma_alpha5_sq;
  out["tau_beta1_sq"] = store_tau_beta1_sq;
  out["tau_beta2_sq"] = store_tau_beta2_sq;
  out["tau_beta3_sq"] = store_tau_beta3_sq;
  out["a"]  = store_a;
  out["b1"] = store_b1;
  out["b2"] = store_b2;
  out["b3"] = store_b3;
  out["b4"] = store_b4;
  out["b5"] = store_b5;
  out["beta4"] = (P4 > 0 && K1m1 > 0) ? wrap(store_beta4_thr) : R_NilValue;
  out["beta5"] = (P5 > 0 && K2m1 > 0) ? wrap(store_beta5_thr) : R_NilValue;
  out["lambda2_mean"]     = store_lambda2_mean;
  out["lambda2"]          = store_lambda2;
  out["lambda2_postmean"] = sum_lambda2 / std::max(n_save, 1);

  // EPA outputs
  out["epa_c"]            = store_c;
  out["epa_sigma"]        = store_sigma;
  out["epa_alpha"]        = store_alpha_epa;
  out["epa_tau"]          = store_tau_epa;
  out["epa_K_plus"]       = store_K_plus;
  out["epa_log_pmf"]      = store_log_epa_pmf;
  out["epa_co_cluster"]   = compute_co_cluster_online ? wrap(co_cluster_pp) : R_NilValue;
  out["epa_delta"]        = delta_epa;

  // EPA / split-merge / permutation diagnostics
  List sm;
  sm["split_attempts"] = sm_split_attempts;
  sm["split_accepts"]  = sm_split_accepts;
  sm["merge_attempts"] = sm_merge_attempts;
  sm["merge_accepts"]  = sm_merge_accepts;
  sm["split_rate"] = (sm_split_attempts > 0)
    ? (double)sm_split_accepts / (double)sm_split_attempts : 0.0;
  sm["merge_rate"] = (sm_merge_attempts > 0)
    ? (double)sm_merge_accepts / (double)sm_merge_attempts : 0.0;
  sm["sigma_swap_attempts"] = (double) sigma_swap_attempts;
  sm["sigma_swap_accepts"]  = (double) sigma_swap_accepts;
  sm["sigma_swap_rate"]     = (sigma_swap_attempts > 0)
    ? (double)sigma_swap_accepts / (double)sigma_swap_attempts : 0.0;
  sm["alpha_epa_attempts"] = alpha_epa_attempts;
  sm["alpha_epa_accepts"]  = alpha_epa_accepts;
  sm["alpha_epa_rate"]     = (alpha_epa_attempts > 0)
    ? (double)alpha_epa_accepts / (double)alpha_epa_attempts : 0.0;
  sm["tau_epa_attempts"]   = tau_epa_attempts;
  sm["tau_epa_accepts"]    = tau_epa_accepts;
  sm["tau_epa_rate"]       = (tau_epa_attempts > 0)
    ? (double)tau_epa_accepts / (double)tau_epa_attempts : 0.0;
  out["epa_diagnostics"] = sm;

  // Acceptance (LSIRM only)
  List acc;
  acc["alpha1"] = acc_alpha1 / n_iter;
  acc["alpha2"] = acc_alpha2 / n_iter;
  acc["alpha3"] = acc_alpha3 / n_iter;
  acc["alpha4"] = acc_alpha4 / n_iter;
  acc["alpha5"] = acc_alpha5 / n_iter;
  acc["beta1"]  = acc_beta1  / n_iter;
  acc["beta2"]  = acc_beta2  / n_iter;
  acc["beta3"]  = acc_beta3  / n_iter;
  acc["log_gamma1"] = acc_log_gamma1 / n_iter;
  acc["log_gamma2"] = acc_log_gamma2 / n_iter;
  acc["log_gamma3"] = acc_log_gamma3 / n_iter;
  acc["log_gamma4"] = acc_log_gamma4 / n_iter;
  acc["log_gamma5"] = acc_log_gamma5 / n_iter;
  acc["log_kappa"]  = acc_log_kappa  / n_iter;
  acc["a"]  = acc_a  / n_iter;
  acc["b1"] = acc_b1 / n_iter;
  acc["b2"] = acc_b2 / n_iter;
  acc["b3"] = acc_b3 / n_iter;
  acc["b4"] = acc_b4 / n_iter;
  acc["b5"] = acc_b5 / n_iter;
  acc["beta4_thr"] = (P4 > 0 && K1m1 > 0) ? wrap(acc_beta4_thr / n_iter) : R_NilValue;
  acc["beta5_thr"] = (P5 > 0 && K2m1 > 0) ? wrap(acc_beta5_thr / n_iter) : R_NilValue;
  out["accept"] = acc;

  return out;
}
