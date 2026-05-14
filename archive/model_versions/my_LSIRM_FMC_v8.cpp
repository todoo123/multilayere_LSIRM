// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// =========================================================
// v8: Joint multilayered LSIRM (5-layer, per-item kappa)
//     + Factor-Mixture Clustering (FMC) in a single MCMC chain.
//
// Each iteration runs:
//   (A) v6 LSIRM updates (verbatim from
//       my_LSIRM_5layered_nonhierarchical_v6.cpp; identical to
//       what v7 inlines):
//         lambda2 -> alpha1..5 -> beta1..3 -> log_gamma1..5
//         -> a -> b1..b5 -> beta4_thr -> beta5_thr
//         -> log_kappa_j (per-item NB) -> Gibbs scalars
//   (B) Build x_{ij} = log max(||a_i - b_concat_j||, FMC_EPS)
//       from the current LSIRM positions
//       (b_concat = rbind(b1..b5)).
//   (C) Factor-mixture clustering Gibbs sweep on x_j:
//         rho -> eta_j -> c_j -> mu_l -> Sigma_l
//         -> delta_i, lambda_i, psi_i -> sigma_delta_sq
//
// IMPORTANT: Coupling is strict ONE-WAY: LSIRM -> FMC.
//   The LSIRM block (A) is byte-identical to v6; it does NOT
//   include the clustering-layer Gaussian factor in its MH
//   acceptance ratios (intentional deviation from
//   factor_mixture_clustering.tex Sec. "Modified LSIRM updates",
//   per user constraint).
//
// The post-hoc Procrustes alignment in the R wrapper is
// orthogonal+translation only, so all FMC parameters that depend
// on log-distances are invariant under that alignment.
// =========================================================

// ===== LSIRM helpers (same as v6 / v7) =====
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

// ===== FMC helpers =====
static const double FMC_EPS = 1e-8;

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

// Sample N_r(m, V) using lower Cholesky of V.
static inline vec mvnrnd_chol(const vec& m, const mat& V) {
  mat L;
  bool ok = arma::chol(L, V, "lower");
  if (!ok) {
    // numerical safety: jitter the diagonal
    mat Vj = V + 1e-8 * arma::eye(V.n_rows, V.n_cols);
    L = arma::chol(Vj, "lower");
  }
  vec z = arma::randn<vec>(m.n_elem);
  return m + L * z;
}

// Sample inverse-Wishart(nu, S). S is the scale matrix (positive-definite).
// Uses arma::iwishrnd; on failure falls back to the prior mean S/(nu - r - 1).
static inline mat sample_iw(double nu, const mat& S) {
  mat out;
  bool ok = arma::iwishrnd(out, S, nu);
  if (!ok || !out.is_finite()) {
    int r = (int)S.n_rows;
    double denom = nu - r - 1.0;
    if (denom <= 0.0) denom = 1.0;
    out = S / denom;
  }
  return out;
}

// log|M| via arma::log_det (works on general PD; portable across versions).
static inline double log_det_pd(const mat& M) {
  double val, sgn;
  arma::log_det(val, sgn, M);
  return val;
}

// Stack b1..b5 into b_all (P_total x d) using the canonical layer order:
//   bin -> con -> cnt -> ord1 -> ord2.
static inline void stack_b_layers(
    const mat& b1, const mat& b2, const mat& b3, const mat& b4, const mat& b5,
    int P1, int P2, int P3, int P4, int P5,
    mat& b_all
) {
  int off = 0;
  if (P1 > 0) { b_all.rows(off, off + P1 - 1) = b1; off += P1; }
  if (P2 > 0) { b_all.rows(off, off + P2 - 1) = b2; off += P2; }
  if (P3 > 0) { b_all.rows(off, off + P3 - 1) = b3; off += P3; }
  if (P4 > 0) { b_all.rows(off, off + P4 - 1) = b4; off += P4; }
  if (P5 > 0) { b_all.rows(off, off + P5 - 1) = b5; off += P5; }
}

// Build x (n x P_total) of log-distances, floored at FMC_EPS.
static inline void build_log_x_matrix(
    const mat& a, const mat& b_all, mat& x_out
) {
  int n = a.n_rows;
  int P_total = b_all.n_rows;
  vec a_sq2 = sum(square(a),     1);
  vec b_sq2 = sum(square(b_all), 1);
  mat cross_ab = a * b_all.t();
  mat sq = repmat(a_sq2, 1, P_total) + repmat(b_sq2.t(), n, 1) - 2.0 * cross_ab;
  sq.elem(find(sq < 0.0)).zeros();
  mat D = sqrt(sq);
  D.elem(find(D < FMC_EPS)).fill(FMC_EPS);
  x_out = arma::log(D);
}

// =========================================================
// Main MCMC: joint LSIRM + factor-mixture clustering
// =========================================================
// [[Rcpp::export]]
List run_lsirm_fmc_v8_cpp(
    mat Y_bin, mat Y_con, mat Y_cnt, IntegerMatrix Y_ord1, IntegerMatrix Y_ord2,
    int d, int n_iter, int burnin, int thin,
    int r_fac, int K_star,
    List lsirm_hyper, List fmc_hyper,
    List lsirm_prop_sd,
    List lsirm_init, List fmc_init,
    bool verbose, bool fix_gamma, double nu2,
    bool save_lambda_full,
    bool save_delta_full,
    bool save_psi_full,
    bool save_eta_full,
    bool compute_co_cluster_online,
    int  fmc_warmup
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

  // 0-based ordinal
  imat Y_ord1_0(n, P4);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < P4; ++j)
      Y_ord1_0(i, j) = Y_ord1(i, j) - 1;
  imat Y_ord2_0(n, P5);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < P5; ++j)
      Y_ord2_0(i, j) = Y_ord2(i, j) - 1;

  if (verbose) {
    Rcout << "[v8 joint] n=" << n
          << " P1=" << P1 << " P2=" << P2 << " P3=" << P3
          << " P4=" << P4 << " P5=" << P5
          << " P_total=" << P_total
          << " K1=" << K1 << " K2=" << K2 << " nu2=" << nu2
          << " r_fac=" << r_fac << " K_star=" << K_star << "\n";
  }

  // ===== LSIRM hyperparameters =====
  double a_sigma = lsirm_hyper["a_sigma"]; double b_sigma = lsirm_hyper["b_sigma"];
  double a_tau1  = lsirm_hyper["a_tau1"];  double b_tau1  = lsirm_hyper["b_tau1"];
  double a_tau2  = lsirm_hyper["a_tau2"];  double b_tau2  = lsirm_hyper["b_tau2"];
  double a_tau3  = lsirm_hyper["a_tau3"];  double b_tau3  = lsirm_hyper["b_tau3"];
  double a_sigma0= lsirm_hyper["a_sigma0"]; double b_sigma0= lsirm_hyper["b_sigma0"];
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

  // ===== FMC hyperparameters =====
  double e0            = fmc_hyper["e0"];
  vec    m0            = as<vec>(fmc_hyper["m0"]);
  mat    V0            = as<mat>(fmc_hyper["V0"]);
  double nu0           = fmc_hyper["nu0"];
  mat    S0            = as<mat>(fmc_hyper["S0"]);
  double tau_lambda_sq = fmc_hyper["tau_lambda_sq"];
  double a_psi         = fmc_hyper["a_psi"];
  double b_psi         = fmc_hyper["b_psi"];
  double a_delta       = fmc_hyper["a_delta"];
  double b_delta       = fmc_hyper["b_delta"];

  if ((int)m0.n_elem != r_fac)
    Rcpp::stop("fmc_hyper$m0 length must equal r_fac");
  if ((int)V0.n_rows != r_fac || (int)V0.n_cols != r_fac)
    Rcpp::stop("fmc_hyper$V0 must be r_fac x r_fac");
  if ((int)S0.n_rows != r_fac || (int)S0.n_cols != r_fac)
    Rcpp::stop("fmc_hyper$S0 must be r_fac x r_fac");

  mat V0_inv = arma::inv_sympd(V0);

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

  // ===== FMC init =====
  vec rho       = as<vec>(fmc_init["rho"]);                  // length K_star
  mat eta       = as<mat>(fmc_init["eta"]);                  // P_total x r_fac
  uvec c_label  = as<uvec>(fmc_init["c"]);                   // length P_total, 0-based
  mat mu_mat    = as<mat>(fmc_init["mu"]);                   // K_star x r_fac
  cube Sigma_arr= as<cube>(fmc_init["Sigma"]);               // r_fac x r_fac x K_star
  mat Lambda_load = as<mat>(fmc_init["Lambda"]);             // n x r_fac
  vec delta     = as<vec>(fmc_init["delta"]);                // length n
  vec psi       = as<vec>(fmc_init["psi"]);                  // length n
  double sigma_delta_sq = as<double>(fmc_init["sigma_delta_sq"]);

  if ((int)rho.n_elem != K_star) Rcpp::stop("fmc_init$rho length must equal K_star");
  if ((int)eta.n_rows != P_total || (int)eta.n_cols != r_fac)
    Rcpp::stop("fmc_init$eta must be P_total x r_fac");
  if ((int)c_label.n_elem != P_total) Rcpp::stop("fmc_init$c length must equal P_total");
  if ((int)mu_mat.n_rows != K_star || (int)mu_mat.n_cols != r_fac)
    Rcpp::stop("fmc_init$mu must be K_star x r_fac");
  if ((int)Sigma_arr.n_rows != r_fac || (int)Sigma_arr.n_cols != r_fac
      || (int)Sigma_arr.n_slices != K_star)
    Rcpp::stop("fmc_init$Sigma must be r_fac x r_fac x K_star");
  if ((int)Lambda_load.n_rows != n || (int)Lambda_load.n_cols != r_fac)
    Rcpp::stop("fmc_init$Lambda must be n x r_fac");
  if ((int)delta.n_elem != n) Rcpp::stop("fmc_init$delta length must equal n");
  if ((int)psi.n_elem   != n) Rcpp::stop("fmc_init$psi length must equal n");

  // ===== Storage: LSIRM (mirrors v7) =====
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

  // ===== Storage: FMC =====
  // Always-on full traces (small / cheap)
  mat  store_rho(n_save, K_star, fill::zeros);
  umat store_c  (n_save, P_total, fill::zeros);
  cube store_mu (K_star, r_fac, n_save, fill::zeros);
  cube store_Sigma(r_fac, r_fac, K_star * std::max(n_save, 1), fill::zeros);
  vec  store_sigma_delta_sq(n_save, fill::zeros);
  vec  store_K_plus(n_save, fill::zeros);

  // Optional full-eta trace
  cube store_eta;
  mat  eta_postmean(P_total, r_fac, fill::zeros);
  if (save_eta_full) store_eta.set_size(P_total, r_fac, n_save);

  // Optional full Lambda trace + always-on posterior mean
  cube store_Lambda;
  mat  Lambda_postmean(n, r_fac, fill::zeros);
  if (save_lambda_full) store_Lambda.set_size(n, r_fac, n_save);

  // Optional full delta trace + always-on posterior mean
  mat  store_delta;
  vec  delta_postmean(n, fill::zeros);
  if (save_delta_full) store_delta.set_size(n_save, n);

  // Optional full psi trace + always-on posterior mean
  mat  store_psi;
  vec  psi_postmean(n, fill::zeros);
  if (save_psi_full) store_psi.set_size(n_save, n);

  // Online co-cluster matrix (P x P, normalised at end)
  umat co_count;
  if (compute_co_cluster_online) co_count.set_size(P_total, P_total);
  if (compute_co_cluster_online) co_count.zeros();

  // ===== Acceptance counters (LSIRM only; FMC is fully Gibbs) =====
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
  mat b_all(P_total, d, fill::zeros);
  mat x_log(n, P_total, fill::zeros);

  // ===== MCMC LOOP =====
  for (int iter = 0; iter < n_iter; ++iter) {
    if (verbose && (iter + 1) % 500 == 0)
      Rcout << "[v8] Iter: " << iter + 1 << " / " << n_iter << "\n";

    // ===================== LSIRM block (verbatim from v6) =====================

    // 0. lambda2 (Gibbs)
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
    // 1c. alpha3 (Cnt — per-item size_j)
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
    // 1d. alpha4 (Ord1 - GRM)
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
    // 1e. alpha5 (Ord2 - GRM)
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
    // 2b. beta2 (robust)
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
    // 2c. beta3 (per-item size_j)
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
        double lp_old = R::dnorm(lg_old, mu_log_gamma1, sd_log_gamma1, 1);
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
        double lp_old = R::dnorm(lg_old, mu_log_gamma2, sd_log_gamma2, 1);
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
        double lp_old = R::dnorm(lg_old, mu_log_gamma3, sd_log_gamma3, 1);
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
        double lp_old = R::dnorm(lg_old, mu_log_gamma4, sd_log_gamma4, 1);
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
        double lp_old = R::dnorm(lg_old, mu_log_gamma5, sd_log_gamma5, 1);
        double lp_new = R::dnorm(lg_prop, mu_log_gamma5, sd_log_gamma5, 1);
        if (log(R::runif(0,1)) < (ll_new + lp_new - ll_old - lp_old)) {
          log_gamma5 = lg_prop; gamma_val5 = g_prop; acc_log_gamma5++;
        }
      }
    }

    // 4. shared a
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

    // 5a. b1
    for (int j = 0; j < P1; ++j) {
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
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) -
           (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b1.row(j) = b_prop; acc_b1(j)++;
      }
    }
    // 5b. b2 (robust)
    for (int j = 0; j < P2; ++j) {
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
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) -
           (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b2.row(j) = b_prop; acc_b2(j)++;
      }
    }
    // 5c. b3 (per-item size_j)
    for (int j = 0; j < P3; ++j) {
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
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) -
           (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b3.row(j) = b_prop; acc_b3(j)++;
      }
    }
    // 5d. b4 (GRM)
    for (int j = 0; j < P4; ++j) {
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
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) -
           (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b4.row(j) = b_prop; acc_b4(j)++;
      }
    }
    // 5e. b5 (GRM)
    for (int j = 0; j < P5; ++j) {
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
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) - 0.5*dot(b_prop,b_prop) -
           (ll_f(b_old) - 0.5*dot(b_old,b_old)))) {
        b5.row(j) = b_prop; acc_b5(j)++;
      }
    }

    // 6a. GRM thresholds L4
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

    // 7. log_kappa per-item (NB)
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
    tau_beta1_sq = 1.0;
    tau_beta2_sq = 1.0;
    tau_beta3_sq = 1.0;

    // ============================================================
    // (B) Build x_{ij} = log max(||a_i - b_concat_j||, FMC_EPS)
    //     and ROW-CENTRE: subtract respondent-mean from each row.
    //
    //     Rationale: without row-centering, delta_i absorbs the
    //     respondent-mean log-distance (sigma_delta_sq ~ var(x_log)),
    //     leaving Lambda*eta with essentially no signal -> FMC
    //     collapses to a single cluster. After row-centering,
    //     sum_j r_{ij} = 0 forces delta_i's posterior toward 0,
    //     so all variation is fit by Lambda*eta and clusters can
    //     capture the column-mean (between-item) structure.
    // ============================================================
    stack_b_layers(b1, b2, b3, b4, b5, P1, P2, P3, P4, P5, b_all);
    build_log_x_matrix(a, b_all, x_log);  // x_log: n x P_total
    {
      arma::vec row_mean = arma::mean(x_log, 1);   // n-vec
      x_log.each_col() -= row_mean;
    }

    // ============================================================
    // (C) Factor-mixture clustering Gibbs sweep
    //     order: rho -> eta_j -> c_j -> mu_l -> Sigma_l
    //          -> (delta_i, lambda_i, psi_i) -> sigma_delta_sq
    //
    //     If iter < fmc_warmup, the FMC block is skipped entirely so
    //     that LSIRM positions converge before clustering starts.
    //     During warm-up, FMC parameters keep their initial values
    //     and the row-centred x_log is computed (above) but unused.
    // ============================================================
    if (iter >= fmc_warmup) {

    // C1. rho ~ Dirichlet(e0 + n_l)
    {
      vec n_l(K_star, fill::zeros);
      for (int j = 0; j < P_total; ++j) n_l(c_label(j)) += 1.0;
      double tot = 0.0;
      for (int l = 0; l < K_star; ++l) {
        rho(l) = R::rgamma(e0 + n_l(l), 1.0);
        tot += rho(l);
      }
      if (tot > 0.0) rho /= tot;
      else rho.fill(1.0 / (double)K_star);
    }

    // Precompute Lambda^T diag(1/psi) Lambda  (r x r) and per-cluster Sigma^-1.
    vec inv_psi = 1.0 / psi;
    mat LtPL    = Lambda_load.t() * arma::diagmat(inv_psi) * Lambda_load;
    cube SigInv(r_fac, r_fac, K_star);
    mat  SigInvMu(r_fac, K_star);
    vec  log_det_S(K_star);
    for (int l = 0; l < K_star; ++l) {
      SigInv.slice(l) = arma::inv_sympd(Sigma_arr.slice(l));
      SigInvMu.col(l) = SigInv.slice(l) * mu_mat.row(l).t();
      log_det_S(l)    = log_det_pd(Sigma_arr.slice(l));
    }

    // C2. eta_j | rest ~ N_r(m_eta, V_eta)
    for (int j = 0; j < P_total; ++j) {
      uword l = c_label(j);
      mat Vinv = LtPL + SigInv.slice(l);
      mat V    = arma::inv_sympd(Vinv);
      vec wr   = (x_log.col(j) - delta) % inv_psi;          // n-vec
      vec rhs  = Lambda_load.t() * wr + SigInvMu.col(l);    // r-vec
      vec m_eta = V * rhs;
      eta.row(j) = mvnrnd_chol(m_eta, V).t();
    }

    // C3. c_j | rest ~ Categorical
    {
      vec log_rho_v = arma::log(rho + FMC_EPS);
      for (int j = 0; j < P_total; ++j) {
        vec lw(K_star);
        rowvec eta_j = eta.row(j);
        for (int l = 0; l < K_star; ++l) {
          rowvec diff = eta_j - mu_mat.row(l);
          double quad = arma::as_scalar(diff * SigInv.slice(l) * diff.t());
          lw(l) = log_rho_v(l) - 0.5 * log_det_S(l) - 0.5 * quad;
        }
        c_label(j) = (uword) sample_log_weights(lw);
      }
    }

    // C4 + C5. mu_l, Sigma_l | rest  (per-cluster Gibbs)
    {
      vec n_l(K_star, fill::zeros);
      for (int j = 0; j < P_total; ++j) n_l(c_label(j)) += 1.0;

      for (int l = 0; l < K_star; ++l) {
        uvec idx = arma::find(c_label == (uword)l);
        mat eta_l;
        if (idx.n_elem > 0) eta_l = eta.rows(idx);

        // mu_l update first (uses CURRENT Sigma_l)
        mat SigInv_l = arma::inv_sympd(Sigma_arr.slice(l));
        mat Vmu_inv  = V0_inv + n_l(l) * SigInv_l;
        mat Vmu      = arma::inv_sympd(Vmu_inv);
        vec sum_eta(r_fac, fill::zeros);
        if (idx.n_elem > 0) sum_eta = arma::sum(eta_l, 0).t();
        vec m_mu = Vmu * (V0_inv * m0 + SigInv_l * sum_eta);
        mu_mat.row(l) = mvnrnd_chol(m_mu, Vmu).t();

        // Sigma_l update (uses NEW mu_l)
        mat Scatter = S0;
        if (idx.n_elem > 0) {
          mat Cm = eta_l.each_row() - mu_mat.row(l);  // n_l x r
          Scatter += Cm.t() * Cm;
        }
        Sigma_arr.slice(l) = sample_iw(nu0 + n_l(l), Scatter);
      }
    }

    // C6/C7/C8. delta_i, lambda_i, psi_i  (per-respondent Gibbs scan)
    mat Sum_eta_etaT = eta.t() * eta;     // r x r, recomputed once
    for (int i = 0; i < n; ++i) {
      // C6. delta_i
      rowvec lam_i = Lambda_load.row(i);
      double s = 0.0;
      for (int j = 0; j < P_total; ++j)
        s += x_log(i, j) - arma::dot(lam_i, eta.row(j));
      double V_d_inv = (double)P_total / psi(i) + 1.0 / sigma_delta_sq;
      double V_d     = 1.0 / V_d_inv;
      double m_d     = V_d * (s / psi(i));
      delta(i) = R::rnorm(m_d, std::sqrt(V_d));

      // C7. lambda_i (uses NEW delta_i)
      vec rhs_lam(r_fac, fill::zeros);
      for (int j = 0; j < P_total; ++j)
        rhs_lam += eta.row(j).t() * (x_log(i, j) - delta(i));
      rhs_lam *= (1.0 / psi(i));
      mat Vlam_inv = (1.0 / tau_lambda_sq) * arma::eye(r_fac, r_fac)
                   + (1.0 / psi(i)) * Sum_eta_etaT;
      mat Vlam     = arma::inv_sympd(Vlam_inv);
      vec m_lam    = Vlam * rhs_lam;
      Lambda_load.row(i) = mvnrnd_chol(m_lam, Vlam).t();

      // C8. psi_i (uses NEW delta_i, lambda_i)
      lam_i = Lambda_load.row(i);
      double rss = 0.0;
      for (int j = 0; j < P_total; ++j) {
        double res = x_log(i, j) - delta(i) - arma::dot(lam_i, eta.row(j));
        rss += res * res;
      }
      double a_post = a_psi + 0.5 * (double)P_total;
      double b_post = b_psi + 0.5 * rss;
      psi(i) = 1.0 / R::rgamma(a_post, 1.0 / b_post);
    }

    // C9. sigma_delta_sq | delta
    {
      double a_post_sd = a_delta + 0.5 * (double)n;
      double b_post_sd = b_delta + 0.5 * arma::dot(delta, delta);
      sigma_delta_sq = 1.0 / R::rgamma(a_post_sd, 1.0 / b_post_sd);
    }

    } // end if (iter >= fmc_warmup)

    // ===================== Save =====================
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      // LSIRM
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

      store_lambda2_mean(save_idx) = accu(lambda2) / (n * P2);
      store_lambda2.slice(save_idx) = lambda2;
      sum_lambda2 += lambda2;

      // FMC
      store_rho.row(save_idx) = rho.t();
      for (int j = 0; j < P_total; ++j) store_c(save_idx, j) = c_label(j);
      store_mu.slice(save_idx) = mu_mat;
      for (int l = 0; l < K_star; ++l)
        store_Sigma.slice(K_star * save_idx + l) = Sigma_arr.slice(l);
      store_sigma_delta_sq(save_idx) = sigma_delta_sq;

      // K_+ (occupied cluster count)
      vec n_l_save(K_star, fill::zeros);
      for (int j = 0; j < P_total; ++j) n_l_save(c_label(j)) += 1.0;
      double Kp = 0.0;
      for (int l = 0; l < K_star; ++l) if (n_l_save(l) > 0.5) Kp += 1.0;
      store_K_plus(save_idx) = Kp;

      // Optional full traces / always-on running means
      if (save_eta_full) store_eta.slice(save_idx) = eta;
      eta_postmean += eta;

      if (save_lambda_full) store_Lambda.slice(save_idx) = Lambda_load;
      Lambda_postmean += Lambda_load;

      if (save_delta_full) store_delta.row(save_idx) = delta.t();
      delta_postmean += delta;

      if (save_psi_full) store_psi.row(save_idx) = psi.t();
      psi_postmean += psi;

      // Online co-cluster matrix (upper triangle + symmetry)
      if (compute_co_cluster_online) {
        for (int j = 0; j < P_total; ++j) {
          uword cj = c_label(j);
          co_count(j, j) += 1u;
          for (int jp = j + 1; jp < P_total; ++jp) {
            if (c_label(jp) == cj) {
              co_count(j, jp)  += 1u;
              co_count(jp, j)  += 1u;
            }
          }
        }
      }

      save_idx++;
    }
  }

  // Finalise running means
  if (n_save > 0) {
    eta_postmean    /= (double)n_save;
    Lambda_postmean /= (double)n_save;
    delta_postmean  /= (double)n_save;
    psi_postmean    /= (double)n_save;
  }

  // Co-cluster: convert counts -> probabilities in [0, 1]
  mat co_cluster_pp;
  if (compute_co_cluster_online) {
    co_cluster_pp.set_size(P_total, P_total);
    if (n_save > 0) co_cluster_pp = arma::conv_to<mat>::from(co_count) / (double)n_save;
    else co_cluster_pp.zeros();
  }

  // ===== Build return list =====
  // Use named-assignment style to avoid Rcpp::List::create arg-count limits.
  List out;

  // LSIRM
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

  // FMC
  out["fmc_rho"]              = store_rho;
  out["fmc_c"]                = store_c;          // 0-based; wrapper +1
  out["fmc_mu"]               = store_mu;
  out["fmc_Sigma"]            = store_Sigma;      // packed (r x r x K*n_save)
  out["fmc_sigma_delta_sq"]   = store_sigma_delta_sq;
  out["fmc_K_plus"]           = store_K_plus;
  out["fmc_eta_postmean"]     = eta_postmean;
  out["fmc_Lambda_postmean"]  = Lambda_postmean;
  out["fmc_delta_postmean"]   = delta_postmean;
  out["fmc_psi_postmean"]     = psi_postmean;
  out["fmc_eta"]              = save_eta_full    ? wrap(store_eta)    : R_NilValue;
  out["fmc_Lambda"]           = save_lambda_full ? wrap(store_Lambda) : R_NilValue;
  out["fmc_delta"]            = save_delta_full  ? wrap(store_delta)  : R_NilValue;
  out["fmc_psi"]              = save_psi_full    ? wrap(store_psi)    : R_NilValue;
  out["fmc_co_cluster"]       = compute_co_cluster_online ? wrap(co_cluster_pp) : R_NilValue;
  out["fmc_r_fac"]            = r_fac;
  out["fmc_K_star"]           = K_star;
  out["fmc_e0"]               = e0;

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
