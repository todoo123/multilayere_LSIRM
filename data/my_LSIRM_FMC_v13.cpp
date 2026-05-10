// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector>
#include <algorithm>

using namespace Rcpp;
using namespace arma;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Fisher-Yates shuffle using R's RNG (so MCMC is reproducible with set.seed).
static inline void rcpp_shuffle(std::vector<int>& v) {
  int n = (int) v.size();
  for (int i = n - 1; i > 0; --i) {
    int j = (int) std::floor(R::runif(0.0, (double)(i + 1)));
    if (j > i) j = i;
    std::swap(v[i], v[j]);
  }
}

// =========================================================
// v13: Joint multilayered LSIRM (5-layer, per-item kappa)
//      + HIERARCHICAL latent-position Gaussian mixture clustering:
//          z_q | r_q, Sigma_b           ~ N_d(r_q, Sigma_b)
//          r_q | c_q = l, mu_l, Sigma_l ~ N_d(mu_l, Sigma_l)
//      + Conjugate NIW prior on (mu_l, Sigma_l) in R^d (on r)
//      + IW (full) or IG (isotropic) prior on global anchor
//        covariance Sigma_b (sigma_b^2 I_d).
//      + Collapsed (NIW posterior-predictive) c_q Gibbs update ON r
//      + Jain-Neal restricted-Gibbs split-merge moves ON r
//
//   Key change vs v11:
//     The mixture is no longer placed directly on the LSIRM positions
//     z_q.  Instead, an auxiliary cluster-aligned position r_q is
//     introduced as an intermediate layer.  This DECOUPLES the role
//     of Sigma_l (cluster shape) from the LSIRM-side anchor noise
//     Sigma_b: tight clusters (small Sigma_l) no longer compress
//     within-cluster pairwise distances among the z_q.
//
//     Marginal: z_q | c_q = l ~ N_d(mu_l, Sigma_l + Sigma_b), so
//     cluster shape heterogeneity is preserved while Sigma_b acts as
//     a global pooled anchor noise informed by all P items.
//
//   b MH update prior term changes from v11's
//     -0.5 * (b - mu_{c_q})^T Sigma_{c_q}^{-1} (b - mu_{c_q})
//   to
//     -0.5 * (b - r_q)^T Sigma_b^{-1} (b - r_q).
//
//   The b_prior_inflation argument from v11 is REMOVED: Sigma_b
//   naturally plays the role of a data-driven anchor strength via
//   its IW (or IG-isotropic) prior, so sampler-level decoupling is
//   no longer needed.
//
//   The LSIRM block (alpha, beta, gamma, a, b-likelihood, beta4_thr,
//   beta5_thr, log_kappa, sigma0_sq, sigma_alpha_l_sq) is kept
//   BYTE-IDENTICAL to v10/v11 except for the b prior term above.
// =========================================================

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

// ===== FMC helpers =====
static const double FMC_EPS = 1e-8;

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

// Sample N_r(m, V) using lower Cholesky of V.
static inline vec mvnrnd_chol(const vec& m, const mat& V) {
  mat L;
  bool ok = arma::chol(L, V, "lower");
  if (!ok) {
    mat Vj = V + 1e-8 * arma::eye(V.n_rows, V.n_cols);
    L = arma::chol(Vj, "lower");
  }
  vec z = arma::randn<vec>(m.n_elem);
  return m + L * z;
}

// Sample inverse-Wishart(nu, S).
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

static inline double log_det_pd(const mat& M) {
  double val, sgn;
  arma::log_det(val, sgn, M);
  return val;
}

// Stack b1..b5 into z_all (canonical order: bin->con->cnt->ord1->ord2).
// z_all has P_total rows of dimension d. This is the global item position
// array used by the mixture clustering layer.
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
// NIW helpers (operating on z_q in R^d, not eta in R^r)
//
// Convention: prior is NIW(m0, kappa0, nu0, S0):
//   Sigma     ~ IW(nu0, S0)
//   mu | Sig  ~ N_d(m0, Sigma / kappa0)
//
// Posterior given z_{J} (size n_J):
//   kappa_n = kappa0 + n_J
//   nu_n    = nu0    + n_J
//   m_n     = (kappa0 m0 + n_J zbar) / kappa_n
//   S_n     = S0 + Q + (kappa0 n_J)/(kappa0 + n_J)
//             (zbar - m0)(zbar - m0)^T
// where Q = sum_{q in J}(z_q - zbar)(z_q - zbar)^T.
// =========================================================

// Multivariate gamma function: log Gamma_d(a)
static inline double log_mv_gamma(double a, int d) {
  double v = 0.25 * (double)(d * (d - 1)) * std::log(M_PI);
  for (int q = 1; q <= d; ++q) {
    v += std::lgamma(a + 0.5 * (1.0 - (double)q));
  }
  return v;
}

// log NIW marginal likelihood m(z_J) for a generic item set J (size n_J).
// If n_J = 0 returns 0 (just the prior, normalised away).
static inline double log_niw_marginal(
    int n_J, int d,
    double kappa0, double nu0, double log_det_S0, double log_mvg_nu0_half,
    double kappa_n, double nu_n, double log_det_S_n
) {
  if (n_J == 0) return 0.0;
  double v = 0.0;
  v += -0.5 * (double)(n_J * d) * std::log(M_PI);
  v += 0.5 * (double)d * (std::log(kappa0) - std::log(kappa_n));
  v += 0.5 * nu0 * log_det_S0;
  v += -0.5 * nu_n * log_det_S_n;
  v += log_mv_gamma(0.5 * nu_n, d);
  v += -log_mvg_nu0_half;
  return v;
}

// Compute NIW posterior parameters for a set of items (rows of z_subset).
static inline void niw_posterior(
    const mat& z_subset, int n_J,
    const vec& m0, double kappa0, double nu0, const mat& S0,
    double& kappa_n, double& nu_n, vec& m_n, mat& S_n
) {
  int d = (int)m0.n_elem;
  kappa_n = kappa0 + (double)n_J;
  nu_n    = nu0    + (double)n_J;
  if (n_J == 0) {
    m_n = m0;
    S_n = S0;
    return;
  }
  vec zbar = arma::sum(z_subset, 0).t() / (double)n_J;
  m_n = (kappa0 * m0 + (double)n_J * zbar) / kappa_n;
  mat Cm = z_subset.each_row() - zbar.t();
  mat Q  = Cm.t() * Cm;
  vec dvec  = zbar - m0;
  double w = (kappa0 * (double)n_J) / kappa_n;
  S_n = S0 + Q + w * (dvec * dvec.t());
  // Symmetrise for numerical safety
  S_n = 0.5 * (S_n + S_n.t());
  (void)d;
}

// Multivariate Student-t log density. x is d-vector, mu is d-vector,
// Sigma is the SCALE matrix (not covariance).
static inline double log_mvt_density(
    const vec& x, const vec& mu, const mat& Sigma, double nu
) {
  int d = (int)x.n_elem;
  vec dvec = x - mu;
  mat L;
  bool ok = arma::chol(L, Sigma, "lower");
  mat Sigma_use = Sigma;
  if (!ok) {
    Sigma_use = Sigma + 1e-8 * arma::eye(d, d);
    L = arma::chol(Sigma_use, "lower");
  }
  vec u = arma::solve(arma::trimatl(L), dvec);
  double quad = arma::dot(u, u);
  double log_det = 2.0 * arma::sum(arma::log(L.diag()));
  double v = std::lgamma(0.5 * (nu + (double)d))
           - std::lgamma(0.5 * nu)
           - 0.5 * (double)d * std::log(nu * M_PI)
           - 0.5 * log_det
           - 0.5 * (nu + (double)d) * std::log1p(quad / nu);
  return v;
}

static inline void niw_posterior_from_indices(
    const mat& z, const std::vector<int>& idx_in_cluster,
    const vec& m0, double kappa0, double nu0, const mat& S0,
    double& kappa_n, double& nu_n, vec& m_n, mat& S_n
) {
  int n_J = (int)idx_in_cluster.size();
  if (n_J == 0) {
    kappa_n = kappa0;
    nu_n    = nu0;
    m_n     = m0;
    S_n     = S0;
    return;
  }
  uvec uidx(n_J);
  for (int t = 0; t < n_J; ++t) uidx(t) = (uword) idx_in_cluster[t];
  mat z_subset = z.rows(uidx);
  niw_posterior(z_subset, n_J, m0, kappa0, nu0, S0, kappa_n, nu_n, m_n, S_n);
}

static inline double log_niw_marginal_from_post(
    int n_A, int d,
    double kappa0, double nu0, double log_det_S0, double log_mvg_nu0_half,
    double kappa_A, double nu_A, const mat& S_A
) {
  if (n_A == 0) return 0.0;
  double log_det_SA = log_det_pd(S_A);
  return log_niw_marginal(n_A, d, kappa0, nu0, log_det_S0, log_mvg_nu0_half,
                          kappa_A, nu_A, log_det_SA);
}

// =========================================================
// Main MCMC
// =========================================================
// [[Rcpp::export]]
List run_lsirm_fmc_v13_cpp(
    mat Y_bin, mat Y_con, mat Y_cnt, IntegerMatrix Y_ord1, IntegerMatrix Y_ord2,
    int d, int n_iter, int burnin, int thin,
    int K_star,
    List lsirm_hyper, List fmc_hyper,
    List lsirm_prop_sd,
    List lsirm_init, List fmc_init,
    bool verbose, bool fix_gamma, double nu2,
    bool compute_co_cluster_online,
    int  fmc_warmup,
    int  n_split_merge
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
    Rcout << "[v13 joint] n=" << n
          << " P1=" << P1 << " P2=" << P2 << " P3=" << P3
          << " P4=" << P4 << " P5=" << P5
          << " P_total=" << P_total
          << " K1=" << K1 << " K2=" << K2 << " nu2=" << nu2
          << " d=" << d << " K_star=" << K_star
          << " M_SM=" << n_split_merge << "\n";
  }

  // Per-layer offsets in the global item index q.
  int off_L1 = 0;
  int off_L2 = off_L1 + P1;
  int off_L3 = off_L2 + P2;
  int off_L4 = off_L3 + P3;
  int off_L5 = off_L4 + P4;

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

  // ===== FMC hyperparameters (NIW in R^d) =====
  double e0     = fmc_hyper["e0"];
  vec    m0     = as<vec>(fmc_hyper["m0"]);
  double kappa0 = fmc_hyper["kappa0"];
  double nu0    = fmc_hyper["nu0"];
  mat    S0     = as<mat>(fmc_hyper["S0"]);

  if ((int)m0.n_elem != d)
    Rcpp::stop("fmc_hyper$m0 length must equal d");
  if ((int)S0.n_rows != d || (int)S0.n_cols != d)
    Rcpp::stop("fmc_hyper$S0 must be d x d");
  if (kappa0 <= 0.0) Rcpp::stop("fmc_hyper$kappa0 must be > 0");
  if (nu0 <= (double)(d - 1)) Rcpp::stop("fmc_hyper$nu0 must be > d - 1");

  // ----- Wishart hyperprior on S0 (Option A: data-driven cluster scale) -----
  //
  //   S0 ~ W(nu_S0, Lambda_S0)
  //   Sigma_l | S0 ~ IW(nu0, S0),  l = 1..K_star
  //
  // Conjugate full conditional (after drawing all Sigma_l):
  //   S0 | {Sigma_l} ~ W(nu_S0 + K_star * nu0,
  //                      (Lambda_S0^{-1} + sum_l Sigma_l^{-1})^{-1}).
  //
  // The hyperprior is enabled iff fmc_hyper has a positive nu_S0; otherwise
  // S0 is held fixed at its initial value (back-compatible behaviour).
  double nu_S0 = 0.0;
  mat    Lambda_S0;
  mat    Lambda_S0_inv;
  bool   use_S0_hyperprior = false;
  if (fmc_hyper.containsElementNamed("nu_S0")) {
    nu_S0 = as<double>(fmc_hyper["nu_S0"]);
    if (nu_S0 > (double)(d - 1)) {
      use_S0_hyperprior = true;
      if (!fmc_hyper.containsElementNamed("Lambda_S0"))
        Rcpp::stop("fmc_hyper$nu_S0 supplied but Lambda_S0 missing");
      Lambda_S0 = as<mat>(fmc_hyper["Lambda_S0"]);
      if ((int)Lambda_S0.n_rows != d || (int)Lambda_S0.n_cols != d)
        Rcpp::stop("fmc_hyper$Lambda_S0 must be d x d");
      Lambda_S0_inv = arma::inv_sympd(Lambda_S0);
    }
  }
  if (verbose) {
    Rcout << "[v13 NIW] S0 hyperprior: "
          << (use_S0_hyperprior ? "ENABLED" : "DISABLED (S0 fixed)")
          << "\n";
  }

  // ----- Anchor covariance Sigma_b: hyperprior + form selector -----
  //
  // Two forms supported:
  //   isotropic:   Sigma_b = sigma_b^2 * I_d,  sigma_b^2 ~ IG(a_b, b_b)
  //   full IW:     Sigma_b ~ IW(nu_b, S_b)
  //
  // Default: isotropic with a_b=3, b_b=0.05 (E[sigma_b^2]=0.025) to match
  // the recommended MIDUS-aligned operating point in model_v13.tex.
  bool sigma_b_isotropic = true;
  bool sigma_b_fixed = false;     // if true, skip Sigma_b conjugate update
  double a_b = 3.0, b_b = 0.05;
  double nu_b = 0.0;
  mat    S_b;
  if (fmc_hyper.containsElementNamed("sigma_b_isotropic"))
    sigma_b_isotropic = as<bool>(fmc_hyper["sigma_b_isotropic"]);
  if (fmc_hyper.containsElementNamed("sigma_b_fixed"))
    sigma_b_fixed = as<bool>(fmc_hyper["sigma_b_fixed"]);
  if (sigma_b_isotropic) {
    if (fmc_hyper.containsElementNamed("a_b")) a_b = as<double>(fmc_hyper["a_b"]);
    if (fmc_hyper.containsElementNamed("b_b")) b_b = as<double>(fmc_hyper["b_b"]);
    if (a_b <= 0.0 || b_b <= 0.0) Rcpp::stop("a_b and b_b must be > 0");
  } else {
    if (!fmc_hyper.containsElementNamed("nu_b") ||
        !fmc_hyper.containsElementNamed("S_b"))
      Rcpp::stop("Full IW Sigma_b requested but nu_b or S_b missing");
    nu_b = as<double>(fmc_hyper["nu_b"]);
    S_b  = as<mat>(fmc_hyper["S_b"]);
    if ((int)S_b.n_rows != d || (int)S_b.n_cols != d)
      Rcpp::stop("fmc_hyper$S_b must be d x d");
    if (nu_b <= (double)(d - 1)) Rcpp::stop("fmc_hyper$nu_b must be > d - 1");
  }

  // Optional Wishart hyperprior on S_b (full IW form only).
  bool   use_S_b_hyperprior = false;
  double nu_S_b = 0.0;
  mat    Lambda_S_b, Lambda_S_b_inv;
  if (!sigma_b_isotropic && fmc_hyper.containsElementNamed("nu_S_b")) {
    nu_S_b = as<double>(fmc_hyper["nu_S_b"]);
    if (nu_S_b > (double)(d - 1)) {
      use_S_b_hyperprior = true;
      if (!fmc_hyper.containsElementNamed("Lambda_S_b"))
        Rcpp::stop("nu_S_b supplied but Lambda_S_b missing");
      Lambda_S_b = as<mat>(fmc_hyper["Lambda_S_b"]);
      if ((int)Lambda_S_b.n_rows != d || (int)Lambda_S_b.n_cols != d)
        Rcpp::stop("Lambda_S_b must be d x d");
      Lambda_S_b_inv = arma::inv_sympd(Lambda_S_b);
    }
  }
  if (verbose) {
    Rcout << "[v13 anchor Sigma_b] form: "
          << (sigma_b_isotropic ? "isotropic (sigma_b^2 * I_d)" : "full IW")
          << (sigma_b_fixed ? " [FIXED, no update]" : " [Gibbs update]")
          << ";  S_b hyperprior: "
          << (use_S_b_hyperprior ? "ENABLED" : "DISABLED")
          << "\n";
  }

  // Cached prior log-quantities (used in NIW marginal for split-merge).
  // log_det_S0 must be refreshed after every S0 update; log_mvg_nu0_half
  // depends only on (nu0, d) and is constant.
  double log_det_S0       = log_det_pd(S0);
  double log_mvg_nu0_half = log_mv_gamma(0.5 * nu0, d);

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

  // ===== FMC init (latent-position mixture) =====
  vec rho       = as<vec>(fmc_init["rho"]);
  uvec c_label  = as<uvec>(fmc_init["c"]);
  mat mu_mat    = as<mat>(fmc_init["mu"]);
  cube Sigma_arr= as<cube>(fmc_init["Sigma"]);

  if ((int)rho.n_elem != K_star) Rcpp::stop("fmc_init$rho length must equal K_star");
  if ((int)c_label.n_elem != P_total) Rcpp::stop("fmc_init$c length must equal P_total");
  if ((int)mu_mat.n_rows != K_star || (int)mu_mat.n_cols != d)
    Rcpp::stop("fmc_init$mu must be K_star x d");
  if ((int)Sigma_arr.n_rows != d || (int)Sigma_arr.n_cols != d
      || (int)Sigma_arr.n_slices != K_star)
    Rcpp::stop("fmc_init$Sigma must be d x d x K_star");

  // ----- Auxiliary positions r and anchor covariance Sigma_b init -----
  //
  // r init: defaults to the stacked initial b positions (z = r at start),
  //   so that the first sweep does not drag r away from its data-anchor
  //   before the FMC has had a chance to learn cluster structure.
  // Sigma_b init: defaults to small isotropic (0.05 * I_d) per the
  //   recommended operating point.
  mat r_all(P_total, d, fill::zeros);
  if (fmc_init.containsElementNamed("r")) {
    r_all = as<mat>(fmc_init["r"]);
    if ((int)r_all.n_rows != P_total || (int)r_all.n_cols != d)
      Rcpp::stop("fmc_init$r must be P_total x d");
  } else {
    int off = 0;
    if (P1 > 0) { r_all.rows(off, off + P1 - 1) = b1; off += P1; }
    if (P2 > 0) { r_all.rows(off, off + P2 - 1) = b2; off += P2; }
    if (P3 > 0) { r_all.rows(off, off + P3 - 1) = b3; off += P3; }
    if (P4 > 0) { r_all.rows(off, off + P4 - 1) = b4; off += P4; }
    if (P5 > 0) { r_all.rows(off, off + P5 - 1) = b5; off += P5; }
  }

  double sigma_b_sq = 0.05;
  mat    Sigma_b(d, d, fill::eye);
  Sigma_b *= sigma_b_sq;
  if (sigma_b_isotropic) {
    if (fmc_init.containsElementNamed("sigma_b_sq"))
      sigma_b_sq = as<double>(fmc_init["sigma_b_sq"]);
    if (sigma_b_sq <= 0.0) Rcpp::stop("fmc_init$sigma_b_sq must be > 0");
    Sigma_b = sigma_b_sq * arma::eye(d, d);
  } else {
    if (fmc_init.containsElementNamed("Sigma_b")) {
      Sigma_b = as<mat>(fmc_init["Sigma_b"]);
      if ((int)Sigma_b.n_rows != d || (int)Sigma_b.n_cols != d)
        Rcpp::stop("fmc_init$Sigma_b must be d x d");
    } else {
      Sigma_b = 0.05 * arma::eye(d, d);
    }
  }
  mat Sigma_b_inv = arma::inv_sympd(Sigma_b);
  double log_det_Sigma_b = log_det_pd(Sigma_b);

  // ===== Storage =====
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

  // FMC storage (hierarchical latent-position mixture: r, Sigma_b added)
  mat  store_rho(n_save, K_star, fill::zeros);
  umat store_c  (n_save, P_total, fill::zeros);
  cube store_mu (K_star, d, n_save, fill::zeros);
  cube store_Sigma(d, d, K_star * std::max(n_save, 1), fill::zeros);
  vec  store_K_plus(n_save, fill::zeros);
  cube store_S0;
  if (use_S0_hyperprior) store_S0.set_size(d, d, n_save);

  // v13-only: r (P_total x d) and Sigma_b (d x d) or sigma_b^2 (scalar)
  cube store_r(P_total, d, n_save, fill::zeros);
  cube store_Sigma_b;
  vec  store_sigma_b_sq;
  if (sigma_b_isotropic) store_sigma_b_sq.set_size(n_save);
  else                   store_Sigma_b.set_size(d, d, n_save);

  umat co_count;
  if (compute_co_cluster_online) {
    co_count.set_size(P_total, P_total);
    co_count.zeros();
  }

  // Split-merge counters (post-warmup only)
  int sm_split_attempts = 0;
  int sm_split_accepts  = 0;
  int sm_merge_attempts = 0;
  int sm_merge_accepts  = 0;

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

  // Reusable workspace: stacked global item positions z = (b1; b2; ...; b5).
  mat z_all(P_total, d, fill::zeros);

  // Scratch for split-merge
  std::vector< std::vector<int> > clusters_idx(K_star);

  // Cached cluster precisions (for the b-update mixture-prior term).
  // Refreshed after every NIW posterior draw of (mu_l, Sigma_l).
  cube SigInv(d, d, K_star);
  for (int l = 0; l < K_star; ++l) {
    SigInv.slice(l) = arma::inv_sympd(Sigma_arr.slice(l));
  }

  // v13: anchor-prior quadratic form (b - r_q)^T Sigma_b^{-1} (b - r_q).
  // Uses GLOBAL Sigma_b (cached as Sigma_b_inv) and ITEM-SPECIFIC r_q.
  // SigInv (per-cluster) remains used by the r_q conjugate update only.
  auto anchor_quad = [&](const rowvec& b_val, int q) {
    rowvec dv = b_val - r_all.row(q);
    return arma::as_scalar(dv * Sigma_b_inv * dv.t());
  };

  // ===== MCMC LOOP =====
  for (int iter = 0; iter < n_iter; ++iter) {
    if (verbose && (iter + 1) % 500 == 0)
      Rcout << "[v13] Iter: " << iter + 1 << " / " << n_iter << "\n";

    // ===================== LSIRM block (verbatim from v10) =====================
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

    // 5a. b1  (prior is mixture: log N_d(b; mu_{c_q}, Sigma_{c_q}))
    for (int j = 0; j < P1; ++j) {
      int q  = off_L1 + j;
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
      double q_old = anchor_quad(b_old,  q);
      double q_new = anchor_quad(b_prop, q);
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) - 0.5 * q_new -
           (ll_f(b_old) - 0.5 * q_old))) {
        b1.row(j) = b_prop; acc_b1(j)++;
      }
    }
    // 5b. b2
    for (int j = 0; j < P2; ++j) {
      int q  = off_L2 + j;
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
      double q_old = anchor_quad(b_old,  q);
      double q_new = anchor_quad(b_prop, q);
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) - 0.5 * q_new -
           (ll_f(b_old) - 0.5 * q_old))) {
        b2.row(j) = b_prop; acc_b2(j)++;
      }
    }
    // 5c. b3
    for (int j = 0; j < P3; ++j) {
      int q  = off_L3 + j;
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
      double q_old = anchor_quad(b_old,  q);
      double q_new = anchor_quad(b_prop, q);
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) - 0.5 * q_new -
           (ll_f(b_old) - 0.5 * q_old))) {
        b3.row(j) = b_prop; acc_b3(j)++;
      }
    }
    // 5d. b4
    for (int j = 0; j < P4; ++j) {
      int q  = off_L4 + j;
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
      double q_old = anchor_quad(b_old,  q);
      double q_new = anchor_quad(b_prop, q);
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) - 0.5 * q_new -
           (ll_f(b_old) - 0.5 * q_old))) {
        b4.row(j) = b_prop; acc_b4(j)++;
      }
    }
    // 5e. b5
    for (int j = 0; j < P5; ++j) {
      int q  = off_L5 + j;
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
      double q_old = anchor_quad(b_old,  q);
      double q_new = anchor_quad(b_prop, q);
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) - 0.5 * q_new -
           (ll_f(b_old) - 0.5 * q_old))) {
        b5.row(j) = b_prop; acc_b5(j)++;
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
    // (B) Build z_q := b_{j(q)}^{(l(q))}, the global LSIRM item
    //     position array.  In v13 the clustering layer operates on
    //     the AUXILIARY positions r_q, not on z_q directly; z_q
    //     enters only through the Sigma_b conjugate update and
    //     through the r_q Gaussian draw.
    // ============================================================
    stack_b_layers(b1, b2, b3, b4, b5, P1, P2, P3, P4, P5, z_all);

    // ============================================================
    // (B') r_q | z_q, c_q, mu_{c_q}, Sigma_{c_q}, Sigma_b -- Gaussian
    //      conjugate draw.  Per-item independent, all P updates can
    //      run in any order.  Uses CURRENT (pre-update) c_q, mu, Sigma.
    //
    //   Lambda_q = Sigma_b^{-1} + Sigma_{c_q}^{-1}
    //   r_hat_q  = Lambda_q^{-1} (Sigma_b^{-1} z_q + Sigma_{c_q}^{-1} mu_{c_q})
    //   r_q ~ N(r_hat_q, Lambda_q^{-1})
    //
    // Numerical: Cholesky-solve to avoid explicit Lambda^{-1}.
    // ============================================================
    if (iter >= fmc_warmup) {
      for (int q = 0; q < P_total; ++q) {
        int cl = (int) c_label(q);
        mat Lambda_q = Sigma_b_inv + SigInv.slice(cl);
        Lambda_q = 0.5 * (Lambda_q + Lambda_q.t());
        mat L_q;
        bool ok_chol = arma::chol(L_q, Lambda_q, "lower");
        if (!ok_chol) {
          mat Lj = Lambda_q + 1e-8 * arma::eye(d, d);
          L_q = arma::chol(Lj, "lower");
        }
        vec rhs = Sigma_b_inv * z_all.row(q).t()
                + SigInv.slice(cl) * mu_mat.row(cl).t();
        // Solve Lambda_q * r_mean = rhs via L L^T r_mean = rhs
        vec y       = arma::solve(arma::trimatl(L_q),    rhs);
        vec r_mean  = arma::solve(arma::trimatu(L_q.t()), y);
        // Sample r ~ N(r_mean, Lambda_q^{-1}) via r = r_mean + L^{-T} z
        vec z_rand  = arma::randn<vec>(d);
        vec eta     = arma::solve(arma::trimatu(L_q.t()), z_rand);
        r_all.row(q) = (r_mean + eta).t();
      }
    }

    // ============================================================
    // (C) Latent-position mixture clustering Gibbs sweep (V13)
    //
    //     Order:
    //       c_q (NIW collapsed on r) -> split-merge (on r)
    //       -> mu_l, Sigma_l (NIW posterior on r)
    //       -> S_0 (Wishart hyperprior, unchanged)
    //       -> Sigma_b (IW or sigma_b^2 IG conjugate from z - r)
    //       -> rho (storage only)
    // ============================================================
    if (iter >= fmc_warmup) {

    // ----------------- Build cluster index lists -----------------
    for (int l = 0; l < K_star; ++l) clusters_idx[l].clear();
    for (int q = 0; q < P_total; ++q)
      clusters_idx[c_label(q)].push_back(q);

    // C1. c_q | rest  -- NIW-collapsed Gibbs (single-site)
    //
    //   Pr(c_q = l | c_-q, z) propto
    //      (n_{l,-q} + e0) * t_{nu_n - d + 1}(z_q; m_n, V_pred_n)
    //   where (m_n, kappa_n, nu_n, S_n) are NIW posteriors using
    //   z items currently in cluster l excluding q.
    {
      vec m_n(d);
      mat S_n(d, d);
      double kappa_n, nu_n;

      for (int q = 0; q < P_total; ++q) {
        uword cq = c_label(q);

        auto& vec_l = clusters_idx[cq];
        vec_l.erase(std::remove(vec_l.begin(), vec_l.end(), q), vec_l.end());

        vec r_q = r_all.row(q).t();
        vec lw(K_star);

        for (int l = 0; l < K_star; ++l) {
          niw_posterior_from_indices(r_all, clusters_idx[l],
                                     m0, kappa0, nu0, S0,
                                     kappa_n, nu_n, m_n, S_n);
          int n_lmq = (int)clusters_idx[l].size();
          double dof = nu_n - (double)d + 1.0;
          mat V_pred = ((kappa_n + 1.0) / (kappa_n * dof)) * S_n;
          double log_w = std::log((double)n_lmq + e0)
                       + log_mvt_density(r_q, m_n, V_pred, dof);
          lw(l) = log_w;
        }
        int new_cl = sample_log_weights(lw);

        c_label(q) = (uword) new_cl;
        clusters_idx[new_cl].push_back(q);
      }
    }

    // C2. Split-merge moves on c (Jain & Neal 2004, restricted Gibbs)
    if (n_split_merge > 0) {
      vec m_n(d), m_a(d), m_b(d);
      mat S_n(d, d), S_a(d, d), S_b(d, d);
      double kappa_n, nu_n, kappa_a, nu_a, kappa_b, nu_b;

      for (int sm = 0; sm < n_split_merge; ++sm) {
        int u = (int) std::floor(R::runif(0.0, (double)P_total));
        if (u >= P_total) u = P_total - 1;
        int v = u;
        while (v == u) {
          v = (int) std::floor(R::runif(0.0, (double)P_total));
          if (v >= P_total) v = P_total - 1;
        }

        if (c_label(u) == c_label(v)) {
          // ---- Split move ----
          sm_split_attempts++;
          int aL = (int) c_label(u);

          int bL = -1;
          int E_c = 0;
          for (int l = 0; l < K_star; ++l) {
            if (clusters_idx[l].empty()) { E_c++; }
          }
          if (E_c == 0) continue;
          int pick = (int) std::floor(R::runif(0.0, (double)E_c));
          if (pick >= E_c) pick = E_c - 1;
          int seen = 0;
          for (int l = 0; l < K_star; ++l) {
            if (clusters_idx[l].empty()) {
              if (seen == pick) { bL = l; break; }
              seen++;
            }
          }
          if (bL < 0) continue;

          std::vector<int> A_idx = clusters_idx[aL];
          std::vector<int> tmp_a, tmp_b;
          tmp_a.push_back(u);
          tmp_b.push_back(v);
          std::vector<int> scan_order;
          for (int idx : A_idx) if (idx != u && idx != v) scan_order.push_back(idx);
          rcpp_shuffle(scan_order);

          double log_q_fwd = -std::log((double)E_c);
          for (int q : scan_order) {
            niw_posterior_from_indices(r_all, tmp_a,
                                       m0, kappa0, nu0, S0,
                                       kappa_a, nu_a, m_a, S_a);
            niw_posterior_from_indices(r_all, tmp_b,
                                       m0, kappa0, nu0, S0,
                                       kappa_b, nu_b, m_b, S_b);
            int n_a = (int) tmp_a.size();
            int n_b = (int) tmp_b.size();
            double dof_a = nu_a - (double)d + 1.0;
            double dof_b = nu_b - (double)d + 1.0;
            mat V_a = ((kappa_a + 1.0) / (kappa_a * dof_a)) * S_a;
            mat V_b = ((kappa_b + 1.0) / (kappa_b * dof_b)) * S_b;
            vec r_q = r_all.row(q).t();
            double lw_a = std::log((double)n_a + e0)
                        + log_mvt_density(r_q, m_a, V_a, dof_a);
            double lw_b = std::log((double)n_b + e0)
                        + log_mvt_density(r_q, m_b, V_b, dof_b);
            double M = std::max(lw_a, lw_b);
            double pa = std::exp(lw_a - M);
            double pb = std::exp(lw_b - M);
            double s = pa + pb; pa /= s; pb /= s;
            double u_rand = R::runif(0.0, 1.0);
            if (u_rand < pa) {
              tmp_a.push_back(q);
              log_q_fwd += std::log(pa);
            } else {
              tmp_b.push_back(q);
              log_q_fwd += std::log(pb);
            }
          }

          double log_q_rev = 0.0;

          int n_A = (int) A_idx.size();
          int n_a_new = (int) tmp_a.size();
          int n_b_new = (int) tmp_b.size();

          niw_posterior_from_indices(r_all, A_idx, m0, kappa0, nu0, S0,
                                     kappa_n, nu_n, m_n, S_n);
          double log_m_A = log_niw_marginal_from_post(n_A, d, kappa0, nu0,
                                                     log_det_S0, log_mvg_nu0_half,
                                                     kappa_n, nu_n, S_n);
          double log_pi_old = std::lgamma((double)n_A + e0) + log_m_A;

          double log_pi_new = 0.0;
          {
            niw_posterior_from_indices(r_all, tmp_a, m0, kappa0, nu0, S0,
                                       kappa_a, nu_a, m_a, S_a);
            niw_posterior_from_indices(r_all, tmp_b, m0, kappa0, nu0, S0,
                                       kappa_b, nu_b, m_b, S_b);
            double log_m_a = log_niw_marginal_from_post(n_a_new, d, kappa0, nu0,
                                                       log_det_S0, log_mvg_nu0_half,
                                                       kappa_a, nu_a, S_a);
            double log_m_b = log_niw_marginal_from_post(n_b_new, d, kappa0, nu0,
                                                       log_det_S0, log_mvg_nu0_half,
                                                       kappa_b, nu_b, S_b);
            log_pi_new = std::lgamma((double)n_a_new + e0)
                       + std::lgamma((double)n_b_new + e0)
                       - std::lgamma(e0)
                       + log_m_a + log_m_b;
          }

          double log_alpha = log_pi_new - log_pi_old + log_q_rev - log_q_fwd;
          if (std::log(R::runif(0.0, 1.0)) < log_alpha) {
            for (int q : tmp_a) c_label(q) = (uword) aL;
            for (int q : tmp_b) c_label(q) = (uword) bL;
            clusters_idx[aL] = tmp_a;
            clusters_idx[bL] = tmp_b;
            sm_split_accepts++;
          }

        } else {
          // ---- Merge move ----
          sm_merge_attempts++;
          int aL = (int) c_label(u);
          int bL = (int) c_label(v);

          std::vector<int> A_idx = clusters_idx[aL];
          std::vector<int> B_idx = clusters_idx[bL];

          double log_q_fwd = 0.0;

          int E_c_after = 0;
          for (int l = 0; l < K_star; ++l) {
            if (clusters_idx[l].empty()) E_c_after++;
          }
          int E_cprime = E_c_after + 1;

          std::vector<int> scan_order;
          for (int idx : A_idx) if (idx != u) scan_order.push_back(idx);
          for (int idx : B_idx) if (idx != v) scan_order.push_back(idx);
          rcpp_shuffle(scan_order);

          std::vector<int> rev_a, rev_b;
          rev_a.push_back(u);
          rev_b.push_back(v);
          double log_q_rev = -std::log((double)E_cprime);

          for (int q : scan_order) {
            bool was_in_a = (std::find(A_idx.begin(), A_idx.end(), q) != A_idx.end());

            niw_posterior_from_indices(r_all, rev_a,
                                       m0, kappa0, nu0, S0,
                                       kappa_a, nu_a, m_a, S_a);
            niw_posterior_from_indices(r_all, rev_b,
                                       m0, kappa0, nu0, S0,
                                       kappa_b, nu_b, m_b, S_b);
            int n_a = (int) rev_a.size();
            int n_b = (int) rev_b.size();
            double dof_a = nu_a - (double)d + 1.0;
            double dof_b = nu_b - (double)d + 1.0;
            mat V_a = ((kappa_a + 1.0) / (kappa_a * dof_a)) * S_a;
            mat V_b = ((kappa_b + 1.0) / (kappa_b * dof_b)) * S_b;
            vec r_q = r_all.row(q).t();
            double lw_a = std::log((double)n_a + e0)
                        + log_mvt_density(r_q, m_a, V_a, dof_a);
            double lw_b = std::log((double)n_b + e0)
                        + log_mvt_density(r_q, m_b, V_b, dof_b);
            double M = std::max(lw_a, lw_b);
            double pa = std::exp(lw_a - M);
            double pb = std::exp(lw_b - M);
            double s = pa + pb; pa /= s; pb /= s;
            if (was_in_a) {
              rev_a.push_back(q);
              log_q_rev += std::log(pa);
            } else {
              rev_b.push_back(q);
              log_q_rev += std::log(pb);
            }
          }

          int n_A = (int) A_idx.size();
          int n_B = (int) B_idx.size();
          int n_AB = n_A + n_B;

          double log_pi_old = 0.0;
          {
            niw_posterior_from_indices(r_all, A_idx, m0, kappa0, nu0, S0,
                                       kappa_a, nu_a, m_a, S_a);
            niw_posterior_from_indices(r_all, B_idx, m0, kappa0, nu0, S0,
                                       kappa_b, nu_b, m_b, S_b);
            double log_m_A = log_niw_marginal_from_post(n_A, d, kappa0, nu0,
                                                       log_det_S0, log_mvg_nu0_half,
                                                       kappa_a, nu_a, S_a);
            double log_m_B = log_niw_marginal_from_post(n_B, d, kappa0, nu0,
                                                       log_det_S0, log_mvg_nu0_half,
                                                       kappa_b, nu_b, S_b);
            log_pi_old = std::lgamma((double)n_A + e0)
                       + std::lgamma((double)n_B + e0)
                       - std::lgamma(e0)
                       + log_m_A + log_m_B;
          }

          double log_pi_new = 0.0;
          {
            std::vector<int> AB_idx = A_idx;
            AB_idx.insert(AB_idx.end(), B_idx.begin(), B_idx.end());
            niw_posterior_from_indices(r_all, AB_idx, m0, kappa0, nu0, S0,
                                       kappa_n, nu_n, m_n, S_n);
            double log_m_AB = log_niw_marginal_from_post(n_AB, d, kappa0, nu0,
                                                        log_det_S0, log_mvg_nu0_half,
                                                        kappa_n, nu_n, S_n);
            log_pi_new = std::lgamma((double)n_AB + e0) + log_m_AB;
          }

          double log_alpha = log_pi_new - log_pi_old + log_q_rev - log_q_fwd;
          if (std::log(R::runif(0.0, 1.0)) < log_alpha) {
            for (int q : B_idx) c_label(q) = (uword) aL;
            std::vector<int> AB = A_idx;
            AB.insert(AB.end(), B_idx.begin(), B_idx.end());
            clusters_idx[aL] = AB;
            clusters_idx[bL].clear();
            sm_merge_accepts++;
          }
        }
      }
    }

    // C3. (mu_l, Sigma_l) | rest  -- NIW posterior draw
    {
      vec m_n(d);
      mat S_n(d, d);
      double kappa_n, nu_n;

      for (int l = 0; l < K_star; ++l) {
        niw_posterior_from_indices(r_all, clusters_idx[l],
                                   m0, kappa0, nu0, S0,
                                   kappa_n, nu_n, m_n, S_n);
        Sigma_arr.slice(l) = sample_iw(nu_n, S_n);
        mat Vmu = Sigma_arr.slice(l) / kappa_n;
        Vmu = 0.5 * (Vmu + Vmu.t());
        mu_mat.row(l) = mvnrnd_chol(m_n, Vmu).t();
        // Refresh cached precision so next iter's b updates use fresh values.
        SigInv.slice(l) = arma::inv_sympd(Sigma_arr.slice(l));
      }
    }

    // C3b. S0 | {Sigma_l}  -- Wishart hyperprior conjugate Gibbs (Option A)
    //   nu_post = nu_S0 + K_star * nu0
    //   V_post  = (Lambda_S0^{-1} + sum_l Sigma_l^{-1})^{-1}
    //   S0' ~ W(nu_post, V_post),  E[S0'] = nu_post * V_post
    if (use_S0_hyperprior) {
      mat sum_Sinv = Lambda_S0_inv;
      for (int l = 0; l < K_star; ++l) sum_Sinv += SigInv.slice(l);
      mat V_post = arma::inv_sympd(sum_Sinv);
      V_post = 0.5 * (V_post + V_post.t());
      double nu_post = nu_S0 + (double)K_star * nu0;
      mat S0_new;
      bool ok = arma::wishrnd(S0_new, V_post, nu_post);
      if (ok && S0_new.is_finite()) {
        S0 = 0.5 * (S0_new + S0_new.t());
        log_det_S0 = log_det_pd(S0);
      }
      // else: keep previous S0 (Wishart sample failure is rare; fall through).
    }

    // C3c. Sigma_b | z, r  -- v13 anchor covariance conjugate update
    //
    //   Pool z_q - r_q deviations across all P items:
    //     W_b = sum_q (z_q - r_q)(z_q - r_q)^T
    //
    //   Isotropic form:
    //     ssq = sum_q ||z_q - r_q||^2
    //     sigma_b^2 ~ IG(a_b + P*d/2, b_b + ssq/2)
    //
    //   Full IW form:
    //     Sigma_b ~ IW(nu_b + P, S_b + W_b)
    //
    //   The cached Sigma_b_inv and log_det_Sigma_b are refreshed for use
    //   in the next sweep's b MH ratio and r_q Gibbs draws.
    //
    //   If sigma_b_fixed is true, this entire block is skipped: Sigma_b
    //   stays at its initial value, treating sigma_b as a fixed
    //   regularization hyperparameter rather than a data-fit quantity.
    if (!sigma_b_fixed) {
      if (sigma_b_isotropic) {
        double ssq = 0.0;
        for (int q = 0; q < P_total; ++q) {
          rowvec dv = z_all.row(q) - r_all.row(q);
          ssq += arma::dot(dv, dv);
        }
        double shape_p = a_b + 0.5 * (double)(P_total * d);
        double rate_p  = b_b + 0.5 * ssq;
        sigma_b_sq = 1.0 / R::rgamma(shape_p, 1.0 / rate_p);
        Sigma_b = sigma_b_sq * arma::eye(d, d);
        Sigma_b_inv = (1.0 / sigma_b_sq) * arma::eye(d, d);
        log_det_Sigma_b = (double)d * std::log(sigma_b_sq);
      } else {
        mat W_b(d, d, fill::zeros);
        for (int q = 0; q < P_total; ++q) {
          rowvec dv = z_all.row(q) - r_all.row(q);
          W_b += dv.t() * dv;
        }
        W_b = 0.5 * (W_b + W_b.t());
        Sigma_b = sample_iw(nu_b + (double)P_total, S_b + W_b);
        Sigma_b = 0.5 * (Sigma_b + Sigma_b.t());
        Sigma_b_inv = arma::inv_sympd(Sigma_b);
        log_det_Sigma_b = log_det_pd(Sigma_b);

        // Optional: S_b Wishart hyperprior conjugate Gibbs
        //   S_b | Sigma_b ~ W(nu_S_b + nu_b, (Lambda_S_b^{-1} + Sigma_b^{-1})^{-1})
        if (use_S_b_hyperprior) {
          mat V_post_Sb = arma::inv_sympd(Lambda_S_b_inv + Sigma_b_inv);
          V_post_Sb = 0.5 * (V_post_Sb + V_post_Sb.t());
          double nu_post_Sb = nu_S_b + nu_b;
          mat S_b_new;
          bool ok_Sb = arma::wishrnd(S_b_new, V_post_Sb, nu_post_Sb);
          if (ok_Sb && S_b_new.is_finite()) {
            S_b = 0.5 * (S_b_new + S_b_new.t());
          }
        }
      }
    }

    // C4. rho ~ Dirichlet(e0 + n_l)  -- storage / monitoring only
    {
      vec n_l(K_star, fill::zeros);
      for (int q = 0; q < P_total; ++q) n_l(c_label(q)) += 1.0;
      double tot = 0.0;
      for (int l = 0; l < K_star; ++l) {
        rho(l) = R::rgamma(e0 + n_l(l), 1.0);
        tot += rho(l);
      }
      if (tot > 0.0) rho /= tot;
      else rho.fill(1.0 / (double)K_star);
    }

    } // end if (iter >= fmc_warmup)

    // ===================== Save =====================
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

      store_rho.row(save_idx) = rho.t();
      for (int q = 0; q < P_total; ++q) store_c(save_idx, q) = c_label(q);
      store_mu.slice(save_idx) = mu_mat;
      for (int l = 0; l < K_star; ++l)
        store_Sigma.slice(K_star * save_idx + l) = Sigma_arr.slice(l);

      vec n_l_save(K_star, fill::zeros);
      for (int q = 0; q < P_total; ++q) n_l_save(c_label(q)) += 1.0;
      double Kp = 0.0;
      for (int l = 0; l < K_star; ++l) if (n_l_save(l) > 0.5) Kp += 1.0;
      store_K_plus(save_idx) = Kp;
      if (use_S0_hyperprior) store_S0.slice(save_idx) = S0;

      // v13: store r and Sigma_b (or sigma_b_sq)
      store_r.slice(save_idx) = r_all;
      if (sigma_b_isotropic) store_sigma_b_sq(save_idx) = sigma_b_sq;
      else                   store_Sigma_b.slice(save_idx) = Sigma_b;

      if (compute_co_cluster_online) {
        for (int q = 0; q < P_total; ++q) {
          uword cq = c_label(q);
          co_count(q, q) += 1u;
          for (int qp = q + 1; qp < P_total; ++qp) {
            if (c_label(qp) == cq) {
              co_count(q, qp)  += 1u;
              co_count(qp, q)  += 1u;
            }
          }
        }
      }

      save_idx++;
    }
  }

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

  out["fmc_rho"]              = store_rho;
  out["fmc_c"]                = store_c;
  out["fmc_mu"]               = store_mu;
  out["fmc_Sigma"]            = store_Sigma;
  out["fmc_K_plus"]           = store_K_plus;
  out["fmc_co_cluster"]       = compute_co_cluster_online ? wrap(co_cluster_pp) : R_NilValue;
  out["fmc_S0"]               = use_S0_hyperprior ? wrap(store_S0) : R_NilValue;
  out["fmc_d"]                = d;
  out["fmc_K_star"]           = K_star;
  out["fmc_e0"]               = e0;
  out["fmc_S0_hyperprior"]    = use_S0_hyperprior;

  // v13-only outputs: auxiliary positions r and anchor covariance Sigma_b
  out["fmc_r"]                  = store_r;
  out["fmc_sigma_b_isotropic"]  = sigma_b_isotropic;
  out["fmc_sigma_b_sq"]         = sigma_b_isotropic ? wrap(store_sigma_b_sq) : R_NilValue;
  out["fmc_Sigma_b"]            = sigma_b_isotropic ? R_NilValue : wrap(store_Sigma_b);
  out["fmc_S_b_hyperprior"]     = use_S_b_hyperprior;

  // Split-merge diagnostics
  List sm;
  sm["split_attempts"] = sm_split_attempts;
  sm["split_accepts"]  = sm_split_accepts;
  sm["merge_attempts"] = sm_merge_attempts;
  sm["merge_accepts"]  = sm_merge_accepts;
  sm["split_rate"] = (sm_split_attempts > 0)
    ? (double)sm_split_accepts / (double)sm_split_attempts : 0.0;
  sm["merge_rate"] = (sm_merge_attempts > 0)
    ? (double)sm_merge_accepts / (double)sm_merge_attempts : 0.0;
  out["fmc_split_merge"] = sm;

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
