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
// v14: Joint multilayered LSIRM (5-layer, per-item kappa)
//      + Hierarchical Mixture-of-Mixtures (MoM) on the global
//        pool {z_q = b_j^(l)}_{q=1..P_total}:
//          z_q | S_q = k, I_q = l ~ N_d(mu_{kl}, Sigma_{kl})
//          S_q ~ Categorical(eta_K),   I_q | S_q=k ~ Categorical(w_k)
//          eta_K ~ Dir_K(gamma_K),     gamma_K = alpha / K   (Stage 1: fixed)
//          w_k  ~ Dir_L(d_0)
//          Sigma_{kl}^{-1} | C_0k       ~ W_d(c_0, C_0k)        [FS conv.]
//          mu_{kl}  | b_0k, Lambda_k    ~ N_d(b_0k, B_tilde_0k)
//          C_0k                         ~ W_d(g_0, G_0)         [FS conv.]
//          b_0k                         ~ N_d(m_0, M_0)
//          lambda_{kj} ~iid Gamma(nu_gig, nu_gig)
//          Lambda_k = diag(lambda_{k1}, ..., lambda_{kd})
//          B_tilde_0k = Lambda_k^{1/2} B_0 Lambda_k^{1/2}   (B_0 diag)
//
// STAGE 1 (this file): K fixed (= K_max), alpha = alpha_const,
//   Variant B b_j MH (partial collapse: S_q fixed, I_q marginalized).
//   No telescoping, no Variant A. Stage 2 / 3 add them.
//
// v14 DIFFERENCES vs v13:
//   - Removes auxiliary positions r_q and global anchor Sigma_b.
//   - Removes single-level NIW + split-merge clustering.
//   - Replaces them with the two-level MoM described above.
//   - b_j MH prior is the marginal-I_q Gaussian mixture over the
//     L subcomponents of cluster S_q.
//
// WISHART CONVENTION:
//   This file uses Fruhwirth-Schnatter (2006) convention:
//     X ~ W_d(c, C)  iff  p(X) prop |X|^{c - (d+1)/2} exp(-tr(CX)),
//     E[X] = c * C^{-1}.
//   Posterior shape after N obs of N_d(mu, Sigma) with Sigma^{-1}~W_d(c,C):
//     shape = c + N/2,   scale = C + 0.5 * sum (z_q - mu)(z_q - mu)^T.
//
//   arma::wishrnd(out, V, df) draws X ~ W with density
//     prop |X|^{(df-d-1)/2} exp(-0.5 tr(V^{-1} X)),  E[X] = df * V.
//
//   Mapping FS (c, C) -> arma (V, df):
//     df = 2 * c
//     V  = (2 * C)^{-1}   (use arma::inv_sympd)
//   See sample_wishart_fs() below.
//
// GIG SAMPLER:
//   Uses GIGrvg::rgig (R package). variants.md GIG(p, a, b) <-> GIGrvg(lambda=p, chi=b, psi=a).
// =========================================================

// ===== LSIRM helpers (verbatim from v13) =====
static inline void rcpp_shuffle(std::vector<int>& v) {
  int n = (int) v.size();
  for (int i = n - 1; i > 0; --i) {
    int j = (int) std::floor(R::runif(0.0, (double)(i + 1)));
    if (j > i) j = i;
    std::swap(v[i], v[j]);
  }
}

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

// ===== MoM helpers =====
static const double V14_EPS = 1e-8;

// Sample categorical from log-weights (returns 0-based index). Uses log-sum-exp.
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

static inline vec mvnrnd_chol(const vec& m, const mat& V) {
  mat L;
  bool ok = arma::chol(L, V, "lower");
  if (!ok) {
    mat Vj = V + V14_EPS * arma::eye(V.n_rows, V.n_cols);
    L = arma::chol(Vj, "lower");
  }
  vec z = arma::randn<vec>(m.n_elem);
  return m + L * z;
}

static inline double log_det_pd(const mat& M) {
  double val, sgn;
  arma::log_det(val, sgn, M);
  return val;
}

// Stack b1..b5 into z_all (canonical order). z_all is P_total x d.
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
// FS Wishart sampler
//   X ~ W_d(c, C),  density prop |X|^{c-(d+1)/2} exp(-tr(CX)),
//   E[X] = c * C^{-1}, posterior shape = c + N/2 under N obs of
//   N_d(mu, X^{-1}).
// Mapping to arma::wishrnd:  df_arma = 2c,  V_arma = (2C)^{-1}.
// =========================================================
static inline mat sample_wishart_fs(double c_fs, const mat& C_fs) {
  int d = (int) C_fs.n_rows;
  mat C_sym = 0.5 * (C_fs + C_fs.t());
  mat V_arma;
  bool ok_inv = arma::inv_sympd(V_arma, 2.0 * C_sym);
  if (!ok_inv) {
    V_arma = arma::inv_sympd(2.0 * C_sym + V14_EPS * arma::eye(d, d));
  }
  V_arma = 0.5 * (V_arma + V_arma.t());
  double df_arma = 2.0 * c_fs;
  mat X;
  bool ok = arma::wishrnd(X, V_arma, df_arma);
  if (!ok || !X.is_finite()) {
    // Fallback: return prior mean c * C^{-1}.
    X = c_fs * arma::inv_sympd(C_sym + V14_EPS * arma::eye(d, d));
  }
  return 0.5 * (X + X.t());
}

// Log multivariate normal density using precision form.
// log N_d(x; mu, Sigma) = -0.5 d log(2 pi) - 0.5 log|Sigma| - 0.5 (x-mu)^T Sigma^{-1} (x-mu).
static inline double log_mvn_density_prec(
    const rowvec& x, const rowvec& mu,
    const mat& SigmaInv, double log_det_Sigma, int d
) {
  rowvec dv = x - mu;
  double quad = arma::as_scalar(dv * SigmaInv * dv.t());
  return -0.5 * (double)d * std::log(2.0 * M_PI)
         - 0.5 * log_det_Sigma
         - 0.5 * quad;
}

// Variant B prior: marginalize I_q over the L subcomponents of cluster k_star.
//   pi_B(b) = sum_{l=1..L} w_{k*,l} * N_d(b; mu_{k*,l}, Sigma_{k*,l})
static inline double log_mom_prior_B(
    const rowvec& b_val, int k_star, int L, int d,
    const mat& w_k, const mat& mu_kl,
    const cube& SigmaInv_kl, const vec& log_det_Sigma_kl
) {
  vec lw(L);
  for (int l = 0; l < L; ++l) {
    int s = k_star * L + l;
    double logphi = log_mvn_density_prec(
      b_val, mu_kl.row(s), SigmaInv_kl.slice(s), log_det_Sigma_kl(s), d
    );
    double w = w_k(k_star, l);
    if (w <= 0.0) w = V14_EPS;
    lw(l) = std::log(w) + logphi;
  }
  double M = lw.max();
  return M + std::log(arma::sum(arma::exp(lw - M)));
}

// =========================================================
// GIG sampler wrapper: lambda_kj ~ GIG(p, a, b) per variants.md Eq (3.5).
//
//   variants.md GIG(p,a,b):  density prop x^{p-1} exp(-0.5 (a*x + b/x))
//   GIGrvg::rgig(n, lambda, chi, psi):
//                            density prop x^{lambda-1} exp(-0.5 (psi*x + chi/x))
//   Mapping: lambda = p, chi = b, psi = a.
//
// We hold a single cached Rcpp::Function pointer (cheap; one R-level call
// per (k, j) in update; ~50 calls/iter, negligible overhead.)
// =========================================================
static inline double gig_sample(Rcpp::Function& rgig_R, double p, double a, double b) {
  // Guard: GIGrvg requires (a >= 0, b >= 0) with (a > 0 if p < 0) etc.
  // For our use, p = nu - L/2 >= 0 (assert at startup), a = 2*nu > 0,
  // b = sum_l (mu_klj - b_0kj)^2 / B_0jj >= 0. b = 0 is OK iff p > 0.
  if (b <= 0.0) b = V14_EPS;
  SEXP out = rgig_R(Named("n") = 1, Named("lambda") = p,
                    Named("chi") = b,  Named("psi") = a);
  double s = Rcpp::as<double>(out);
  if (!std::isfinite(s) || s <= 0.0) s = V14_EPS;
  return s;
}

// =========================================================
// Main MCMC
// =========================================================
// [[Rcpp::export]]
List run_lsirm_fmc_v14_cpp(
    mat Y_bin, mat Y_con, mat Y_cnt, IntegerMatrix Y_ord1, IntegerMatrix Y_ord2,
    int d, int n_iter, int burnin, int thin,
    int K_max, int L,
    List lsirm_hyper, List fmc_hyper,
    List lsirm_prop_sd,
    List lsirm_init, List fmc_init,
    bool verbose, bool fix_gamma, double nu2,
    bool compute_co_cluster_online,
    int  fmc_warmup,
    double alpha_const,
    bool telescoping_on,
    int  b_variant
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

  if (telescoping_on) Rcpp::stop("v14 Stage 1: telescoping_on must be FALSE.");
  if (b_variant != 1) Rcpp::stop("v14 Stage 1: b_variant must be 1 (Variant B).");
  if (K_max < 1)      Rcpp::stop("K_max must be >= 1.");
  if (L < 1)          Rcpp::stop("L must be >= 1.");

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
    Rcout << "[v14 stage1] n=" << n
          << " P1=" << P1 << " P2=" << P2 << " P3=" << P3
          << " P4=" << P4 << " P5=" << P5
          << " P_total=" << P_total
          << " K1=" << K1 << " K2=" << K2 << " nu2=" << nu2
          << " d=" << d << " K_max=" << K_max << " L=" << L
          << " alpha_const=" << alpha_const << "\n";
  }

  // Per-layer offsets in the global item index q.
  int off_L1 = 0;
  int off_L2 = off_L1 + P1;
  int off_L3 = off_L2 + P2;
  int off_L4 = off_L3 + P3;
  int off_L5 = off_L4 + P4;

  // ===== LSIRM hyperparameters (verbatim from v13) =====
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

  // ===== MoM hyperparameters (variants.md §1.2) =====
  vec    m_0     = as<vec>(fmc_hyper["m_0"]);
  mat    M_0     = as<mat>(fmc_hyper["M_0"]);
  vec    B_0_diag = as<vec>(fmc_hyper["B_0_diag"]);
  mat    G_0     = as<mat>(fmc_hyper["G_0"]);
  double g_0     = as<double>(fmc_hyper["g_0"]);
  double c_0     = as<double>(fmc_hyper["c_0"]);
  double nu_gig  = as<double>(fmc_hyper["nu_gig"]);
  double d_0     = as<double>(fmc_hyper["d_0"]);

  // Validation
  if ((int)m_0.n_elem != d)
    Rcpp::stop("fmc_hyper$m_0 length must equal d");
  if ((int)M_0.n_rows != d || (int)M_0.n_cols != d)
    Rcpp::stop("fmc_hyper$M_0 must be d x d");
  if ((int)B_0_diag.n_elem != d)
    Rcpp::stop("fmc_hyper$B_0_diag length must equal d");
  if ((int)G_0.n_rows != d || (int)G_0.n_cols != d)
    Rcpp::stop("fmc_hyper$G_0 must be d x d");
  // Wishart shape sanity (FS conv: shape > (d-1)/2 for proper density)
  if (c_0 <= 0.5 * (double)(d - 1))
    Rcpp::stop("c_0 must be > (d-1)/2 (FS Wishart shape boundary).");
  if (g_0 <= 0.5 * (double)(d - 1))
    Rcpp::stop("g_0 must be > (d-1)/2 (FS Wishart shape boundary).");
  if (nu_gig <= 0.0)
    Rcpp::stop("nu_gig must be > 0.");
  // GIG shape p_kL = nu_gig - L/2 must be non-negative for stable sampling.
  double p_kL = nu_gig - 0.5 * (double)L;
  if (p_kL < 0.0) {
    Rcpp::stop("GIG shape p_kL = nu_gig - L/2 must be >= 0. Increase nu_gig or decrease L.");
  }
  if (d_0 <= 0.0)
    Rcpp::stop("d_0 must be > 0.");
  if (alpha_const <= 0.0)
    Rcpp::stop("alpha_const must be > 0.");

  // Cached M_0^{-1} for Step 3.7
  mat M_0_inv = arma::inv_sympd(M_0 + V14_EPS * arma::eye(d, d));

  // Cache GIGrvg::rgig once outside the loop.
  Rcpp::Function rgig_R("rgig", Rcpp::Environment::namespace_env("GIGrvg"));

  if (verbose) {
    Rcout << "[v14 hyper] c_0=" << c_0 << " g_0=" << g_0
          << " nu_gig=" << nu_gig << " p_kL=" << p_kL
          << " d_0=" << d_0 << "\n";
  }

  // ===== LSIRM init (verbatim from v13) =====
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

  // ===== MoM state (preallocated to K_max) =====
  // S_q, I_q (0-based)
  uvec S_q(P_total);
  uvec I_q(P_total);
  if (fmc_init.containsElementNamed("S") && fmc_init.containsElementNamed("I")) {
    S_q = as<uvec>(fmc_init["S"]);
    I_q = as<uvec>(fmc_init["I"]);
    if ((int)S_q.n_elem != P_total) Rcpp::stop("fmc_init$S length must equal P_total");
    if ((int)I_q.n_elem != P_total) Rcpp::stop("fmc_init$I length must equal P_total");
  } else {
    // Random init
    for (int q = 0; q < P_total; ++q) {
      S_q(q) = (uword) std::floor(R::runif(0.0, (double)K_max));
      if ((int)S_q(q) >= K_max) S_q(q) = K_max - 1;
      I_q(q) = (uword) std::floor(R::runif(0.0, (double)L));
      if ((int)I_q(q) >= L) I_q(q) = L - 1;
    }
  }

  // Counts
  ivec N_k(K_max, fill::zeros);
  ivec N_kl(K_max * L, fill::zeros);
  for (int q = 0; q < P_total; ++q) {
    N_k((int)S_q(q)) += 1;
    N_kl((int)S_q(q) * L + (int)I_q(q)) += 1;
  }

  // Cluster random hyperparameters
  mat  b_0k(K_max, d);
  cube C_0k(d, d, K_max);
  mat  Lambda_k(K_max, d, fill::ones);
  mat  B_tilde_0k_diag(K_max, d);
  for (int k = 0; k < K_max; ++k) {
    b_0k.row(k) = mvnrnd_chol(m_0, M_0).t();
    C_0k.slice(k) = sample_wishart_fs(g_0, G_0);
    for (int j = 0; j < d; ++j) {
      Lambda_k(k, j) = R::rgamma(nu_gig, 1.0 / nu_gig);  // shape, scale=1/rate
      B_tilde_0k_diag(k, j) = Lambda_k(k, j) * B_0_diag(j);
    }
  }

  // Subcomponent means / covariances
  mat  mu_kl(K_max * L, d);
  cube Sigma_kl(d, d, K_max * L);
  cube SigmaInv_kl(d, d, K_max * L);
  vec  log_det_Sigma_kl(K_max * L);
  for (int k = 0; k < K_max; ++k) {
    for (int l = 0; l < L; ++l) {
      int s = k * L + l;
      // Sigma_kl^{-1} ~ W_d(c_0, C_0k)  (FS conv)
      mat SigInv = sample_wishart_fs(c_0, C_0k.slice(k));
      SigmaInv_kl.slice(s) = SigInv;
      mat Sig = arma::inv_sympd(SigInv);
      Sig = 0.5 * (Sig + Sig.t());
      Sigma_kl.slice(s) = Sig;
      log_det_Sigma_kl(s) = log_det_pd(Sig);
      // mu_kl ~ N_d(b_0k, B_tilde_0k)  (diag)
      vec eps(d);
      for (int j = 0; j < d; ++j)
        eps(j) = R::rnorm(0.0, std::sqrt(B_tilde_0k_diag(k, j)));
      mu_kl.row(s) = b_0k.row(k) + eps.t();
    }
  }

  // Mixture weights
  vec eta_K(K_max);
  mat w_k(K_max, L);
  {
    double tot = 0;
    for (int k = 0; k < K_max; ++k) { eta_K(k) = R::rgamma(alpha_const / K_max, 1.0); tot += eta_K(k); }
    if (tot > 0) eta_K /= tot;
    else eta_K.fill(1.0 / (double)K_max);
    for (int k = 0; k < K_max; ++k) {
      double tw = 0;
      for (int l = 0; l < L; ++l) { w_k(k, l) = R::rgamma(d_0, 1.0); tw += w_k(k, l); }
      if (tw > 0) w_k.row(k) /= tw;
      else w_k.row(k).fill(1.0 / (double)L);
    }
  }

  int K_cur = K_max;  // Stage 1: K fixed
  int K_plus = 0;     // recomputed each iter
  double alpha_cur = alpha_const;

  // ===== Storage =====
  int n_save = (n_iter - burnin) / thin;
  if (n_save < 1) n_save = 1;

  // LSIRM storage (same as v13)
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
  mat sum_lambda2(n, P2, fill::zeros);

  // MoM storage
  umat store_S(n_save, P_total, fill::zeros);
  umat store_I(n_save, P_total, fill::zeros);
  vec  store_K_plus(n_save, fill::zeros);
  vec  store_K(n_save, fill::zeros);
  vec  store_alpha(n_save, fill::zeros);
  mat  store_eta_K(n_save, K_max, fill::zeros);
  cube store_w_k(K_max, L, n_save, fill::zeros);          // K_max x L per save
  cube store_mu_kl(K_max * L, d, n_save, fill::zeros);
  cube store_Sigma_kl(d, d, K_max * L * n_save, fill::zeros);
  cube store_b_0k(K_max, d, n_save, fill::zeros);
  cube store_C_0k(d, d, K_max * n_save, fill::zeros);
  cube store_Lambda_k(K_max, d, n_save, fill::zeros);

  umat co_count;
  if (compute_co_cluster_online) {
    co_count.set_size(P_total, P_total);
    co_count.zeros();
  }

  // ===== Acceptance counters =====
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

  mat z_all(P_total, d, fill::zeros);

  // Variant B prior for b_j MH (uses current S_q for item q).
  auto mom_prior_B = [&](const rowvec& b_val, int q) {
    int k_star = (int) S_q(q);
    return log_mom_prior_B(b_val, k_star, L, d,
                           w_k, mu_kl, SigmaInv_kl, log_det_Sigma_kl);
  };

  // ===== MCMC LOOP =====
  for (int iter = 0; iter < n_iter; ++iter) {
    if (verbose && (iter + 1) % 500 == 0)
      Rcout << "[v14] Iter: " << iter + 1 << " / " << n_iter << "\n";

    // ===================== LSIRM block (verbatim from v13) =====================
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
    // 1b. alpha2
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
    // 1c. alpha3
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
    // 1d. alpha4
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
    // 1e. alpha5
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

    // ====== b_j MH (Variant B, collapsed over I_q given S_q) ======
    // Prior: pi_B(b) = sum_l w_{S_q, l} N_d(b; mu_{S_q,l}, Sigma_{S_q,l}).
    // 5a. b1
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
      double lpri_old = mom_prior_B(b_old,  q);
      double lpri_new = mom_prior_B(b_prop, q);
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) + lpri_new -
           (ll_f(b_old)  + lpri_old))) {
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
      double lpri_old = mom_prior_B(b_old,  q);
      double lpri_new = mom_prior_B(b_prop, q);
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) + lpri_new -
           (ll_f(b_old)  + lpri_old))) {
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
      double lpri_old = mom_prior_B(b_old,  q);
      double lpri_new = mom_prior_B(b_prop, q);
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) + lpri_new -
           (ll_f(b_old)  + lpri_old))) {
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
      double lpri_old = mom_prior_B(b_old,  q);
      double lpri_new = mom_prior_B(b_prop, q);
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) + lpri_new -
           (ll_f(b_old)  + lpri_old))) {
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
      double lpri_old = mom_prior_B(b_old,  q);
      double lpri_new = mom_prior_B(b_prop, q);
      if (log(R::runif(0,1)) <
          (ll_f(b_prop) + lpri_new -
           (ll_f(b_old)  + lpri_old))) {
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
    // (B) Build z_q := stacked b1..b5
    // ============================================================
    stack_b_layers(b1, b2, b3, b4, b5, P1, P2, P3, P4, P5, z_all);

    // ============================================================
    // MoM block (variants.md §3.1-§3.4, §3.13, §3.14; Stage 1: K fixed)
    // ============================================================
    if (iter >= fmc_warmup) {

      // ---- Step 3.1: (S_q, I_q) joint allocation (variants.md Eq 3.2a/b) ----
      // Two-stage form: marginalize I to get S, then I | S.
      vec lw_up(K_cur);
      vec lw_low(L);
      for (int q = 0; q < P_total; ++q) {
        rowvec z_q = z_all.row(q);
        // Upper: log eta_k + log sum_l w_kl phi(z; mu_kl, Sigma_kl)
        for (int k = 0; k < K_cur; ++k) {
          vec lwi(L);
          for (int l = 0; l < L; ++l) {
            int s = k * L + l;
            double logphi = log_mvn_density_prec(
              z_q, mu_kl.row(s), SigmaInv_kl.slice(s),
              log_det_Sigma_kl(s), d
            );
            double w = w_k(k, l);
            if (w <= 0.0) w = V14_EPS;
            lwi(l) = std::log(w) + logphi;
          }
          double Mi = lwi.max();
          double lse_k = Mi + std::log(arma::sum(arma::exp(lwi - Mi)));
          double e = eta_K(k);
          if (e <= 0.0) e = V14_EPS;
          lw_up(k) = std::log(e) + lse_k;
        }
        int k_new = sample_log_weights(lw_up);

        // Lower: I_q | S_q = k_new
        for (int l = 0; l < L; ++l) {
          int s = k_new * L + l;
          double logphi = log_mvn_density_prec(
            z_q, mu_kl.row(s), SigmaInv_kl.slice(s),
            log_det_Sigma_kl(s), d
          );
          double w = w_k(k_new, l);
          if (w <= 0.0) w = V14_EPS;
          lw_low(l) = std::log(w) + logphi;
        }
        int l_new = sample_log_weights(lw_low);

        S_q(q) = (uword) k_new;
        I_q(q) = (uword) l_new;
      }

      // Recount.
      N_k.zeros();
      N_kl.zeros();
      for (int q = 0; q < P_total; ++q) {
        N_k((int)S_q(q)) += 1;
        N_kl((int)S_q(q) * L + (int)I_q(q)) += 1;
      }
      K_plus = 0;
      for (int k = 0; k < K_cur; ++k) if (N_k(k) > 0) K_plus++;

      // ---- Step 3.2: (mu_kl, Sigma_kl) update (variants.md Eq 3.3, 3.4) ----
      // For each (k, l): conditional posterior. When N_kl = 0 these reduce
      // to draws from the prior (b_0k, B_tilde_0k) and (c_0, C_0k).
      for (int k = 0; k < K_cur; ++k) {
        // Build B_tilde_0k_inv (diagonal)
        rowvec Btilde_diag = B_tilde_0k_diag.row(k);
        for (int l = 0; l < L; ++l) {
          int s = k * L + l;
          int n_kl = N_kl(s);
          // ----- Sigma_kl^{-1} | rest ~ W_d(c_0 + n_kl/2, C_0k + 0.5 * sum (z-mu)(z-mu)^T)
          mat SS(d, d, fill::zeros);
          if (n_kl > 0) {
            rowvec mu_old = mu_kl.row(s);
            for (int q = 0; q < P_total; ++q) {
              if ((int)S_q(q) == k && (int)I_q(q) == l) {
                rowvec dv = z_all.row(q) - mu_old;
                SS += dv.t() * dv;
              }
            }
          }
          double c_post = c_0 + 0.5 * (double)n_kl;
          mat    C_post = C_0k.slice(k) + 0.5 * SS;
          C_post = 0.5 * (C_post + C_post.t());
          mat SigInv = sample_wishart_fs(c_post, C_post);
          SigmaInv_kl.slice(s) = SigInv;
          mat Sig = arma::inv_sympd(SigInv + V14_EPS * arma::eye(d, d));
          Sig = 0.5 * (Sig + Sig.t());
          Sigma_kl.slice(s) = Sig;
          log_det_Sigma_kl(s) = log_det_pd(Sig);

          // ----- mu_kl | rest ~ N(B_kl ( B_tilde_0k_inv b_0k + SigInv * n_kl * zbar ), B_kl)
          // B_kl = ( B_tilde_0k_inv + n_kl * SigInv )^{-1}
          mat Btilde_inv = arma::diagmat(1.0 / Btilde_diag);
          mat prec_post = Btilde_inv + (double)n_kl * SigInv;
          prec_post = 0.5 * (prec_post + prec_post.t());
          mat B_kl = arma::inv_sympd(prec_post + V14_EPS * arma::eye(d, d));
          B_kl = 0.5 * (B_kl + B_kl.t());
          vec sum_z(d, fill::zeros);
          if (n_kl > 0) {
            for (int q = 0; q < P_total; ++q) {
              if ((int)S_q(q) == k && (int)I_q(q) == l) {
                sum_z += z_all.row(q).t();
              }
            }
          }
          vec rhs = Btilde_inv * b_0k.row(k).t() + SigInv * sum_z;
          vec mu_mean = B_kl * rhs;
          mu_kl.row(s) = mvnrnd_chol(mu_mean, B_kl).t();
        }
      }

      // ---- Step 3.3: (b_0k, C_0k, Lambda_k) update (variants.md Eq 3.5, 3.6, 3.7) ----
      for (int k = 0; k < K_cur; ++k) {
        // ----- 3.5: lambda_kj ~ GIG(p_kL, 2 nu, b_kj),  b_kj = sum_l (mu_klj - b_0kj)^2 / B_0jj
        for (int j = 0; j < d; ++j) {
          double bkj = 0.0;
          for (int l = 0; l < L; ++l) {
            int s = k * L + l;
            double diff = mu_kl(s, j) - b_0k(k, j);
            bkj += diff * diff;
          }
          bkj /= B_0_diag(j);
          double lam_new = gig_sample(rgig_R, p_kL, 2.0 * nu_gig, bkj);
          Lambda_k(k, j) = lam_new;
          B_tilde_0k_diag(k, j) = lam_new * B_0_diag(j);
        }

        // ----- 3.6: C_0k ~ W_d(g_0 + L * c_0, G_0 + sum_l SigInv_kl)
        mat sum_SigInv(d, d, fill::zeros);
        for (int l = 0; l < L; ++l) {
          int s = k * L + l;
          sum_SigInv += SigmaInv_kl.slice(s);
        }
        sum_SigInv = 0.5 * (sum_SigInv + sum_SigInv.t());
        double cC_shape = g_0 + (double)L * c_0;
        mat    cC_scale = G_0 + sum_SigInv;
        cC_scale = 0.5 * (cC_scale + cC_scale.t());
        C_0k.slice(k) = sample_wishart_fs(cC_shape, cC_scale);

        // ----- 3.7: b_0k ~ N_d(m_k_post, M_k_post)  (B_tilde diag => coord separable)
        rowvec Btilde_diag = B_tilde_0k_diag.row(k);
        mat Btilde_inv = arma::diagmat(1.0 / Btilde_diag);
        mat M_post_prec = M_0_inv + (double)L * Btilde_inv;
        M_post_prec = 0.5 * (M_post_prec + M_post_prec.t());
        mat M_post = arma::inv_sympd(M_post_prec + V14_EPS * arma::eye(d, d));
        M_post = 0.5 * (M_post + M_post.t());
        vec sum_mu(d, fill::zeros);
        for (int l = 0; l < L; ++l) {
          int s = k * L + l;
          sum_mu += mu_kl.row(s).t();
        }
        vec rhs = M_0_inv * m_0 + Btilde_inv * sum_mu;
        vec m_post = M_post * rhs;
        b_0k.row(k) = mvnrnd_chol(m_post, M_post).t();
      }

      // ---- Step 3.13: eta_K | K, alpha, S ~ Dir_K(alpha/K + N_k) ----
      {
        double tot = 0;
        double gamma_K = alpha_cur / (double)K_cur;
        for (int k = 0; k < K_cur; ++k) {
          eta_K(k) = R::rgamma(gamma_K + (double)N_k(k), 1.0);
          tot += eta_K(k);
        }
        if (tot > 0) {
          for (int k = 0; k < K_cur; ++k) eta_K(k) /= tot;
        } else {
          eta_K.fill(1.0 / (double)K_cur);
        }
      }

      // ---- Step 3.14: w_k | I, S ~ Dir_L(d_0 + N_kl) ----
      for (int k = 0; k < K_cur; ++k) {
        double tw = 0;
        for (int l = 0; l < L; ++l) {
          int s = k * L + l;
          w_k(k, l) = R::rgamma(d_0 + (double)N_kl(s), 1.0);
          tw += w_k(k, l);
        }
        if (tw > 0) {
          for (int l = 0; l < L; ++l) w_k(k, l) /= tw;
        } else {
          for (int l = 0; l < L; ++l) w_k(k, l) = 1.0 / (double)L;
        }
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
      sum_lambda2 += lambda2;

      // MoM state
      for (int q = 0; q < P_total; ++q) {
        store_S(save_idx, q) = S_q(q);
        store_I(save_idx, q) = I_q(q);
      }
      store_K_plus(save_idx) = (double)K_plus;
      store_K(save_idx)      = (double)K_cur;
      store_alpha(save_idx)  = alpha_cur;
      store_eta_K.row(save_idx) = eta_K.t();
      for (int k = 0; k < K_max; ++k)
        for (int l = 0; l < L; ++l)
          store_w_k(k, l, save_idx) = w_k(k, l);
      store_mu_kl.slice(save_idx) = mu_kl;
      for (int s = 0; s < K_max * L; ++s)
        store_Sigma_kl.slice(K_max * L * save_idx + s) = Sigma_kl.slice(s);
      store_b_0k.slice(save_idx) = b_0k;
      for (int k = 0; k < K_max; ++k)
        store_C_0k.slice(K_max * save_idx + k) = C_0k.slice(k);
      store_Lambda_k.slice(save_idx) = Lambda_k;

      if (compute_co_cluster_online) {
        for (int q = 0; q < P_total; ++q) {
          uword sq = S_q(q);
          co_count(q, q) += 1u;
          for (int qp = q + 1; qp < P_total; ++qp) {
            if (S_q(qp) == sq) {
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
  out["lambda2_postmean"] = sum_lambda2 / std::max(n_save, 1);

  // MoM outputs
  out["fmc_S"]                  = store_S;
  out["fmc_I"]                  = store_I;
  out["fmc_K_plus"]             = store_K_plus;
  out["fmc_K"]                  = store_K;
  out["fmc_alpha"]              = store_alpha;
  out["fmc_eta_K"]              = store_eta_K;
  out["fmc_w_k"]                = store_w_k;
  out["fmc_mu_kl"]              = store_mu_kl;
  out["fmc_Sigma_kl"]           = store_Sigma_kl;
  out["fmc_b_0k"]               = store_b_0k;
  out["fmc_C_0k"]               = store_C_0k;
  out["fmc_Lambda_k"]           = store_Lambda_k;
  out["fmc_co_cluster"]         = compute_co_cluster_online ? wrap(co_cluster_pp) : R_NilValue;
  out["fmc_d"]                  = d;
  out["fmc_K_max"]              = K_max;
  out["fmc_L"]                  = L;
  out["fmc_alpha_const"]        = alpha_const;
  out["fmc_b_variant"]          = b_variant;
  out["fmc_telescoping_on"]     = telescoping_on;

  // Acceptance
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


// =========================================================
// Auxiliary export: GIG-sampler wiring sanity check.
//
// Calls the R-side GIGrvg::rgig from within the cpp, used by
// test/test_v14_gig.R to verify the wiring and the moment of
// GIG(lambda, chi, psi).
// =========================================================
// [[Rcpp::export]]
NumericVector v14_gig_sanity(int n, double lambda, double chi, double psi) {
  Rcpp::Function rgig_R("rgig", Rcpp::Environment::namespace_env("GIGrvg"));
  NumericVector out(n);
  for (int i = 0; i < n; ++i) {
    out[i] = gig_sample(rgig_R, lambda, psi, chi);
    // gig_sample(R, p=lambda, a=psi, b=chi) -> variants.md (p,a,b)
  }
  return out;
}

// =========================================================
// Auxiliary export: FS Wishart sampler sanity check.
//
// Draws n samples from W_d(c, C) under FS convention.
// Returns a (d, d, n) cube. Used by test/test_v14_wishart_convention.R.
// =========================================================
// [[Rcpp::export]]
arma::cube v14_wishart_fs_sanity(int n, double c_fs, arma::mat C_fs) {
  int d = (int) C_fs.n_rows;
  arma::cube out(d, d, n);
  for (int i = 0; i < n; ++i) {
    out.slice(i) = sample_wishart_fs(c_fs, C_fs);
  }
  return out;
}
