// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// =========================================================
// Bipartite SBM for strictly-positive distance-type edge weights.
// Consumes posterior distance samples D^(s) from a fitted multilayered
// LSIRM (one SBM Gibbs sweep per LSIRM sample).
//
// Model (paper/bipartite_SBM.tex §1.2):
//   X^(s)_ij | z_i = q, w_j = l, kappa, lambda_ql
//       ~ Gamma(kappa, lambda_ql)
//   lambda_ql | kappa ~ Gamma(kappa * r, r * Xbar^(s))
//   z_i ~ Categorical(pi);  pi ~ Dirichlet(1,...,1)
//   w_j ~ Categorical(rho); rho ~ Dirichlet(1,...,1)
//   log kappa ~ N(mu_log_kappa, sd_log_kappa^2)
//
// One sweep per LSIRM sample s = 0..m-1:
//   pi -> rho -> Lambda -> z -> w -> log_kappa
// =========================================================

static const double EPS = 1e-12;

// Compute log(sum(exp(x))) stably given a vec
static inline double log_sum_exp(const vec& x) {
  double M = x.max();
  return M + std::log(arma::sum(arma::exp(x - M)));
}

// Sample categorical given log-weights (numerically stable)
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

// =========================================================
// Helper: build (n x p x m) distance cube from
//   a_arr: (n x d x m) cube
//   b_arr: (p x d x m) cube
// returned cube: D.slice(s)(i,j) = || a_arr.slice(s).row(i) - b_arr.slice(s).row(j) ||
// =========================================================
// [[Rcpp::export]]
arma::cube compute_distance_cube(const arma::cube& a_arr,
                                 const arma::cube& b_arr) {
  if (a_arr.n_cols != b_arr.n_cols)
    Rcpp::stop("compute_distance_cube: a_arr and b_arr must share dim d (cols).");
  if (a_arr.n_slices != b_arr.n_slices)
    Rcpp::stop("compute_distance_cube: a_arr and b_arr must share m (slices).");

  const uword n = a_arr.n_rows;
  const uword p = b_arr.n_rows;
  const uword m = a_arr.n_slices;

  arma::cube D(n, p, m, fill::zeros);

  for (uword s = 0; s < m; ++s) {
    const mat& A = a_arr.slice(s);  // n x d
    const mat& B = b_arr.slice(s);  // p x d

    //  ||a_i - b_j||^2 = ||a_i||^2 + ||b_j||^2 - 2 a_i . b_j
    vec a_sq = sum(square(A), 1);                       // n
    vec b_sq = sum(square(B), 1);                       // p
    mat cross = A * B.t();                              // n x p
    mat sq = repmat(a_sq, 1, p) + repmat(b_sq.t(), n, 1) - 2.0 * cross;
    sq.elem(find(sq < 0.0)).zeros();                    // numerical safety
    D.slice(s) = sqrt(sq);
  }

  return D;
}

// =========================================================
// Main MCMC: one SBM sweep per slice of D.
// =========================================================
// [[Rcpp::export]]
List run_bipartite_sbm_cpp(
    const arma::cube& D,        // n x p x m
    int Q, int L,
    List hyper,
    List prop_sd,
    List init,
    bool verbose
) {
  const int n = (int)D.n_rows;
  const int p = (int)D.n_cols;
  const int m = (int)D.n_slices;

  // --- Hyperparameters ---
  double r              = hyper["r"];
  double mu_log_kappa   = hyper["mu_log_kappa"];
  double sd_log_kappa   = hyper["sd_log_kappa"];

  // --- Proposal SD ---
  double sd_log_kappa_prop = prop_sd["log_kappa"];

  // --- Initialize state ---
  uvec z      = as<uvec>(init["z"]);          // length n,  values 0..Q-1
  uvec w      = as<uvec>(init["w"]);          // length p,  values 0..L-1
  vec  pi_v   = as<vec >(init["pi"]);         // length Q
  vec  rho_v  = as<vec >(init["rho"]);        // length L
  mat  Lambda = as<mat >(init["Lambda"]);     // Q x L
  double log_kappa = as<double>(init["log_kappa"]);
  double kappa     = std::exp(log_kappa);

  if ((int)z.n_elem != n)      Rcpp::stop("init$z length != n");
  if ((int)w.n_elem != p)      Rcpp::stop("init$w length != p");
  if ((int)pi_v.n_elem != Q)   Rcpp::stop("init$pi length != Q");
  if ((int)rho_v.n_elem != L)  Rcpp::stop("init$rho length != L");
  if ((int)Lambda.n_rows != Q || (int)Lambda.n_cols != L)
    Rcpp::stop("init$Lambda must be Q x L");

  // --- Storage ---
  umat store_z(m, n, fill::zeros);
  umat store_w(m, p, fill::zeros);
  mat  store_pi(m, Q, fill::zeros);
  mat  store_rho(m, L, fill::zeros);
  cube store_Lambda(Q, L, m, fill::zeros);
  vec  store_log_kappa(m, fill::zeros);
  vec  store_Xbar(m, fill::zeros);
  double acc_log_kappa = 0.0;

  // --- MCMC loop: one sweep per LSIRM distance sample ---
  for (int s = 0; s < m; ++s) {
    if (verbose && ((s + 1) % 500 == 0))
      Rcout << "[bipartite_SBM] sweep " << (s + 1) << " / " << m << "\n";

    const mat& X = D.slice(s);
    double Xbar = arma::mean(arma::vectorise(X));
    if (Xbar <= 0.0) Xbar = EPS;

    // ---------- 1. pi | z  (Dirichlet(1 + n_q)) ----------
    vec n_q(Q, fill::zeros);
    for (int i = 0; i < n; ++i) n_q(z(i)) += 1.0;
    for (int q = 0; q < Q; ++q)
      pi_v(q) = R::rgamma(1.0 + n_q(q), 1.0);
    pi_v /= arma::sum(pi_v);

    // ---------- 2. rho | w  (Dirichlet(1 + m_l)) ----------
    vec m_l(L, fill::zeros);
    for (int j = 0; j < p; ++j) m_l(w(j)) += 1.0;
    for (int l = 0; l < L; ++l)
      rho_v(l) = R::rgamma(1.0 + m_l(l), 1.0);
    rho_v /= arma::sum(rho_v);

    // ---------- 3. lambda_ql | rest  (Gamma) ----------
    mat N_ql(Q, L, fill::zeros);
    mat S_ql(Q, L, fill::zeros);
    for (int i = 0; i < n; ++i) {
      const uword zi = z(i);
      for (int j = 0; j < p; ++j) {
        const uword wj = w(j);
        N_ql(zi, wj) += 1.0;
        S_ql(zi, wj) += X(i, j);
      }
    }
    for (int q = 0; q < Q; ++q) {
      for (int l = 0; l < L; ++l) {
        double shape = kappa * (r + N_ql(q, l));
        double rate  = r * Xbar + S_ql(q, l);
        if (rate < EPS) rate = EPS;
        Lambda(q, l) = R::rgamma(shape, 1.0 / rate);
        if (Lambda(q, l) < EPS) Lambda(q, l) = EPS;
      }
    }

    // ---------- 4. z_i | rest  (categorical) ----------
    vec log_pi  = arma::log(pi_v + EPS);
    mat log_Lam = arma::log(Lambda + EPS);   // Q x L

    for (int i = 0; i < n; ++i) {
      vec lw(Q, fill::zeros);
      for (int q = 0; q < Q; ++q) {
        double v = log_pi(q);
        for (int j = 0; j < p; ++j) {
          const uword wj = w(j);
          v += kappa * log_Lam(q, wj) - Lambda(q, wj) * X(i, j);
        }
        lw(q) = v;
      }
      z(i) = (uword)sample_log_weights(lw);
    }

    // ---------- 5. w_j | rest  (categorical) ----------
    vec log_rho = arma::log(rho_v + EPS);

    for (int j = 0; j < p; ++j) {
      vec lw(L, fill::zeros);
      for (int l = 0; l < L; ++l) {
        double v = log_rho(l);
        for (int i = 0; i < n; ++i) {
          const uword zi = z(i);
          v += kappa * log_Lam(zi, l) - Lambda(zi, l) * X(i, j);
        }
        lw(l) = v;
      }
      w(j) = (uword)sample_log_weights(lw);
    }

    // ---------- 6. log_kappa | rest  (RW-MH on log scale) ----------
    auto log_kernel = [&](double k_val, double lk_val) {
      // Likelihood: sum_{i,j} log Gamma(X_ij | shape=k_val, rate=Lambda_{z_i,w_j})
      double ll = 0.0;
      double lgk = std::lgamma(k_val);
      for (int i = 0; i < n; ++i) {
        const uword zi = z(i);
        for (int j = 0; j < p; ++j) {
          const uword wj = w(j);
          double lam = Lambda(zi, wj);
          double xij = X(i, j); if (xij < EPS) xij = EPS;
          ll += k_val * std::log(lam + EPS)
              - lgk
              + (k_val - 1.0) * std::log(xij)
              - lam * xij;
        }
      }
      // Lambda prior: Gamma(lambda_ql | shape=k_val*r, rate=r*Xbar)
      double sh = k_val * r;
      double ra = r * Xbar; if (ra < EPS) ra = EPS;
      double lgsh = std::lgamma(sh);
      for (int q = 0; q < Q; ++q) {
        for (int l = 0; l < L; ++l) {
          double lam = Lambda(q, l); if (lam < EPS) lam = EPS;
          ll += sh * std::log(ra)
              - lgsh
              + (sh - 1.0) * std::log(lam)
              - ra * lam;
        }
      }
      // Prior on log_kappa: Gaussian in xi-space (no Jacobian needed)
      ll += R::dnorm(lk_val, mu_log_kappa, sd_log_kappa, 1);
      return ll;
    };

    double lk_old   = log_kappa;
    double lk_prop  = R::rnorm(lk_old, sd_log_kappa_prop);
    double k_old    = kappa;
    double k_prop   = std::exp(lk_prop);
    double ll_old   = log_kernel(k_old,  lk_old);
    double ll_new   = log_kernel(k_prop, lk_prop);
    if (std::log(R::runif(0.0, 1.0)) < (ll_new - ll_old)) {
      log_kappa = lk_prop;
      kappa     = k_prop;
      acc_log_kappa += 1.0;
    }

    // ---------- 7. Store ----------
    for (int i = 0; i < n; ++i) store_z(s, i) = z(i);
    for (int j = 0; j < p; ++j) store_w(s, j) = w(j);
    store_pi.row(s)        = pi_v.t();
    store_rho.row(s)       = rho_v.t();
    store_Lambda.slice(s)  = Lambda;
    store_log_kappa(s)     = log_kappa;
    store_Xbar(s)          = Xbar;
  }

  return List::create(
    Named("z")             = store_z,
    Named("w")             = store_w,
    Named("pi")            = store_pi,
    Named("rho")           = store_rho,
    Named("Lambda")        = store_Lambda,
    Named("log_kappa")     = store_log_kappa,
    Named("Xbar")          = store_Xbar,
    Named("acc_log_kappa") = acc_log_kappa / std::max(1, m),
    Named("Q")             = Q,
    Named("L")             = L,
    Named("n")             = n,
    Named("p")             = p,
    Named("m")             = m
  );
}
