// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rmath.h>

using namespace Rcpp;

// ---------- helpers ----------
inline double log1pexp(double x){
  if (x > 0) return x + std::log1p(std::exp(-x));
  else       return std::log1p(std::exp(x));
}

// R array index for dims (n_save, n, d): [ss, i, k]
inline R_xlen_t idx3(R_xlen_t ss, R_xlen_t i, R_xlen_t k,
                     R_xlen_t n_save, R_xlen_t n){
  return ss + n_save * ( i + n * k );
}

// R array index for dims (n_save, P, d): [ss, j, k]
inline R_xlen_t idx3P(R_xlen_t ss, R_xlen_t j, R_xlen_t k,
                      R_xlen_t n_save, R_xlen_t P){
  return ss + n_save * ( j + P * k );
}

arma::mat dist_mat(const arma::mat& A, const arma::mat& B){
  // returns nrow(A) x nrow(B)
  const int nA = A.n_rows;
  const int nB = B.n_rows;
  arma::mat D(nA, nB, arma::fill::zeros);
  for(int i=0;i<nA;i++){
    for(int j=0;j<nB;j++){
      double acc=0.0;
      for(unsigned int k=0;k<A.n_cols;k++){
        const double diff = A(i,k) - B(j,k);
        acc += diff*diff;
      }
      D(i,j) = std::sqrt(acc);
    }
  }
  return D;
}

// [[Rcpp::export]]
Rcpp::List basic_lsirm_cpp(
    const arma::mat& Y_bin,   // n x P (0/1 but stored double ok)
    const int d,
    const int n_iter,
    const int burnin,
    const int thin,
    Rcpp::List hyper,
    Rcpp::List prop_sd,
    Rcpp::Nullable<Rcpp::List> init = R_NilValue,
    const bool verbose = true,
    const bool fix_gamma = false
){
  RNGScope scope;
  
  // -------- data dims --------
  const int n = Y_bin.n_rows;
  const int P = Y_bin.n_cols;
  
  // -------- hyper --------
  const double a_sigma      = as<double>(hyper["a_sigma"]);
  const double b_sigma      = as<double>(hyper["b_sigma"]);
  const double a_tau        = as<double>(hyper["a_tau"]);
  const double b_tau        = as<double>(hyper["b_tau"]);
  const double mu_log_gamma = as<double>(hyper["mu_log_gamma"]);
  const double sd_log_gamma = as<double>(hyper["sd_log_gamma"]);
  
  // -------- proposal sds --------
  const double ps_alpha     = as<double>(prop_sd["alpha"]);
  const double ps_beta      = as<double>(prop_sd["beta"]);
  const double ps_log_gamma = as<double>(prop_sd["log_gamma"]);
  const double ps_a         = as<double>(prop_sd["a"]);
  const double ps_b         = as<double>(prop_sd["b"]);
  
  // -------- init params --------
  arma::vec alpha(n), beta(P);
  arma::mat a(n, d), b(P, d);
  
  double log_gamma, gamma;
  double sigma_alpha_sq, tau_beta_sq;
  
  if (init.isNotNull()){
    Rcpp::List ini(init);
    
    alpha = as<arma::vec>(ini["alpha"]);
    beta  = as<arma::vec>(ini["beta"]);
    
    a = as<arma::mat>(ini["a"]);
    b = as<arma::mat>(ini["b"]);
    
    log_gamma = as<double>(ini["log_gamma"]);
    gamma     = std::exp(log_gamma);
    
    sigma_alpha_sq = as<double>(ini["sigma_alpha_sq"]);
    tau_beta_sq    = as<double>(ini["tau_beta_sq"]);
  } else {
    for(int i=0;i<n;i++) alpha(i) = R::rnorm(0.0, 0.1);
    for(int j=0;j<P;j++) beta(j)  = R::rnorm(0.0, 0.1);
    
    for(int i=0;i<n;i++) for(int k=0;k<d;k++) a(i,k) = R::rnorm(0.0, 0.5);
    for(int j=0;j<P;j++) for(int k=0;k<d;k++) b(j,k) = R::rnorm(0.0, 0.5);
    
    log_gamma = 0.0; gamma = 1.0;
    sigma_alpha_sq = 1.0;
    tau_beta_sq    = 1.0;
  }
  
  // -------- storage dims --------
  const int n_save = (n_iter > burnin) ? ( (n_iter - burnin) / thin ) : 0;
  
  NumericMatrix s_alpha(n_save, n);
  NumericMatrix s_beta(n_save, P);
  
  NumericVector s_log_gamma(n_save);
  NumericVector s_sigma_alpha_sq(n_save), s_tau_beta_sq(n_save);
  NumericVector s_loglik(n_save);
  
  // positions as arrays with dim (n_save, n, d) and (n_save, P, d)
  NumericVector s_a(n_save * n * d);
  NumericVector s_b(n_save * P * d);
  
  // compatibility outputs (z,a1,a2,a3) = a  (원래 네 코드 관례 유지)
  NumericVector s_z (n_save * n * d);
  NumericVector s_a1(n_save * n * d);
  NumericVector s_a2(n_save * n * d);
  NumericVector s_a3(n_save * n * d);
  
  s_a.attr("dim")  = IntegerVector::create(n_save, n, d);
  s_b.attr("dim")  = IntegerVector::create(n_save, P, d);
  
  s_z.attr("dim")  = IntegerVector::create(n_save, n, d);
  s_a1.attr("dim") = IntegerVector::create(n_save, n, d);
  s_a2.attr("dim") = IntegerVector::create(n_save, n, d);
  s_a3.attr("dim") = IntegerVector::create(n_save, n, d);
  
  // -------- acceptance counters --------
  arma::vec acc_alpha(n, arma::fill::zeros);
  arma::vec acc_beta(P, arma::fill::zeros);
  double acc_log_gamma = 0.0;
  arma::vec acc_a(n, arma::fill::zeros);
  arma::vec acc_b(P, arma::fill::zeros);
  
  // -------- distance matrix --------
  arma::mat D = dist_mat(a, b); // n x P
  
  int save_idx = 0;
  
  // ==========================
  // MCMC
  // ==========================
  for(int iter=1; iter<=n_iter; iter++){
    if(verbose && (iter % 500 == 0)){
      Rcout << "Iter: " << iter << " / " << n_iter << "\n";
    }
    
    // 1) alpha_i : MH
    for(int i=0;i<n;i++){
      double ll_cur = 0.0;
      for(int j=0;j<P;j++){
        const double eta = alpha(i) - beta(j) - gamma * D(i,j);
        ll_cur += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_cur = -0.5 * (alpha(i)*alpha(i)) / sigma_alpha_sq;
      const double logpost_cur = ll_cur + lp_cur;
      
      const double a_prop = R::rnorm(alpha(i), ps_alpha);
      
      double ll_p = 0.0;
      for(int j=0;j<P;j++){
        const double eta = a_prop - beta(j) - gamma * D(i,j);
        ll_p += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_p = -0.5 * (a_prop*a_prop) / sigma_alpha_sq;
      const double logpost_p = ll_p + lp_p;
      
      const double log_acc = logpost_p - logpost_cur;
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        alpha(i) = a_prop;
        acc_alpha(i) += 1.0;
      }
    }
    
    // 2) beta_j : MH
    for(int j=0;j<P;j++){
      double ll_cur = 0.0;
      for(int i=0;i<n;i++){
        const double eta = alpha(i) - beta(j) - gamma * D(i,j);
        ll_cur += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_cur = -0.5 * (beta(j)*beta(j)) / tau_beta_sq;
      
      const double b_prop = R::rnorm(beta(j), ps_beta);
      
      double ll_p = 0.0;
      for(int i=0;i<n;i++){
        const double eta = alpha(i) - b_prop - gamma * D(i,j);
        ll_p += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_p = -0.5 * (b_prop*b_prop) / tau_beta_sq;
      
      const double log_acc = (ll_p + lp_p) - (ll_cur + lp_cur);
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        beta(j) = b_prop;
        acc_beta(j) += 1.0;
      }
    }
    
    // 3) log_gamma : MH  (전체 likelihood 재계산)
    if(fix_gamma){
      acc_log_gamma += 1.0;
    } else {
      double ll_cur = 0.0;
      for(int i=0;i<n;i++) for(int j=0;j<P;j++){
        const double eta = alpha(i) - beta(j) - gamma * D(i,j);
        ll_cur += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_cur = R::dnorm(log_gamma, mu_log_gamma, sd_log_gamma, 1);
      const double logpost_cur = ll_cur + lp_cur;
      
      const double log_gamma_prop = R::rnorm(log_gamma, ps_log_gamma);
      const double gamma_prop = std::exp(log_gamma_prop);
      
      double ll_p = 0.0;
      for(int i=0;i<n;i++) for(int j=0;j<P;j++){
        const double eta = alpha(i) - beta(j) - gamma_prop * D(i,j);
        ll_p += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_p = R::dnorm(log_gamma_prop, mu_log_gamma, sd_log_gamma, 1);
      const double logpost_p = ll_p + lp_p;
      
      const double log_acc = logpost_p - logpost_cur;
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        log_gamma = log_gamma_prop;
        gamma = gamma_prop;
        acc_log_gamma += 1.0;
      }
    }
    
    // 4) respondent positions a_i : MH, prior N(0,I)
    for(int i=0;i<n;i++){
      arma::rowvec a_cur = a.row(i);
      
      double ll_cur = 0.0;
      for(int j=0;j<P;j++){
        const double eta = alpha(i) - beta(j) - gamma * D(i,j);
        ll_cur += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_cur = -0.5 * arma::dot(a_cur, a_cur);
      const double logpost_cur = ll_cur + lp_cur;
      
      arma::rowvec a_prop = a_cur;
      for(int k=0;k<d;k++) a_prop(k) += R::rnorm(0.0, ps_a);
      
      // recompute distances for row i
      arma::rowvec dp(P);
      for(int j=0;j<P;j++){
        double acc=0.0;
        for(int k=0;k<d;k++){
          const double diff = a_prop(k) - b(j,k);
          acc += diff*diff;
        }
        dp(j) = std::sqrt(acc);
      }
      
      double ll_p = 0.0;
      for(int j=0;j<P;j++){
        const double eta = alpha(i) - beta(j) - gamma * dp(j);
        ll_p += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_p = -0.5 * arma::dot(a_prop, a_prop);
      const double logpost_p = ll_p + lp_p;
      
      const double log_acc = logpost_p - logpost_cur;
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        a.row(i) = a_prop;
        D.row(i) = dp;
        acc_a(i) += 1.0;
      }
    }
    
    // 5) item positions b_j : MH, prior N(0,I)
    for(int j=0;j<P;j++){
      arma::rowvec b_cur = b.row(j);
      
      double ll_cur = 0.0;
      for(int i=0;i<n;i++){
        const double eta = alpha(i) - beta(j) - gamma * D(i,j);
        ll_cur += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_cur = -0.5 * arma::dot(b_cur, b_cur);
      
      arma::rowvec b_prop = b_cur;
      for(int k=0;k<d;k++) b_prop(k) += R::rnorm(0.0, ps_b);
      
      arma::vec dp(n);
      for(int i=0;i<n;i++){
        double acc=0.0;
        for(int k=0;k<d;k++){
          const double diff = a(i,k) - b_prop(k);
          acc += diff*diff;
        }
        dp(i) = std::sqrt(acc);
      }
      
      double ll_p = 0.0;
      for(int i=0;i<n;i++){
        const double eta = alpha(i) - beta(j) - gamma * dp(i);
        ll_p += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_p = -0.5 * arma::dot(b_prop, b_prop);
      
      const double log_acc = (ll_p + lp_p) - (ll_cur + lp_cur);
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        b.row(j) = b_prop;
        D.col(j) = dp;
        acc_b(j) += 1.0;
      }
    }
    
    // 6) Gibbs: sigma_alpha_sq, tau_beta_sq  (Inv-Gamma)
    {
      double ss=0.0;
      for(int i=0;i<n;i++) ss += alpha(i)*alpha(i);
      const double shape = a_sigma + ((double)n)/2.0;
      const double rate  = b_sigma + 0.5 * ss;
      sigma_alpha_sq = 1.0 / R::rgamma(shape, 1.0/rate);
    }
    {
      double ss=0.0;
      for(int j=0;j<P;j++) ss += beta(j)*beta(j);
      const double shape = a_tau + ((double)P)/2.0;
      const double rate  = b_tau + 0.5 * ss;
      tau_beta_sq = 1.0 / R::rgamma(shape, 1.0/rate);
    }
    
    // 7) save
    if(iter > burnin && ((iter - burnin) % thin == 0)){
      const int ss = save_idx; // 0-based
      
      for(int i=0;i<n;i++) s_alpha(ss, i) = alpha(i);
      for(int j=0;j<P;j++) s_beta(ss, j)  = beta(j);
      
      s_log_gamma[ss] = log_gamma;
      s_sigma_alpha_sq[ss] = sigma_alpha_sq;
      s_tau_beta_sq[ss]    = tau_beta_sq;
      
      // positions
      for(int i=0;i<n;i++){
        for(int k=0;k<d;k++){
          const double val = a(i,k);
          const R_xlen_t pos = idx3(ss, i, k, n_save, n);
          s_a[pos]  = val;
          s_z[pos]  = val;
          s_a1[pos] = val;
          s_a2[pos] = val;
          s_a3[pos] = val;
        }
      }
      for(int j=0;j<P;j++){
        for(int k=0;k<d;k++){
          s_b[idx3P(ss, j, k, n_save, P)] = b(j,k);
        }
      }
      
      // loglik (Bernoulli-logit only)
      double ll = 0.0;
      for(int i=0;i<n;i++) for(int j=0;j<P;j++){
        const double eta = alpha(i) - beta(j) - gamma * D(i,j);
        ll += Y_bin(i,j)*eta - log1pexp(eta);
      }
      s_loglik[ss] = ll;
      
      save_idx++;
    }
  }
  
  // acceptance ratios
  NumericVector accept_alpha(n), accept_beta(P), accept_a(n), accept_b(P);
  for(int i=0;i<n;i++){
    accept_alpha[i] = acc_alpha(i) / (double)n_iter;
    accept_a[i]     = acc_a(i)     / (double)n_iter;
  }
  for(int j=0;j<P;j++){
    accept_beta[j] = acc_beta(j) / (double)n_iter;
    accept_b[j]    = acc_b(j)    / (double)n_iter;
  }
  
  Rcpp::List samples = Rcpp::List::create(
    _["alpha"] = s_alpha,
    _["beta"]  = s_beta,
    _["log_gamma"] = s_log_gamma,
    _["sigma_alpha_sq"] = s_sigma_alpha_sq,
    _["tau_beta_sq"]    = s_tau_beta_sq,
    _["a"] = s_a,
    _["b"] = s_b,
    _["z"]  = s_z,
    _["a1"] = s_a1,
    _["a2"] = s_a2,
    _["a3"] = s_a3,
    _["loglik"] = s_loglik
  );
  
  Rcpp::List accept = Rcpp::List::create(
    _["alpha"] = accept_alpha,
    _["beta"]  = accept_beta,
    _["log_gamma"] = acc_log_gamma / (double)n_iter,
    _["a"] = accept_a,
    _["b"] = accept_b
  );
  
  return Rcpp::List::create(
    _["samples"] = samples,
    _["accept"]  = accept,
    _["hyper"]   = hyper,
    _["prop_sd"] = prop_sd
  );
}
