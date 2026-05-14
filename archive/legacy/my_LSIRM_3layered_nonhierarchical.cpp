// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rmath.h>

using namespace Rcpp;

// ---------- helpers ----------
inline double log1pexp(double x){
  // stable log(1+exp(x))
  if (x > 0) return x + std::log1p(std::exp(-x));
  else       return std::log1p(std::exp(x));
}

// R array index for dims (n_save, n, d): [ss, i, k]
inline R_xlen_t idx3(R_xlen_t ss, R_xlen_t i, R_xlen_t k,
                     R_xlen_t n_save, R_xlen_t n){
  // column-major: ss + n_save*( i + n*k )
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
Rcpp::List lsirm_sharedpos_layer3_cpp(
    const arma::mat& Y_bin,   // n x P1 (0/1)
    const arma::mat& Y_con,   // n x P2
    const arma::mat& Y_cnt,   // n x P3 (counts, but stored as double)
    const int d,
    const int n_iter,
    const int burnin,
    const int thin,
    Rcpp::List hyper,
    Rcpp::List prop_sd,
    Rcpp::Nullable<Rcpp::List> init = R_NilValue,
    const bool verbose = true
){
  RNGScope scope;
  
  // -------- data dims --------
  const int n  = Y_bin.n_rows;
  const int P1 = Y_bin.n_cols;
  const int P2 = Y_con.n_cols;
  const int P3 = Y_cnt.n_cols;
  
  arma::mat Z_con = Y_con; // 그대로
  
  // -------- hyper --------
  const double a_sigma   = as<double>(hyper["a_sigma"]);
  const double b_sigma   = as<double>(hyper["b_sigma"]);
  const double a_tau1    = as<double>(hyper["a_tau1"]);
  const double b_tau1    = as<double>(hyper["b_tau1"]);
  const double a_tau2    = as<double>(hyper["a_tau2"]);
  const double b_tau2    = as<double>(hyper["b_tau2"]);
  const double a_tau3    = as<double>(hyper["a_tau3"]);
  const double b_tau3    = as<double>(hyper["b_tau3"]);
  const double a_sigma0  = as<double>(hyper["a_sigma0"]);
  const double b_sigma0  = as<double>(hyper["b_sigma0"]);
  const double mu_log_gamma = as<double>(hyper["mu_log_gamma"]);
  const double sd_log_gamma = as<double>(hyper["sd_log_gamma"]);
  const double mu_log_kappa = as<double>(hyper["mu_log_kappa"]);
  const double sd_log_kappa = as<double>(hyper["sd_log_kappa"]);
  
  // -------- proposal sds --------
  const double ps_alpha     = as<double>(prop_sd["alpha"]);
  const double ps_beta1     = as<double>(prop_sd["beta1"]);
  const double ps_beta2     = as<double>(prop_sd["beta2"]);
  const double ps_beta3     = as<double>(prop_sd["beta3"]);
  const double ps_log_gamma = as<double>(prop_sd["log_gamma"]);
  const double ps_log_kappa = as<double>(prop_sd["log_kappa"]);
  const double ps_a         = as<double>(prop_sd["a"]);
  const double ps_b1        = as<double>(prop_sd["b1"]);
  const double ps_b2        = as<double>(prop_sd["b2"]);
  const double ps_b3        = as<double>(prop_sd["b3"]);
  
  // -------- init params --------
  arma::vec alpha(n), beta1(P1), beta2(P2), beta3(P3);
  arma::mat a(n, d), b1(P1, d), b2(P2, d), b3(P3, d);
  
  double log_gamma, gamma;
  double log_kappa, kappa;
  
  double sigma_alpha_sq, tau_beta1_sq, tau_beta2_sq, tau_beta3_sq, sigma0_sq;
  
  if (init.isNotNull()){
    Rcpp::List ini(init);
    
    alpha = as<arma::vec>(ini["alpha"]);
    beta1 = as<arma::vec>(ini["beta1"]);
    beta2 = as<arma::vec>(ini["beta2"]);
    beta3 = as<arma::vec>(ini["beta3"]);
    
    a  = as<arma::mat>(ini["a"]);
    b1 = as<arma::mat>(ini["b1"]);
    b2 = as<arma::mat>(ini["b2"]);
    b3 = as<arma::mat>(ini["b3"]);
    
    log_gamma = as<double>(ini["log_gamma"]);
    gamma     = std::exp(log_gamma);
    
    log_kappa = as<double>(ini["log_kappa"]);
    kappa     = std::exp(log_kappa);
    
    sigma_alpha_sq = as<double>(ini["sigma_alpha_sq"]);
    tau_beta1_sq   = as<double>(ini["tau_beta1_sq"]);
    tau_beta2_sq   = as<double>(ini["tau_beta2_sq"]);
    tau_beta3_sq   = as<double>(ini["tau_beta3_sq"]);
    sigma0_sq      = as<double>(ini["sigma0_sq"]);
  } else {
    for(int i=0;i<n;i++) alpha(i) = R::rnorm(0.0, 0.1);
    for(int j=0;j<P1;j++) beta1(j) = R::rnorm(0.0, 0.1);
    for(int j=0;j<P2;j++) beta2(j) = R::rnorm(0.0, 0.1);
    for(int j=0;j<P3;j++) beta3(j) = R::rnorm(0.0, 0.1);
    
    for(int i=0;i<n;i++) for(int k=0;k<d;k++) a(i,k)  = R::rnorm(0.0, 0.5);
    for(int j=0;j<P1;j++) for(int k=0;k<d;k++) b1(j,k) = R::rnorm(0.0, 0.5);
    for(int j=0;j<P2;j++) for(int k=0;k<d;k++) b2(j,k) = R::rnorm(0.0, 0.5);
    for(int j=0;j<P3;j++) for(int k=0;k<d;k++) b3(j,k) = R::rnorm(0.0, 0.5);
    
    log_gamma = 0.0; gamma = 1.0;
    log_kappa = 0.0; kappa = 1.0;
    
    sigma_alpha_sq = 1.0;
    tau_beta1_sq   = 1.0;
    tau_beta2_sq   = 1.0;
    tau_beta3_sq   = 1.0;
    sigma0_sq      = 1.0;
  }
  
  // -------- storage dims --------
  const int n_save = (n_iter > burnin) ? ( (n_iter - burnin) / thin ) : 0;
  
  NumericMatrix s_alpha(n_save, n);
  NumericMatrix s_beta1(n_save, P1);
  NumericMatrix s_beta2(n_save, P2);
  NumericMatrix s_beta3(n_save, P3);
  
  NumericVector s_log_gamma(n_save), s_log_kappa(n_save);
  NumericVector s_sigma_alpha_sq(n_save), s_tau_beta1_sq(n_save), s_tau_beta2_sq(n_save), s_tau_beta3_sq(n_save), s_sigma0_sq(n_save);
  NumericVector s_loglik(n_save);
  
  // positions as arrays with dim (n_save, n, d) etc
  NumericVector s_a(n_save * n  * d);
  NumericVector s_b1(n_save * P1 * d);
  NumericVector s_b2(n_save * P2 * d);
  NumericVector s_b3(n_save * P3 * d);
  
  // compatibility outputs (z,a1,a2,a3) = a
  NumericVector s_z (n_save * n * d);
  NumericVector s_a1(n_save * n * d);
  NumericVector s_a2(n_save * n * d);
  NumericVector s_a3(n_save * n * d);
  
  s_a.attr("dim")  = IntegerVector::create(n_save, n,  d);
  s_b1.attr("dim") = IntegerVector::create(n_save, P1, d);
  s_b2.attr("dim") = IntegerVector::create(n_save, P2, d);
  s_b3.attr("dim") = IntegerVector::create(n_save, P3, d);
  
  s_z.attr("dim")  = IntegerVector::create(n_save, n, d);
  s_a1.attr("dim") = IntegerVector::create(n_save, n, d);
  s_a2.attr("dim") = IntegerVector::create(n_save, n, d);
  s_a3.attr("dim") = IntegerVector::create(n_save, n, d);
  
  // -------- acceptance counters --------
  arma::vec acc_alpha(n, arma::fill::zeros);
  arma::vec acc_beta1(P1, arma::fill::zeros);
  arma::vec acc_beta2(P2, arma::fill::zeros);
  arma::vec acc_beta3(P3, arma::fill::zeros);
  double acc_log_gamma = 0.0;
  double acc_log_kappa = 0.0;
  arma::vec acc_a(n, arma::fill::zeros);
  arma::vec acc_b1(P1, arma::fill::zeros);
  arma::vec acc_b2(P2, arma::fill::zeros);
  arma::vec acc_b3(P3, arma::fill::zeros);
  
  // -------- distance matrices --------
  arma::mat D1 = dist_mat(a, b1);
  arma::mat D2 = dist_mat(a, b2);
  arma::mat D3 = dist_mat(a, b3);
  
  int save_idx = 0;
  
  // ==========================
  // MCMC
  // ==========================
  for(int iter=1; iter<=n_iter; iter++){
    if(verbose && (iter % 500 == 0)){
      Rcout << "Iter: " << iter << " / " << n_iter << "\n";
    }
    
    // 1) alpha_i (shared): MH using all layers
    for(int i=0;i<n;i++){
      // current ll
      double ll1=0.0, ll2=0.0, ll3=0.0;
      
      // layer1
      for(int j=0;j<P1;j++){
        const double eta = alpha(i) + beta1(j) - gamma * D1(i,j);
        ll1 += Y_bin(i,j)*eta - log1pexp(eta);
      }
      // layer2
      for(int j=0;j<P2;j++){
        const double mu = alpha(i) + beta2(j) - gamma * D2(i,j);
        const double r  = Z_con(i,j) - mu;
        ll2 += -0.5 * (r*r) / sigma0_sq;
      }
      // layer3
      const double size = 1.0 / kappa;
      for(int j=0;j<P3;j++){
        const double mu = std::exp(alpha(i) + beta3(j) - gamma * D3(i,j));
        ll3 += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
      }
      
      const double lp_cur = -0.5 * (alpha(i)*alpha(i)) / sigma_alpha_sq;
      const double logpost_cur = ll1 + ll2 + ll3 + lp_cur;
      
      // proposal
      const double a_prop = R::rnorm(alpha(i), ps_alpha);
      
      double ll1p=0.0, ll2p=0.0, ll3p=0.0;
      for(int j=0;j<P1;j++){
        const double eta = a_prop + beta1(j) - gamma * D1(i,j);
        ll1p += Y_bin(i,j)*eta - log1pexp(eta);
      }
      for(int j=0;j<P2;j++){
        const double mu = a_prop + beta2(j) - gamma * D2(i,j);
        const double r  = Z_con(i,j) - mu;
        ll2p += -0.5 * (r*r) / sigma0_sq;
      }
      for(int j=0;j<P3;j++){
        const double mu = std::exp(a_prop + beta3(j) - gamma * D3(i,j));
        ll3p += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
      }
      
      const double lp_p = -0.5 * (a_prop*a_prop) / sigma_alpha_sq;
      const double logpost_p = ll1p + ll2p + ll3p + lp_p;
      
      const double log_acc = logpost_p - logpost_cur;
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        alpha(i) = a_prop;
        acc_alpha(i) += 1.0;
      }
    }
    
    // 2) beta^(l): MH
    // beta1
    for(int j=0;j<P1;j++){
      double ll_cur=0.0;
      for(int i=0;i<n;i++){
        const double eta = alpha(i) + beta1(j) - gamma * D1(i,j);
        ll_cur += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_cur = -0.5 * (beta1(j)*beta1(j)) / tau_beta1_sq;
      
      const double b_prop = R::rnorm(beta1(j), ps_beta1);
      double ll_p=0.0;
      for(int i=0;i<n;i++){
        const double eta = alpha(i) + b_prop - gamma * D1(i,j);
        ll_p += Y_bin(i,j)*eta - log1pexp(eta);
      }
      const double lp_p = -0.5 * (b_prop*b_prop) / tau_beta1_sq;
      
      const double log_acc = (ll_p + lp_p) - (ll_cur + lp_cur);
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        beta1(j) = b_prop;
        acc_beta1(j) += 1.0;
      }
    }
    
    // beta2
    for(int j=0;j<P2;j++){
      double ll_cur=0.0;
      for(int i=0;i<n;i++){
        const double mu = alpha(i) + beta2(j) - gamma * D2(i,j);
        const double r  = Z_con(i,j) - mu;
        ll_cur += -0.5 * (r*r) / sigma0_sq;
      }
      const double lp_cur = -0.5 * (beta2(j)*beta2(j)) / tau_beta2_sq;
      
      const double b_prop = R::rnorm(beta2(j), ps_beta2);
      double ll_p=0.0;
      for(int i=0;i<n;i++){
        const double mu = alpha(i) + b_prop - gamma * D2(i,j);
        const double r  = Z_con(i,j) - mu;
        ll_p += -0.5 * (r*r) / sigma0_sq;
      }
      const double lp_p = -0.5 * (b_prop*b_prop) / tau_beta2_sq;
      
      const double log_acc = (ll_p + lp_p) - (ll_cur + lp_cur);
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        beta2(j) = b_prop;
        acc_beta2(j) += 1.0;
      }
    }
    
    // beta3
    {
      const double size = 1.0 / kappa;
      for(int j=0;j<P3;j++){
        double ll_cur=0.0;
        for(int i=0;i<n;i++){
          const double mu = std::exp(alpha(i) + beta3(j) - gamma * D3(i,j));
          ll_cur += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
        }
        const double lp_cur = -0.5 * (beta3(j)*beta3(j)) / tau_beta3_sq;
        
        const double b_prop = R::rnorm(beta3(j), ps_beta3);
        double ll_p=0.0;
        for(int i=0;i<n;i++){
          const double mu = std::exp(alpha(i) + b_prop - gamma * D3(i,j));
          ll_p += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
        }
        const double lp_p = -0.5 * (b_prop*b_prop) / tau_beta3_sq;
        
        const double log_acc = (ll_p + lp_p) - (ll_cur + lp_cur);
        if(std::log(R::runif(0.0,1.0)) < log_acc){
          beta3(j) = b_prop;
          acc_beta3(j) += 1.0;
        }
      }
    }
    
    // 3) log_gamma: MH
    {
      double ll1=0.0, ll2=0.0, ll3=0.0;
      // layer1
      for(int i=0;i<n;i++) for(int j=0;j<P1;j++){
        const double eta = alpha(i) + beta1(j) - gamma*D1(i,j);
        ll1 += Y_bin(i,j)*eta - log1pexp(eta);
      }
      // layer2
      for(int i=0;i<n;i++) for(int j=0;j<P2;j++){
        const double mu = alpha(i) + beta2(j) - gamma*D2(i,j);
        const double r  = Z_con(i,j) - mu;
        ll2 += -0.5 * (r*r) / sigma0_sq;
      }
      // layer3
      {
        const double size = 1.0 / kappa;
        for(int i=0;i<n;i++) for(int j=0;j<P3;j++){
          const double mu = std::exp(alpha(i) + beta3(j) - gamma*D3(i,j));
          ll3 += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
        }
      }
      
      const double lp_cur = R::dnorm(log_gamma, mu_log_gamma, sd_log_gamma, 1);
      const double logpost_cur = ll1 + ll2 + ll3 + lp_cur;
      
      const double log_gamma_prop = R::rnorm(log_gamma, ps_log_gamma);
      const double gamma_prop = std::exp(log_gamma_prop);
      
      double ll1p=0.0, ll2p=0.0, ll3p=0.0;
      for(int i=0;i<n;i++) for(int j=0;j<P1;j++){
        const double eta = alpha(i) + beta1(j) - gamma_prop*D1(i,j);
        ll1p += Y_bin(i,j)*eta - log1pexp(eta);
      }
      for(int i=0;i<n;i++) for(int j=0;j<P2;j++){
        const double mu = alpha(i) + beta2(j) - gamma_prop*D2(i,j);
        const double r  = Z_con(i,j) - mu;
        ll2p += -0.5 * (r*r) / sigma0_sq;
      }
      {
        const double size = 1.0 / kappa;
        for(int i=0;i<n;i++) for(int j=0;j<P3;j++){
          const double mu = std::exp(alpha(i) + beta3(j) - gamma_prop*D3(i,j));
          ll3p += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
        }
      }
      
      const double lp_p = R::dnorm(log_gamma_prop, mu_log_gamma, sd_log_gamma, 1);
      const double logpost_p = ll1p + ll2p + ll3p + lp_p;
      
      const double log_acc = logpost_p - logpost_cur;
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        log_gamma = log_gamma_prop;
        gamma = gamma_prop;
        acc_log_gamma += 1.0;
      }
    }
    
    // 4) shared respondent position a_i : MH, prior N(0,I)
    for(int i=0;i<n;i++){
      arma::rowvec a_cur = a.row(i);
      
      // current ll
      double ll1=0.0, ll2=0.0, ll3=0.0;
      for(int j=0;j<P1;j++){
        const double eta = alpha(i) + beta1(j) - gamma*D1(i,j);
        ll1 += Y_bin(i,j)*eta - log1pexp(eta);
      }
      for(int j=0;j<P2;j++){
        const double mu = alpha(i) + beta2(j) - gamma*D2(i,j);
        const double r  = Z_con(i,j) - mu;
        ll2 += -0.5 * (r*r) / sigma0_sq;
      }
      {
        const double size = 1.0 / kappa;
        for(int j=0;j<P3;j++){
          const double mu = std::exp(alpha(i) + beta3(j) - gamma*D3(i,j));
          ll3 += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
        }
      }
      double lp_cur = -0.5 * arma::dot(a_cur, a_cur);
      double logpost_cur = ll1 + ll2 + ll3 + lp_cur;
      
      // proposal
      arma::rowvec a_prop = a_cur;
      for(int k=0;k<d;k++) a_prop(k) += R::rnorm(0.0, ps_a);
      
      // recompute distances for row i
      arma::rowvec d1p(P1), d2p(P2), d3p(P3);
      for(int j=0;j<P1;j++){
        double acc=0.0;
        for(int k=0;k<d;k++){
          double diff = a_prop(k) - b1(j,k);
          acc += diff*diff;
        }
        d1p(j) = std::sqrt(acc);
      }
      for(int j=0;j<P2;j++){
        double acc=0.0;
        for(int k=0;k<d;k++){
          double diff = a_prop(k) - b2(j,k);
          acc += diff*diff;
        }
        d2p(j) = std::sqrt(acc);
      }
      for(int j=0;j<P3;j++){
        double acc=0.0;
        for(int k=0;k<d;k++){
          double diff = a_prop(k) - b3(j,k);
          acc += diff*diff;
        }
        d3p(j) = std::sqrt(acc);
      }
      
      // proposed ll
      double ll1p=0.0, ll2p=0.0, ll3p=0.0;
      for(int j=0;j<P1;j++){
        const double eta = alpha(i) + beta1(j) - gamma*d1p(j);
        ll1p += Y_bin(i,j)*eta - log1pexp(eta);
      }
      for(int j=0;j<P2;j++){
        const double mu = alpha(i) + beta2(j) - gamma*d2p(j);
        const double r  = Z_con(i,j) - mu;
        ll2p += -0.5 * (r*r) / sigma0_sq;
      }
      {
        const double size = 1.0 / kappa;
        for(int j=0;j<P3;j++){
          const double mu = std::exp(alpha(i) + beta3(j) - gamma*d3p(j));
          ll3p += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
        }
      }
      
      double lp_p = -0.5 * arma::dot(a_prop, a_prop);
      double logpost_p = ll1p + ll2p + ll3p + lp_p;
      
      const double log_acc = logpost_p - logpost_cur;
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        a.row(i) = a_prop;
        D1.row(i) = d1p;
        D2.row(i) = d2p;
        D3.row(i) = d3p;
        acc_a(i) += 1.0;
      }
    }
    
    // 5) item positions b^(l)_j : MH, prior N(0,I)
    // b1
    for(int j=0;j<P1;j++){
      arma::rowvec b_cur = b1.row(j);
      
      double ll_cur=0.0;
      for(int i=0;i<n;i++){
        const double eta = alpha(i) + beta1(j) - gamma*D1(i,j);
        ll_cur += Y_bin(i,j)*eta - log1pexp(eta);
      }
      double lp_cur = -0.5 * arma::dot(b_cur, b_cur);
      
      arma::rowvec b_prop = b_cur;
      for(int k=0;k<d;k++) b_prop(k) += R::rnorm(0.0, ps_b1);
      
      arma::vec d1p(n);
      for(int i=0;i<n;i++){
        double acc=0.0;
        for(int k=0;k<d;k++){
          double diff = a(i,k) - b_prop(k);
          acc += diff*diff;
        }
        d1p(i) = std::sqrt(acc);
      }
      
      double ll_p=0.0;
      for(int i=0;i<n;i++){
        const double eta = alpha(i) + beta1(j) - gamma*d1p(i);
        ll_p += Y_bin(i,j)*eta - log1pexp(eta);
      }
      double lp_p = -0.5 * arma::dot(b_prop, b_prop);
      
      const double log_acc = (ll_p + lp_p) - (ll_cur + lp_cur);
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        b1.row(j) = b_prop;
        D1.col(j) = d1p;
        acc_b1(j) += 1.0;
      }
    }
    
    // b2
    for(int j=0;j<P2;j++){
      arma::rowvec b_cur = b2.row(j);
      
      double ll_cur=0.0;
      for(int i=0;i<n;i++){
        const double mu = alpha(i) + beta2(j) - gamma*D2(i,j);
        const double r  = Z_con(i,j) - mu;
        ll_cur += -0.5 * (r*r) / sigma0_sq;
      }
      double lp_cur = -0.5 * arma::dot(b_cur, b_cur);
      
      arma::rowvec b_prop = b_cur;
      for(int k=0;k<d;k++) b_prop(k) += R::rnorm(0.0, ps_b2);
      
      arma::vec d2p(n);
      for(int i=0;i<n;i++){
        double acc=0.0;
        for(int k=0;k<d;k++){
          double diff = a(i,k) - b_prop(k);
          acc += diff*diff;
        }
        d2p(i) = std::sqrt(acc);
      }
      
      double ll_p=0.0;
      for(int i=0;i<n;i++){
        const double mu = alpha(i) + beta2(j) - gamma*d2p(i);
        const double r  = Z_con(i,j) - mu;
        ll_p += -0.5 * (r*r) / sigma0_sq;
      }
      double lp_p = -0.5 * arma::dot(b_prop, b_prop);
      
      const double log_acc = (ll_p + lp_p) - (ll_cur + lp_cur);
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        b2.row(j) = b_prop;
        D2.col(j) = d2p;
        acc_b2(j) += 1.0;
      }
    }
    
    // b3
    {
      const double size = 1.0 / kappa;
      for(int j=0;j<P3;j++){
        arma::rowvec b_cur = b3.row(j);
        
        double ll_cur=0.0;
        for(int i=0;i<n;i++){
          const double mu = std::exp(alpha(i) + beta3(j) - gamma*D3(i,j));
          ll_cur += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
        }
        double lp_cur = -0.5 * arma::dot(b_cur, b_cur);
        
        arma::rowvec b_prop = b_cur;
        for(int k=0;k<d;k++) b_prop(k) += R::rnorm(0.0, ps_b3);
        
        arma::vec d3p(n);
        for(int i=0;i<n;i++){
          double acc=0.0;
          for(int k=0;k<d;k++){
            double diff = a(i,k) - b_prop(k);
            acc += diff*diff;
          }
          d3p(i) = std::sqrt(acc);
        }
        
        double ll_p=0.0;
        for(int i=0;i<n;i++){
          const double mu = std::exp(alpha(i) + beta3(j) - gamma*d3p(i));
          ll_p += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
        }
        double lp_p = -0.5 * arma::dot(b_prop, b_prop);
        
        const double log_acc = (ll_p + lp_p) - (ll_cur + lp_cur);
        if(std::log(R::runif(0.0,1.0)) < log_acc){
          b3.row(j) = b_prop;
          D3.col(j) = d3p;
          acc_b3(j) += 1.0;
        }
      }
    }
    
    // 6) log_kappa: MH (count layer only)
    {
      // mu3 depends on alpha,beta3,gamma,D3; recompute ll for current/proposal
      const double lp_cur = R::dnorm(log_kappa, mu_log_kappa, sd_log_kappa, 1);
      
      double ll_cur=0.0;
      {
        const double size = 1.0 / kappa;
        for(int i=0;i<n;i++) for(int j=0;j<P3;j++){
          const double mu = std::exp(alpha(i) + beta3(j) - gamma*D3(i,j));
          ll_cur += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
        }
      }
      const double logpost_cur = ll_cur + lp_cur;
      
      const double log_kappa_prop = R::rnorm(log_kappa, ps_log_kappa);
      const double kappa_prop = std::exp(log_kappa_prop);
      
      const double lp_p = R::dnorm(log_kappa_prop, mu_log_kappa, sd_log_kappa, 1);
      double ll_p=0.0;
      {
        const double size_p = 1.0 / kappa_prop;
        for(int i=0;i<n;i++) for(int j=0;j<P3;j++){
          const double mu = std::exp(alpha(i) + beta3(j) - gamma*D3(i,j));
          ll_p += R::dnbinom_mu(Y_cnt(i,j), size_p, mu, 1);
        }
      }
      const double logpost_p = ll_p + lp_p;
      
      const double log_acc = logpost_p - logpost_cur;
      if(std::log(R::runif(0.0,1.0)) < log_acc){
        log_kappa = log_kappa_prop;
        kappa = kappa_prop;
        acc_log_kappa += 1.0;
      }
    }
    
    // 7) Gibbs: sigma0_sq, sigma_alpha_sq, tau_beta*_sq
    // sigma0_sq
    {
      double SSE=0.0;
      for(int i=0;i<n;i++) for(int j=0;j<P2;j++){
        const double mu = alpha(i) + beta2(j) - gamma*D2(i,j);
        const double r  = Z_con(i,j) - mu;
        SSE += r*r;
      }
      const double shape = a_sigma0 + ( (double)n * (double)P2 ) / 2.0;
      const double rate  = b_sigma0 + 0.5 * SSE;
      sigma0_sq = 1.0 / R::rgamma(shape, 1.0/rate); // rgamma(shape, scale)
    }
    
    // sigma_alpha_sq
    {
      double ss=0.0;
      for(int i=0;i<n;i++) ss += alpha(i)*alpha(i);
      const double shape = a_sigma + ((double)n)/2.0;
      const double rate  = b_sigma + 0.5 * ss;
      sigma_alpha_sq = 1.0 / R::rgamma(shape, 1.0/rate);
    }
    
    // tau_beta1_sq
    {
      double ss=0.0;
      for(int j=0;j<P1;j++) ss += beta1(j)*beta1(j);
      const double shape = a_tau1 + ((double)P1)/2.0;
      const double rate  = b_tau1 + 0.5 * ss;
      tau_beta1_sq = 1.0 / R::rgamma(shape, 1.0/rate);
    }
    
    // tau_beta2_sq
    {
      double ss=0.0;
      for(int j=0;j<P2;j++) ss += beta2(j)*beta2(j);
      const double shape = a_tau2 + ((double)P2)/2.0;
      const double rate  = b_tau2 + 0.5 * ss;
      tau_beta2_sq = 1.0 / R::rgamma(shape, 1.0/rate);
    }
    
    // tau_beta3_sq
    {
      double ss=0.0;
      for(int j=0;j<P3;j++) ss += beta3(j)*beta3(j);
      const double shape = a_tau3 + ((double)P3)/2.0;
      const double rate  = b_tau3 + 0.5 * ss;
      tau_beta3_sq = 1.0 / R::rgamma(shape, 1.0/rate);
    }
    
    // 8) save
    if(iter > burnin && ((iter - burnin) % thin == 0)){
      const int ss = save_idx; // 0-based for C++
      // alpha/betas
      for(int i=0;i<n;i++)  s_alpha(ss, i) = alpha(i);
      for(int j=0;j<P1;j++) s_beta1(ss, j) = beta1(j);
      for(int j=0;j<P2;j++) s_beta2(ss, j) = beta2(j);
      for(int j=0;j<P3;j++) s_beta3(ss, j) = beta3(j);
      
      s_log_gamma[ss] = log_gamma;
      s_log_kappa[ss] = log_kappa;
      
      s_sigma_alpha_sq[ss] = sigma_alpha_sq;
      s_tau_beta1_sq[ss]   = tau_beta1_sq;
      s_tau_beta2_sq[ss]   = tau_beta2_sq;
      s_tau_beta3_sq[ss]   = tau_beta3_sq;
      s_sigma0_sq[ss]      = sigma0_sq;
      
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
      for(int j=0;j<P1;j++) for(int k=0;k<d;k++){
        s_b1[idx3P(ss, j, k, n_save, P1)] = b1(j,k);
      }
      for(int j=0;j<P2;j++) for(int k=0;k<d;k++){
        s_b2[idx3P(ss, j, k, n_save, P2)] = b2(j,k);
      }
      for(int j=0;j<P3;j++) for(int k=0;k<d;k++){
        s_b3[idx3P(ss, j, k, n_save, P3)] = b3(j,k);
      }
      
      // loglik (same convention as your R code: Bern + Normal(full const) + NB)
      double ll1=0.0, ll2=0.0, ll3=0.0;
      for(int i=0;i<n;i++) for(int j=0;j<P1;j++){
        const double eta = alpha(i) + beta1(j) - gamma*D1(i,j);
        ll1 += Y_bin(i,j)*eta - log1pexp(eta);
      }
      double SSE=0.0;
      for(int i=0;i<n;i++) for(int j=0;j<P2;j++){
        const double mu = alpha(i) + beta2(j) - gamma*D2(i,j);
        const double r  = Z_con(i,j) - mu;
        SSE += r*r;
      }
      ll2 = -0.5 * SSE / sigma0_sq - 0.5 * (double)n * (double)P2 * std::log(2.0*M_PI*sigma0_sq);
      
      {
        const double size = 1.0 / kappa;
        for(int i=0;i<n;i++) for(int j=0;j<P3;j++){
          const double mu = std::exp(alpha(i) + beta3(j) - gamma*D3(i,j));
          ll3 += R::dnbinom_mu(Y_cnt(i,j), size, mu, 1);
        }
      }
      s_loglik[ss] = ll1 + ll2 + ll3;
      
      save_idx++;
      if(save_idx >= n_save) {
        // safety
      }
    }
  }
  
  // acceptance ratios (divide by n_iter like your R code)
  NumericVector accept_alpha(n), accept_beta1(P1), accept_beta2(P2), accept_beta3(P3);
  NumericVector accept_a(n), accept_b1(P1), accept_b2(P2), accept_b3(P3);
  for(int i=0;i<n;i++){
    accept_alpha[i] = acc_alpha(i) / (double)n_iter;
    accept_a[i]     = acc_a(i)     / (double)n_iter;
  }
  for(int j=0;j<P1;j++){
    accept_beta1[j] = acc_beta1(j) / (double)n_iter;
    accept_b1[j]    = acc_b1(j)    / (double)n_iter;
  }
  for(int j=0;j<P2;j++){
    accept_beta2[j] = acc_beta2(j) / (double)n_iter;
    accept_b2[j]    = acc_b2(j)    / (double)n_iter;
  }
  for(int j=0;j<P3;j++){
    accept_beta3[j] = acc_beta3(j) / (double)n_iter;
    accept_b3[j]    = acc_b3(j)    / (double)n_iter;
  }
  
  Rcpp::List samples = Rcpp::List::create(
    _["alpha"] = s_alpha,
    _["beta1"] = s_beta1,
    _["beta2"] = s_beta2,
    _["beta3"] = s_beta3,
    _["log_gamma"] = s_log_gamma,
    _["log_kappa"] = s_log_kappa,
    _["sigma_alpha_sq"] = s_sigma_alpha_sq,
    _["tau_beta1_sq"]   = s_tau_beta1_sq,
    _["tau_beta2_sq"]   = s_tau_beta2_sq,
    _["tau_beta3_sq"]   = s_tau_beta3_sq,
    _["sigma0_sq"]      = s_sigma0_sq,
    _["a"]  = s_a,
    _["b1"] = s_b1,
    _["b2"] = s_b2,
    _["b3"] = s_b3,
    _["z"]  = s_z,
    _["a1"] = s_a1,
    _["a2"] = s_a2,
    _["a3"] = s_a3,
    _["loglik"] = s_loglik
  );
  
  Rcpp::List accept = Rcpp::List::create(
    _["alpha"] = accept_alpha,
    _["beta1"] = accept_beta1,
    _["beta2"] = accept_beta2,
    _["beta3"] = accept_beta3,
    _["log_gamma"] = acc_log_gamma / (double)n_iter,
    _["log_kappa"] = acc_log_kappa / (double)n_iter,
    _["a"]  = accept_a,
    _["b1"] = accept_b1,
    _["b2"] = accept_b2,
    _["b3"] = accept_b3
  );
  
  return Rcpp::List::create(
    _["samples"] = samples,
    _["accept"]  = accept,
    _["hyper"]   = hyper,
    _["prop_sd"] = prop_sd
  );
}
