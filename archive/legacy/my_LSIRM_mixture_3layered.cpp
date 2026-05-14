#include <Rcpp.h>
using namespace Rcpp;

inline double log1pexp_cpp(double x) {
  if (x > 0) {
    return x + std::log1p(std::exp(-x));
  } else {
    return std::log1p(std::exp(x));
  }
}

// [[Rcpp::export]]
Rcpp::List lsirm_global_local_layer3_cpp(
    Rcpp::NumericMatrix Y_bin,   // n x P1, 0/1
    Rcpp::NumericMatrix Y_con,   // n x P2, >0 (continuous)
    Rcpp::NumericMatrix Y_cnt,   // n x P3, 0,1,2,... (count)
    int d,                       // latent dimension
    int n_iter,
    int burnin,
    int thin,
    Rcpp::List hyper,
    Rcpp::List prop_sd,
    Rcpp::Nullable<Rcpp::List> init = R_NilValue,
    bool verbose = true
) {
  using namespace Rcpp;
  
  int n  = Y_bin.nrow();
  int P1 = Y_bin.ncol();
  int P2 = Y_con.ncol();
  int P3 = Y_cnt.ncol();
  
  // Continuous layer (여기서는 그대로 사용)
  NumericMatrix Z_con = clone(Y_con);
  
  // --- Hyperparameters ---
  double a_sigma1 = as<double>(hyper["a_sigma1"]);
  double b_sigma1 = as<double>(hyper["b_sigma1"]);
  double a_sigma2 = as<double>(hyper["a_sigma2"]);
  double b_sigma2 = as<double>(hyper["b_sigma2"]);
  double a_sigma3 = as<double>(hyper["a_sigma3"]);
  double b_sigma3 = as<double>(hyper["b_sigma3"]);
  
  double a_tau1 = as<double>(hyper["a_tau1"]);
  double b_tau1 = as<double>(hyper["b_tau1"]);
  double a_tau2 = as<double>(hyper["a_tau2"]);
  double b_tau2 = as<double>(hyper["b_tau2"]);
  double a_tau3 = as<double>(hyper["a_tau3"]);
  double b_tau3 = as<double>(hyper["b_tau3"]);
  
  double a_sigma0 = as<double>(hyper["a_sigma0"]);
  double b_sigma0 = as<double>(hyper["b_sigma0"]);
  
  double mu_log_alpha  = as<double>(hyper["mu_log_alpha"]);
  double sd_log_alpha  = as<double>(hyper["sd_log_alpha"]);
  
  double a_sigma_global = as<double>(hyper["a_sigma_global"]);
  double b_sigma_global = as<double>(hyper["b_sigma_global"]);
  
  double mu_log_gamma1 = as<double>(hyper["mu_log_gamma1"]);
  double sd_log_gamma1 = as<double>(hyper["sd_log_gamma1"]);
  double mu_log_gamma2 = as<double>(hyper["mu_log_gamma2"]);
  double sd_log_gamma2 = as<double>(hyper["sd_log_gamma2"]);
  double mu_log_gamma3 = as<double>(hyper["mu_log_gamma3"]);
  double sd_log_gamma3 = as<double>(hyper["sd_log_gamma3"]);
  
  // --- Proposal sds ---
  double ps_alpha1     = as<double>(prop_sd["alpha1"]);
  double ps_beta1      = as<double>(prop_sd["beta1"]);
  double ps_alpha2     = as<double>(prop_sd["alpha2"]);
  double ps_beta2      = as<double>(prop_sd["beta2"]);
  double ps_alpha3     = as<double>(prop_sd["alpha3"]);
  double ps_beta3      = as<double>(prop_sd["beta3"]);
  double ps_log_gamma1 = as<double>(prop_sd["log_gamma1"]);
  double ps_log_gamma2 = as<double>(prop_sd["log_gamma2"]);
  double ps_log_gamma3 = as<double>(prop_sd["log_gamma3"]);
  double ps_log_alpha  = as<double>(prop_sd["log_alpha"]);
  double ps_a1         = as<double>(prop_sd["a1"]);
  double ps_a2         = as<double>(prop_sd["a2"]);
  double ps_a3         = as<double>(prop_sd["a3"]);
  double ps_b1         = as<double>(prop_sd["b1"]);
  double ps_b2         = as<double>(prop_sd["b2"]);
  double ps_b3         = as<double>(prop_sd["b3"]);
  
  // --- Latent parameters ---
  NumericVector alpha1(n), alpha2(n), alpha3(n);
  NumericVector beta1(P1), beta2(P2), beta3(P3);
  double log_gamma1, log_gamma2, log_gamma3;
  double log_alpha;
  
  NumericMatrix a1(n, d), a2(n, d), a3(n, d);
  NumericMatrix b1(P1, d), b2(P2, d), b3(P3, d);
  NumericMatrix z(n, d);
  
  double sigma_alpha1_sq, sigma_alpha2_sq, sigma_alpha3_sq;
  double tau_beta1_sq, tau_beta2_sq, tau_beta3_sq;
  double sigma0_sq, sigma1_sq;
  
  // --- Initialization ---
  if (init.isNull()) {
    // alpha/beta
    for (int i = 0; i < n; ++i) {
      alpha1[i] = R::rnorm(0.0, 0.1);
      alpha2[i] = R::rnorm(0.0, 0.1);
      alpha3[i] = R::rnorm(0.0, 0.1);
    }
    for (int j = 0; j < P1; ++j) beta1[j] = R::rnorm(0.0, 0.1);
    for (int j = 0; j < P2; ++j) beta2[j] = R::rnorm(0.0, 0.1);
    for (int j = 0; j < P3; ++j) beta3[j] = R::rnorm(0.0, 0.1);
    
    log_gamma1 = 0.0;
    log_gamma2 = 0.0;
    log_gamma3 = 0.0;
    log_alpha  = 0.0;
    
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < d; ++k) {
        a1(i,k) = R::rnorm(0.0, 0.5);
        a2(i,k) = R::rnorm(0.0, 0.5);
        a3(i,k) = R::rnorm(0.0, 0.5);
      }
    }
    for (int j = 0; j < P1; ++j)
      for (int k = 0; k < d; ++k)
        b1(j,k) = R::rnorm(0.0, 0.5);
    for (int j = 0; j < P2; ++j)
      for (int k = 0; k < d; ++k)
        b2(j,k) = R::rnorm(0.0, 0.5);
    for (int j = 0; j < P3; ++j)
      for (int k = 0; k < d; ++k)
        b3(j,k) = R::rnorm(0.0, 0.5);
    
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < d; ++k)
        z(i,k) = R::rnorm(0.0, 1.0);
    
    sigma_alpha1_sq = 1.0;
    sigma_alpha2_sq = 1.0;
    sigma_alpha3_sq = 1.0;
    tau_beta1_sq    = 1.0;
    tau_beta2_sq    = 1.0;
    tau_beta3_sq    = 1.0;
    sigma0_sq       = 1.0;
    sigma1_sq       = 0.5;
    
  } else {
    List initL(init);
    alpha1 = as<NumericVector>(initL["alpha1"]);
    beta1  = as<NumericVector>(initL["beta1"]);
    alpha2 = as<NumericVector>(initL["alpha2"]);
    beta2  = as<NumericVector>(initL["beta2"]);
    alpha3 = as<NumericVector>(initL["alpha3"]);
    beta3  = as<NumericVector>(initL["beta3"]);
    
    log_gamma1 = as<double>(initL["log_gamma1"]);
    log_gamma2 = as<double>(initL["log_gamma2"]);
    log_gamma3 = as<double>(initL["log_gamma3"]);
    log_alpha  = as<double>(initL["log_alpha"]);
    
    a1 = as<NumericMatrix>(initL["a1"]);
    a2 = as<NumericMatrix>(initL["a2"]);
    a3 = as<NumericMatrix>(initL["a3"]);
    b1 = as<NumericMatrix>(initL["b1"]);
    b2 = as<NumericMatrix>(initL["b2"]);
    b3 = as<NumericMatrix>(initL["b3"]);
    z  = as<NumericMatrix>(initL["z"]);
    
    sigma_alpha1_sq = as<double>(initL["sigma_alpha1_sq"]);
    sigma_alpha2_sq = as<double>(initL["sigma_alpha2_sq"]);
    sigma_alpha3_sq = as<double>(initL["sigma_alpha3_sq"]);
    tau_beta1_sq    = as<double>(initL["tau_beta1_sq"]);
    tau_beta2_sq    = as<double>(initL["tau_beta2_sq"]);
    tau_beta3_sq    = as<double>(initL["tau_beta3_sq"]);
    sigma0_sq       = as<double>(initL["sigma0_sq"]);
    sigma1_sq       = as<double>(initL["sigma1_sq"]);
  }
  
  double gamma1 = std::exp(log_gamma1);
  double gamma2 = std::exp(log_gamma2);
  double gamma3 = std::exp(log_gamma3);
  double alpha  = std::exp(log_alpha);
  
  // --- Storage ---
  int n_save = (n_iter - burnin) / thin;
  
  NumericVector alpha1_save(n_save * n);
  NumericVector beta1_save(n_save * P1);
  NumericVector alpha2_save(n_save * n);
  NumericVector beta2_save(n_save * P2);
  NumericVector alpha3_save(n_save * n);
  NumericVector beta3_save(n_save * P3);
  
  alpha1_save.attr("dim") = IntegerVector::create(n_save, n);
  beta1_save.attr("dim")  = IntegerVector::create(n_save, P1);
  alpha2_save.attr("dim") = IntegerVector::create(n_save, n);
  beta2_save.attr("dim")  = IntegerVector::create(n_save, P2);
  alpha3_save.attr("dim") = IntegerVector::create(n_save, n);
  beta3_save.attr("dim")  = IntegerVector::create(n_save, P3);
  
  NumericVector log_gamma1_save(n_save);
  NumericVector log_gamma2_save(n_save);
  NumericVector log_gamma3_save(n_save);
  
  NumericVector sigma_alpha1_sq_save(n_save);
  NumericVector sigma_alpha2_sq_save(n_save);
  NumericVector sigma_alpha3_sq_save(n_save);
  NumericVector tau_beta1_sq_save(n_save);
  NumericVector tau_beta2_sq_save(n_save);
  NumericVector tau_beta3_sq_save(n_save);
  NumericVector sigma0_sq_save(n_save);
  NumericVector sigma1_sq_save(n_save);
  NumericVector log_alpha_save(n_save);
  
  NumericVector a1_save(n_save * n * d);
  NumericVector a2_save(n_save * n * d);
  NumericVector a3_save(n_save * n * d);
  NumericVector b1_save(n_save * P1 * d);
  NumericVector b2_save(n_save * P2 * d);
  NumericVector b3_save(n_save * P3 * d);
  NumericVector z_save(n_save * n * d);
  
  a1_save.attr("dim") = IntegerVector::create(n_save, n, d);
  a2_save.attr("dim") = IntegerVector::create(n_save, n, d);
  a3_save.attr("dim") = IntegerVector::create(n_save, n, d);
  b1_save.attr("dim") = IntegerVector::create(n_save, P1, d);
  b2_save.attr("dim") = IntegerVector::create(n_save, P2, d);
  b3_save.attr("dim") = IntegerVector::create(n_save, P3, d);
  z_save.attr("dim")  = IntegerVector::create(n_save, n, d);
  
  NumericVector loglik_save(n_save);
  
  // --- Accept counters ---
  NumericVector acc_alpha1(n), acc_beta1(P1);
  NumericVector acc_alpha2(n), acc_beta2(P2);
  NumericVector acc_alpha3(n), acc_beta3(P3);
  double acc_log_gamma1 = 0.0, acc_log_gamma2 = 0.0, acc_log_gamma3 = 0.0;
  double acc_log_alpha  = 0.0;
  NumericVector acc_a1(n), acc_a2(n), acc_a3(n);
  NumericVector acc_b1(P1), acc_b2(P2), acc_b3(P3);
  
  // constants
  const double PI = 3.14159265358979323846;
  const double LOG2PI = std::log(2.0 * PI);
  
  int save_idx = 0;
  
  // ====================
  // MCMC
  // ====================
  for (int iter = 1; iter <= n_iter; ++iter) {
    if (verbose && iter % 500 == 0) {
      Rcpp::Rcout << "Iter: " << iter << " / " << n_iter << std::endl;
    }
    
    // -------------------------------
    // 1. Binary layer (ℓ = 1)
    // -------------------------------
    NumericMatrix D1(n, P1);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < P1; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
          double diff = a1(i,k) - b1(j,k);
          s += diff * diff;
        }
        D1(i,j) = std::sqrt(s);
      }
    }
    
    // (1) alpha1_i
    for (int i = 0; i < n; ++i) {
      double loglik_current = 0.0;
      for (int j = 0; j < P1; ++j) {
        double eta = alpha1[i] + beta1[j] - gamma1 * D1(i,j);
        double yij = Y_bin(i,j);
        loglik_current += yij * eta - log1pexp_cpp(eta);
      }
      double logprior_current = -0.5 * alpha1[i] * alpha1[i] / sigma_alpha1_sq;
      
      double alpha_prop = R::rnorm(alpha1[i], ps_alpha1);
      double loglik_prop = 0.0;
      for (int j = 0; j < P1; ++j) {
        double eta = alpha_prop + beta1[j] - gamma1 * D1(i,j);
        double yij = Y_bin(i,j);
        loglik_prop += yij * eta - log1pexp_cpp(eta);
      }
      double logprior_prop = -0.5 * alpha_prop * alpha_prop / sigma_alpha1_sq;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        alpha1[i] = alpha_prop;
        acc_alpha1[i] += 1.0;
      }
    }
    
    // (2) beta1_j
    for (int j = 0; j < P1; ++j) {
      double loglik_current = 0.0;
      for (int i = 0; i < n; ++i) {
        double eta = alpha1[i] + beta1[j] - gamma1 * D1(i,j);
        double yij = Y_bin(i,j);
        loglik_current += yij * eta - log1pexp_cpp(eta);
      }
      double logprior_current = -0.5 * beta1[j] * beta1[j] / tau_beta1_sq;
      
      double beta_prop = R::rnorm(beta1[j], ps_beta1);
      double loglik_prop = 0.0;
      for (int i = 0; i < n; ++i) {
        double eta = alpha1[i] + beta_prop - gamma1 * D1(i,j);
        double yij = Y_bin(i,j);
        loglik_prop += yij * eta - log1pexp_cpp(eta);
      }
      double logprior_prop = -0.5 * beta_prop * beta_prop / tau_beta1_sq;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        beta1[j] = beta_prop;
        acc_beta1[j] += 1.0;
      }
    }
    
    // (3) log_gamma1
    {
      double loglik_current = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P1; ++j) {
          double eta = alpha1[i] + beta1[j] - gamma1 * D1(i,j);
          double yij = Y_bin(i,j);
          loglik_current += yij * eta - log1pexp_cpp(eta);
        }
      }
      double logprior_current = R::dnorm4(log_gamma1, mu_log_gamma1, sd_log_gamma1, 1);
      
      double log_gamma_prop = R::rnorm(log_gamma1, ps_log_gamma1);
      double gamma_prop     = std::exp(log_gamma_prop);
      double loglik_prop    = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P1; ++j) {
          double eta = alpha1[i] + beta1[j] - gamma_prop * D1(i,j);
          double yij = Y_bin(i,j);
          loglik_prop += yij * eta - log1pexp_cpp(eta);
        }
      }
      double logprior_prop = R::dnorm4(log_gamma_prop, mu_log_gamma1, sd_log_gamma1, 1);
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        log_gamma1 = log_gamma_prop;
        gamma1     = gamma_prop;
        acc_log_gamma1 += 1.0;
      }
    }
    
    // (4) a1_i
    for (int i = 0; i < n; ++i) {
      double loglik_current = 0.0;
      for (int j = 0; j < P1; ++j) {
        double eta = alpha1[i] + beta1[j] - gamma1 * D1(i,j);
        double yij = Y_bin(i,j);
        loglik_current += yij * eta - log1pexp_cpp(eta);
      }
      double logprior_current = 0.0;
      for (int k = 0; k < d; ++k) {
        double diff = a1(i,k) - z(i,k);
        logprior_current += diff * diff;
      }
      logprior_current *= -0.5 / sigma1_sq;
      
      NumericVector a_prop(d);
      for (int k = 0; k < d; ++k) {
        a_prop[k] = a1(i,k) + R::rnorm(0.0, ps_a1);
      }
      
      std::vector<double> d_prop(P1);
      for (int j = 0; j < P1; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
          double diff = a_prop[k] - b1(j,k);
          s += diff * diff;
        }
        d_prop[j] = std::sqrt(s);
      }
      
      double loglik_prop = 0.0;
      for (int j = 0; j < P1; ++j) {
        double eta = alpha1[i] + beta1[j] - gamma1 * d_prop[j];
        double yij = Y_bin(i,j);
        loglik_prop += yij * eta - log1pexp_cpp(eta);
      }
      double logprior_prop = 0.0;
      for (int k = 0; k < d; ++k) {
        double diff = a_prop[k] - z(i,k);
        logprior_prop += diff * diff;
      }
      logprior_prop *= -0.5 / sigma1_sq;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        for (int k = 0; k < d; ++k) {
          a1(i,k) = a_prop[k];
        }
        for (int j = 0; j < P1; ++j) {
          D1(i,j) = d_prop[j];
        }
        acc_a1[i] += 1.0;
      }
    }
    
    // (5) b1_j
    for (int j = 0; j < P1; ++j) {
      double loglik_current = 0.0;
      for (int i = 0; i < n; ++i) {
        double eta = alpha1[i] + beta1[j] - gamma1 * D1(i,j);
        double yij = Y_bin(i,j);
        loglik_current += yij * eta - log1pexp_cpp(eta);
      }
      double logprior_current = 0.0;
      for (int k = 0; k < d; ++k) {
        logprior_current += b1(j,k) * b1(j,k);
      }
      logprior_current *= -0.5;
      
      NumericVector b_prop(d);
      for (int k = 0; k < d; ++k) {
        b_prop[k] = b1(j,k) + R::rnorm(0.0, ps_b1);
      }
      
      std::vector<double> d_prop(n);
      for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
          double diff = a1(i,k) - b_prop[k];
          s += diff * diff;
        }
        d_prop[i] = std::sqrt(s);
      }
      
      double loglik_prop = 0.0;
      for (int i = 0; i < n; ++i) {
        double eta = alpha1[i] + beta1[j] - gamma1 * d_prop[i];
        double yij = Y_bin(i,j);
        loglik_prop += yij * eta - log1pexp_cpp(eta);
      }
      double logprior_prop = 0.0;
      for (int k = 0; k < d; ++k) {
        logprior_prop += b_prop[k] * b_prop[k];
      }
      logprior_prop *= -0.5;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        for (int k = 0; k < d; ++k) {
          b1(j,k) = b_prop[k];
        }
        for (int i = 0; i < n; ++i) {
          D1(i,j) = d_prop[i];
        }
        acc_b1[j] += 1.0;
      }
    }
    
    // (6) variance params binary
    {
      double shape = a_sigma1 + n / 2.0;
      double rate  = b_sigma1 + 0.5 * sum(alpha1 * alpha1);
      sigma_alpha1_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
      
      shape = a_tau1 + P1 / 2.0;
      rate  = b_tau1 + 0.5 * sum(beta1 * beta1);
      tau_beta1_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
    }
    
    // -------------------------------
    // 2. Continuous layer (ℓ = 2)
    // -------------------------------
    NumericMatrix D2(n, P2);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < P2; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
          double diff = a2(i,k) - b2(j,k);
          s += diff * diff;
        }
        D2(i,j) = std::sqrt(s);
      }
    }
    
    // (1) alpha2_i
    for (int i = 0; i < n; ++i) {
      double loglik_current = 0.0;
      for (int j = 0; j < P2; ++j) {
        double mu = alpha2[i] + beta2[j] - gamma2 * D2(i,j);
        double resid = Z_con(i,j) - mu;
        loglik_current += -0.5 * (resid * resid) / sigma0_sq;
      }
      double logprior_current = -0.5 * alpha2[i] * alpha2[i] / sigma_alpha2_sq;
      
      double alpha_prop = R::rnorm(alpha2[i], ps_alpha2);
      double loglik_prop = 0.0;
      for (int j = 0; j < P2; ++j) {
        double mu = alpha_prop + beta2[j] - gamma2 * D2(i,j);
        double resid = Z_con(i,j) - mu;
        loglik_prop += -0.5 * (resid * resid) / sigma0_sq;
      }
      double logprior_prop = -0.5 * alpha_prop * alpha_prop / sigma_alpha2_sq;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        alpha2[i] = alpha_prop;
        acc_alpha2[i] += 1.0;
      }
    }
    
    // (2) beta2_j
    for (int j = 0; j < P2; ++j) {
      double loglik_current = 0.0;
      for (int i = 0; i < n; ++i) {
        double mu = alpha2[i] + beta2[j] - gamma2 * D2(i,j);
        double resid = Z_con(i,j) - mu;
        loglik_current += -0.5 * (resid * resid) / sigma0_sq;
      }
      double logprior_current = -0.5 * beta2[j] * beta2[j] / tau_beta2_sq;
      
      double beta_prop = R::rnorm(beta2[j], ps_beta2);
      double loglik_prop = 0.0;
      for (int i = 0; i < n; ++i) {
        double mu = alpha2[i] + beta_prop - gamma2 * D2(i,j);
        double resid = Z_con(i,j) - mu;
        loglik_prop += -0.5 * (resid * resid) / sigma0_sq;
      }
      double logprior_prop = -0.5 * beta_prop * beta_prop / tau_beta2_sq;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        beta2[j] = beta_prop;
        acc_beta2[j] += 1.0;
      }
    }
    
    // (3) log_gamma2
    {
      double loglik_current = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P2; ++j) {
          double mu = alpha2[i] + beta2[j] - gamma2 * D2(i,j);
          double resid = Z_con(i,j) - mu;
          loglik_current += -0.5 * (resid * resid) / sigma0_sq;
        }
      }
      double logprior_current = R::dnorm4(log_gamma2, mu_log_gamma2, sd_log_gamma2, 1);
      
      double log_gamma_prop = R::rnorm(log_gamma2, ps_log_gamma2);
      double gamma_prop     = std::exp(log_gamma_prop);
      double loglik_prop    = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P2; ++j) {
          double mu = alpha2[i] + beta2[j] - gamma_prop * D2(i,j);
          double resid = Z_con(i,j) - mu;
          loglik_prop += -0.5 * (resid * resid) / sigma0_sq;
        }
      }
      double logprior_prop = R::dnorm4(log_gamma_prop, mu_log_gamma2, sd_log_gamma2, 1);
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        log_gamma2 = log_gamma_prop;
        gamma2     = gamma_prop;
        acc_log_gamma2 += 1.0;
      }
    }
    
    // (4) a2_i
    for (int i = 0; i < n; ++i) {
      double loglik_current = 0.0;
      for (int j = 0; j < P2; ++j) {
        double mu = alpha2[i] + beta2[j] - gamma2 * D2(i,j);
        double resid = Z_con(i,j) - mu;
        loglik_current += -0.5 * (resid * resid) / sigma0_sq;
      }
      double logprior_current = 0.0;
      for (int k = 0; k < d; ++k) {
        double diff = a2(i,k) - z(i,k);
        logprior_current += diff * diff;
      }
      logprior_current *= -0.5 / sigma1_sq;
      
      NumericVector a_prop(d);
      for (int k = 0; k < d; ++k) {
        a_prop[k] = a2(i,k) + R::rnorm(0.0, ps_a2);
      }
      
      std::vector<double> d_prop(P2);
      for (int j = 0; j < P2; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
          double diff = a_prop[k] - b2(j,k);
          s += diff * diff;
        }
        d_prop[j] = std::sqrt(s);
      }
      
      double loglik_prop = 0.0;
      for (int j = 0; j < P2; ++j) {
        double mu = alpha2[i] + beta2[j] - gamma2 * d_prop[j];
        double resid = Z_con(i,j) - mu;
        loglik_prop += -0.5 * (resid * resid) / sigma0_sq;
      }
      double logprior_prop = 0.0;
      for (int k = 0; k < d; ++k) {
        double diff = a_prop[k] - z(i,k);
        logprior_prop += diff * diff;
      }
      logprior_prop *= -0.5 / sigma1_sq;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        for (int k = 0; k < d; ++k) {
          a2(i,k) = a_prop[k];
        }
        for (int j = 0; j < P2; ++j) {
          D2(i,j) = d_prop[j];
        }
        acc_a2[i] += 1.0;
      }
    }
    
    // (5) b2_j
    for (int j = 0; j < P2; ++j) {
      double loglik_current = 0.0;
      for (int i = 0; i < n; ++i) {
        double mu = alpha2[i] + beta2[j] - gamma2 * D2(i,j);
        double resid = Z_con(i,j) - mu;
        loglik_current += -0.5 * (resid * resid) / sigma0_sq;
      }
      double logprior_current = 0.0;
      for (int k = 0; k < d; ++k) {
        logprior_current += b2(j,k) * b2(j,k);
      }
      logprior_current *= -0.5;
      
      NumericVector b_prop(d);
      for (int k = 0; k < d; ++k) {
        b_prop[k] = b2(j,k) + R::rnorm(0.0, ps_b2);
      }
      
      std::vector<double> d_prop(n);
      for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
          double diff = a2(i,k) - b_prop[k];
          s += diff * diff;
        }
        d_prop[i] = std::sqrt(s);
      }
      
      double loglik_prop = 0.0;
      for (int i = 0; i < n; ++i) {
        double mu = alpha2[i] + beta2[j] - gamma2 * d_prop[i];
        double resid = Z_con(i,j) - mu;
        loglik_prop += -0.5 * (resid * resid) / sigma0_sq;
      }
      double logprior_prop = 0.0;
      for (int k = 0; k < d; ++k) {
        logprior_prop += b_prop[k] * b_prop[k];
      }
      logprior_prop *= -0.5;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        for (int k = 0; k < d; ++k) {
          b2(j,k) = b_prop[k];
        }
        for (int i = 0; i < n; ++i) {
          D2(i,j) = d_prop[i];
        }
        acc_b2[j] += 1.0;
      }
    }
    
    // (6) sigma0^2
    {
      double SSE = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P2; ++j) {
          double mu = alpha2[i] + beta2[j] - gamma2 * D2(i,j);
          double resid = Z_con(i,j) - mu;
          SSE += resid * resid;
        }
      }
      double shape = a_sigma0 + n * P2 / 2.0;
      double rate  = b_sigma0 + 0.5 * SSE;
      sigma0_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
    }
    
    // (7) variance params continuous
    {
      double shape = a_sigma2 + n / 2.0;
      double rate  = b_sigma2 + 0.5 * sum(alpha2 * alpha2);
      sigma_alpha2_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
      
      shape = a_tau2 + P2 / 2.0;
      rate  = b_tau2 + 0.5 * sum(beta2 * beta2);
      tau_beta2_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
    }
    
    // -------------------------------
    // 3. Count layer (ℓ = 3)
    // -------------------------------
    NumericMatrix D3(n, P3);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < P3; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
          double diff = a3(i,k) - b3(j,k);
          s += diff * diff;
        }
        D3(i,j) = std::sqrt(s);
      }
    }
    
    // (1) alpha3_i
    for (int i = 0; i < n; ++i) {
      double loglik_current = 0.0;
      for (int j = 0; j < P3; ++j) {
        double eta = alpha3[i] + beta3[j] - gamma3 * D3(i,j);
        double mu  = std::exp(eta);
        double yij = Y_cnt(i,j);
        loglik_current += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
      }
      double logprior_current = -0.5 * alpha3[i] * alpha3[i] / sigma_alpha3_sq;
      
      double alpha_prop = R::rnorm(alpha3[i], ps_alpha3);
      double loglik_prop = 0.0;
      for (int j = 0; j < P3; ++j) {
        double eta = alpha_prop + beta3[j] - gamma3 * D3(i,j);
        double mu  = std::exp(eta);
        double yij = Y_cnt(i,j);
        loglik_prop += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
      }
      double logprior_prop = -0.5 * alpha_prop * alpha_prop / sigma_alpha3_sq;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        alpha3[i] = alpha_prop;
        acc_alpha3[i] += 1.0;
      }
    }
    
    // (2) beta3_j
    for (int j = 0; j < P3; ++j) {
      double loglik_current = 0.0;
      for (int i = 0; i < n; ++i) {
        double eta = alpha3[i] + beta3[j] - gamma3 * D3(i,j);
        double mu  = std::exp(eta);
        double yij = Y_cnt(i,j);
        loglik_current += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
      }
      double logprior_current = -0.5 * beta3[j] * beta3[j] / tau_beta3_sq;
      
      double beta_prop = R::rnorm(beta3[j], ps_beta3);
      double loglik_prop = 0.0;
      for (int i = 0; i < n; ++i) {
        double eta = alpha3[i] + beta_prop - gamma3 * D3(i,j);
        double mu  = std::exp(eta);
        double yij = Y_cnt(i,j);
        loglik_prop += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
      }
      double logprior_prop = -0.5 * beta_prop * beta_prop / tau_beta3_sq;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        beta3[j] = beta_prop;
        acc_beta3[j] += 1.0;
      }
    }
    
    // (3) log_gamma3
    {
      double loglik_current = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P3; ++j) {
          double eta = alpha3[i] + beta3[j] - gamma3 * D3(i,j);
          double mu  = std::exp(eta);
          double yij = Y_cnt(i,j);
          loglik_current += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
        }
      }
      double logprior_current = R::dnorm4(log_gamma3, mu_log_gamma3, sd_log_gamma3, 1);
      
      double log_gamma_prop = R::rnorm(log_gamma3, ps_log_gamma3);
      double gamma_prop     = std::exp(log_gamma_prop);
      double loglik_prop    = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P3; ++j) {
          double eta = alpha3[i] + beta3[j] - gamma_prop * D3(i,j);
          double mu  = std::exp(eta);
          double yij = Y_cnt(i,j);
          loglik_prop += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
        }
      }
      double logprior_prop = R::dnorm4(log_gamma_prop, mu_log_gamma3, sd_log_gamma3, 1);
      
      double log_acc = (loglik_prop + logprior_prop) -
        (loglik_current + logprior_current) + (log_gamma3 - log_gamma_prop);
      
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        log_gamma3 = log_gamma_prop;
        gamma3     = gamma_prop;
        acc_log_gamma3 += 1.0;
      }
    }
    
    // (4) a3_i
    for (int i = 0; i < n; ++i) {
      double loglik_current = 0.0;
      for (int j = 0; j < P3; ++j) {
        double eta = alpha3[i] + beta3[j] - gamma3 * D3(i,j);
        double mu  = std::exp(eta);
        double yij = Y_cnt(i,j);
        loglik_current += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
      }
      double logprior_current = 0.0;
      for (int k = 0; k < d; ++k) {
        double diff = a3(i,k) - z(i,k);
        logprior_current += diff * diff;
      }
      logprior_current *= -0.5 / sigma1_sq;
      
      NumericVector a_prop(d);
      for (int k = 0; k < d; ++k) {
        a_prop[k] = a3(i,k) + R::rnorm(0.0, ps_a3);
      }
      
      std::vector<double> d_prop(P3);
      for (int j = 0; j < P3; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
          double diff = a_prop[k] - b3(j,k);
          s += diff * diff;
        }
        d_prop[j] = std::sqrt(s);
      }
      
      double loglik_prop = 0.0;
      for (int j = 0; j < P3; ++j) {
        double eta = alpha3[i] + beta3[j] - gamma3 * d_prop[j];
        double mu  = std::exp(eta);
        double yij = Y_cnt(i,j);
        loglik_prop += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
      }
      double logprior_prop = 0.0;
      for (int k = 0; k < d; ++k) {
        double diff = a_prop[k] - z(i,k);
        logprior_prop += diff * diff;
      }
      logprior_prop *= -0.5 / sigma1_sq;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        for (int k = 0; k < d; ++k) {
          a3(i,k) = a_prop[k];
        }
        for (int j = 0; j < P3; ++j) {
          D3(i,j) = d_prop[j];
        }
        acc_a3[i] += 1.0;
      }
    }
    
    // (5) b3_j
    for (int j = 0; j < P3; ++j) {
      double loglik_current = 0.0;
      for (int i = 0; i < n; ++i) {
        double eta = alpha3[i] + beta3[j] - gamma3 * D3(i,j);
        double mu  = std::exp(eta);
        double yij = Y_cnt(i,j);
        loglik_current += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
      }
      double logprior_current = 0.0;
      for (int k = 0; k < d; ++k) {
        logprior_current += b3(j,k) * b3(j,k);
      }
      logprior_current *= -0.5;
      
      NumericVector b_prop(d);
      for (int k = 0; k < d; ++k) {
        b_prop[k] = b3(j,k) + R::rnorm(0.0, ps_b3);
      }
      
      std::vector<double> d_prop(n);
      for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
          double diff = a3(i,k) - b_prop[k];
          s += diff * diff;
        }
        d_prop[i] = std::sqrt(s);
      }
      
      double loglik_prop = 0.0;
      for (int i = 0; i < n; ++i) {
        double eta = alpha3[i] + beta3[j] - gamma3 * d_prop[i];
        double mu  = std::exp(eta);
        double yij = Y_cnt(i,j);
        loglik_prop += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
      }
      double logprior_prop = 0.0;
      for (int k = 0; k < d; ++k) {
        logprior_prop += b_prop[k] * b_prop[k];
      }
      logprior_prop *= -0.5;
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        for (int k = 0; k < d; ++k) {
          b3(j,k) = b_prop[k];
        }
        for (int i = 0; i < n; ++i) {
          D3(i,j) = d_prop[i];
        }
        acc_b3[j] += 1.0;
      }
    }
    
    // (6) alpha (dispersion)
    {
      double loglik_current = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P3; ++j) {
          double eta = alpha3[i] + beta3[j] - gamma3 * D3(i,j);
          double mu  = std::exp(eta);
          double yij = Y_cnt(i,j);
          loglik_current += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
        }
      }
      double logprior_current = R::dnorm4(log_alpha, mu_log_alpha, sd_log_alpha, 1);
      
      double log_alpha_prop = R::rnorm(log_alpha, ps_log_alpha);
      double alpha_prop     = std::exp(log_alpha_prop);
      double loglik_prop    = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P3; ++j) {
          double eta = alpha3[i] + beta3[j] - gamma3 * D3(i,j);
          double mu  = std::exp(eta);
          double yij = Y_cnt(i,j);
          loglik_prop += R::dnbinom_mu(yij, 1.0/alpha_prop, mu, 1);
        }
      }
      double logprior_prop = R::dnorm4(log_alpha_prop, mu_log_alpha, sd_log_alpha, 1);
      
      double log_acc = (loglik_prop + logprior_prop) - (loglik_current + logprior_current);
      if (std::log(R::runif(0.0, 1.0)) < log_acc) {
        log_alpha = log_alpha_prop;
        alpha     = alpha_prop;
        acc_log_alpha += 1.0;
      }
    }
    
    // (7) variance params count
    {
      double shape = a_sigma3 + n / 2.0;
      double rate  = b_sigma3 + 0.5 * sum(alpha3 * alpha3);
      sigma_alpha3_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
      
      shape = a_tau3 + P3 / 2.0;
      rate  = b_tau3 + 0.5 * sum(beta3 * beta3);
      tau_beta3_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
    }
    
    // -------------------------------
    // 4. Global positions z, sigma1_sq
    // -------------------------------
    {
      double var_z_post = 1.0 / (1.0 + 3.0 / sigma1_sq);
      double sd_z_post  = std::sqrt(var_z_post);
      
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < d; ++k) {
          double mean_z_post = var_z_post * (a1(i,k) + a2(i,k) + a3(i,k)) / sigma1_sq;
          z(i,k) = R::rnorm(mean_z_post, sd_z_post);
        }
      }
      
      double sse_z = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < d; ++k) {
          double diff1 = a1(i,k) - z(i,k);
          double diff2 = a2(i,k) - z(i,k);
          double diff3 = a3(i,k) - z(i,k);
          sse_z += diff1*diff1 + diff2*diff2 + diff3*diff3;
        }
      }
      double total_obs = n * 3.0 * d;
      double shape = a_sigma_global + total_obs / 2.0;
      double rate  = b_sigma_global + sse_z / 2.0;
      sigma1_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
    }
    
    // -------------------------------
    // Save samples
    // -------------------------------
    if (iter > burnin && ((iter - burnin) % thin == 0)) {
      int si = save_idx; // 0-based
      
      // alphas/betas
      for (int i = 0; i < n; ++i) {
        alpha1_save[si + n_save * i] = alpha1[i];
        alpha2_save[si + n_save * i] = alpha2[i];
        alpha3_save[si + n_save * i] = alpha3[i];
      }
      for (int j = 0; j < P1; ++j)
        beta1_save[si + n_save * j] = beta1[j];
      for (int j = 0; j < P2; ++j)
        beta2_save[si + n_save * j] = beta2[j];
      for (int j = 0; j < P3; ++j)
        beta3_save[si + n_save * j] = beta3[j];
      
      log_gamma1_save[si] = log_gamma1;
      log_gamma2_save[si] = log_gamma2;
      log_gamma3_save[si] = log_gamma3;
      
      sigma_alpha1_sq_save[si] = sigma_alpha1_sq;
      sigma_alpha2_sq_save[si] = sigma_alpha2_sq;
      sigma_alpha3_sq_save[si] = sigma_alpha3_sq;
      tau_beta1_sq_save[si]    = tau_beta1_sq;
      tau_beta2_sq_save[si]    = tau_beta2_sq;
      tau_beta3_sq_save[si]    = tau_beta3_sq;
      sigma0_sq_save[si]       = sigma0_sq;
      sigma1_sq_save[si]       = sigma1_sq;
      log_alpha_save[si]       = log_alpha;
      
      // 3D arrays: (n_save, n, d) etc
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < d; ++k) {
          a1_save[si + n_save * (i + n * k)] = a1(i,k);
          a2_save[si + n_save * (i + n * k)] = a2(i,k);
          a3_save[si + n_save * (i + n * k)] = a3(i,k);
          z_save[ si + n_save * (i + n * k)] = z(i,k);
        }
      }
      for (int j = 0; j < P1; ++j) {
        for (int k = 0; k < d; ++k) {
          b1_save[si + n_save * (j + P1 * k)] = b1(j,k);
        }
      }
      for (int j = 0; j < P2; ++j) {
        for (int k = 0; k < d; ++k) {
          b2_save[si + n_save * (j + P2 * k)] = b2(j,k);
        }
      }
      for (int j = 0; j < P3; ++j) {
        for (int k = 0; k < d; ++k) {
          b3_save[si + n_save * (j + P3 * k)] = b3(j,k);
        }
      }
      
      // log-likelihood
      double loglik_bin = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P1; ++j) {
          double eta = alpha1[i] + beta1[j] - gamma1 * D1(i,j);
          double yij = Y_bin(i,j);
          loglik_bin += yij * eta - log1pexp_cpp(eta);
        }
      }
      
      double loglik_con = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P2; ++j) {
          double mu = alpha2[i] + beta2[j] - gamma2 * D2(i,j);
          double resid = Z_con(i,j) - mu;
          loglik_con += -0.5 * (resid * resid) / sigma0_sq;
        }
      }
      loglik_con += -0.5 * n * P2 * (LOG2PI + std::log(sigma0_sq));
      
      double loglik_cnt = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < P3; ++j) {
          double eta = alpha3[i] + beta3[j] - gamma3 * D3(i,j);
          double mu  = std::exp(eta);
          double yij = Y_cnt(i,j);
          loglik_cnt += R::dnbinom_mu(yij, 1.0/alpha, mu, 1);
        }
      }
      
      loglik_save[si] = loglik_bin + loglik_con + loglik_cnt;
      
      save_idx++;
    }
  } // end iter
  
  // normalize accept
  double nit = static_cast<double>(n_iter);
  for (int i = 0; i < n; ++i) {
    acc_alpha1[i] /= nit;
    acc_alpha2[i] /= nit;
    acc_alpha3[i] /= nit;
    acc_a1[i]     /= nit;
    acc_a2[i]     /= nit;
    acc_a3[i]     /= nit;
  }
  for (int j = 0; j < P1; ++j) {
    acc_beta1[j] /= nit;
    acc_b1[j]    /= nit;
  }
  for (int j = 0; j < P2; ++j) {
    acc_beta2[j] /= nit;
    acc_b2[j]    /= nit;
  }
  for (int j = 0; j < P3; ++j) {
    acc_beta3[j] /= nit;
    acc_b3[j]    /= nit;
  }
  acc_log_gamma1 /= nit;
  acc_log_gamma2 /= nit;
  acc_log_gamma3 /= nit;
  acc_log_alpha  /= nit;
  
  List samples = List::create(
    _["alpha1"] = alpha1_save,
    _["beta1"]  = beta1_save,
    _["alpha2"] = alpha2_save,
    _["beta2"]  = beta2_save,
    _["alpha3"] = alpha3_save,
    _["beta3"]  = beta3_save,
    _["log_gamma1"] = log_gamma1_save,
    _["log_gamma2"] = log_gamma2_save,
    _["log_gamma3"] = log_gamma3_save,
    _["sigma_alpha1_sq"] = sigma_alpha1_sq_save,
    _["sigma_alpha2_sq"] = sigma_alpha2_sq_save,
    _["sigma_alpha3_sq"] = sigma_alpha3_sq_save,
    _["tau_beta1_sq"]    = tau_beta1_sq_save,
    _["tau_beta2_sq"]    = tau_beta2_sq_save,
    _["tau_beta3_sq"]    = tau_beta3_sq_save,
    _["sigma0_sq"]       = sigma0_sq_save,
    _["sigma1_sq"]       = sigma1_sq_save,
    _["log_alpha"]       = log_alpha_save,
    _["a1"] = a1_save,
    _["a2"] = a2_save,
    _["a3"] = a3_save,
    _["b1"] = b1_save,
    _["b2"] = b2_save,
    _["b3"] = b3_save,
    _["z"]  = z_save,
    _["loglik"] = loglik_save
  );
  
  List accept = List::create(
    _["alpha1"]     = acc_alpha1,
    _["beta1"]      = acc_beta1,
    _["alpha2"]     = acc_alpha2,
    _["beta2"]      = acc_beta2,
    _["alpha3"]     = acc_alpha3,
    _["beta3"]      = acc_beta3,
    _["log_gamma1"] = acc_log_gamma1,
    _["log_gamma2"] = acc_log_gamma2,
    _["log_gamma3"] = acc_log_gamma3,
    _["log_alpha"]  = acc_log_alpha,
    _["a1"]         = acc_a1,
    _["a2"]         = acc_a2,
    _["a3"]         = acc_a3,
    _["b1"]         = acc_b1,
    _["b2"]         = acc_b2,
    _["b3"]         = acc_b3
  );
  
  return List::create(
    _["samples"] = samples,
    _["accept"]  = accept,
    _["hyper"]   = hyper,
    _["prop_sd"] = prop_sd
  );
}
