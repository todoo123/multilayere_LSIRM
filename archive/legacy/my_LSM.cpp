#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// --- C++ Helper Functions ---

/*
 * 유클리드 거리 (L2) 행렬 계산
 * R의 L2_dist()와 동일
 */
arma::mat L2_dist_cpp(const arma::mat& z) {
  int n = z.n_rows;
  arma::mat D(n, n, arma::fill::zeros);
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double d = arma::norm(z.row(i) - z.row(j), 2); 
      D(i, j) = d;
      D(j, i) = d;
    }
  }
  return D;
}

/*
 * 네트워크의 로그 가능도(log-likelihood) 계산
 * R의 loglik_binary_network()와 동일
 */
double loglik_binary_network_cpp(const arma::mat& Y, const arma::mat& z, double alpha, double beta) {
  arma::mat D = L2_dist_cpp(z);
  arma::mat eta = alpha - beta * D;
  
  // plogis (logistic function)
  arma::mat pi = 1.0 / (1.0 + arma::exp(-eta)); 
  
  int n = Y.n_rows;
  double ll = 0.0;
  
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double p = pi(i, j);
      if (p < 1e-11) p = 1e-11;       // 수치적 안정성
      if (p > (1.0 - 1e-11)) p = 1.0 - 1e-11;
      
      if (Y(i, j) == 1.0) {
        ll += log(p);
      } else {
        ll += log(1.0 - p);
      }
    }
  }
  return ll;
}

/*
 * Z_i의 로그 사전 확률(log prior)
 * R의 log_prior_z_i()와 동일
 */
double log_prior_z_i_cpp(const arma::rowvec& z_i, double sigma2) {
  double log_prior = 0.0;
  double sd = sqrt(sigma2);
  // arma::rowvec를 순회
  for(double val : z_i) {
    log_prior += R::dnorm(val, 0.0, sd, true); // log = true
  }
  return log_prior;
}

/*
 * alpha의 로그 사전 확률
 * R의 log_prior_alpha()와 동일
 */
double log_prior_alpha_cpp(double alpha, double xi, double psi2) {
  return R::dnorm(alpha, xi, sqrt(psi2), true); // log = true
}

/*
 * beta의 로그 사전 확률
 * R의 log_prior_beta()와 동일
 * R의 dgamma(rate = b)는 Rcpp::dgamma(scale = 1.0/b)와 동일
 */
double log_prior_beta_cpp(double beta, double a, double b) {
  if (beta <= 0) return R_NegInf; // 0 또는 음수 값에 대한 방어
  return R::dgamma(beta, a, 1.0 / b, true); // log = true
}

// --- C++ MCMC Update Functions ---

// depreacated: 
// /*
//  * Z_j 업데이트 (Metropolis-Hastings)
//  * R의 update_z_j()와 동일
//  * z와 accepted를 레퍼런스(&)로 받아 직접 수정
//  */
// void update_z_j_cpp(int j, arma::mat& z, const arma::mat& Y, double alpha, 
//                     double sigma2, double prop_z, double beta, int d, 
//                     bool& accepted) {
//   
//   accepted = false; // 기본값
//   arma::rowvec z_j = z.row(j);
//   
//   arma::mat z_prop = z;
//   // arma::randn<rowvec>로 정규분포 난수 생성
//   arma::rowvec prop_z_j = z_j + (arma::randn<arma::rowvec>(d) * sqrt(prop_z));
//   z_prop.row(j) = prop_z_j;
//   
//   // log-posterior 계산 (로그 가능도 + 로그 사전확률)
//   double current_logpost = loglik_binary_network_cpp(Y, z, alpha, beta) + 
//     log_prior_z_i_cpp(z_j, sigma2);
//   double prop_logpost = loglik_binary_network_cpp(Y, z_prop, alpha, beta) + 
//     log_prior_z_i_cpp(prop_z_j, sigma2);
//   
//   double acc_ratio = prop_logpost - current_logpost;
//   
//   if (log(R::runif(0, 1)) < acc_ratio) {
//     z = z_prop; // z를 제안 값으로 업데이트
//     accepted = true;
//   }
//   // 기각되면 z는 변경되지 않음
// }


/*
 * z(j,k) 원소 하나만 Metropolis-Hastings로 업데이트
 *  - j: 노드 인덱스 (0-based)
 *  - k: 좌표 차원 인덱스 (0-based)
 *  - prop_z: 제안분포 분산(여기선 sd = sqrt(prop_z))
 *  - accepted: 수용 여부
 *
 * 구현 단순화를 위해 현재는 전체 로그가능도를 재계산한다.
 * 속도가 필요하면 D/eta를 캐시하고 j행/열만 증분 갱신하는 방식으로 최적화 가능.
 */
void update_z_elem_cpp(int j, int k,
                       arma::mat& z,
                       const arma::mat& Y,
                       double alpha,
                       double sigma2,
                       double prop_z,
                       double beta,
                       bool& accepted) {
  accepted = false;
  
  // 현재 값과 제안값
  double z_jk_old  = z(j, k);
  double z_jk_prop = z_jk_old + R::rnorm(0.0, std::sqrt(prop_z));
  
  // 제안 z 행렬 (원소 한 개만 바꿈)
  arma::mat z_prop = z;
  z_prop(j, k) = z_jk_prop;
  
  // 로그 사후(= 로그가능도 + 로그사전) 비교
  // 가능도는 전체 z 의존 → 안전하게 전체 재계산
  arma::rowvec z_j_old  = z.row(j);
  arma::rowvec z_j_prop = z_prop.row(j);
  
  double current_logpost =
    loglik_binary_network_cpp(Y, z,      alpha, beta) +
    log_prior_z_i_cpp(z_j_old,  sigma2);
  
  double prop_logpost =
    loglik_binary_network_cpp(Y, z_prop, alpha, beta) +
    log_prior_z_i_cpp(z_j_prop, sigma2);
  
  double acc_ratio = prop_logpost - current_logpost;
  
  if (std::log(R::runif(0, 1)) < acc_ratio) {
    z(j, k) = z_jk_prop;  // 수용 시 실제 z 갱신
    accepted = true;
  }
}


/*
 * alpha 업데이트 (Metropolis-Hastings)
 * R의 update_alpha()와 동일
 */
void update_alpha_cpp(double& alpha, const arma::mat& z, const arma::mat& Y, 
                      double xi, double psi2, double prop_alpha, double beta, 
                      bool& accepted) {
  
  accepted = false;
  double ll_curr = loglik_binary_network_cpp(Y, z, alpha, beta) + 
    log_prior_alpha_cpp(alpha, xi, psi2);
  
  double alpha_prop = alpha + R::rnorm(0, prop_alpha);
  
  double ll_prop = loglik_binary_network_cpp(Y, z, alpha_prop, beta) + 
    log_prior_alpha_cpp(alpha_prop, xi, psi2);
  
  double acc_ratio = ll_prop - ll_curr;
  
  if (log(R::runif(0, 1)) < acc_ratio) {
    alpha = alpha_prop;
    accepted = true;
  }
}

/*
 * beta 업데이트 (Metropolis-Hastings, Log-Normal proposal)
 * R의 update_beta()와 동일
 */
void update_beta_cpp(double& beta, const arma::mat& z, const arma::mat& Y, 
                     double alpha, double prop_beta, double a, double b, 
                     bool& accepted) {
  
  accepted = false;
  // log(beta)를 포함한 Jacobian 항 포함
  double curr = loglik_binary_network_cpp(Y, z, alpha, beta) + 
    log_prior_beta_cpp(beta, a, b) + log(beta);
  
  double beta_prop = exp(log(beta) + R::rnorm(0, prop_beta));
  
  // 0 또는 음수 방지
  if (beta_prop <= 0) {
    return; // 기각
  }
  
  double prop = loglik_binary_network_cpp(Y, z, alpha, beta_prop) + 
    log_prior_beta_cpp(beta_prop, a, b) + log(beta_prop);
  
  if (log(R::runif(0, 1)) < (prop - curr)) {
    beta = beta_prop;
    accepted = true;
  }
}

/*
 * xi, psi2 업데이트 (Gibbs Sampler)
 * R의 update_xi_psi2()와 동일
 * (스칼라 alpha에 대한 계층적 구조)
 */
List update_xi_psi2_cpp(double alpha, double xi, double psi2, 
                        double m0 = 0.0, double kappa0 = 0.5, 
                        double alpha0 = 3.0, double beta0 = 0.5) {
  
  double n = 1.0; // R 코드에서 length(alpha) == 1 이므로
  double abar = alpha;
  
  // 1) xi | psi, alpha
  double kappa_n = 1.0 / (n/psi2 + 1.0/kappa0);
  double m_n = kappa_n * (m0/kappa0 + (n * abar)/psi2);
  double xi_new = R::rnorm(m_n, sqrt(kappa_n));
  
  // 2) psi | xi, alpha
  double shape = alpha0 + n/2.0;
  double scale_param = beta0 + 0.5 * pow(alpha - xi_new, 2); // R의 scale
  
  // R의 rgamma(shape, rate=scale_param)는
  // Rcpp::rgamma(shape, scale=1.0/scale_param)와 동일
  double psi2_new = 1.0 / R::rgamma(shape, 1.0/scale_param);
  
  return List::create(Named("xi") = xi_new, Named("psi2") = psi2_new);
}


// --- Main Rcpp Function ---

// [[Rcpp::export]]
List LSM_cpp(arma::mat Y, int iter = 5000, int burnin = 1000, int thinning = 5,
             double sigma2 = 1.0, double a0 = 0.5, double b0 = 0.5, int d = 2,
             double prop_z = 0.05, double prop_alpha = 0.05, double prop_beta = 0.05,
             double alpha_init = 0.0, double beta_init = 1.0, double xi_init = 0.0,
             double psi2_init = 1.0, Nullable<arma::mat> z_init = R_NilValue, 
             bool verbose = true) {
  
  // data validation
  int n = Y.n_rows;
  if (Y.n_cols != n) {
    stop("Y must be a square matrix.");
  }
  if (arma::accu(arma::abs(Y - Y.t())) > 1e-8) {
    stop("Y must be symmetric.");
  }
  
  // init
  arma::mat z;
  if (z_init.isNotNull()) {
    z = as<arma::mat>(z_init);
    if (z.n_rows != n || z.n_cols != d) {
      stop("z_init has incorrect dimensions.");
    }
  } else {
    z = arma::randn(n, d) * sqrt(sigma2);
  }
  
  double alpha = alpha_init;
  double beta = beta_init;
  double xi = xi_init;
  double psi2 = psi2_init;
  
  // storage
  // R의 seq() 로직과 일치하도록 저장 크기 계산
  int ns = static_cast<int>(floor(static_cast<double>(iter - 1 - burnin) / thinning)) + 1;
  if (iter <= burnin) ns = 0;
  
  arma::cube z_save(n, d, ns); // (rows, cols, slices)
  arma::vec alpha_save(ns);
  arma::vec xi_save(ns);
  arma::vec psi2_save(ns);
  arma::vec beta_save(ns);
  arma::vec ll_save(ns);
  
  // acceptance rate
  // arma::vec acc_z(n, arma::fill::zeros);
  arma::mat acc_z_elem(n, d, arma::fill::zeros);
  double acc_alpha = 0.0;
  double acc_beta = 0.0;
  
  int s = 0; // storage index
  
  // MCMC Sampler
  for (int i = 0; i < iter; ++i) {
    
    // update z_i (MH)
    // bool accepted_z_j = false;
    // for (int j = 0; j < n; ++j) {
    //   update_z_j_cpp(j, z, Y, alpha, sigma2, prop_z, beta, d, accepted_z_j);
    //   if (accepted_z_j) {
    //     acc_z(j)++;
    //   }
    // }
    // (변경) z 원소 단위 업데이트
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < d; ++k) {
        bool accepted_elem = false;
        update_z_elem_cpp(j, k, z, Y, alpha, sigma2, prop_z, beta, accepted_elem);
        if (accepted_elem) {
          acc_z_elem(j, k) += 1.0;
        }
      }
    }
    
    // update alpha (MH)
    bool accepted_alpha = false;
    update_alpha_cpp(alpha, z, Y, xi, psi2, prop_alpha, beta, accepted_alpha);
    if (accepted_alpha) {
      acc_alpha++;
    }
    
    // update beta (MH)
    bool accepted_beta = false;
    update_beta_cpp(beta, z, Y, alpha, prop_beta, a0, b0, accepted_beta);
    if (accepted_beta) {
      acc_beta++;
    }
    
    // !!! 원본 R 코드의 버그/테스트 코드 반영 !!!
    // 원본 코드:
    // # test: remove beta
    // beta <- 1
    // acc_beta <- acc_beta + 1L
    beta = 1.0;
    acc_beta++;
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    // Gibbs for (xi, psi2)
    List up_xi_psi2 = update_xi_psi2_cpp(alpha, xi, psi2, 0.0, 0.5, 3.0, 0.5);
    xi = as<double>(up_xi_psi2["xi"]);
    psi2 = as<double>(up_xi_psi2["psi2"]);
    
    // save
    // R의 1-based index (burnin + 1 부터 시작)와 동일한 로직
    if (i >= burnin && (i - burnin) % thinning == 0) {
      z_save.slice(s) = z;
      alpha_save(s) = alpha;
      xi_save(s) = xi;
      psi2_save(s) = psi2;
      beta_save(s) = beta;
      ll_save(s) = loglik_binary_network_cpp(Y, z, alpha, beta);
      s++;
    }
    
    if (verbose && ((i + 1) % 100 == 0)) {
      Rprintf("iteration: %d, alpha: %.4f, beta: %.4f, ll: %.4f\n", 
              i + 1, alpha, beta, loglik_binary_network_cpp(Y, z, alpha, beta));
    }
    
    // 사용자가 Ctrl+C (SIGINT)를 눌렀는지 확인
    Rcpp::checkUserInterrupt();
  }
  
  // Procrustes 매칭은 R에서 수행 (원본 코드와 동일)
  
  // 결과 반환
  return List::create(
    Named("samples") = List::create(
      Named("z") = z_save,
      Named("alpha") = alpha_save,
      Named("xi") = xi_save,
      Named("psi2") = psi2_save,
      Named("beta") = beta_save,
      Named("ll") = ll_save
    ),
    Named("accept") = List::create(
      // Named("z") = acc_z / iter,
      Named("z_elem") = acc_z_elem / iter,     // (n x d) 원소별 수용률
      Named("alpha") = acc_alpha / iter,
      Named("beta") = acc_beta / iter
    ),
    Named("config") = List::create(
      Named("iter") = iter, 
      Named("burnin") = burnin, 
      Named("thinning") = thinning,
      Named("sigma2") = sigma2,
      Named("prop_sd") = List::create(
        Named("z") = prop_z, 
        Named("alpha") = prop_alpha, 
        Named("beta") = prop_beta
      ),
      Named("a0") = a0,
      Named("b0") = b0
    )
  );
}