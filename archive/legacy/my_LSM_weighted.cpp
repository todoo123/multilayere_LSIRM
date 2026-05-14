#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// --- C++ Helper Functions ---

/*
 * 유클리드 거리 (L2) 행렬 계산
 * (이전 코드와 동일)
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
 * 연속형(가중치) 네트워크의 로그 가능도(log-likelihood) 계산
 * R의 loglik_conti_network() (dlnorm 사용)와 동일
 */
double loglik_conti_network_cpp(const arma::mat& Y, const arma::mat& z, 
                                double alpha, double beta, double kappa, 
                                double eps = 1e-12) {
  arma::mat D = L2_dist_cpp(z);
  // 사용하는 weight 의 성질에 따라서 부호 조정해야 함.
  arma::mat eta = alpha + beta * D;
  
  int n = Y.n_rows;
  double ll = 0.0;
  double sd_log = sqrt(kappa); // log-normal의 sd (log scale)
  
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double y = Y(i, j);
      if (y < eps) y = eps; // R의 pmax(Y[upper], eps)와 동일
      
      double mu_log = eta(i, j) - kappa / 2.0; // log-normal의 mean (log scale)
      
      // R::dlnorm(x, meanlog, sdlog, log = true)
      ll += R::dlnorm(y, mu_log, sd_log, true);
    }
  }
  return ll;
}

/*
 * Z_i의 로그 사전 확률
 */
double log_prior_z_i_cpp(const arma::rowvec& z_i, double sigma2) {
  double log_prior = 0.0;
  double sd = sqrt(sigma2);
  for(double val : z_i) {
    log_prior += R::dnorm(val, 0.0, sd, true);
  }
  return log_prior;
}

/*
 * alpha의 로그 사전 확률
 */
double log_prior_alpha_cpp(double alpha, double xi, double psi2) {
  return R::dnorm(alpha, xi, sqrt(psi2), true);
}

/*
 * beta의 로그 사전 확률
 */
double log_prior_beta_cpp(double beta, double a, double b) {
  if (beta <= 0) return R_NegInf;
  return R::dgamma(beta, a, 1.0 / b, true);
}

/*
 * kappa의 로그 사전 확률
 * R의 log_prior_kappa()와 동일 (log(kappa) ~ N(a, b))
 */
double log_prior_kappa_cpp(double kappa, double a, double b) {
  if (kappa <= 0) return R_NegInf;
  return R::dnorm(log(kappa), a, sqrt(b), true); // log = true
}

// --- C++ MCMC Update Functions ---

/*
 * deprecated: vector-at-once
 * Z_j 업데이트 (Metropolis-Hastings)
 * R의 update_z_j()와 동일 (연속형 가능도 사용)
 */
// void update_z_j_cpp(int j, arma::mat& z, const arma::mat& Y, double alpha, 
//                     double sigma2, double prop_z, double beta, double kappa, int d, 
//                     bool& accepted) {
//   
//   accepted = false;
//   arma::rowvec z_j = z.row(j);
//   
//   arma::mat z_prop = z;
//   arma::rowvec prop_z_j = z_j + (arma::randn<arma::rowvec>(d) * sqrt(prop_z));
//   z_prop.row(j) = prop_z_j;
//   
//   // 연속형 가능도 함수로 변경
//   double current_logpost = loglik_conti_network_cpp(Y, z, alpha, beta, kappa) + 
//     log_prior_z_i_cpp(z_j, sigma2);
//   double prop_logpost = loglik_conti_network_cpp(Y, z_prop, alpha, beta, kappa) + 
//     log_prior_z_i_cpp(prop_z_j, sigma2);
//   
//   double acc_ratio = prop_logpost - current_logpost;
//   
//   if (log(R::runif(0, 1)) < acc_ratio) {
//     z = z_prop;
//     accepted = true;
//   }
// }

/*
 * z(j,k) 원소 하나만 Metropolis-Hastings로 업데이트
 *  - j: 노드 인덱스 (0-based)
 *  - k: 좌표 차원 인덱스 (0-based)
 *  - prop_z: 제안분포 표준편차^2 (여기선 sd = sqrt(prop_z))
 *  - accepted: 수용 여부 리턴
 */
void update_z_elem_cpp(int j, int k,
                       arma::mat& z,
                       const arma::mat& Y,
                       double alpha,
                       double sigma2,
                       double prop_z,
                       double beta,
                       double kappa,
                       bool& accepted) {
  
  accepted = false;
  
  // 현재 값 보관
  double z_jk_old = z(j, k);
  
  // 제안값 생성 (정규 제안)
  double z_jk_prop = z_jk_old + R::rnorm(0.0, std::sqrt(prop_z));
  
  // 제안 z 행렬 만들기
  arma::mat z_prop = z;
  z_prop(j, k) = z_jk_prop;
  
  // 로그 사후(= 로그가능도 + 로그사전) 비교
  //  - 가능도는 전체 z에 의존하므로 안전하게 전체를 다시 계산 (간단 구현)
  //  - 사전은 z_j 행만 바뀌므로 z_j만 넣어도 무방
  arma::rowvec z_j_old = z.row(j);
  arma::rowvec z_j_prop = z_prop.row(j);
  
  double current_logpost =
    loglik_conti_network_cpp(Y, z,      alpha, beta, kappa) +
    log_prior_z_i_cpp(z_j_old, sigma2);
  
  double prop_logpost =
    loglik_conti_network_cpp(Y, z_prop, alpha, beta, kappa) +
    log_prior_z_i_cpp(z_j_prop, sigma2);
  
  double acc_ratio = prop_logpost - current_logpost;
  
  if (std::log(R::runif(0, 1)) < acc_ratio) {
    z(j, k) = z_jk_prop;  // 수용: 실제 z를 갱신
    accepted = true;
  }
}



/*
 * alpha 업데이트 (Metropolis-Hastings)
 * R의 update_alpha()와 동일 (연속형 가능도 사용)
 */
void update_alpha_cpp(double& alpha, const arma::mat& z, const arma::mat& Y, 
                      double xi, double psi2, double prop_alpha, double beta, double kappa,
                      bool& accepted) {
  
  accepted = false;
  // 연속형 가능도 함수로 변경
  double ll_curr = loglik_conti_network_cpp(Y, z, alpha, beta, kappa) + 
    log_prior_alpha_cpp(alpha, xi, psi2);
  
  double alpha_prop = alpha + R::rnorm(0, prop_alpha);
  
  double ll_prop = loglik_conti_network_cpp(Y, z, alpha_prop, beta, kappa) + 
    log_prior_alpha_cpp(alpha_prop, xi, psi2);
  
  double acc_ratio = ll_prop - ll_curr;
  
  if (log(R::runif(0, 1)) < acc_ratio) {
    alpha = alpha_prop;
    accepted = true;
  }
}

/*
 * beta 업데이트 (Metropolis-Hastings, Log-Normal proposal)
 * R의 update_beta()와 동일 (연속형 가능도 사용)
 */
void update_beta_cpp(double& beta, const arma::mat& z, const arma::mat& Y, 
                     double alpha, double prop_beta, double a, double b, double kappa,
                     bool& accepted) {
  
  accepted = false;
  // 연속형 가능도 함수로 변경
  double curr = loglik_conti_network_cpp(Y, z, alpha, beta, kappa) + 
    log_prior_beta_cpp(beta, a, b) + log(beta);
  
  double beta_prop = exp(log(beta) + R::rnorm(0, prop_beta));
  
  if (beta_prop <= 0) {
    return;
  }
  
  double prop = loglik_conti_network_cpp(Y, z, alpha, beta_prop, kappa) + 
    log_prior_beta_cpp(beta_prop, a, b) + log(beta_prop);
  
  if (log(R::runif(0, 1)) < (prop - curr)) {
    beta = beta_prop;
    accepted = true;
  }
}

/*
 * kappa 업데이트 (Metropolis-Hastings, Log-Normal proposal)
 * R의 update_kappa()와 동일
 */
void update_kappa_cpp(double& kappa, const arma::mat& z, const arma::mat& Y, 
                      double alpha, double beta, double prop_kappa, 
                      double m_kappa, double s_kappa, bool& accepted) {
  
  accepted = false;
  // R 코드와 동일 (Jacobian 항 불필요, log-scale에서 대칭 제안)
  double curr = loglik_conti_network_cpp(Y, z, alpha, beta, kappa) + 
    log_prior_kappa_cpp(kappa, m_kappa, s_kappa);
  
  double kappa_prop = exp(log(kappa) + R::rnorm(0, prop_kappa));
  
  if (kappa_prop <= 0) {
    return;
  }
  
  double prop = loglik_conti_network_cpp(Y, z, alpha, beta, kappa_prop) + 
    log_prior_kappa_cpp(kappa_prop, m_kappa, s_kappa);
  
  if (log(R::runif(0, 1)) < (prop - curr)) {
    kappa = kappa_prop;
    accepted = true;
  }
}


/*
 * xi, psi2 업데이트 (Gibbs Sampler)
 * (이전 코드와 동일)
 */
List update_xi_psi2_cpp(double alpha, double xi, double psi2, 
                        double m0 = 0.0, double kappa0 = 0.5, 
                        double alpha0 = 3.0, double beta0 = 0.5) {
  
  double n = 1.0; 
  double abar = alpha;
  
  double kappa_n = 1.0 / (n/psi2 + 1.0/kappa0);
  double m_n = kappa_n * (m0/kappa0 + (n * abar)/psi2);
  double xi_new = R::rnorm(m_n, sqrt(kappa_n));
  
  double shape = alpha0 + n/2.0;
  double scale_param = beta0 + 0.5 * pow(alpha - xi_new, 2); 
  
  double psi2_new = 1.0 / R::rgamma(shape, 1.0/scale_param);
  
  return List::create(Named("xi") = xi_new, Named("psi2") = psi2_new);
}


// --- Main Rcpp Function ---

// [[Rcpp::export]]
List LSM_conti_cpp(arma::mat Y, int iter = 5000, int burnin = 1000, int thinning = 5, int d = 2,
                   double sigma2 = 1.0, double a0 = 0.5, double b0 = 0.5, 
                   double m_kappa = 0.0, double s_kappa = 1.0,
                   double prop_z = 0.05, double prop_alpha = 0.05, 
                   double prop_beta = 0.05, double prop_kappa = 0.2,
                   double alpha_init = 0.0, double beta_init = 1.0, 
                   double xi_init = 0.0, double kappa_init = 1.0,
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
  double kappa = kappa_init;
  
  // storage
  int ns = static_cast<int>(floor(static_cast<double>(iter - 1 - burnin) / thinning)) + 1;
  if (iter <= burnin) ns = 0;
  
  arma::cube z_save(n, d, ns);
  arma::vec alpha_save(ns);
  arma::vec xi_save(ns);
  arma::vec psi2_save(ns);
  arma::vec beta_save(ns);
  arma::vec kappa_save(ns); // kappa 저장 공간 추가
  arma::vec ll_save(ns);
  
  // acceptance rate
  // arma::vec acc_z(n, arma::fill::zeros);
  // --- storage (추가/수정) ---
  arma::mat acc_z_elem(n, d, arma::fill::zeros);  // 각 원소 수용 카운트
  
  double acc_alpha = 0.0;
  double acc_beta = 0.0;
  double acc_kappa = 0.0; // kappa 수용률 추가
  
  int s = 0; // storage index
  
  // MCMC Sampler
  for (int i = 0; i < iter; ++i) {
    
    // update z_i (MH)
    // bool accepted_z_j = false;
    // for (int j = 0; j < n; ++j) {
    //   update_z_j_cpp(j, z, Y, alpha, sigma2, prop_z, beta, kappa, d, accepted_z_j);
    //   if (accepted_z_j) {
    //     acc_z(j)++;
    //   }
    // }
    // (변경) z 원소 단위 업데이트
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < d; ++k) {
        bool accepted_elem = false;
        update_z_elem_cpp(j, k, z, Y, alpha, sigma2, prop_z, beta, kappa, accepted_elem);
        if (accepted_elem) {
          acc_z_elem(j, k) += 1.0;
        }
      }
    }
    
    // update alpha (MH)
    bool accepted_alpha = false;
    update_alpha_cpp(alpha, z, Y, xi, psi2, prop_alpha, beta, kappa, accepted_alpha);
    if (accepted_alpha) {
      acc_alpha++;
    }
    
    // update beta (MH)
    bool accepted_beta = false;
    update_beta_cpp(beta, z, Y, alpha, prop_beta, a0, b0, kappa, accepted_beta);
    if (accepted_beta) {
      acc_beta++;
    }
    
    // !!! 원본 R 코드의 테스트 코드 반영 !!!
    // R 코드:
    // # test: remove beta
    // beta <- 1
    // # acc_beta <- acc_beta + 1L (주석 처리됨)
    beta = 1.0;
    // acc_beta는 R 코드와 동일하게 증가시키지 않음
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    // update kappa (MH)
    bool accepted_kappa = false;
    update_kappa_cpp(kappa, z, Y, alpha, beta, prop_kappa, m_kappa, s_kappa, accepted_kappa);
    if (accepted_kappa) {
      acc_kappa++;
    }
    
    // Gibbs for (xi, psi2)
    List up_xi_psi2 = update_xi_psi2_cpp(alpha, xi, psi2, 0.0, 0.5, 3.0, 0.5);
    xi = as<double>(up_xi_psi2["xi"]);
    psi2 = as<double>(up_xi_psi2["psi2"]);
    
    // save
    if (i >= burnin && (i - burnin) % thinning == 0) {
      z_save.slice(s) = z;
      alpha_save(s) = alpha;
      xi_save(s) = xi;
      psi2_save(s) = psi2;
      beta_save(s) = beta;
      kappa_save(s) = kappa; // kappa 저장
      ll_save(s) = loglik_conti_network_cpp(Y, z, alpha, beta, kappa); // 연속형 가능도 저장
      s++;
    }
    
    if (verbose && ((i + 1) % 100 == 0)) {
      double current_ll;
      if (s > 0) {
        current_ll = ll_save(s - 1);
      } else {
        current_ll = loglik_conti_network_cpp(Y, z, alpha, beta, kappa);
      }
      Rprintf("iteration: %d, alpha: %.4f, beta: %.4f, kappa: %.4f, ll: %.4f\n",
              i + 1, alpha, beta, kappa, current_ll);
    }
    
    Rcpp::checkUserInterrupt();
  }
  
  // Procrustes 매칭은 R에서 수행
  
  // 결과 반환
  return List::create(
    Named("samples") = List::create(
      Named("z") = z_save,
      Named("alpha") = alpha_save,
      Named("xi") = xi_save,
      Named("psi2") = psi2_save,
      Named("beta") = beta_save,
      Named("kappa") = kappa_save, // kappa 반환
      Named("ll") = ll_save
    ),
    Named("accept") = List::create(
      Named("z_elem") = acc_z_elem / iter,     // (n x d) 각 원소별 수용률
      Named("alpha") = acc_alpha / iter,
      Named("beta") = acc_beta / iter,
      Named("kappa") = acc_kappa / iter // kappa 수용률 반환
    ),
    Named("config") = List::create(
      Named("iter") = iter, 
      Named("burnin") = burnin, 
      Named("thinning") = thinning,
      Named("sigma2") = sigma2,
      Named("prop_sd") = List::create(
        Named("z") = prop_z, 
        Named("alpha") = prop_alpha, 
        Named("beta") = prop_beta,
        Named("kappa") = prop_kappa // kappa 제안 표준편차 반환
      ),
      Named("a0") = a0,
      Named("b0") = b0,
      Named("m_kappa") = m_kappa, // kappa 사전확률 모수 반환
      Named("s_kappa") = s_kappa
    )
  );
}