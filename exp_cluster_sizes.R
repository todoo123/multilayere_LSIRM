## Verify cluster-size distribution: K_+ trace says ~4, but PSM=0.74 implies
## K_eff (1/HHI) = 1.34.  Hypothesis: typical c sample is one giant cluster
## + few tiny ones.  Re-run alpha=1 quickly and save c_samples for analysis.

suppressPackageStartupMessages({ library(Rcpp); library(vegan) })
proj_dir <- "/Users/todoo/Desktop/학교/대학원/Research/joint_LSIRM"
data_dir <- file.path(proj_dir, "data")
setwd(data_dir)
source(file.path(data_dir, "my_LSIRM_FMC_cpp_v12.R"))

set.seed(20260501)
n  <- 150; P1 <- 30; P2 <- 30; P3 <- 30; P4 <- 30
P_total <- P1+P2+P3+P4; d <- 2L; K_true <- 4L; K1 <- 5L
gamma_true <- 1.0; sigma0_sq_true <- 1.0; kappa_true <- 1.0
nu2_true <- 5; nu2_fit <- 4; sigma_b <- 1.0
centers_meta <- rbind(c(-1,1),c(1,1),c(1,-1),c(-1,-1))
sigma_meta <- list(diag(c(0.05,0.05)),diag(c(0.07,0.07)),diag(c(0.03,0.03)),diag(c(0.06,0.06)))
B1_meta <- rep_len(1:K_true,P1); B2_meta <- rep_len(1:K_true,P2)
B3_meta <- rep_len(1:K_true,P3); B4_meta <- rep_len(1:K_true,P4)
true_item_cluster <- c(B1_meta,B2_meta,B3_meta,B4_meta)
centers_resp <- centers_meta*0.7; sd_cluster_resp <- 0.3
sample_around <- function(C,ids,sd) C[ids,,drop=FALSE]+matrix(rnorm(length(ids)*ncol(C),0,sd),length(ids),ncol(C))
sample_sigma <- function(C,ids,Sl) {
  out <- matrix(0,length(ids),ncol(C))
  for (k in seq_along(Sl)) { idx<-which(ids==k); if (!length(idx)) next
    L<-chol(Sl[[k]]); out[idx,] <- matrix(C[k,],length(idx),ncol(C),byrow=TRUE)+matrix(rnorm(length(idx)*ncol(C)),length(idx),ncol(C))%*%L }
  out
}
dist_mat<-function(A,B){o<-matrix(0,nrow(A),nrow(B));for(j in 1:nrow(B))o[,j]<-sqrt(rowSums((A-matrix(B[j,],nrow(A),ncol(A),byrow=TRUE))^2));o}
invlogit<-function(x)1/(1+exp(-x))

resp_cl<-sample.int(nrow(centers_resp),n,replace=TRUE)
A_true<-sample_around(centers_resp,resp_cl,sd_cluster_resp)
B1_true<-sample_sigma(centers_meta,B1_meta,sigma_meta);B2_true<-sample_sigma(centers_meta,B2_meta,sigma_meta)
B3_true<-sample_sigma(centers_meta,B3_meta,sigma_meta);B4_true<-sample_sigma(centers_meta,B4_meta,sigma_meta)
alpha1_true<-rnorm(n,0,1);alpha2_true<-rnorm(n,0,1);alpha3_true<-rnorm(n,0,1);alpha4_true<-rnorm(n,0,1)
beta1_true<-rnorm(P1,0,1);beta2_true<-rnorm(P2,0,1);beta3_true<-rnorm(P3,0,0.5)
gen_thr<-function(P,K){o<-matrix(NA_real_,P,K-1);for(j in 1:P)o[j,]<-sort(rnorm(K-1,0,1.5),decreasing=TRUE);o}
beta4_true<-gen_thr(P4,K1)
D1<-dist_mat(A_true,B1_true);D2<-dist_mat(A_true,B2_true);D3<-dist_mat(A_true,B3_true);D4<-dist_mat(A_true,B4_true)
ETA1<-outer(alpha1_true,rep(1,P1))-outer(rep(1,n),beta1_true)-gamma_true*D1
ETA2<-outer(alpha2_true,rep(1,P2))-outer(rep(1,n),beta2_true)-gamma_true*D2
ETA3<-outer(alpha3_true,rep(1,P3))-outer(rep(1,n),beta3_true)-gamma_true*D3
ETA4<-outer(alpha4_true,rep(1,P4))-gamma_true*D4
Y_bin<-matrix(rbinom(n*P1,1,as.vector(invlogit(ETA1))),n,P1)
lambda_true<-matrix(rgamma(n*P2,nu2_true/2,nu2_true/2),n,P2)
Y_con<-ETA2+matrix(rnorm(n*P2,0,sqrt(sigma0_sq_true)),n,P2)/sqrt(lambda_true);storage.mode(Y_con)<-"numeric"
Y_cnt<-matrix(rnbinom(n*P3,size=1/kappa_true,mu=as.vector(exp(ETA3))),n,P3)
gen_grm<-function(ETA,thr,K){Y<-matrix(NA_integer_,nrow(ETA),ncol(ETA))
  for(j in 1:ncol(ETA))for(i in 1:nrow(ETA)){
    pge<-invlogit(ETA[i,j]+thr[j,]);p<-numeric(K);p[1]<-1-pge[1]
    for(k in 2:(K-1))p[k]<-pge[k-1]-pge[k];p[K]<-pge[K-1]
    p[p<0]<-0;ps<-sum(p);if(ps<=0){p<-rep(0,K);p[round(K/2)]<-1}else p<-p/ps
    Y[i,j]<-sample.int(K,1,prob=p)
  };storage.mode(Y)<-"integer";Y}
Y_ord1<-gen_grm(ETA4,beta4_true,K1);Y_ord2<-matrix(0L,n,0)

init_pca<-function(Y_bin,Y_con,Y_cnt,Y_ord1,Y_ord2,d){bl<-list()
  if(ncol(Y_bin)>0)bl<-c(bl,list(scale(Y_bin)));if(ncol(Y_con)>0)bl<-c(bl,list(scale(Y_con)))
  if(ncol(Y_cnt)>0)bl<-c(bl,list(scale(log1p(Y_cnt))));if(ncol(Y_ord1)>0)bl<-c(bl,list(scale(as.matrix(Y_ord1))))
  X<-do.call(cbind,bl);X[is.na(X)]<-0;pc<-prcomp(t(X),center=TRUE,scale.=FALSE)
  rk<-min(d,ncol(pc$x));o<-matrix(0,ncol(X),d);o[,seq_len(rk)]<-pc$x[,seq_len(rk),drop=FALSE];o}
b_pca<-init_pca(Y_bin,Y_con,Y_cnt,Y_ord1,Y_ord2,d)
set.seed(0502); km<-kmeans(b_pca,centers=K_true,nstart=25,iter.max=50)
random_sigma<-{set.seed(0502);sample.int(P_total,P_total)}

set.seed(20260507)
res<-lsirm_epa_v12_cpp(
  Y_bin=Y_bin,Y_con=Y_con,Y_cnt=Y_cnt,Y_ord1=Y_ord1,Y_ord2=Y_ord2,
  d=d,n_iter=20000,burnin=8000,thin=5,nu2=nu2_fit,
  lsirm_hyper=list(a_sigma=1,b_sigma=1,a_tau1=1,b_tau1=1,a_tau2=1,b_tau2=1,a_tau3=1,b_tau3=1,
                   a_sigma0=1,b_sigma0=1,
                   mu_log_gamma1=0,sd_log_gamma1=0.5,mu_log_gamma2=0,sd_log_gamma2=0.5,
                   mu_log_gamma3=0,sd_log_gamma3=0.5,mu_log_gamma4=0,sd_log_gamma4=0.5,
                   mu_log_gamma5=0,sd_log_gamma5=0.5,mu_log_kappa=0,sd_log_kappa=0.1,
                   mu_beta4=0,sd_beta4=2,mu_beta5=0,sd_beta5=2),
  epa_hyper=list(a_alpha=1,b_alpha=1,a_tau=1,b_tau=1,delta=0),
  lsirm_prop_sd=list(alpha1=1.15,alpha2=0.48,alpha3=1.15,alpha4=0.92,alpha5=0.5,
                     log_gamma1=0.07,log_gamma2=0.018,log_gamma3=0.05,log_gamma4=0.05,log_gamma5=0.05,
                     a=0.26,beta1=0.44,beta2=0.13,beta3=0.37,beta4=0.30,beta5=0.30,
                     b1=0.33,b2=0.13,b3=0.30,b4=0.26,b5=0.20,log_kappa=0.20),
  epa_prop_sd=list(log_alpha=0.5,log_tau=0.5),
  lsirm_init=NULL,
  epa_init=list(c=as.integer(km$cluster),sigma=as.integer(random_sigma),alpha=1,tau=1),
  sigma_b=sigma_b,compute_co_cluster_online=TRUE,epa_warmup=0L,
  n_split_merge=1L,n_split_merge_R=5L,n_perm_swaps=as.integer(P_total),
  b_epa_coupling=TRUE,update_alpha=FALSE,update_tau=FALSE,
  verbose=FALSE,fix_gamma=FALSE,procrustes_target=NULL
)

c_samps <- res$epa_c   ## (n_save, P_total) integer matrix
n_save <- nrow(c_samps); P <- ncol(c_samps)
cat(sprintf("\n# saved iters: %d, P=%d\n", n_save, P))

## Per-iter cluster size distribution
size_summary <- t(apply(c_samps, 1, function(v) {
  tt <- sort(tabulate(v), decreasing=TRUE)
  tt <- tt[tt > 0]
  K  <- length(tt)
  hhi <- sum((tt/P)^2)
  largest_share <- tt[1]/P
  c(K=K, HHI=hhi, K_eff=1/hhi, largest=largest_share,
    second=if (K>=2) tt[2]/P else 0)
}))

cat("\n=== per-iter cluster geometry (across saved iters) ===\n")
cat(sprintf("K_+ (= K from c_samps): mean=%.2f median=%g sd=%.2f range=[%g,%g]\n",
            mean(size_summary[,"K"]), median(size_summary[,"K"]),
            sd(size_summary[,"K"]),
            min(size_summary[,"K"]), max(size_summary[,"K"])))
cat(sprintf("HHI:                    mean=%.3f median=%.3f sd=%.3f range=[%.3f,%.3f]\n",
            mean(size_summary[,"HHI"]), median(size_summary[,"HHI"]),
            sd(size_summary[,"HHI"]),
            min(size_summary[,"HHI"]), max(size_summary[,"HHI"])))
cat(sprintf("K_eff (=1/HHI):         mean=%.2f median=%.2f range=[%.2f,%.2f]\n",
            mean(size_summary[,"K_eff"]), median(size_summary[,"K_eff"]),
            min(size_summary[,"K_eff"]), max(size_summary[,"K_eff"])))
cat(sprintf("Largest-cluster share:  mean=%.2f median=%.2f sd=%.2f range=[%.2f,%.2f]\n",
            mean(size_summary[,"largest"]), median(size_summary[,"largest"]),
            sd(size_summary[,"largest"]),
            min(size_summary[,"largest"]), max(size_summary[,"largest"])))
cat(sprintf("2nd largest share:      mean=%.2f median=%.2f sd=%.2f range=[%.2f,%.2f]\n",
            mean(size_summary[,"second"]), median(size_summary[,"second"]),
            sd(size_summary[,"second"]),
            min(size_summary[,"second"]), max(size_summary[,"second"])))

## Distribution of cluster size profiles
cat("\n=== sample of cluster-size profiles (first 10 saved iters) ===\n")
for (s in 1:10) {
  tt <- sort(tabulate(c_samps[s,]), decreasing=TRUE)
  tt <- tt[tt>0]
  cat(sprintf("  iter %d: K=%d  sizes=%s\n", s, length(tt), paste(tt, collapse=",")))
}

cat("\n=== histogram of largest-cluster share ===\n")
print(table(cut(size_summary[,"largest"], breaks=c(0,0.3,0.5,0.7,0.85,0.95,1.0))))
