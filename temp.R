library(lsirm12pl)

install.packages('lsirm12pl')

Y_ord2

# generate example ordinal item response matrix
set.seed(123)
nsample <- 50
nitem <- 10
data <- matrix(sample(1:5, nsample * nitem, replace = TRUE), nrow = nsample)
data <- lsirm_all$Y_ord1
# Fit GRM LSIRM using direct function call
fit <- lsirmgrm(data, niter = 100000, nburn = 10000, nthin = 10)
plot(fit)


colnames(lsirm_all$Y_ord1)[13]
colnames(lsirm_all$Y_ord1)[25]


base::plot(fit$w_estimate)
text(x = fit$w_estimate[,1], y = fit$w_estimate[,2], labels = colnames(fit$data), pos = 3, cex = 0.7)


ts.plot(fit$w[,1,2])


