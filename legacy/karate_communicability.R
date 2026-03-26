library(igraph)
library(network)
library(brainGraph)
g <- make_graph("Zachary")
comm       <- communicability(g)
comm_angle <- comm / sqrt(outer(diag(comm), diag(comm)))
comm_dist  <- sqrt(outer(diag(comm), diag(comm), "+") - 2*comm)
comm_dist.scaled  <- matrix(scale(as.numeric(comm_dist)), nrow = 34, ncol = 34)

temp <- comm_dist
hist(temp)
temp[temp<8] <- 0
diag(temp) <- 0
hist(temp[temp != 0])
hist(log(temp[temp != 0]))
network_obj <- network(temp,
                       directed = FALSE, 
                       loops = FALSE,
                       ignore.eval = FALSE,
                       names.eval = "value")
rowSums(temp)[1]
rowSums(temp)[33]
rowSums(temp)[34]

# 1, 33
color <- rep('blue', 34)
color[1] <- 'red'
color[33] <- 'red'
color[34] <- 'red'
plot(network_obj, vertex.col = color, edge.lwd = scale(temp), displaylabels = TRUE)

library(latentnet)
g1 = as.network(igraph::as_adjacency_matrix(g,sparse = F))
network::set.vertex.attribute(g1, "label", 1:network.size(g1))

fit1 <- ergmm(g1 ~ euclidean(d = 2),
              control = ergmm.control(burnin = 2000,
                                      sample.size = 6000,
                                      interval = 5))
plot(g)
plot(fit1, labels = TRUE)


motif_GOF_test(fit1, 
               igraph::as_adjacency_matrix(g,sparse = F), 
               B = 500, custom_text = paste0('karate network'), arg = 'simple_motif')

temp<-igraph::as_adjacency_matrix(g,sparse = F)


fit2 <- ergmm(
  network_obj ~ euclidean(d = 2),
  family = "normal",
  response = "value",
  fam.par = list(prior.var = 1, prior.var.df = 1),
  control = ergmm.control(sample.size = 2500, burnin = 5000),
  verbose = T
)

plot(fit2, labels = TRUE)

motif_GOF_test(fit2, 
               igraph::as_adjacency_matrix(g,sparse = F), 
               B = 500, custom_text = paste0('karate network'), arg = 'simple_motif')






# 노드 40명, 2개 community (20명씩)
sizes <- c(10, 10)
# 블록별 확률행렬 (대각선 = within, 비대각선 = between)
pref.matrix <- matrix(c(0.30, 0.01,
                        0.01, 0.20), nrow = 2)

g_sbm <- sample_sbm(sum(sizes), pref.matrix, block.sizes = sizes)

plot(g_sbm,
     vertex.color = membership(cluster_louvain(g_sbm)),
     vertex.label = NA,
     vertex.size = 6,
     layout = layout_with_fr)


g_sbm_net <- as.network(igraph::as_adjacency_matrix(g_sbm,sparse = F))
fit1 <- ergmm(g_sbm_net ~ euclidean(d = 2),
              control = ergmm.control(burnin = 2000,
                                      sample.size = 6000,
                                      interval = 5))
plot(fit1, labels = T)
plot(g_sbm)
A <- igraph::as_adjacency_matrix(g_sbm,sparse = F)

comm       <- communicability(igraph::graph_from_adjacency_matrix(A))
comm_angle <- comm / sqrt(outer(diag(comm), diag(comm)))
comm_dist  <- sqrt(outer(diag(comm), diag(comm), "+") - 2*comm)
comm_dist.scaled  <- matrix(scale(as.numeric(comm_dist)), nrow = 34, ncol = 34)

temp <- comm_dist
hist(temp)
temp[temp<1.5] <- 0
diag(temp) <- 0
hist(temp)
hist(log(temp[temp != 0]))
network_obj <- network(temp,
                       directed = FALSE,
                       loops = FALSE,
                       ignore.eval = FALSE,
                       names.eval = "value")

network.vertex.names(network_obj) <- 1:network.size(network_obj)
plot(network_obj, displaylabels = TRUE, edge.lwd = scale(temp))




# 원본
g_sbm
# 원본 LSM
fit1

# communicability(with threshold)
temp
# communicability LSM
