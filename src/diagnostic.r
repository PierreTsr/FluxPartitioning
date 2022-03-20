library(RcppCNPy)
library(coda)
library(ggplot2)


files <- list.files(path="etc/diagnostic", pattern="*.npy", full.names=TRUE, recursive=FALSE)
diagnostic.plots <- function(file){
  x <- npyLoad(file)
  y <- mcmc.list(apply(x, 2, function(col) {mcmc(col)}, simplify = FALSE))
  name <- basename(file)
  name <- substr(name, 1, nchar(name) - 4)
  geweke.plot(y[[1]], main="Geweke test", sub=name)
  gelman.plot(y, main="Gelman test", sub=name, autoburnin = FALSE)
  autocorr.plot(y[[1]],lag.max=dim(x)[1]/4, main="Auto-correlation plot", sub=name)
}

lapply(files, diagnostic.plots)
