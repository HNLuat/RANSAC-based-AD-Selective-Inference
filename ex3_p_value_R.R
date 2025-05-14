library(reticulate)
# Using environment that already install numpy and mpmath
reticulate::use_condaenv("my_r_env", required = TRUE)

gendata <- import("gendata")
util <- import("util")
regressor <- import("regressor")
CTRL_RANSAC <- import("CTRL_RANSAC")

n <- 100
p <- 5
B <- 10
t <- 2
num_outliers <- 0
delta <- 0

data <- gendata$gen(as.integer(n), as.integer(p), as.integer(num_outliers), delta)
X <- data[[1]]
y <- data[[2]]

model <- regressor$RANSAC(as.integer(B), as.integer(t))
model$fit(X, y)

list_of_outliers <- model$best_outliers

for (j in list_of_outliers) {
  res <- util$calculate_SI_essentials(X, y, list_of_outliers, j)
  etaT_yobs <- res[[1]]
  etaT_Sigma_eta <- res[[2]]
  a <- res[[3]]
  b <- res[[4]]

  truncatedRegion <- CTRL_RANSAC$identifying_truncated_region(model, X, a, b)
  selective_pvalue <- util$calculate_p_value(truncatedRegion, etaT_yobs, etaT_Sigma_eta)

  cat(sprintf("The p value of the %d(th) instance is %f\n", j, selective_pvalue))
}
