#' Semi-Parametric Mixed Effects Random Forest
#'
#' @param X A data frame or matrix of fixed effects covariates.
#' @param y A numeric vector of the response variable.
#' @param id A vector indicating subject/group IDs for random effects.
#' @param time A numeric vector of time points for each observation.
#' @param iter Integer, maximum number of iterations. Default is 100.
#' @param delta Numeric, convergence threshold. Default is 0.001.
#' @param ntree Integer, number of trees for the random forest. Default is 500.
#' @param importance Character, importance type for ranger ('none', 'impurity', 'permutation'). Default is 'permutation'.
#' @param smooth_method Character, smooth estimation engine to use: either \code{"gam"} or \code{"bam"}. Default is \code{"gam"}.
#' @param num.threads Number of threads for ranger and mgcv parallelization. Defaults to 1.
#'
#' @return A list containing the fitted random forest model, the GAM smooth model, predictions, and MSE history.
#' @export
SP_MERF <- function(X, y, id, time, iter = 100, delta = 0.001, 
                    ntree = 500, importance = 'permutation', 
                    smooth_method = "gam", num.threads = 1) {
  
  # 1. Preprocessing & Parameter Validation
  N <- length(y)
  id <- as.factor(id)
  df_internal <- data.frame(y = y, id = id, time = time, X)
  
  # Validate and normalize smooth choice
  smooth_method <- match.arg(tolower(smooth_method), c("gam", "bam"))
  
  # Initialization
  pred_smooth <- rep(0, N) 
  pred_fixed <- rep(0, N)
  mse_history <- numeric(iter)
  
  # Define Formula for Forest
  fixed_vars <- colnames(X)
  rf_formula <- as.formula(paste("y_adj ~", paste(fixed_vars, collapse = " + ")))
  
  cat(sprintf("Starting SP-MERF iterations using [%s] for smooth estimation...\n", toupper(smooth_method)))
  
  # 2. EM-like Iterative Loop
  for (r in 1:iter) {
    
    # Step A: Update Fixed Effects (Random Forest)
    # Target: y - current individual non-linear smooths
    df_internal$y_adj <- df_internal$y - pred_smooth
    
    rf_model <- ranger::ranger(
      formula = rf_formula,
      data = df_internal,
      num.trees = ntree,
      importance = 'none',
      num.threads = num.threads
    )
    
    # Update population predictions
    pred_fixed <- predict(rf_model, data = df_internal)$predictions
    
    # Step B: Update Random Effects (Spline Smooths)
    # Target: y - current population forest predictions
    df_internal$residuals <- df_internal$y - pred_fixed
    
    # Factor-smooth interaction allows unique non-linear curves per PID
    if (smooth_method == "gam") {
      smooth_model <- mgcv::gam(
        residuals ~ s(time, id, bs = "fs"), 
        data = df_internal,
        method = "fREML",
        nthreads = num.threads
      )
    } else {
      # Optimized Smooth Update using bam() with discretization
      smooth_model <- mgcv::bam(
        residuals ~ s(time, id, bs = "fs"), 
        data = df_internal,
        method = "fREML",
        discrete = TRUE,
        nthreads = num.threads
      )
    }
    
    pred_smooth <- as.numeric(predict(smooth_model, newdata = df_internal))
    
    # Step C: Convergence Check
    current_mse <- mean((df_internal$y - (pred_fixed + pred_smooth))^2)
    mse_history[r] <- current_mse
    
    if (r > 1 && abs(mse_history[r-1] - current_mse) < delta) {
      cat("Convergence reached at iteration", r, "\n")
      mse_history <- mse_history[1:r]
      break
    }
  }
  
  # 3. Final Importance Calculation Pass
  if (importance != 'none') {
    df_internal$y_adj <- df_internal$y - pred_smooth
    
    rf_model <- ranger::ranger(
      formula = rf_formula,
      data = df_internal,
      num.trees = ntree,
      importance = importance, # Executes permutation math exactly once here
      num.threads = num.threads
    )
    # Sync final population predictions
    pred_fixed <- predict(rf_model, data = df_internal)$predictions
  }
  
  # 4. Return objects
  results <- list(
    forest = rf_model,
    smooth_model = smooth_model,
    fitted = pred_fixed + pred_smooth,
    fixed_fitted = pred_fixed,
    random_fitted = pred_smooth,
    mse_history = mse_history
  )
  
  return(results)
}