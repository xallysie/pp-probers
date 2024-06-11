
library(tidyverse)  # data hygiene
library(readr)      # code hygiene
library(rlang)      # code hygiene
library(scales)     # to scale response variable
library(lme4)       # multilevel models
library(brms)       # bayesian multilevel models
library(foreach)    # foreach() function
library(doParallel) # parallel computing
library(parallel)   # parallel computing
library(furrr)      # parallel computing
library(caret)      # for cross-validation
library(MuMIn)      # for model evaluation

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 
set.seed(57812623)
options(scipen=999,digits=6)

# import training data
load("train_data_transformations_raised.Rdata")
train_data = train_data_extended

# ols regression
ols_regression_fit <- function(data, formula_str, weights=NULL, ...){
  .formula <- as.formula(formula_str)
  if (is.null(weights)) {
    model <- lm(formula = .formula, data=data)
  } else {
    predictor_vars <- all.vars(.formula)[-1] # get predictor variables from formula
    predictors <- data[, predictor_vars, drop=FALSE]
    weighted_preds <- sapply(seq_along(weights), function(x) {predictors[,x]*weights[x]})
    weights_preds_sum <- rowSums(data.frame(weighted_preds))
    model <- lm(formula = .formula, data=data, weights=abs(weights_preds_sum))
  }
  return(model)
}

# mixed-effects models
lmer_regression_fit <- function(data, formula_str, weights=NULL, ...){
  .formula <- as.formula(formula_str)
  if (is.null(weights)) {
    model <- lmer(.formula, data=data, REML=FALSE)
  } else {
    predictor_vars <- all.vars(.formula)[-1] # get predictor variables from formula
    predictors <- data[, predictor_vars, drop=FALSE]
    weighted_preds <- sapply(seq_along(weights), function(x) {predictors[,x]*weights[x]})
    weights_preds_sum <- rowSums(data.frame(weighted_preds))
    model <- lmer(.formula, data=data, weights=abs(weights_preds_sum), REML=FALSE)
  }
  return(model)
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# functions for model evaluation ------------------------------------------

# define wrapper function to evaluate models
evaluate_model <- function(model, val_fold, outcome_var) {
  val_predictions <- predict(model, newdata = val_fold, allow.new.levels = TRUE) # fit model on validation dataset
  
  # calculate root mean squared error
  val_rmse <- sqrt(mean((val_predictions - val_fold[[outcome_var]])^2)) # obtain RMSE on validation dataset
  
  # calculate adjusted R-squared based on model type 
  # **CHANGEME** if you add different types of models not shown here, please adjust this to recognize your model type
  if (inherits(model, "lm")) { # if linear regression
    val_adj_r_squared <- summary(model)$adj.r.squared
  } else if (inherits(model, "lmerMod")) { # if mixed-effects model
    val_adj_r_squared <- r.squaredGLMM(model)[2] # get conditional r-squared
  } else if (inherits(model, "randomForest")) {
    val_adj_r_squared <- 1 - (sum((val_predictions - val_fold[[outcome_var]])^2) / 
                                sum((val_fold[[outcome_var]] - mean(val_fold[[outcome_var]]))^2))
  } else {
    val_adj_r_squared <- NA # **CHANGEME** update this if you add different types of models not shown here
    print("could not detect model type")
  }
  
  return(list(val_rmse = val_rmse,
              val_adj_r_squared = val_adj_r_squared))
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# functions for cross-validation ------------------------------------------

# currently supports k-fold cv and leave-one-out cv
# define wrapper function for k-fold cross-validation
evaluate_models_kfold <- function(data, outcome_var, formula_str, model_function, k=5, weights=NULL, description="NA", ...){
  
  # print diagnostic msg
  print(paste("Outcome variable:", outcome_var))
  print(paste("Formula:", formula_str))
  print(paste("Weights:", paste(weights,collapse=",")))
  
  folds <- createFolds(data[[outcome_var]], k = k, list=TRUE)
  
  results <- map_dfr(1:k, function(i) {
    tryCatch({
      # check if the fold is not empty
      if (length(folds[[i]]) == 0) {
        stop(paste("Fold", i, "is empty"))
      }
      # create folds
      train_fold <- data[-folds[[i]], ]
      val_fold <- data[folds[[i]], ]
      
      # check dims of train_fold and val_fold
      if (ncol(train_fold) < 2 || ncol(val_fold) < 2) {
        stop(paste("Fold", i, "train or validation fold has insufficient dimensions"))
      }
      
      model <- model_function(data = train_fold, formula_str = formula_str, ...)
      evaluation <- evaluate_model(model, val_fold, outcome_var) 
      
      data.frame(fold = i, val_rmse = evaluation$val_rmse, val_adj_r_squared = evaluation$val_adj_r_squared)
    }, error = function(e) {
      message(paste("Error in fold", i, ":", e$message))
      NULL
    })
  })
  
  # calculate mean RMSE and adjusted r-squared across all folds
  mean_rmse <- mean(results$val_rmse, na.rm=TRUE)
  mean_adj_r_squared <- mean(results$val_adj_r_squared, na.rm=TRUE)
  
  final_model <- model_function(data = data, formula_str = formula_str, ...)
  
  results_list <- list(
    model = final_model, 
    model_formula = formula_str,
    description = description,
    weights = weights,
    outcome = outcome_var,
    cross_val_results = results, 
    mean_rmse = mean_rmse, 
    mean_adj_r_squared = mean_adj_r_squared)
  
  print(paste0("Description: ", description))
  print(paste0("Mean RMSE: ", results_list$mean_rmse))
  print(paste0("Adjusted R-squared: ", results_list$mean_adj_r_squared))
  
  return(results_list)
}



# define wrapper function for leave-one-out cross-validation
evaluate_models_loo <- function(data, outcome_var, formula_str, model_function, description="NA", weights=NULL, ...){
  
  # print diagnostic msg
  print(paste("Outcome variable:", outcome_var))
  print(paste("Formula:", formula_str))
  print(paste("Weights:", paste(weights,collapse=",")))
  
  results <- map_dfr(1:nrow(data), function(i) {
    tryCatch({
      # create folds
      train_fold <- data[-i, , drop=FALSE]  # remove i-th row to create training fold
      val_fold <- data[i, , drop=FALSE]     # validation fold
      
      # check dims of train_fold and val_fold
      if (ncol(train_fold) < 2 || ncol(val_fold) < 2) {
        stop(paste("Fold", i, "train or validation fold has insufficient dimensions"))
      }
      
      model <- model_function(data = train_fold, formula_str = formula_str, ...)
      evaluation <- evaluate_model(model, val_fold, outcome_var) 
      
      data.frame(fold = i, val_rmse = evaluation$val_rmse, val_adj_r_squared = evaluation$val_adj_r_squared)
    }, error = function(e) {
      message(paste("Error in fold", i, ":", e$message))
      NULL
    })
  })
  
  # calculate mean RMSE and adjusted r-squared across all folds
  mean_rmse <- mean(results$val_rmse, na.rm=TRUE)
  mean_adj_r_squared <- mean(results$val_adj_r_squared, na.rm=TRUE)
  
  final_model <- model_function(data = data, formula_str = formula_str, ...)
  
  results_list <- list(
    model = final_model, 
    model_formula = formula_str,
    weights = weights,
    description = formula_str,
    outcome = outcome_var,
    cross_val_results = results, 
    mean_rmse = mean_rmse, 
    mean_adj_r_squared = mean_adj_r_squared)
  
  print(paste0("Description: ", description))
  print(paste0("Mean RMSE: ", results_list$mean_rmse))
  print(paste0("Adjusted R-squared: ", results_list$mean_adj_r_squared))
  
  return(results_list)
}


base_formula_elements = c('generalized', 'symbolic', 'contact_quality', 'contact_friendsz', 'identification_selfinvestment')
versions <- lapply(base_formula_elements, function(element) {
  c(element, paste0(element, "_squared"), paste0(element, "_cubid"))
})
combinations <- expand.grid(versions)

formula_strs = c()
for (i in 1:nrow(combinations)) {
  formula_elements = c()
  x = combinations[i,]
  for (j in 1:length(x)) {
    formula_elements = c(formula_elements, as.character(x[[j]]))
  }
  formula_str <- paste(c('bias ~', paste(formula_elements, collapse = ' + ')), collapse = ' ')
  formula_strs <- c(formula_strs, formula_str)
}

test_formula = formula_strs[1]


# function to run a single parameter combination
single_run <- function(
    data,           # training data
    outcome_var,    # name of outcome var (string)
    formula_str,    # formula (as a string)
    model_function, # name of model fitting function
    cv_function,    # name of cross-validation function
    description,    # string description of model for ez comparison
    k=5,            # if using k-fold cv, number of folds
    weights=NULL,   # input parameter weights to test
    ...){
  
  result <- cv_function(data = data, outcome_var = outcome_var, formula_str = formula_str, 
                        model_function = model_function, k = k, description = description, weights = weights,...)
  
  # write results to .csv
  results_csv <- data.frame(matrix(nrow=1, ncol=6))
  names(results_csv) <- c("Time","Model","Description","Weights","mean_rmse","mean_adj_r_squared")
  results_csv$Time <- Sys.time()
  results_csv$Model <- result$model_formula
  # results_csv$Description <- result$description
  # results_csv$Weights <- paste(result$weights, collapse=",")
  results_csv$mean_rmse <- result$mean_rmse
  results_csv$mean_adj_r_squared <- result$mean_adj_r_squared
  write_csv(results_csv, file=paste0("parameter_search_out/",description,"_",generate_random_string(16),".csv"))
  return(results_csv)
}
generate_random_string <- function(n) {
  chars <- c(letters, LETTERS, 0:9)
  unique_seed <- as.integer(Sys.time()) + Sys.getpid()
  set.seed(unique_seed)
  random_string <- paste0(sample(chars, n, replace = TRUE), collapse = "")
  return(random_string)
}

cluster_run <- function(formula) {
  single_run(
    data = train_data,
    outcome_var = "bias",
    formula_str = formula,
    model_function = ols_regression_fit,
    cv_function = evaluate_models_loo,
    description = "bias_ols_weighted",
    k=20)
}


