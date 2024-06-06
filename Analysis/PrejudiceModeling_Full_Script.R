# model comparisons for prejudice modeling competition
# last updated 2024/06/05 -sally

# setup -------------------------------------------------------------------
# load libraries
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
#library(randomForest) # for random forests 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 
set.seed(57812623)
options(scipen=999,digits=6)

# import training data
train_data <- read_csv("../TrainingData/train.csv") %>%
  dplyr::select(Outgroup, bias, outgroup_att, everything())

# fix training data by recoding reversed values
train_data <- train_data %>%
  mutate(sThreat3 = 8 - sThreat3,
         Agreeable1 = 6 - Agreeable1)

# outcome variables
# (1) outgroup_att = factor score calculated from WarmOG, PositiveOG, LikeOG
# (2) bias = factor score calculated from WarmIG-WarmOG, PositiveIG-PositiveeOG, LikeIG-LikeOG (positive score is more ingroup positivity bias)
raw_predictors_key <- c("sThreat1", "sThreat2", "sThreat3", "sThreat4",
                        "Identification1", "Identification2", "Identification3", "Identification4", "Identification5",
                        "Identification6", "Identification7", "Identification8", "Identification9", "Identification10",
                        "ContactQ1", "ContactQ2", "ContactQ3", "ContactN1",
                        "Agreeable1", "Agreeable2",
                        "rThreatIG1", "rThreatIG2", "rThreatOG1", "rThreatOG2",
                        "DisgustP1", "DisgustP2", "DisgustP3", "DisgustP4", "DisgustP5", "DisgustP6",
                        "DisgustS1", "DisgustS2", "DisgustS3", "DisgustS4",
                        "DisgustR1", "DisgustR2")
transformed_vars_key <- c("diff_warm", # WarmIG - WarmOG
                          "diff_pos",  # PositiveIG - PositiveOG
                          "diff_like", # LikeIG - LikeOG
                          "contact_friendsz", # same as contact_friends but z-scored
                          "generalized_challdiff", # rThreatOG1 - rThreatIG1
                          "generalized_probdiff"  # rThreatOG2 - rThreatIG2
                          )
latent_vars_key <- c(
  "symbolic", "identification_sol", "identification_sat", "identification_cen",
  "identification_selfinvestment", "contact_quality",
  "b5a",
  "generalized", "disgust_p", "disgust_s", "disgust_r"
)

# remove variables that were used to construct the outcome variables (bias and outgroup_att)
# also remove reverse-coded variables that I already manipulated
train_data <- train_data %>%
  dplyr::select(everything(), -WarmIG, -WarmOG, -PositiveIG, -PositiveOG, -LikeIG, -LikeOG, -diff_warm, -diff_pos, -diff_like, -sThreat3R, -agreeable1r) %>%
  mutate(Outgroup = as.factor(Outgroup))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# evaluate baseline model -------------------------------------------------

# create weights for OLS regression (Prej Model 1.0)
baseline_model <- train_data %>%
  rowwise() %>%
  mutate(
    weights_bias = .120*generalized + .173*symbolic - .209*contact_quality - .119*contact_friendsz + .523*identification_selfinvestment,
    weights_oA = .243*symbolic - .462*contact_quality - .119*contact_friendsz - .128*b5a) %>%
  ungroup() %>%
  mutate(
    baseline_preds_bias = weights_bias + mean(bias),
    baseline_preds_oA = weights_oA + mean(outgroup_att),
    SS_total_bias = (bias - mean(bias))^2,
    SS_regress_bias = (baseline_preds_bias - mean(bias))^2,
    SS_total_oA = (outgroup_att - mean(outgroup_att))^2,
    SS_regress_oA = (baseline_preds_oA - mean(outgroup_att))^2
  )

# here are the baseline values to beat for bias

# r-squared
baseline_model %>% summarise(sum(SS_regress_bias)/sum(SS_total_bias)) # R_squared for Bias: .382
# adjusted r-squared***
baseline_model %>% summarise(1 - ((1-(sum(SS_regress_bias)/sum(SS_total_bias)))*(nrow(baseline_model)-1)/(nrow(baseline_model)-5-1))) # Adjusted R-Sq: .380
# root mean squared error***
baseline_model %>% summarise(sqrt(mean((bias - baseline_preds_bias)^2))) # RMSE = .913

# baseline values to beat for outgroup attitudes:
# r-squared
baseline_model %>% summarise(sum(SS_regress_oA)/sum(SS_total_oA)) # R_squared for Bias: .391
# adjusted r-squared***
baseline_model %>% summarise(1 - ((1-(sum(SS_regress_oA)/sum(SS_total_oA)))*(nrow(baseline_model)-1)/(nrow(baseline_model)-4-1))) # Adjusted R-Sq: .390
# root mean squared error***
baseline_model %>% summarise(sqrt(mean((outgroup_att - baseline_preds_oA)^2))) # RMSE = 1.90

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# functions to fit models -------------------------------------------------

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
    description = description,
    outcome = outcome_var,
    cross_val_results = results, 
    mean_rmse = mean_rmse, 
    mean_adj_r_squared = mean_adj_r_squared)
  
  print(paste0("Description: ", description))
  print(paste0("Mean RMSE: ", results_list$mean_rmse))
  print(paste0("Adjusted R-squared: ", results_list$mean_adj_r_squared))
  
  return(results_list)
}

# example usage with OLS regression on base model
results <- evaluate_models_kfold( # k-fold cv
  data = train_data,
  outcome_var = "bias", 
  formula_str = "bias ~ generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment", 
  description = "bias ~ .120*generalized + .173*symbolic + .209*contact_quality + .119*contact_friendsz + .523*identification_selfinvestment", 
  weights = c(.120, .173, .209, .119, .523),
  model_function = ols_regression_fit,
  k = 20)
results <- evaluate_models_loo( # leave-one-out cv
  data = train_data,
  outcome_var = "bias", 
  formula_str = "bias ~ generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment", 
  description = "bias ~ .120*generalized + .173*symbolic + .209*contact_quality + .119*contact_friendsz + .523*identification_selfinvestment", 
  weights = c(.120, .173, .209, .119, .523),
  model_function = ols_regression_fit)

# example usage with linear mixed-effects model
results <- evaluate_models_kfold(
  data = train_data,
  outcome_var = "bias", 
  formula_str = "bias ~ generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment + (generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment | Outgroup)", 
  description = "lmerMod with weights .120*generalized + .173*symbolic + .209*contact_quality + .119*contact_friendsz + .523*identification_selfinvestment", 
  weights = c(.120, .173, .209, .119, .523),
  model_function = lmer_regression_fit,
  k = 5)
results <- evaluate_models_loo( # DON'T RUN THIS, THIS TAKES A LONG TIME
  data = train_data,
  outcome_var = "bias", 
  formula_str = "bias ~ generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment + (generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment | Outgroup)", 
  description = "lmerMod with weights .120*generalized + .173*symbolic + .209*contact_quality + .119*contact_friendsz + .523*identification_selfinvestment", 
  weights = c(.120, .173, .209, .119, .523),
  model_function = lmer_regression_fit)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# convenience function: randomly generate string --------------------------

generate_random_string <- function(n) {
  chars <- c(letters, LETTERS, 0:9)
  unique_seed <- as.integer(Sys.time()) + Sys.getpid()
  set.seed(unique_seed)
  random_string <- paste0(sample(chars, n, replace = TRUE), collapse = "")
  return(random_string)
}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# single run: Wrap all steps, parallelize parameter search ----------------

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
  results_csv$Description <- result$description
  results_csv$Weights <- paste(result$weights, collapse=",")
  results_csv$mean_rmse <- result$mean_rmse
  results_csv$mean_adj_r_squared <- result$mean_adj_r_squared
  write_csv(results_csv, file=paste0("parameter_search_out/",description,"_",generate_random_string(16),".csv"))
  return(result)
}

# test single run
test <- single_run(
  data = train_data,
  outcome_var = "bias", 
  formula_str = "bias ~ generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment", 
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_weighted", 
  k=20,
  weights = c(.120, .173, .209, .119, .523))

# parallelize setup
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# function for parameter search -------------------------------------------

# define parameter grid for parallel execution
# load list of variables and starting values
load("train_data_transformations_V1.Rda")

variables_to_consider <- read_csv("output/ElasticNet_NOTmultilevel_ParameterWeights_Transformed_V1.csv") %>%
  filter(ParamWeight != 0) #%>%
  #filter(abs(ParamWeight) >= 0.005)

train_data_trans_subset <- train_data_trans %>%
  select(Outgroup, bias, all_of(variables_to_consider$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 55 vars retained from elastic net", 
  k=5)
# try multilevel unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + "), paste(" + ("), paste(variables_to_consider$Variable, collapse = " + "), paste(" | Outgroup)")),
  model_function = lmer_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_lmer_unweighted with 55 vars retained from elastic net and 55 random intercepts", 
  k=5)

# generate weight ranges
generate_weight_range <- function(start_val, step = 0.1, range = 0.10) {
  seq(from = start_val - range, to = start_val + range, by = step)
}
weight_ranges <- lapply(variables_to_consider$ParamWeight, generate_weight_range)

# create all possible combinations of weights
generate_parameter_grid <- function(ranges, current_combination = list(), index = 1) {
  if (index > length(ranges)) {
    return(list(current_combination))
  } else {
    result <- list()
    for (value in ranges[[index]]) {
      new_combination <- c(current_combination, value)
      result <- c(result, generate_parameter_grid(ranges, new_combination, index + 1))
    }
    return(result)
  }
}
all_param_comboss <- generate_parameter_grid(weight_ranges)

parameter_grid <- list(
  list(formula_str = "bias ~ generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment", 
       description = "bias ~ .120*generalized + .173*symbolic + .209*contact_quality + .119*contact_friendsz + .523*identification_selfinvestment", 
       weights = c(.120, .173, .209, .119, .523)),
  # Add more parameter combinations here
)

results <- foreach(param = parameter_grid, .packages = c("lme4", "MuMIn", "caret", "dplyr")) %dopar% {
  single_run(
    data = train_data,
    outcome_var = "bias",
    formula_str = param$formula_str,
    model_function = ols_regression_fit,
    cv_function = evaluate_models_kfold,
    description = param$description,
    k = 20,
    weights = param$weights
  )
}

stopCluster(cl)

# Print results
print(results)

