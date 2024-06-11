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
library(caret)      # cross-validation
library(MuMIn)      # model evaluation
library(mgcv)       # general additive models
library(MASS)       # robust linear models
select <- dplyr::select
#library(randomForest) # random forests 

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

# how many n per group? (K = 15, n=134, balanced)
train_data %>% select(Outgroup) %>%
  group_by(Outgroup) %>%
  summarise(n = n(), .groups = "drop")

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
baseline_model %>% 
  group_by(Outgroup) %>%
  summarise(sum(SS_regress_bias)/sum(SS_total_bias)) %>%
  ungroup() %>% select(-Outgroup) %>% unlist() %>% mean() # R_squared for Bias: .396
# adjusted r-squared***
baseline_model %>% 
  group_by(Outgroup) %>%
  summarise(1 - ((1-(sum(SS_regress_bias)/sum(SS_total_bias)))*(nrow(baseline_model)-1)/(nrow(baseline_model)-5-1))) %>%
  ungroup() %>% select(-Outgroup) %>% unlist() %>% mean() # Adjusted R-Sq: .394
# root mean squared error***
baseline_model %>% 
  group_by(Outgroup) %>%
  summarise(sqrt(mean((bias - baseline_preds_bias)^2))) %>%
  ungroup() %>% select(-Outgroup) %>% unlist() %>% mean() # RMSE = .903

# baseline values to beat for outgroup attitudes:
# r-squared
baseline_model %>% 
  group_by(Outgroup) %>%
  summarise(sum(SS_regress_oA)/sum(SS_total_oA)) %>%
  ungroup() %>% select(-Outgroup) %>% unlist() %>% mean() # R_squared for Bias: .391
# adjusted r-squared***
baseline_model %>% 
  group_by(Outgroup) %>% 
  summarise(1 - ((1-(sum(SS_regress_oA)/sum(SS_total_oA)))*(nrow(baseline_model)-1)/(nrow(baseline_model)-4-1))) %>%
  ungroup() %>% select(-Outgroup) %>% unlist() %>% mean() # Adjusted R-Sq: .390
# root mean squared error***
baseline_model %>% 
  group_by(Outgroup) %>%
  summarise(sqrt(mean((outgroup_att - baseline_preds_oA)^2))) %>%
  ungroup() %>% select(-Outgroup) %>% unlist() %>% mean() # RMSE = 1.87

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

# ols regression revised to assess avg model performance across clusters
ols_regression_fit <- function(data, formula_str, cluster_name, weights=NULL, ...){
  results <- as.data.frame(matrix(ncol=3,nrow=length(levels(as.factor(data[[cluster_name]])))))
  names(results) <- c("Outgroup", "RMSE", "Adj_R_sq")
  results[[1]] <- levels(as.factor(data[[cluster_name]]))
  .formula <- as.formula(formula_str)
  xvars <- ### CONTINUE HERE
  yvar <- str_split(gsub(" ", "", formula_str), "~")[[1]][1]
    weights
  
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

# robust linear models
rlm_regression_fit <- function(data, formula_str, weights=NULL, ...){
  .formula <- as.formula(formula_str)
  if (is.null(weights)) {
    model <- rlm(formula = .formula, data=data,
                 method=c("M"),
                 wt.method = c("inv.var"),
                 model = TRUE, x.ret = TRUE, y.ret = FALSE, contrasts = NULL)
  } else {
    predictor_vars <- all.vars(.formula)[-1] # get predictor variables from formula
    predictors <- data[, predictor_vars, drop=FALSE]
    weighted_preds <- sapply(seq_along(weights), function(x) {predictors[,x]*weights[x]})
    weights_preds_sum <- rowSums(data.frame(weighted_preds))
    model <- rlm(formula = .formula, data=data, weights=abs(weights_preds_sum),
                      method=c("M"),
                      wt.method = c("inv.var"),
                      model = TRUE, x.ret = TRUE, y.ret = FALSE, contrasts = NULL)
  }
  return(model)
}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# functions for model evaluation ------------------------------------------

# define wrapper function to evaluate models
evaluate_model <- function(model, val_fold, outcome_var, cluster_name) {
  val_predictions <- predict(model, newdata = val_fold, allow.new.levels = TRUE) # fit model on validation dataset

  # calculate root mean squared error
  val_rmse <- sqrt(mean((val_predictions - val_fold[[outcome_var]])^2)) # obtain RMSE on validation dataset
  
  # calculate adjusted R-squared based on model type 
  # **CHANGEME** if you add different types of models not shown here, please adjust this to recognize your model type
  if (inherits(model, "lm") && !"rlm" %in% class(model)) { # if linear regression and NOT robust method
    val_adj_r_squared <- summary(model)$adj.r.squared
  } else if (inherits(model, "lmerMod")) { # if mixed-effects model
    val_adj_r_squared <- r.squaredGLMM(model)[2] # get conditional r-squared
  } else if (inherits(model, "randomForest")) {
    val_adj_r_squared <- 1 - (sum((val_predictions - val_fold[[outcome_var]])^2) / 
                                sum((val_fold[[outcome_var]] - mean(val_fold[[outcome_var]]))^2))
  } else if (inherits(model, "rlm")) { # if robust linear regression
    resids <- val_predictions - val_fold[[outcome_var]] # residuals
    rss <- sum(resids^2) # residual sum of squares
    tss <- sum((val_fold[[outcome_var]] - mean(val_fold[[outcome_var]]))^2) # total sum of squares
    r_squared <- 1 - (rss/tss) # calculate r-squared
    n <- nrow(val_fold) # number of observations
    p <- length(coef(model)) - 1 # number of predictors (subtract 1 for intercept)
    val_adj_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
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

# example usage with robust linear model
results <- evaluate_models_kfold( # k-fold cv
  data = train_data,
  outcome_var = "bias", 
  formula_str = "bias ~ generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment", 
  description = "bias ~ .120*generalized + .173*symbolic + .209*contact_quality + .119*contact_friendsz + .523*identification_selfinvestment", 
  weights = c(.120, .173, .209, .119, .523),
  model_function = rlm_regression_fit,
  k = 20)

# example usage with linear mixed-effects model
#results <- evaluate_models_kfold(
#  data = train_data,
#  outcome_var = "bias", 
#  formula_str = "bias ~ generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment + (generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment | Outgroup)", 
#  description = "lmerMod with weights .120*generalized + .173*symbolic + .209*contact_quality + .119*contact_friendsz + .523*identification_selfinvestment", 
#  weights = c(.120, .173, .209, .119, .523),
#  model_function = lmer_regression_fit,
#  k = 5)
#results <- evaluate_models_loo( # DON'T RUN THIS, THIS TAKES A LONG TIME
#  data = train_data,
#  outcome_var = "bias", 
#  formula_str = "bias ~ generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment + (generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment | Outgroup)", 
#  description = "lmerMod with weights .120*generalized + .173*symbolic + .209*contact_quality + .119*contact_friendsz + .523*identification_selfinvestment", 
#  weights = c(.120, .173, .209, .119, .523),
#  model_function = lmer_regression_fit)

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
# test some models --------------------------------------------------------

# BIAS MODEL 2.0
# take 55 variables from elastic net (whittled down from ~1400) and test 

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
# try weighted model
test_weighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider$ParamWeight,
  description = "bias_ols_weighted with 55 vars retained from elastic net", 
  k=5)
# prune model based on sig vals
train_data_trans_subset2 <- train_data_trans %>%
  select(Outgroup,bias,
         rThreatIG1_log, sThreat3_log, contact_quality_gmc, identification_sol_gmc, symbolic_gmc,
         generalized_gmc, identification_selfinvestment_gmc, contact_friendsz_gmc, identification_sat_gmc,
         sThreat3_gmc, Identification5_gmc, DisgustS4_gmc, Identification1_gmc_x_Identification8_gmc, 
         sThreat2_gmc_x_sThreat4_gmc, Identification2_gmc_x_identification_sol_gmc, Identification7_gmc_x_generalized_probdiff_gmc,
         Identification6_gmc_x_contact_quality_gmc, DisgustP6_gmc_x_disgust_r_gmc, Identification4_gmc_x_DisgustP1_gmc,
         sThreat3_gmc_x_DisgustP5_gmc, sThreat3_gmc_x_rThreatIG1_gmc, Identification4_gmc_x_Identification5_gmc, 
         DisgustS3_gmc_x_DisgustS4_gmc, rThreatIG2_gmc_x_generalized_gmc, Identification1_gmc_x_Identification6_gmc,
         rThreatIG1_gmc_x_contact_friends_gmc, rThreatOG1_gmc_x_contact_friends_gmc)
test_unweighted_2 <- single_run(
  data = train_data_trans_subset2,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(names(train_data_trans_subset2[3:29]), collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 24 vars retained from sig values from elastic net", 
  k=10)
test_weighted_2 <- single_run(
  data = train_data_trans_subset2,
  outcome_var = "bias",
  formula_str = paste("bias ~", paste(names(train_data_trans_subset2[3:29]), collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider[which(variables_to_consider$Variable %in% names(train_data_trans_subset2[3:29])),]$ParamWeight,
  description = "bias_ols_weighted with 24 vars retained from sig values from elastic net",
  k=10)

# BIAS MODEL 2.5 *** BEST SO FAR
# take 55 + 58 variables from two versions of elastic nets
# 55 whittled down from ~1400, 58 whittled down from ~2600
# total 68 vars
stopCluster(cl)

# load list of variables and starting values
load("train_data_transformations_V2_with2659vars.Rda")

variables_to_consider_1 <- read_csv("output/ElasticNet_NOTmultilevel_ParameterWeights_Transformed_V1.csv") %>%
  filter(ParamWeight != 0)
variables_to_consider_2 <- read_csv("output/ElasticNet_NOTmultilevel_ParameterWeights_Transformed_V2_with2653preds.csv") %>%
  filter(ParamWeight != 0)
new_variables_to_consider <- variables_to_consider_2 %>%
  filter(!Variable %in% variables_to_consider_1$Variable)
old_variables_dropped <- variables_to_consider_1 %>%
  filter(!Variable %in% variables_to_consider_2$Variable)

variables_to_consider <- rbind(variables_to_consider_2, old_variables_dropped) %>%
  rbind(c("extreme_proportion_q_gmc",.005),c("item_variances_gmc",.005)) %>%
  mutate(ParamWeight= as.numeric(ParamWeight))

train_data_trans_subset <- train_data_trans %>%
  dplyr::select(Outgroup, bias, all_of(variables_to_consider$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 70 vars retained from elastic net", 
  k=20) 
# try weighted 
test_weighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider$ParamWeight,
  description = "bias_ols_weighted with 70 vars retained from elastic net", 
  k=20) 
# try robust ver
test_unweighted_r <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = rlm_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_rlm_unweighted with 70 vars retained from elastic net", 
  k=20) 


# BIAS MODEL 2.5B 
# take 55 + 58 variables from two versions of elastic nets
# total 68+2 vars
# also replace squared vals with abs()
stopCluster(cl)

# load list of variables and starting values
load("train_data_transformations_V3_with2658vars.Rda")

variables_to_consider_1 <- read_csv("output/ElasticNet_NOTmultilevel_ParameterWeights_Transformed_V1.csv") %>%
  filter(ParamWeight != 0)
variables_to_consider_2 <- read_csv("output/ElasticNet_NOTmultilevel_ParameterWeights_Transformed_V2_with2653preds.csv") %>%
  filter(ParamWeight != 0)
new_variables_to_consider <- variables_to_consider_2 %>%
  filter(!Variable %in% variables_to_consider_1$Variable)
old_variables_dropped <- variables_to_consider_1 %>%
  filter(!Variable %in% variables_to_consider_2$Variable)

variables_to_consider <- rbind(variables_to_consider_2, old_variables_dropped) %>%
  rbind(c("extreme_proportion_q_gmc",.005),c("item_variances_gmc",.005))

train_data_trans_subset <- train_data_trans %>%
  select(Outgroup, bias, all_of(variables_to_consider$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 70 vars retained from elastic net", 
  k=20)
# try weighted
test_weighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider$ParamWeight,
  description = "bias_ols_weighted with 70 vars retained from elastic net", 
  k=20) 

# BIAS MODEL 2.7
# applied interaction, polynomial, & MD transformation only to (sub)factors
# total 282 vars, whittled down to 54+2
# also replace squared vals with abs()
stopCluster(cl)

# load list of variables and starting values
load("train_data_transformationssubfactors_V3_with288vars.Rda")

variables_to_consider <- read_csv("output/ElasticNet_NOTmultilevel_Subfactors_Transformed_with288preds.csv") %>%
  filter(Variable != "item_variances" & Variable != "extreme_proportion_q") %>%
  filter(ParamWeight != 0) %>%
  rbind(c("extreme_proportion_q_gmc",.005), c("item_variances_gmc",.005))

train_data_trans_subset <- train_data_trans %>%
  select(Outgroup, bias, all_of(variables_to_consider$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 56 vars retained from elastic net", 
  k=20)
# try weighted
test_weighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider$ParamWeight,
  description = "bias_ols_weighted with 70 vars retained from elastic net", 
  k=20) 

# BIAS MODEL 2.8
# applied interaction, polynomial, & MD transformation TO ALL VARS
# total 2750 vars, whittled down to 52+2
# also replace squared vals with abs()
stopCluster(cl)

# load list of variables and starting values
load("train_data_transformations_V3_with2750vars.Rda")

variables_to_consider <- read_csv("output/ElasticNet_NOTmultilevel_ALL_Transformed_with2750preds.csv") %>%
  filter(Variable != "item_variances" & Variable != "extreme_proportion_q") %>%
  filter(ParamWeight != 0) %>%
  rbind(c("extreme_proportion_q_gmc",.005), c("item_variances_gmc",.005))

train_data_trans_subset <- train_data_trans %>%
  select(Outgroup, bias, all_of(variables_to_consider$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 54 vars retained from elastic net", 
  k=20)
# try weighted
test_weighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider$ParamWeight,
  description = "bias_ols_weighted with 54 vars retained from elastic net", 
  k=20)

summary(lm(
  paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  data=train_data_trans_subset
))

# BIAS MODEL 2.6
# take 383 variables, whittled down to 83
# (re-created factor/subfactors as sum scores, applied transformations mostly
# to sum scores)
stopCluster(cl)

# load list of variables and starting values
load("train_data_sumscores_Factortransformations_with383vars.Rda")

variables_to_consider <- read_csv("output/ElasticNet_NOTmultilevel_SumScoreFactors_Transformed_with383preds.csv") %>%
  filter(ParamWeight != 0)

variables_to_consider <- variables_to_consider %>%
  filter(Variable != "item_variances" & Variable != "extreme_proportion_sd")

train_data_trans_subset <- train_data_sumscores_trans %>%
  select(Outgroup, bias, all_of(variables_to_consider$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 93 vars retained from elastic net", 
  k=20)
# try weighted
test_weighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider$ParamWeight,
  description = "bias_ols_weighted with 93 vars retained from elastic net", 
  k=20) 

# BIAS MODEL 2.7
# take 2794 variables, whittled down to 77 + 2
# (re-created factor/subfactors as sum scores, applied transformations)
stopCluster(cl)

# load list of variables and starting values
load("train_data_sumscores_transformations_with2794vars.Rda")

variables_to_consider <- read_csv("output/ElasticNet_NOTmultilevel_SumScore_Transformed_with2794preds.csv") %>%
  filter(ParamWeight != 0) %>%
  rbind(c("extreme_proportion_q_gmc",.005),c("item_variances_gmc",.005))

train_data_trans_subset <- train_data_sumscores_trans %>%
  select(Outgroup, bias, all_of(variables_to_consider$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 77 vars retained from elastic net", 
  k=20)
# try weighted
test_weighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider$ParamWeight,
  description = "bias_ols_weighted with 93 vars retained from elastic net", 
  k=20) 

# BIAS MODEL 6.0
# take vars from stricter versions of elastic nets
# 11 whittled down from ~1400, 58 whittled down from ~2600

# load list of variables and starting values
load("train_data_transformations_V2_with2653vars.Rda")

variables_to_consider_3 <- read_csv("output/ElasticNet_Lambda0.50_Alpha0.70_11vars.csv") %>%
  filter(ParamWeight != 0)
train_data_trans_subset <- train_data_trans %>%
  select(Outgroup, bias, all_of(variables_to_consider_3$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider_3$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 11 vars retained from elastic net", 
  k=20)
# try weighted
test_weighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider_3$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider_3$ParamWeight,
  description = "bias_ols_weighted with 11 vars retained from elastic net", 
  k=20) 


predicted_values <- predict(ols_regression_fit(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider_3$Variable, collapse = " + "))
), newdata = train_data_trans_subset, se.fit = TRUE)
preds <- train_data_trans_subset %>%
  select(Outgroup,bias) %>%
  cbind(predicted_values$fit)




lasso_preds11 = as.matrix(train_data_trans %>%
                          select(
                            all_of(variables_to_consider_3$Variable))
) %*% variables_to_consider_3$ParamWeight

theme_set(theme_classic())

library(data.table)
lasso_model11 = data.table(
  data.frame(lasso11 = lasso_preds11,
             lasso = lasso_preds,
             baseline = baseline_model$baseline_preds_bias,
             bias = train_data$bias)
)

# lasso works well, but
ggplot(lasso_model11, aes(x = lasso11, y= lasso)) +
  geom_point(alpha = .2)  + geom_smooth(method = 'lm')

ggplot(lasso_model11, aes(x = bias - lasso, y= lasso11)) +
  geom_point(alpha = .2)  + geom_smooth(method = 'lm')



# BIAS 4.0 without MDstuff
variables_to_consider_7 <- elastic_net_weights %>%
  filter(ParamWeight != 0)
train_data_trans_subset <- train_data_trans %>%
  select(Outgroup, bias, all_of(variables_to_consider_7$Variable))

test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider_7$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 47 vars retained from elastic net", 
  k=20)


predicted_values <- predict(ols_regression_fit(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider_7$Variable, collapse = " + "))
), newdata = train_data_trans_subset, se.fit = TRUE)
preds <- train_data_trans_subset %>%
  select(Outgroup,bias) %>%
  cbind(predicted_values$fit)




lasso_preds = as.matrix(train_data_trans %>%
                          select(
                            all_of(variables_to_consider_7$Variable))
) %*% variables_to_consider_7$ParamWeight

theme_set(theme_classic())

library(data.table)
lasso_model = data.table(
  data.frame(lasso = lasso_preds,
             baseline = baseline_model$baseline_preds_bias,
             bias = train_data$bias)
)

# lasso works well, but
ggplot(lasso_model, aes(x = bias, y= lasso)) +
  geom_point(alpha = .2)  + geom_smooth(method = 'lm')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# test with outgroup attitudes

# OUTGROUP ATTITUDES MODEL 3.0
# take vars from stricter versions of elastic nets
# 55 whittled down from ~1400, 58 whittled down from ~2600

# load list of variables and starting values
load("train_data_transformations_V2_with2653vars.Rda")
load("train_data_transformations_V3_with2750vars.Rda")

variables_to_consider_4 <- read_csv("output/ElasticNet_OutgroupAttitudes_ParameterWeights_Transformed_V2_with2653preds.csv") %>%
  filter(ParamWeight != 0) 
train_data_trans_subset <- train_data_trans %>%
  select(Outgroup, outgroup_att, all_of(variables_to_consider_4$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "outgroup_att", 
  formula_str = paste("outgroup_att ~", paste(variables_to_consider_4$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "outgroup_att_ols_unweighted with 70 vars retained from elastic net", 
  k=20) # rmse = .723, adjr = .674



# try weighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "outgroup_att", 
  formula_str = paste("outgroup_att ~", paste(variables_to_consider_4$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider_4$ParamWeight,
  description = "outgroup_att_ols_weighted with 70 vars retained from elastic net", 
  k=20)

# try model with stricter lambda
variables_to_consider_5 <- read_csv("output/ElasticNet_OutgroupAttitudes_Lambda0.50_Alpha0.70_11vars.csv") %>%
  filter(ParamWeight != 0)
train_data_trans_subset <- train_data_trans %>%
  select(Outgroup, outgroup_att, all_of(variables_to_consider_5$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "outgroup_att", 
  formula_str = paste("outgroup_att ~", paste(variables_to_consider_5$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "outgroup_att_ols_unweighted with 5 vars retained from elastic net", 
  k=20)




# OUTGROUP ATT 4.0
# applied interaction, polynomial, & MD transformation TO ALL VARS
# total 2750 vars, whittled down to 52+2
# also replace squared vals with abs()
stopCluster(cl)

# load list of variables and starting values
load("train_data_transformations_V3_with2750vars.Rda")

variables_to_consider <- read_csv("output/ElasticNet_NOTmultilevel_OutgroupAtt_Transformed_with2750preds.csv") %>%
  filter(Variable != "item_variances" & Variable != "extreme_proportion_q") %>%
  filter(ParamWeight != 0) %>%
  rbind(c("extreme_proportion_q_gmc",.005), c("item_variances_gmc",.005))

train_data_trans_subset <- train_data_trans %>%
  select(Outgroup, outgroup_att, all_of(variables_to_consider$Variable))

# try unweighted model
test_unweighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "outgroup_att", 
  formula_str = paste("outgroup_att ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "outgroup_att_ols_unweighted with 70 vars retained from elastic net", 
  k=20)


# get coefs separately for each group -------------------------------------

coefsss <- list()
for (k in train_data_trans_subset[["Outgroup"]]){
  coeffs.df <- data.frame(matrix(ncol=3,nrow=length(variables_to_consider$Variable)-3))
  names(coeffs.df) <- c("Outgroup","Variable","Coefficient")
  coeffs.df$Outgroup <- k
  dat. <- train_data_trans_subset[train_data_trans_subset[["Outgroup"]] == k,]
  model. <- lm(formula = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
            data=dat.)
  coeffs.df$Variable <- rownames(summary(model.)$coefficients[-1,])
  coeffs.df$Coefficient <- data.frame(summary(model.)$coefficients[-1,])$Estimate
  coefsss[[k]] <- coeffs.df
}
coefficientz <- do.call(rbind, coefsss)
coefficientz.summary <- coefficientz %>%
  group_by(Variable) %>%
  summarize(ParamWeight = mean(Coefficient, na.rm=TRUE))



# test models with paper_data ---------------------------------------------

paper_data <- list.files(path = "../TrainingData/paper_data/", pattern = "*.csv") %>% 
  str_subset("CorrPlot", negate=TRUE) %>%
  map_df(~read_csv(paste0("../TrainingData/paper_data/",.), show_col_types=F))


# OLD STUFF TO TRY

# try weighted model
test_weighted <- single_run(
  data = train_data_trans_subset,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider$ParamWeight,
  description = "bias_ols_weighted with 55 vars retained from elastic net", 
  k=5)
# prune model based on sig vals
train_data_trans_subset2 <- train_data_trans %>%
  select(Outgroup,bias,
         rThreatIG1_log, sThreat3_log, contact_quality_gmc, identification_sol_gmc, symbolic_gmc,
         generalized_gmc, identification_selfinvestment_gmc, contact_friendsz_gmc, identification_sat_gmc,
         sThreat3_gmc, Identification5_gmc, DisgustS4_gmc, Identification1_gmc_x_Identification8_gmc, 
         sThreat2_gmc_x_sThreat4_gmc, Identification2_gmc_x_identification_sol_gmc, Identification7_gmc_x_generalized_probdiff_gmc,
         Identification6_gmc_x_contact_quality_gmc, DisgustP6_gmc_x_disgust_r_gmc, Identification4_gmc_x_DisgustP1_gmc,
         sThreat3_gmc_x_DisgustP5_gmc, sThreat3_gmc_x_rThreatIG1_gmc, Identification4_gmc_x_Identification5_gmc, 
         DisgustS3_gmc_x_DisgustS4_gmc, rThreatIG2_gmc_x_generalized_gmc, Identification1_gmc_x_Identification6_gmc,
         rThreatIG1_gmc_x_contact_friends_gmc, rThreatOG1_gmc_x_contact_friends_gmc)
test_unweighted_2 <- single_run(
  data = train_data_trans_subset2,
  outcome_var = "bias", 
  formula_str = paste("bias ~", paste(names(train_data_trans_subset2[3:29]), collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  description = "bias_ols_unweighted with 24 vars retained from sig values from elastic net", 
  k=10)
test_weighted_2 <- single_run(
  data = train_data_trans_subset2,
  outcome_var = "bias",
  formula_str = paste("bias ~", paste(names(train_data_trans_subset2[3:29]), collapse = " + ")),
  model_function = ols_regression_fit,
  cv_function = evaluate_models_kfold,
  weights = variables_to_consider[which(variables_to_consider$Variable %in% names(train_data_trans_subset2[3:29])),]$ParamWeight,
  description = "bias_ols_weighted with 24 vars retained from sig values from elastic net",
  k=10)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# function for parameter search -------------------------------------------

######## TO-DO, NOT YET IMPLEMENTED ####################
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

