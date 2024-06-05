# model comparisons for prejudice modeling competition


# setup -------------------------------------------------------------------
# load libraries
library(scales)     # to scale response variable
library(tidyverse)  # data hygiene
library(readr)      # code hygiene
library(rlang)      # code hygiene
library(psych)      # descriptive functions
library(lme4)       # multilevel models
library(lmerTest)   # quick-and-dirty p-vals for fixed effects
library(MASS)       # simulate from a multivariate normal distribution (MASS::mvrnorm)
library(foreach)    # foreach() function
library(doParallel) # parallel computing
library(parallel)   # parallel computing
library(furrr)      # parallel computing
library(ggplot2)    # data viz
library(patchwork)  # data viz
library(colorspace) # nice colours
library(caret)      # for cross-validation
library(MuMIn)      # for model evaluation

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # MAKE SURE I'M IN THE SAME FOLDER AS DATA
setseed(57812623)
options(scipen=999,digits=6)

# import training data
train_data <- read_csv("../TrainingData/train.csv") %>%
  dplyr::select(Outgroup, bias, outgroup_att, everything())

# outcome variables
# (1) outgroup attitude = factor score calculated from WarmOG, PositiveOG, LikeOG
# (2) bias = factor score calculated from WarmIG-WarmOG, PositiveIG-PositiveeOG, LikeIG-LikeOG (positive score is more ingroup positivity bias)
raw_predictors_key <- names(train_data)[10:45]

# define cores for parallel computing
# **CHANGEME** update this when submitting job to computing closer
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

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
    #r_sq = sum(SS_regress_bias)/sum(SS_total_bias), 
    #r_sqadj = 1 - ((1-r_sq)*(nrow(baseline_model)-1)/(nrow(baseline_model)-5-1)),
    #rmse = sqrt(mean((bias - baseline_preds_bias)^2)),
    SS_total_oA = (outgroup_att - mean(outgroup_att))^2,
    SS_regress_oA = (baseline_preds_oA - mean(outgroup_att))^2
  )

# baseline "values to beat"
# adjusted R-sq and RMSE for bias(prejudice)
baseline_model %>% summarise(sum(SS_regress_bias)/sum(SS_total_bias)) # R_squared for Bias: .382
baseline_model %>% summarise(1 - ((1-(sum(SS_regress_bias)/sum(SS_total_bias)))*(nrow(baseline_model)-1)/(nrow(baseline_model)-5-1))) # Adjusted R-Sq: .380
baseline_model %>% summarise(sqrt(mean((bias - baseline_preds_bias)^2))) # .913

# adjusted R-sq and RMSE for outgroup attitudes
baseline_model %>% summarise(sum(SS_regress_oA)/sum(SS_total_oA)) # R_squared for Bias: .391
baseline_model %>% summarise(1 - ((1-(sum(SS_regress_oA)/sum(SS_total_oA)))*(nrow(baseline_model)-1)/(nrow(baseline_model)-4-1))) # Adjusted R-Sq: .390
baseline_model %>% summarise(sqrt(mean((outgroup_att - baseline_preds_oA)^2))) # 1.90

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# function for model evaluation -------------------------------------------

# define function for model evaluation (w/ k-fold cross-validation)
evaluate_models_kfold <- function(data,           # training data (dataframe)
                                  outcome_var,    # name of outcome variable (character)
                                  model_function, # function for building & fitting model
                                  k = 5,          # folds
                                  ...) {
  
  # perform k-fold cv
  folds <- createFolds(data[[outcome_var]], k = k, list = TRUE)
  results <- foreach(i = 1:k, .combine = rbind, .packages = c('tidyverse')) %dopar% {
    train_fold <- data[-folds[[i]], ]
    val_fold <- data[folds[[i]], ]
    
    # fit model
    model <- model_function(data = train_fold)
    
    # predict on validation set
    val_predictions <- predict(model, newdata = val_fold, allow.new.levels=TRUE)
    
    # calculate RMSE on validation set
    val_rmse <- sqrt(mean((val_predictions - val_fold$prejudice)^2))
    
    # calculate adjusted R-squared or conditional R-squared
    val_adj_r_squared <- summary(model)$adj.r.squared
    
    data.frame(fold = i, val_rmse = val_rmse, val_adj_r_squared = val_adj_r_squared)
  }
  
  # Calculate mean RMSE and Adjusted R-squared across all folds
  mean_rmse <- mean(results$val_rmse)
  mean_adj_r_squared <- mean(results$val_adj_r_squared)
  
  # Fit the final model on the entire dataset
  final_model <- model_function(data = data)
  
  return(list(model = final_model, 
              outcome = outcome_var,
              cross_val_results = results, 
              mean_rmse = mean_rmse, 
              mean_adj_r_squared = mean_adj_r_squared))
}

# Example usage with training data and baseline model (Prejudice 1.0) 
baselinemodel_outgroupattitude <- lm()
baseline_model <- function(data){
  model <- lm()
}

# Define the formula for the model
formula <- as.formula('prejudice ~ .')

# Evaluate the model
results <- evaluate_models(data, formula, k = 5)

# Print the results
print(paste("Cross-validated Mean RMSE: ", results$mean_rmse))
print(paste("Cross-validated Mean Adjusted R-squared: ", results$mean_adj_r_squared))

# Stop the parallel cluster
stopCluster(cl)


