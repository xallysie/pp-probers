# L1, L2, elastic net regularization
# last updated 2024/06/05 -sally

# setup -------------------------------------------------------------------
# load libraries
library(tidyverse)  # data hygiene
library(glmnet)     # elastic net
library(glmmLasso)  # elastic net with mixed-effects models
library(foreach)    # foreach() function
library(doParallel) # parallel computing
library(parallel)   # parallel computing
library(furrr)      # parallel computing
library(caret)      # cross-validation
library(MuMIn)      # model evaluation
library(Metrics)    # model evaluation

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
#raw_predictors_key <- names(train_data)[10:45]
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


# prepare data ------------------------------------------------------------

# remove variables that were used to construct the outcome variables (bias and outgroup_att)
train_data <- train_data %>%
  dplyr::select(everything(), -WarmIG, -WarmOG, -PositiveIG, -PositiveOG, -LikeIG, -LikeOG, -diff_warm, -diff_pos, -diff_like, -sThreat3R, -agreeable1r) %>%
  mutate(Outgroup = as.factor(Outgroup))


# create matrix of predictors, removing outcome variables and cluster(Outgroup)
predictors <- train_data %>%
  dplyr::select(-bias, -outgroup_att, -Outgroup) %>%
  as.matrix()

outcome_bias <- train_data[["bias"]]
outcome_outgroup_att <- train_data[["outgroup_att"]]
group <- train_data[["Outgroup"]] # these data are nested within the Outgroup cluster, which contains 15 unique groups


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# BASIC MODEL -------------------------------------------------------------

# let's try this with just the 36 raw measures + latent vars that came with
# the original dataset

# eventually we'll run this with the transformed vars that we create

# > elastic net without multilevel ----------------------------------------

# Register parallel backend
cl <- makeCluster(detectCores() - 1) # leave one core free
registerDoParallel(cl)

# cross-validation for elastic net
cv_fit <- cv.glmnet(x=predictors, y=outcome_bias, alpha=0.5, parallel=TRUE)

# best lambda
best_lambda <- cv_fit$lambda.min

# fit final model
elastic_net_model <- glmnet(x=predictors, y=outcome_bias, alpha = 0.5, lambda=best_lambda)

# Stop the cluster
stopCluster(cl)

# Summary of the model
print(elastic_net_model$beta)

# order by abs magnitude of weight
elastic_net_weights <- matrix(elastic_net_model$beta, dimnames=elastic_net_model$beta@Dimnames)
elastic_net_weights <- as.data.frame(elastic_net_weights)
elastic_net_weights <- cbind(rownames(elastic_net_weights), data.frame(elastic_net_weights, row.names=NULL))
names(elastic_net_weights) <- c("Variable","ParamWeight")
elastic_net_weights <- elastic_net_weights[order(abs(elastic_net_weights$ParamWeight), decreasing=T), ]

write_csv(elastic_net_weights, file="output/ElasticNet_NOTmultilevel_ParameterWeights_AllOriginalMeasures.csv")


# > L1 regularization w/ multilevel structure -----------------------------

# keep only predictors with non-zero weights from elastic net
elastic_net_approved_preds <- elastic_net_weights %>%
  filter(ParamWeight != 0) %>%
  select(Variable) %>%
  unlist() %>% as.character()

# define the fixed-effects formula (exclude random eff)
formula <- as.formula(
  paste("bias ~ ", paste(elastic_net_approved_preds, collapse = " + ")))
formula

train_data_elasticizedvars <- train_data %>%
  select(Outgroup, bias, outgroup_att, all_of(elastic_net_approved_preds), -contact_friends)

predictors_elasticizedvars <- train_data_elasticizedvars %>%
  select(-Outgroup, -bias, -outgroup_att) %>%
  as.matrix()

# Register parallel backend
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# define k for k-fold cv
k = 10
folds = createFolds(outcome_bias, k=k, list=T, returnTrain=T)

# define a range of lambda values to test
lambda_values <- seq(0.01, 20, by = 0.05)
lambda_values <- seq(0.2,1, by=0.2)

# cross-validation in parallel to tune lambda
tuning_results <- foreach(lambda = lambda_values, 
                          .packages = c('glmmLasso','Metrics','foreach'), .combine = rbind) %dopar% {
                            
  fold_rmse <- foreach(fold_idx = 1:k, .packages = c('glmmLasso','Metrics','foreach'), .combine = c) %do% {
    train_idx <- folds[[fold_idx]]
    val_idx <- setdiff(seq_len(nrow(train_data_elasticizedvars)), train_idx)
    
    train_fold <- train_data_elasticizedvars[train_idx, ]
    val_fold <- train_data_elasticizedvars[val_idx, ]
    
    model <- glmmLasso(fix = formula, rnd = list(Outgroup = ~1), data=train_data_elasticizedvars,
                       lambda = lambda, family = gaussian(link = "identity"))
    
    preds <- predict(model, newdata = val_fold)
    rmse_val <- rmse(val_fold$bias, preds)
    return(rmse_val)
  }
  
  mean_rmse <- mean(fold_rmse)
  data.frame(lambda = lambda, mean_rmse = mean_rmse)
}

# stop cluster
stopCluster(cl)

# find best lambda
best_lambda <- tuning_results[which.min(tuning_results$mean_rmse), "lambda"]

# fit final model w best lambda: fixed lmer w/ lasso regularization
final_model <- glmmLasso(fix = formula, rnd = list(Outgroup = ~1), data = train_data, 
                         lambda = best_lambda, family = gaussian(link = "identity"))

# summary of the final model
summary(final_model)

# predict on the training data (as a stand-in for evaluation since we don't have a separate test set)
train_data_elasticizedvars$final_preds <- predict(final_model, newdata = data.frame(train_data_elasticizedvars))

# get rmse and r-squared for the final model
final_rmse <- rmse(train_data_elasticizedvars$bias, train_data_elasticizedvars$final_preds)
final_r2 <- R2(train_data_elasticizedvars$final_preds, train_data_elasticizedvars$bias)

# Print the final evaluation metrics
cat("Final Mixed-Effects Lasso RMSE:", final_rmse, "R2:", final_r2, "\n")

# order by abs magnitude of weight
glmmlasso_weights <- data.frame(final_model$coefficients)
glmmlasso_weights <- cbind(rownames(glmmlasso_weights), data.frame(glmmlasso_weights, row.names=NULL))
names(glmmlasso_weights) <- c("Variable","ParamWeight")
glmmlasso_weights <- glmmlasso_weights[order(abs(glmmlasso_weights$ParamWeight), decreasing=T), ]

write_csv(glmmlasso_weights, file="output/glmmLasso_multilevel_ParameterWeights_OriginalMeasuresElasticized.csv")



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# APPLY VAR TRANSFORMATIONS -----------------------------------------------

train_data_trans <- train_data

predictors_trans <- train_data_trans %>%
  select(-Outgroup,-bias,-outgroup_att)

# create a bunch of new variables and see what sticks

# loop through dataframe and cgrand-mean-center predictors (we will cluster mean-center later in mlms)
train_data_gmc <- train_data_trans
for (var_name in names(predictors_trans)) {
  var_name <- ensym(var_name)
  new_var_name <- paste0(rlang::as_string(var_name),"_gmc") # grand-mean-center
  train_data_gmc <- train_data_gmc %>%
    mutate(!!new_var_name := !!var_name - mean(!!var_name))
}
train_data_gmc <- train_data_gmc %>% select(Outgroup, bias, outgroup_att, contains("_gmc"))
predictors_gmc <- train_data_gmc %>% select(-Outgroup, -bias, -outgroup_att, contains("_gmc"))

# create 2-way interactions between all unique pairs of variables
interactions <- data.frame(matrix(nrow=2010))
for (i in seq_along(predictors_gmc)) {
  for (j in seq_along(predictors_gmc)) {
    if (i < j) {  # to avoid duplicate pairs
      col1 <- names(predictors_gmc)[i]
      col2 <- names(predictors_gmc)[j]
      interaction_name <- paste0(col1, "_x_", col2)
      interactions[[interaction_name]] <- predictors_gmc[[col1]] * predictors_gmc[[col2]]
    }
  }
}
interactions <- interactions[,-1] # remove NA column

# loop through each variable (not centered) and create a log-transformed version
train_data_log <- train_data_trans
for (var_name in names(predictors_trans)) {
  var_name <- ensym(var_name)
  new_var_name <- paste0(rlang::as_string(var_name),"_log")
  train_data_log <- train_data_log %>%
    mutate(!!new_var_name := log10(!!var_name))
}
# find and drop columns with NAN values
log_vars_with_NaN <- train_data_log %>%
  summarise_all(~ any(is.na(.))) %>%
  unlist() %>%
  which()
train_data_log <- train_data_log %>%
  select(-all_of(log_vars_with_NaN)) %>%
  select(contains("_log"))

# combine these transformed variables
train_data_trans <- cbind(train_data_gmc, interactions) %>%
  cbind(train_data_log)

# > elastic net without multilevel ----------------------------------------

# redefine predictors
predictors <- train_data_trans %>%
  dplyr::select(-bias, -outgroup_att, -Outgroup) %>%
  as.matrix()

# Register parallel backend
cl <- makeCluster(detectCores() - 1) # leave one core free
registerDoParallel(cl)

# cross-validation for elastic net
cv_fit <- cv.glmnet(x=predictors, y=outcome_bias, alpha=0.5, parallel=TRUE)

# best lambda
best_lambda <- cv_fit$lambda.min

# fit final model
elastic_net_model <- glmnet(x=predictors, y=outcome_bias, alpha = 0.5, lambda=best_lambda)

# Stop the cluster
stopCluster(cl)

# Summary of the model
print(elastic_net_model$beta)

# order by abs magnitude of weight
elastic_net_weights <- matrix(elastic_net_model$beta, dimnames=elastic_net_model$beta@Dimnames)
elastic_net_weights <- as.data.frame(elastic_net_weights)
elastic_net_weights <- cbind(rownames(elastic_net_weights), data.frame(elastic_net_weights, row.names=NULL))
names(elastic_net_weights) <- c("Variable","ParamWeight")
elastic_net_weights <- elastic_net_weights[order(abs(elastic_net_weights$ParamWeight), decreasing=T), ]

write_csv(elastic_net_weights, file="output/ElasticNet_NOTmultilevel_ParameterWeights_Transformed_V1.csv")


# APPLY MORE VAR TRANSFORMATIONS ------------------------------------------

# added 2024/06/9

load("train_data_transformations_V2_with2653vars.Rda")

# redefine predictors
predictors <- train_data_trans %>%
  dplyr::select(-bias, -outgroup_att, -Outgroup) %>%
  as.matrix()

# Register parallel backend
cl <- makeCluster(detectCores() - 1) # leave one core free
registerDoParallel(cl)

# cross-validation for elastic net
cv_fit <- cv.glmnet(x=predictors, y=outcome_bias, alpha=0.5, parallel=TRUE)

# best lambda
best_lambda <- cv_fit$lambda.min

# fit final model
elastic_net_model <- glmnet(x=predictors, y=outcome_bias, alpha = 0.5, lambda=best_lambda)

# Stop the cluster
stopCluster(cl)

# Summary of the model
print(elastic_net_model$beta)

# order by abs magnitude of weight
elastic_net_weights <- matrix(elastic_net_model$beta, dimnames=elastic_net_model$beta@Dimnames)
elastic_net_weights <- as.data.frame(elastic_net_weights)
elastic_net_weights <- cbind(rownames(elastic_net_weights), data.frame(elastic_net_weights, row.names=NULL))
names(elastic_net_weights) <- c("Variable","ParamWeight")
elastic_net_weights <- elastic_net_weights[order(abs(elastic_net_weights$ParamWeight), decreasing=T), ]

write_csv(elastic_net_weights, file="output/ElasticNet_NOTmultilevel_ParameterWeights_Transformed_V2_with2653preds.csv")


# > TO-DO: L1 regularization w/ multilevel structure -----------------------------
# THIS HASN'T BEEN IMPLEMENTED YET #############################################
########### CODE BELOW IS COPY/PASTED FROM ABOVE ###############################
# keep only predictors with non-zero weights from elastic net
elastic_net_approved_preds <- elastic_net_weights %>%
  filter(ParamWeight != 0) %>%
  select(Variable) %>%
  unlist() %>% as.character()

# define the fixed-effects formula (exclude random eff)
formula <- as.formula(
  paste("bias ~ ", paste(elastic_net_approved_preds, collapse = " + ")))
formula

train_data_elasticizedvars <- train_data %>%
  select(Outgroup, bias, outgroup_att, all_of(elastic_net_approved_preds), -contact_friends)

predictors_elasticizedvars <- train_data_elasticizedvars %>%
  select(-Outgroup, -bias, -outgroup_att) %>%
  as.matrix()

# Register parallel backend
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# define k for k-fold cv
k = 10
folds = createFolds(outcome_bias, k=k, list=T, returnTrain=T)

# define a range of lambda values to test
lambda_values <- seq(0.01, 20, by = 0.05)
lambda_values <- seq(0.2,1, by=0.2)

# cross-validation in parallel to tune lambda
tuning_results <- foreach(lambda = lambda_values, 
                          .packages = c('glmmLasso','Metrics','foreach'), .combine = rbind) %dopar% {
                            
                            fold_rmse <- foreach(fold_idx = 1:k, .packages = c('glmmLasso','Metrics','foreach'), .combine = c) %do% {
                              train_idx <- folds[[fold_idx]]
                              val_idx <- setdiff(seq_len(nrow(train_data_elasticizedvars)), train_idx)
                              
                              train_fold <- train_data_elasticizedvars[train_idx, ]
                              val_fold <- train_data_elasticizedvars[val_idx, ]
                              
                              model <- glmmLasso(fix = formula, rnd = list(Outgroup = ~1), data=train_data_elasticizedvars,
                                                 lambda = lambda, family = gaussian(link = "identity"))
                              
                              preds <- predict(model, newdata = val_fold)
                              rmse_val <- rmse(val_fold$bias, preds)
                              return(rmse_val)
                            }
                            
                            mean_rmse <- mean(fold_rmse)
                            data.frame(lambda = lambda, mean_rmse = mean_rmse)
                          }

# stop cluster
stopCluster(cl)

# find best lambda
best_lambda <- tuning_results[which.min(tuning_results$mean_rmse), "lambda"]

# fit final model w best lambda: fixed lmer w/ lasso regularization
final_model <- glmmLasso(fix = formula, rnd = list(Outgroup = ~1), data = train_data, 
                         lambda = best_lambda, family = gaussian(link = "identity"))

# summary of the final model
summary(final_model)

# predict on the training data (as a stand-in for evaluation since we don't have a separate test set)
train_data$final_preds <- predict(final_model, newdata = data.frame(train_data))

# get rmse and r-squared for the final model
final_rmse <- rmse(train_data$bias, train_data$final_preds)
final_r2 <- R2(train_data$final_preds, train_data$bias)

# Print the final evaluation metrics
cat("Final Mixed-Effects Lasso RMSE:", final_rmse, "R2:", final_r2, "\n")

# order by abs magnitude of weight
glmmlasso_weights <- data.frame(final_model$coefficients)
glmmlasso_weights <- cbind(rownames(glmmlasso_weights), data.frame(glmmlasso_weights, row.names=NULL))
names(glmmlasso_weights) <- c("Variable","ParamWeight")
glmmlasso_weights <- glmmlasso_weights[order(abs(glmmlasso_weights$ParamWeight), decreasing=T), ]

write_csv(glmmlasso_weights, file="output/glmmLasso_multilevel_ParameterWeights_Phase2Regulariation.csv")




