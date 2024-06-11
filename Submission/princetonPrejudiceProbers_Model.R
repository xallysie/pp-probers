# model comparisons for prejudice modeling competition
# last updated 2024/06/05
# team: Princeton Prejudice Probers
# members: Sally Xie, Kerem Oktar, Bonan Zhao

# setup -------------------------------------------------------------------
# load libraries
library(tidyverse)  # data hygiene
library(readr)      # code hygiene
library(rlang)      # code hygiene
library(scales)     # to scale response variable
library(lme4)       # multilevel models
library(foreach)    # foreach() function
library(doParallel) # parallel computing
library(parallel)   # parallel computing
library(furrr)      # parallel computing
library(caret)      # for cross-validation
library(MuMIn)      # for model evaluation

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 
options(scipen=999,digits=6)

# import test data (assumes test data is in same directory as this .R script)
test_data <- read_csv("test.csv") %>%
  dplyr::select(Outgroup, bias, outgroup_att, everything())

# modify test data by recoding reversed values and removing the original (R) vers
test_data <- test_data %>%
  mutate(sThreat3 = 8 - sThreat3,
         Agreeable1 = 6 - Agreeable1) %>%
  select(everything(), -sThreat3R, -agreeable1r)

# outcome variables
# (1) outgroup_att = factor score calculated from WarmOG, PositiveOG, LikeOG
# (2) bias = factor score calculated from WarmIG-WarmOG, PositiveIG-PositiveeOG, LikeIG-LikeOG (positive score is more ingroup positivity bias)

# predictors
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
test_data <- test_data %>%
  dplyr::select(everything(), -WarmIG, -WarmOG, -PositiveIG, -PositiveOG, -LikeIG, -LikeOG, -diff_warm, -diff_pos, -diff_like) %>%
  mutate(Outgroup = as.factor(Outgroup))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# data transformations ----------------------------------------------------

# create matrix of predictors, removing outcome variables and cluster(Outgroup)
predictors <- test_data %>%
  dplyr::select(-bias, -outgroup_att, -Outgroup) %>%
  as.matrix()
outcome_bias <- test_data[["bias"]]
outcome_outgroup_att <- test_data[["outgroup_att"]]
group <- test_data[["Outgroup"]]# 15 unique groups

test_data_trans <- test_data

predictors_trans <- test_data_trans %>%
  dplyr::select(-Outgroup,-bias,-outgroup_att)

# 1. grand mean center all predictors
# loop through dataframe and grand-mean-center predictors (we will cluster mean-center later in mlms)
test_data_gmc <- test_data_trans
for (var_name in names(predictors_trans)) {
  var_name <- ensym(var_name)
  new_var_name <- paste0(rlang::as_string(var_name),"_gmc") # grand-mean-center
  test_data_gmc <- test_data_gmc %>%
    mutate(!!new_var_name := !!var_name - mean(!!var_name))
}
test_data_gmc <- test_data_gmc %>% dplyr::select(Outgroup, bias, outgroup_att, contains("_gmc"))
predictors_gmc <- test_data_gmc %>% dplyr::select(-Outgroup, -bias, -outgroup_att, contains("_gmc"))

# 2. create 2-way interactions between all unique pairs of variables
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

# 3. log10-transform all valid predictors

# loop through each variable (not centered) and create a log-transformed version
test_data_log <- test_data_trans
for (var_name in names(predictors_trans)) {
  var_name <- ensym(var_name)
  new_var_name <- paste0(rlang::as_string(var_name),"_log")
  test_data_log <- test_data_log %>%
    mutate(!!new_var_name := log10(!!var_name))
}
# find and drop columns with NAN values
log_vars_with_NaN <- test_data_log %>%
  summarise_all(~ any(is.na(.))) %>%
  unlist() %>%
  which()
test_data_log <- test_data_log %>%
  dplyr::select(-all_of(log_vars_with_NaN)) %>%
  dplyr::select(contains("_log"))

# 4. compute similarity metrics 

# compute similarity based on mahalanobis distance between pairwise variables,
# by loop through each pair of predictors
mahalanobis_distances <- data.frame(matrix(nrow=nrow(predictors_trans)))
for (i in seq_along(predictors_trans)) {
  for (j in seq_along(predictors_trans)) {
    if (i < j) {  # to avoid duplicate pairs
      col1 <- names(predictors_trans)[i]
      col2 <- names(predictors_trans)[j]
      var_pair <- paste0(col1, "_d_", col2, "_mD_log")
      tryCatch({
        # compute mahalanobis distance, return NaN if covmatrix is singular
        mahalanobis_distances[[var_pair]] <- mahalanobis(predictors_trans[,c(col1,col2)],
                                                         center = colMeans(predictors_trans[,c(col1,col2)]),
                                                         cov = cov(predictors_trans[,c(col1,col2)]))
      }, error = function(e){
        # handle error if covariance matrix is singular
        mahalanobis_distances[[var_pair]] <- NaN
      })
      tryCatch({
        mahalanobis_distances[[var_pair]] <- log10(mahalanobis_distances[[var_pair]])
      }, error = function(e){
        mahalanobis_distances[[var_pair]] <- NaN
      })
    }
  }
}
mahalanobis_distances <- mahalanobis_distances[,-1] # remove NA column
# remove cols NaN values
mahalanobis_distances_with_NaN <- mahalanobis_distances %>%
  summarise_all(~ any(is.na(.))) %>%
  unlist() %>%
  which()

# 5. compute item variance for each row
# how much variance in each participant's responses? 
individual_variances <- predictors_trans %>%
  rowwise() %>%
  mutate(item_variances = var(c_across(everything()))) %>% select(item_variances) %>%
  ungroup() %>%
  mutate(item_variances_gmc = item_variances - mean(item_variances))

# 6. identify extreme response styles

# compute quartiles and iqr for each scale
compute_quartiles <- function(predictor) {
  quantiles <- quantile(predictor, probs = c(0.25, 0.50, 0.75), na.rm=TRUE)
  iqr <- quantiles[3] - quantiles[1]
  list(Q1 = quantiles[1], median = quantiles[2], Q3 = quantiles[3], IQR=iqr)
}
# apply this function to each scale
quartiles <- lapply(test_data[,raw_predictors_key], compute_quartiles)

# calculate quartile-based extreme and midpoint responses for each row
calculate_response_patterns_quartiles <- function(row, quartiles, data){
  counts <- list(extreme = 0, midpoint = 0, total = 0)
  for (col in colnames(data)){
    Q1 <- quartiles[[col]]$Q1
    Q3 <- quartiles[[col]]$Q3
    value <- row[[col]]
    
    if (!is.na(value)){
      if (value < Q1 || value > Q3) {
        counts$extreme <- counts$extreme + 1
      } else {
        counts$midpoint <- counts$midpoint + 1
      }
    }
    counts$total <- counts$total + 1
  }
  return(counts)
}
test_data_ <- test_data[,raw_predictors_key]
response_patterns <- apply(test_data_, 1, function(row) calculate_response_patterns_quartiles(row, quartiles, data=test_data_))
response_patterns_df <- do.call(rbind, lapply(response_patterns,as.data.frame))
response_patterns_df <- data.frame(response_patterns_df)

# calc proportions
response_patterns_df <- response_patterns_df %>%
  mutate(extreme_proportion_q = extreme/total,
         midpoint_proportion_q = midpoint/total,
         extreme_proportion_q_gmc = extreme_proportion_q - mean(extreme_proportion_q)) %>%
  select(extreme_proportion_q,extreme_proportion_q_gmc,midpoint_proportion_q)

# 7. create polynomials

polynomials_gmc <- predictors_gmc
for (var_name in names(polynomials_gmc)) {
  var_name <- ensym(var_name)
  var_name_squared <- paste0(rlang::as_string(var_name),"_squared")
  var_name_cubid <- paste0(rlang::as_string(var_name),"_cubid")
  polynomials_gmc <- polynomials_gmc %>%
    mutate(!!var_name_squared := (!!var_name)^2,
           !!var_name_cubid := (!!var_name)^3)
}
polynomials_gmc <- polynomials_gmc %>%
  select(contains("squared"), contains("cubid"))

# combine all data transformations
test_data <- cbind(test_data_gmc, interactions) %>%
  cbind(test_data_log) %>% 
  cbind(mahalanobis_distances) %>%
  cbind(individual_variances) %>%
  cbind(polynomials_gmc) %>%
  cbind(response_patterns_df)

# import parameter weights ------------------------------------------------

# bias model parameter weights
paramWeights_bias <- read_csv("paramWeights_bias.csv") # make sure in same folder

# outgroup_att parameter weights
paramWeights_outgroup_att <- read_csv("paramWeights_outgroup_att.csv") 

# BIAS MODEL --------------------------------------------------------------

# bias model
bias_model <- lm(formula = paste("bias ~", paste(paramWeights_bias$Variable, collapse = " + ")),
                 data = test_data)
summary(bias_model)
bias_adj_r_squared <- summary(bias_model)$adj.r.squared; print(paste0("Bias Model Adj R-sq: ",bias_adj_r_squared))
bias_rmse <- sqrt(mean(bias_model$residuals^2)); print(paste0("Bias Model RMSE: ", bias_rmse))

# OUTGROUP ATT MODEL ------------------------------------------------------

# outgroup attitudes model
outgroup_att_model <- lm(
  formula = paste("outgroup_att ~", paste(paramWeights_outgroup_att$Variable, collapse = " + ")),
  data = test_data
  )

outgroup_att_adj_r_squared <- summary(outgroup_att_model)$adj.r.squared; print(paste0("outgroup_att Model Adj R-sq: ",outgroup_att_adj_r_squared))
outgroup_att_rmse <- sqrt(mean(outgroup_att_model$residuals^2)); print(paste0("outgroup_att Model RMSE: ", outgroup_att_rmse))

#dput(as.numeric(bias_model$coefficients))
model_weights = 
c(-0.225841899111369, 0.534394191245823, -0.442006670304571, 
  0.298713348167119, 0.149977180981614, -0.321612471783982, 0.168313750035044, 
  -0.117798666140286, 0.0790534909927289, 0.0729420641931729, -0.111316225076459, 
  0.0278834108319874, 0.00747722721674714, 0.0377138225295037, 
  -0.000365397275756211, 0.0961084821220865, 0.00756116211656676, 
  -0.0133641637081985, -0.00829108372721134, 0.0481885052693005, 
  0.0268654934643008, 0.0474875280078807, -0.0168218721429416, 
  0.00441304137028173, 0.0166543265029762, 0.0211462963629962, 
  0.0527420690392894, 0.0165966141497899, 0.0415227079271951, -0.00675835832291348, 
  -0.0190561536943706, -0.00541008503036683, -0.00898822566003744, 
  0.0169476198362784, -0.303700938543064, -0.0200872494821351, 
  0.07771194094425, -0.00529311382338792, -0.0119092254141884, 
  0.0274039144494538, 0.00757470358954235, -0.0451282821591128, 
  0.0237352786642476, 0.00683651420877748, -0.00707196428907954, 
  -0.221326489847204, -0.0287526801784395, -0.0142656854289765, 
  0.0135166161581917, -0.0391821067627707, 0.0102633673918376, 
  0.0196981829520311, -0.0231478912909736, 0.00470157333636713, 
  -0.00944230982143102, -0.00205928119159055, 0.00378785154037736, 
  -0.0805074197800774, 0.00845251785576558, 0.0212159573857118, 
  0.00125071200414619, 0.00981354077337553, 0.0081182404539431, 
  -0.0219614331082759, 0, -0.0267896813600218, -0.0128540554104536, 
  0.0254322879126361, -0.00908735427603479, 0, 0.0052481237712854
)

#predictor matrix, PM. PM %*% model_weights (+ intercept ie mean?)


