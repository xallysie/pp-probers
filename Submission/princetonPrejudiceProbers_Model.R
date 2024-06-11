# model comparisons for prejudice modeling competition
# last updated 2024/06/05
# team: Princeton Prejudice Probers
# members: Sally Xie, Kerem Oktar, Bonan Zhao

# setup ------------------------------------------------------------------- ####
# load libraries
library(tidyverse)  # data hygiene
library(readr)      # code hygiene
library(rlang)      # code hygiene
library(scales)     # to scale response variable
library(caret)      # for cross-validation
library(MuMIn)      # for model evaluation
library(MASS)       # for robust linear models
select <- dplyr::select

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 
options(scipen=999,digits=6)

# import test data (assumes test data is in same directory as this .R script)
test_data <- read_csv("test.csv") %>%
  dplyr::select(Outgroup, bias, outgroup_att, everything())

# modify test data by recoding reversed values, removing the "R" reversed vers,
# and removing variables that were used to construct the outcome variables
test_data <- test_data %>%
  mutate(sThreat3 = 8 - sThreat3,
         Agreeable1 = 6 - Agreeable1) %>%
  select(everything(), -sThreat3R, -agreeable1r,
         -WarmIG, -WarmOG, -PositiveIG, -PositiveOG, -LikeIG, -LikeOG, -diff_warm, -diff_pos, -diff_like) %>%
  mutate(Outgroup = as.factor(Outgroup))


# outcome variables
# (1) outgroup_att = factor score calculated from WarmOG, PositiveOG, LikeOG
# (2) bias = factor score calculated from WarmIG-WarmOG, PositiveIG-PositiveeOG, LikeIG-LikeOG (positive score is more ingroup positivity bias)

# variable names for predictors
raw_predictors_key <- c("sThreat1", "sThreat2", "sThreat3", "sThreat4",
                        "Identification1", "Identification2", "Identification3", "Identification4", "Identification5",
                        "Identification6", "Identification7", "Identification8", "Identification9", "Identification10",
                        "ContactQ1", "ContactQ2", "ContactQ3", "ContactN1",
                        "Agreeable1", "Agreeable2",
                        "rThreatIG1", "rThreatIG2", "rThreatOG1", "rThreatOG2",
                        "DisgustP1", "DisgustP2", "DisgustP3", "DisgustP4", "DisgustP5", "DisgustP6",
                        "DisgustS1", "DisgustS2", "DisgustS3", "DisgustS4",
                        "DisgustR1", "DisgustR2")
transformed_vars_key <- c(#"diff_warm", # WarmIG - WarmOG
                          #"diff_pos",  # PositiveIG - PositiveOG
                          #"diff_like", # LikeIG - LikeOG
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# data transformations ------------------------------------------------- #######

transform_data <- function(test_data){
  # create matrix of predictors, removing outcome variables and cluster(Outgroup)
  predictors <- test_data %>%
    dplyr::select(-bias, -outgroup_att, -Outgroup) %>%
    as.matrix()
  outcome_bias <- test_data[["bias"]]
  outcome_outgroup_att <- test_data[["outgroup_att"]]
  group <- test_data[["Outgroup"]] # 15 unique groups
  nobs <- nrow(test_data) # get number of rows
  
  test_data_trans <- test_data
  
  predictors_trans <- test_data_trans %>%
    dplyr::select(-Outgroup,-bias,-outgroup_att)
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # 2. create 2-way interactions between all unique pairs of variables
  
  interactions <- data.frame(matrix(nrow=nobs))
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
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # 3. log10-transform all valid predictors
  
  # loop through each variable (not centered) and create a log-trans'd version
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
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # 4. compute pairwise similarity metrics 
  
  # compute similarity based on mahalanobis distance between pairs of variables,
  # by looping through each unique pair of predictors
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
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # 5. compute item variance for each row (participant)
  # how much variance across participant's responses on all items?
  individual_variances <- predictors_trans %>%
    rowwise() %>%
    mutate(item_variances = var(c_across(everything()))) %>% select(item_variances) %>%
    ungroup() %>%
    mutate(item_variances_gmc = item_variances - mean(item_variances))
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # combine all data transformations
  test_data_transformed <- cbind(test_data_gmc, interactions) %>%
    cbind(test_data_log) %>% 
    cbind(mahalanobis_distances) %>%
    cbind(individual_variances) %>%
    cbind(polynomials_gmc) %>%
    cbind(response_patterns_df)
  
  return(test_data_transformed)
}

# apply function to transform test data
test_data_transformed <- transform_data(test_data)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# import parameter weights ------------------------------------------- ####
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 

# bias model parameter weights
paramWeights_bias <- read_csv("paramWeights_bias.csv") %>% # make sure in same folder
  mutate(ParamWeight = as.numeric(ParamWeight))

# outgroup_att parameter weights
paramWeights_outgroup_att <- read_csv("paramWeights_outgroup_att_archived.csv") %>%
  mutate(ParamWeight = as.numeric(ParamWeight))

# if the competition rules do not allow other files to be uploaded with code,
# then here are the variable names

xvars_bias <- dput(paramWeights_bias$Variable)
xweights_bias <- dput(as.numeric(paramWeights_bias$ParamWeight))
xvars_outgroup_att <- dput(paramWeights_outgroup_att$Variable)
xweights_outgroup_att <- dput(as.numeric(paramWeights_outgroup_att$ParamWeight))

#xvars <- gsub("(_squared)", "_gmc\\1", xvars)
#xvars <- gsub("(_cubid)", "_gmc\\1", xvars)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# model evaluation function, avg across groups --------------------- ####
evaluate_model_avg <- function(xvars, yvar, weights, data, cluster_name="clusterName"){
  
  # create results dataframe with 3 columns: Group Name, RMSE, Adjusted R-sq
  results <- as.data.frame(matrix(ncol=3,nrow=length(levels(as.factor(data[[cluster_name]])))))
  names(results) <- c("Outgroup", "RMSE", "Adj_R_sq")
  results[[1]] <- levels(as.factor(data[[cluster_name]]))
  
  # for each outgroup, compute RMSE and Adj_R_Sq separately
  for (k in results[[1]]){
    # subset data for outgroup k
    data_subset <- data[data[[cluster_name]] == k,]
    
    # make sure data structure is appropriate
    x_mat <- data_subset[xvars] %>% as.matrix()
    y_vec <- data_subset[yvar] %>% unlist() %>% as.numeric()
    weights <- as.numeric(weights)
    
    # get fitted values
    preds <- x_mat %*% weights
    
    # build model dataset
    model_data <- as.data.frame(cbind(y_vec, preds))
    names(model_data) <- c(yvar, "preds")
    
    # build model object
    model <- lm(formula = paste(yvar, "~", "preds"), data = model_data)
    
    # get rmse
    rmse <- sqrt(mean(model$residuals^2))
    
    # get adjusted r-squared
    r_squared <- summary(model)$r.squared
    n <- nrow(data) # number of obs = total rows in dataset
    p <- length(xvars) # number of predictors = length of x variable list
    adj_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
    
    # output to results dataframe
    results[results[[cluster_name]] == k, "RMSE"] <- rmse
    results[results[[cluster_name]] == k, "Adj_R_sq"] <- adj_r_squared 
  }
  
  # test model on entire dataset 
  # first compute RMSE and Adj_R_Sq over entire dataset
  data_subset <- data
  x_mat <- data_subset[xvars] %>% as.matrix()
  y_vec <- data_subset[yvar] %>% unlist() %>% as.numeric()
  preds <- x_mat %*% weights
  model_data <- as.data.frame(cbind(y_vec, preds))
  names(model_data) <- c(yvar, "preds")
  model <- lm(formula = paste(yvar, "~", "preds"), data = model_data)
  rmse <- sqrt(mean(model$residuals^2))
  r_squared <- summary(model)$r.squared
  n <- nrow(data) # number of obs = total rows in dataset
  p <- length(xvars) # number of predictors = length of x variable list
  adj_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
  
  print(paste("RMSE (entire dataset):", rmse))
  print(paste("Adj_R_Sq (entire dataset):", adj_r_squared))
  print("~~~~~~~~~~~~~~")
  print(paste("*** Mean RMSE (avg over all groups):", mean(results$RMSE))) # averaged across all
  print(paste("*** Mean Adj_R_sq (avg over all groups):", mean(results$Adj_R_sq)))
  
  return(results)
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# test function against baseline ---------------------------------------- ####

# test function on baseline model (Bias Model 1.0)
baseline_test <- evaluate_model_avg(xvars = c("generalized","symbolic","contact_quality","contact_friendsz","identification_selfinvestment"),
                                    yvar = "bias",
                                    weights = c(.120, .173, -.209, -.119, .523),
                                    data = test_data,
                                    cluster_name = "Outgroup")

# test function on baseline model (Outgroup Attitudes 1.0)
baseline_test_2 <- evaluate_model_avg(xvars = c("symbolic","contact_quality","contact_friendsz","b5a"),
                                    yvar = "outgroup_att",
                                    weights = c(.243, -.462, -.119, -.128),
                                    data = test_data,
                                    cluster_name = "Outgroup")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BIAS MODEL ----------------------------------------------------------- ####

# ppprobers bias model 3000
bias_model_results <- evaluate_model_avg(xvars = xvars_bias,
                                         yvar = "bias",
                                         weights = xweights_bias,
                                         data = test_data_transformed,
                                         cluster_name="Outgroup")


# OUTGROUP ATT MODEL ---------------------------------------------- ####

# ppprobers outgroup_att model 3000
outgroup_att_model_results <- evaluate_model_avg(xvars = xvars_outgroup_att,
                                         yvar = "outgroup_att",
                                         weights = xweights_outgroup_att,
                                         data = test_data_transformed,
                                         cluster_name="Outgroup")

# final models for submission
bias_model_results
outgroup_att_model_results