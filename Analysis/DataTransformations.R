# how many ways can we torture the data!?!?!?
# play around with data transformations
# last updated 2024/06/06 -sally

# setup -------------------------------------------------------------------
# load libraries
library(tidyverse)  # data hygiene
library(readr)      # code hygiene
library(rlang)      # code hygiene

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
transformed_vars_key <- c(
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
# Apply transformations ---------------------------------------------------

train_data_trans <- train_data

predictors_trans <- train_data_trans %>%
  dplyr::select(-Outgroup,-bias,-outgroup_att)

# create a bunch of new variables and see what sticks!


# 1. grand-mean-center all predictors -------------------------------------

# loop through dataframe and grand-mean-center predictors (we will cluster mean-center later in mlms)
train_data_gmc <- train_data_trans
for (var_name in names(predictors_trans)) {
  var_name <- ensym(var_name)
  new_var_name <- paste0(rlang::as_string(var_name),"_gmc") # grand-mean-center
  train_data_gmc <- train_data_gmc %>%
    mutate(!!new_var_name := !!var_name - mean(!!var_name))
}
train_data_gmc <- train_data_gmc %>% dplyr::select(Outgroup, bias, outgroup_att, contains("_gmc"))
predictors_gmc <- train_data_gmc %>% dplyr::select(-Outgroup, -bias, -outgroup_att, contains("_gmc"))



# 2. two-way interactions -------------------------------------------------

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


# 3. log10-transform all valid predictors ---------------------------------

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
  dplyr::select(-all_of(log_vars_with_NaN)) %>%
  dplyr::select(contains("_log"))

# combine these transformed variables
train_data_trans <- cbind(train_data_gmc, interactions) %>%
  cbind(train_data_log)

# 6. similarities between variables ---------------------------------------

# create similarity metrics between different variables to try to identify
# response profiles

# compute similarity based on mahalanobis distance between pairwise variables

# loop through each pair of predictors and compute mahalanobis distance
#mahalanobis_distances <- data.frame(matrix(nrow=2010))
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

# compute variance in responses for each participant
individual_variances <- predictors_trans %>%
  rowwise() %>%
  mutate(item_variances = var(c_across(everything()))) %>% select(item_variances) %>%
  ungroup() %>%
  mutate(item_variances_gmc = item_variances - mean(item_variances))


# load changes from cubed/squared vers
load("train_data_transformations_raised.Rdata")
train_data_extended <- train_data_extended[,66:76]

train_data_squared <- train_data_extended %>% select(contains("squared")) %>%
  mutate_all(abs)
train_data_cubid <- train_data_extended %>% select(contains("cubid"))

train_data_extended <- cbind(train_data_squared,train_data_cubid)

# combine these transformed variables
train_data_trans <- cbind(train_data_gmc, interactions) %>%
  cbind(train_data_log) %>% 
  cbind(mahalanobis_distances) %>%
  cbind(individual_variances) %>%
  cbind(train_data_extended)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 7. identify extremity of response ---------------------------------------

# use quartiles or sd to capture the distribution characteristics of responses
# can identify extreme responders/outliers

# first, define which columns correspond to which scale
scale_1_to_7 <- c(
  raw_predictors_key[raw_predictors_key %in% str_subset(raw_predictors_key,"sThreat")], 
  raw_predictors_key[raw_predictors_key %in% str_subset(raw_predictors_key,"Identification")],
  raw_predictors_key[raw_predictors_key %in% str_subset(raw_predictors_key, "Contact")]
)
scale_1_to_9 <- c(
  raw_predictors_key[raw_predictors_key %in% str_subset(raw_predictors_key,"rThreat")]
)
scale_1_to_5 <- c(
  raw_predictors_key[raw_predictors_key %in% str_subset(raw_predictors_key,"Disgust")],
  raw_predictors_key[raw_predictors_key %in% str_subset(raw_predictors_key,"Agreeable")]
)

# compute quartiles and iqr for each scale
compute_quartiles <- function(predictor) {
  quantiles <- quantile(predictor, probs = c(0.25, 0.50, 0.75), na.rm=TRUE)
  iqr <- quantiles[3] - quantiles[1]
  list(Q1 = quantiles[1], median = quantiles[2], Q3 = quantiles[3], IQR=iqr)
}

# apply this function to each scale
quartiles <- lapply(train_data[,raw_predictors_key], compute_quartiles)

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
train_data_ <- train_data[,raw_predictors_key]
response_patterns <- apply(train_data_, 1, function(row) calculate_response_patterns_quartiles(row, quartiles, data=train_data_))
response_patterns_df <- do.call(rbind, lapply(response_patterns,as.data.frame))
response_patterns_df <- data.frame(response_patterns_df)

# calc proportions
response_patterns_df <- response_patterns_df %>%
  mutate(extreme_proportion_q = extreme/total,
         midpoint_proportion_q = midpoint/total,
         extreme_proportion_q_gmc = extreme_proportion_q - mean(extreme_proportion_q)) %>%
  select(extreme_proportion_q,extreme_proportion_q_gmc,midpoint_proportion_q)

# combine these transformed variables
train_data_trans <- cbind(train_data_gmc, interactions) %>%
  cbind(train_data_log) %>% 
  cbind(mahalanobis_distances) %>%
  cbind(individual_variances) %>%
  cbind(train_data_extended) %>%
  cbind(response_patterns_df)

# now try this with a SD-based approach
compute_mean_sd <- function(predictor) {
  mean_val <- mean(predictor, na.rm=TRUE)
  sd_val <- sd(predictor, na.rm=TRUE)
  list(mean = mean_val, sd = sd_val)
}
# apply this function to each var
mean_sds <- lapply(train_data[, raw_predictors_key], compute_mean_sd)

# calculate sd-based extreme and midpoint response styles
calculate_response_patterns_sd <- function(row, data) {
  counts <- list(extreme = 0, midpoint = 0, total = 0)
  
  # extreme = more than 1 SD away from mean
  # midpoint = within 1 SD of mean
  is_extreme <- function(val, mean, sd) abs(val - mean) > sd
  is_midpoint <- function(val, mean, sd) abs(val - mean) <= sd
  
  for (col in names(row)){
    mean_val <- mean_sds[[col]]$mean
    sd_val <- mean_sds[[col]]$sd
    val <- row[[col]]
    
    if (!is.na(val)) {
      if (is_extreme(val, mean_val, sd_val)) {
        counts$extreme <- counts$extreme + 1
      }
      if (is_midpoint(val, mean_val, sd_val)) {
        counts$midpoint <- counts$midpoint + 1
      }
      counts$total <- counts$total + 1
    }
  }
  return(counts)
}
response_patterns_sd <- apply(train_data_,1,function(row) calculate_response_patterns_sd(row, mean_sds))
response_patterns_sd_df <- do.call(rbind, lapply(response_patterns_sd, function(x){
  data.frame(extreme = x$extreme, midpoint = x$midpoint, total = x$total)
}))

# calc proportions
response_patterns_sd_df <- response_patterns_sd_df %>%
  mutate(extreme_proportion_sd = extreme/total,
         midpoint_proportion_sd = midpoint/total,
         extreme_proportion_sd_gmc = extreme_proportion_sd - mean(extreme_proportion_sd)) %>%
  select(extreme_proportion_sd,extreme_proportion_sd_gmc,midpoint_proportion_sd)


# combine these transformed variables
train_data_trans <- cbind(train_data_gmc, interactions) %>%
  cbind(train_data_log) %>% 
  cbind(mahalanobis_distances) %>%
  cbind(individual_variances) %>%
  cbind(train_data_extended) %>%
  cbind(response_patterns_df) %>%
  cbind(response_patterns_sd_df)

# 
#summary(
#  lm(formula = paste("bias ~", paste(variables_to_consider$Variable, collapse = " + "), "+ extreme_proportion_q_gmc", "+ item_variances_gmc"),
#     data=train_data_trans)
#)

#summary(
#  lm(bias ~ extreme_proportion_q_gmc + item_variances_gmc, data=train_data_trans)
#)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# 8. apply polynomials gmc to all -----------------------------------------

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

train_data_trans <- cbind(train_data_gmc, interactions) %>%
  cbind(train_data_log) %>% 
  cbind(mahalanobis_distances) %>%
  cbind(individual_variances) %>%
  cbind(polynomials_gmc) %>%
  cbind(response_patterns_df) %>%
  cbind(response_patterns_sd_df)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# 9. transform factors/subfactors only ------------------------------------

subfactors_gmc <- predictors_gmc %>%
  select(contains(transformed_vars_key),contains(latent_vars_key))
subfactors_trans <- predictors_trans %>%
  select(contains(transformed_vars_key),contains(latent_vars_key))

# create 2-way interactions between all unique factors/subfactors
interactions <- data.frame(matrix(nrow=2010))
for (i in seq_along(subfactors_gmc)) {
  for (j in seq_along(subfactors_gmc)) {
    if (i < j) {  # to avoid duplicate pairs
      col1 <- names(subfactors_gmc)[i]
      col2 <- names(subfactors_gmc)[j]
      interaction_name <- paste0(col1, "_x_", col2)
      interactions[[interaction_name]] <- subfactors_gmc[[col1]] * subfactors_gmc[[col2]]
    }
  }
}
interactions <- interactions[,-1] # remove NA column

#compute similarity based on mahalanobis distance between pairwise factors
# loop through each pair of (sub)factors and compute mahalanobis distance
mahalanobis_distances <- data.frame(matrix(nrow=nrow(subfactors_gmc)))
for (i in seq_along(subfactors_trans)) {
  for (j in seq_along(subfactors_trans)) {
    if (i < j) {  # to avoid duplicate pairs
      col1 <- names(subfactors_trans)[i]
      col2 <- names(subfactors_trans)[j]
      var_pair <- paste0(col1, "_d_", col2, "_mD_log")
      tryCatch({
        # compute mahalanobis distance, return NaN if covmatrix is singular
        mahalanobis_distances[[var_pair]] <- mahalanobis(subfactors_trans[,c(col1,col2)],
                                                         center = colMeans(subfactors_trans[,c(col1,col2)]),
                                                         cov = cov(subfactors_trans[,c(col1,col2)]))
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

# combine these transformed variables
train_data_trans <- cbind(train_data_gmc, interactions) %>%
  cbind(train_data_log) %>% 
  cbind(mahalanobis_distances) %>%
  cbind(individual_variances) %>%
  cbind(train_data_extended) %>%
  cbind(response_patterns_df) %>%
  cbind(response_patterns_sd_df)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# 10. [archive] create sum scores instead of factor scores ----------------

sum_scores <- train_data %>%
  select(all_of(raw_predictors_key), all_of(transformed_vars_key)) %>%
  rowwise() %>%
  mutate(
    symbolic = sum(c_across(contains("sThreat"))),
    identification_sol = sum(c_across(c("Identification1","Identification2","Identification3"))),
    identification_sat = sum(c_across(c("Identification4","Identification5","Identification6","Identification7"))),
    identification_cen = sum(c_across(c("Identification8","Identification9","Identification10"))),
    identification_selfinvestment = sum(c_across(c("identification_sol","identification_sat","identification_cen"))),
    contact_quality = sum(c_across(c("ContactQ1","ContactQ2","ContactQ3"))),
    b5a = sum(c_across(c("Agreeable1","Agreeable2"))),
    generalized = sum(c_across(c("generalized_challdiff","generalized_probdiff"))),
    disgust_p = sum(c_across(contains("DisgustP"))),
    disgust_s = sum(c_across(contains("DisgustS"))),
    disgust_r = sum(c_across(contains("DisgustR"))),
    threat_og = sum(c_across(c("rThreatOG1","rThreatOG2"))),
    threat_ig = sum(c_across(c("rThreatIG1","rThreatIG2")))
  ) %>%
  ungroup()

# grand mean center all vars including sum scores
# loop through dataframe and grand-mean-center
sum_scores_gmc <- sum_scores
for (var_name in names(sum_scores)) {
  var_name <- ensym(var_name)
  new_var_name <- paste0(rlang::as_string(var_name),"_gmc") # grand-mean-center
  sum_scores_gmc <- sum_scores_gmc %>%
    mutate(!!new_var_name := !!var_name - mean(!!var_name))
}
# add outcomes
train_data_sumscores <- train_data %>%
  select(Outgroup,bias,outgroup_att) %>%
  cbind(sum_scores_gmc) %>%
  select(Outgroup, bias, outgroup_att, all_of(contains("_gmc")), -all_of(raw_predictors_key))

sum_scores_gmc <- sum_scores_gmc %>%
  select(all_of(contains("_gmc")))

# restrict to factors only
factors_sumscores <- sum_scores_gmc %>%
  select(contains("gmc")) %>%
  select(-contains(raw_predictors_key))

# create 2-way interactions between all unique pairs of factors
interactions_sumscores <- data.frame(matrix(nrow=2010))
for (i in seq_along(sum_scores_gmc)) {
  for (j in seq_along(sum_scores_gmc)) {
    if (i < j) {  # to avoid duplicate pairs
      col1 <- names(sum_scores_gmc)[i]
      col2 <- names(sum_scores_gmc)[j]
      interaction_name <- paste0(col1, "_x_", col2)
      interactions_sumscores[[interaction_name]] <- sum_scores_gmc[[col1]] * sum_scores_gmc[[col2]]
    }
  }
}
interactions_sumscores <- interactions_sumscores[,-1] # remove NA column

# loop through each variable (not centered) and create a log-transformed version
sum_scores_log <- sum_scores
for (var_name in names(sum_scores_log)) {
  var_name <- ensym(var_name)
  new_var_name <- paste0(rlang::as_string(var_name),"_log")
  sum_scores_log <- sum_scores_log %>%
    mutate(!!new_var_name := log10(!!var_name))
}
# find and drop columns with NAN values
log_vars_with_NaN <- sum_scores_log %>%
  summarise_all(~ any(is.na(.))) %>%
  unlist() %>%
  which()
sum_scores_log <- sum_scores_log %>%
  dplyr::select(-all_of(log_vars_with_NaN)) %>%
  dplyr::select(contains("_log"))

# compute similarity based on mahalanobis distance between pairwise vars
# loop through each pair of vars and compute mahalanobis distance
sum_score_mds_log <- data.frame(matrix(nrow=nrow(sum_scores)))
for (i in seq_along(sum_scores)) {
  for (j in seq_along(sum_scores)) {
    if (i < j) {  # to avoid duplicate pairs
      col1 <- names(sum_scores)[i]
      col2 <- names(sum_scores)[j]
      var_pair <- paste0(col1, "_d_", col2, "_mD_log")
      tryCatch({
        # compute mahalanobis distance, return NaN if covmatrix is singular
        sum_score_mds_log[[var_pair]] <- mahalanobis(sum_scores[,c(col1,col2)],
                                                         center = colMeans(sum_scores[,c(col1,col2)]),
                                                         cov = cov(sum_scores[,c(col1,col2)]))
      }, error = function(e){
        # handle error if covariance matrix is singular
        sum_score_mds_log[[var_pair]] <- NaN
      })
      tryCatch({
        sum_score_mds_log[[var_pair]] <- log10(sum_score_mds_log[[var_pair]])
      }, error = function(e){
        sum_score_mds_log[[var_pair]] <- NaN
      })
    }
  }
}
sum_score_mds_log <- sum_score_mds_log[,-1] # remove NA column
# remove cols NaN values
sum_score_mds_log_with_NaN <- sum_score_mds_log %>%
  summarise_all(~ any(is.na(.))) %>%
  unlist() %>%
  which()

# polynomials: square and cube the mean-centered factors
sum_score_polynomials <- factors_sumscores
for (var_name in names(sum_score_polynomials)) {
  var_name <- ensym(var_name)
  var_name_squared <- paste0(rlang::as_string(var_name),"_squared")
  var_name_cubid <- paste0(rlang::as_string(var_name),"_cubid")
  sum_score_polynomials <- sum_score_polynomials %>%
    mutate(!!var_name_squared := (!!var_name)^2,
           !!var_name_cubid := (!!var_name)^3)
}
sum_score_polynomials <- sum_score_polynomials %>%
  select(contains("squared"), contains("cubid"))

# combine all 
train_data_sumscores_trans <- cbind(train_data_sumscores, interactions_sumscores) %>%
  cbind(sum_scores_log) %>% 
  cbind(sum_score_mds_log) %>%
  cbind(individual_variances) %>%
  cbind(sum_score_polynomials) %>%
  cbind(response_patterns_df) %>%
  cbind(response_patterns_sd_df)

length(names(train_data_sumscores_trans))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# now let's create "multilevel model"-friendly versions of this!

# 7. cluster-mean-center variables ----------------------------------------

train_data_trans_ml <- train_data

predictors_trans_ml <- train_data_trans_ml %>%
  dplyr::select(-Outgroup,-bias,-outgroup_att)

# loop through dataframe and cluster-mean-center predictors (so we can disentangle within-group and between-group effs)
train_data_cmc <- train_data_trans_ml
for (var_name in names(predictors_trans_ml)) {
  var_name <- ensym(var_name)
  cluster_mean <- paste0(rlang::as_string(var_name),"_cm") # cluster mean
  cwc <- paste0(rlang::as_string(var_name),"_cwc") # centered-within-cluster
  train_data_cmc <- train_data_cmc %>%
    group_by(Outgroup) %>%
    mutate(!!cluster_mean := mean(!!var_name, na.rm=TRUE)) %>%
    ungroup() %>%
    mutate(!!cwc := !!var_name - !!sym(cluster_mean))
}
train_data_cmc <- train_data_cmc %>% dplyr::select(Outgroup, bias, outgroup_att, contains("_cm"), contains("_cwc"))
predictors_cmc <- train_data_cmc %>% dplyr::select(-Outgroup, -bias, -outgroup_att, contains("_cm"), contains("_cwc"))


# 8. two-way interactions (cluster-mean-centered) -------------------------

# create 2-way interactions between all unique pairs of variables (cluster means)
interactions_cm <- data.frame(matrix(nrow=2010))
predictors_cm <- predictors_cmc %>% dplyr::select(contains("_cm"))
for (i in seq_along(predictors_cm)) {
  for (j in seq_along(predictors_cm)) {
    if (i < j) {  # to avoid duplicate pairs
      col1 <- names(predictors_cm)[i]
      col2 <- names(predictors_cm)[j]
      interaction_name <- paste0(col1, "_x_", col2)
      interactions_cm[[interaction_name]] <- predictors_cm[[col1]] * predictors_cm[[col2]]
    }
  }
}
interactions_cm <- interactions_cm[,-1] # remove NA column

# create 2-way interactions between all unique pairs of variables (centered within clusters)
interactions_cwc <- data.frame(matrix(nrow=2010))
predictors_cwc <- predictors_cmc %>% dplyr::select(contains("_cwc"))
for (i in seq_along(predictors_cwc)) {
  for (j in seq_along(predictors_cwc)) {
    if (i < j) {  # to avoid duplicate pairs
      col1 <- names(predictors_cwc)[i]
      col2 <- names(predictors_cwc)[j]
      interaction_name <- paste0(col1, "_x_", col2)
      interactions_cwc[[interaction_name]] <- predictors_cwc[[col1]] * predictors_cwc[[col2]]
    }
  }
}
interactions_cwc <- interactions_cwc[,-1] # remove NA column
interactions <- cbind(interactions_cm,interactions_cwc)

# combine these transformed variables
train_data_trans_ml <- cbind(train_data_cmc, interactions) %>%
  cbind(train_data_log)




# save data ---------------------------------------------------------------

save(train_data_trans, file="train_data_transformations_V3_with2750vars.Rda")
save(train_data_trans, file="train_data_transformationssubfactors_V3_with288vars.Rda")
save(train_data_sumscores_trans, file="train_data_sumscores_transformations_with2794vars.Rda")
save(train_data_trans_ml, file="train_data_transformations_V1_multilevel.Rda")
