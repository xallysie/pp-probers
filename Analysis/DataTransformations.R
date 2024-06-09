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
tidy_mahalanobis <- function(...) {
  variables <- cbind(...)
  if (sum(is.na(variables)) != 0)
  {
    stop("Your data has some NAs, which will cause `tidy_mahalnobis` to crash. Try removing NAs before running `tidy_mahalanobis`.")
  }
  # return 0 if there are too few points
  if(nrow(variables) < 5) {
    return(rep(0, nrow(variables)))
  }
  # *Tips hat to Joe Fruehwald via Renwick for this line of code.*
  t_params <- MASS::cov.trob(variables)
  mahalanobis(variables,
              center = t_params$center,
              cov    = t_params$cov)
}

train_data_trans %>%
  mutate(d_mahalanobis = tidy_mahalanobis(symbolic, contact_friendsz)) %>%
  select(d_mahalanobis) %>% unlist() %>% mean()

# loop through each pair of predictors and compute mahalanobis distance
#mahalanobis_distances <- data.frame(matrix(nrow=2010))
mahalanobis_distances <- data.frame(matrix(nrow=nrow(predictors_trans)))
for (i in seq_along(predictors_trans)) {
  for (j in seq_along(predictors_trans)) {
    if (i < j) {  # to avoid duplicate pairs
      col1 <- names(predictors_trans)[i]
      col2 <- names(predictors_trans)[j]
      var_pair <- paste0(col1, "~", col2, "_mD_log")
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


# combine these transformed variables
train_data_trans <- cbind(train_data_gmc, interactions) %>%
  cbind(train_data_log) %>% 
  cbind(mahalanobis_distances) %>%
  cbind(individual_variances) %>%
  cbind(train_data_extended)


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

save(train_data_trans, file="train_data_transformations_V2_with2653vars.Rda")
save(train_data_trans_ml, file="train_data_transformations_V1_multilevel.Rda")
