# randomForest to explain variance
# last updated 2024/06/06 -sally

# goal of this is to test the "upper bound" of how much variance we can explain
# with any fixed set of variables. for example, if i got 55 vars from elastic
# net, how good are these predictors? 
# if it's pretty good, then we can try to optimize the parameter weights using
# gradient descent and other search straetgies; if these 55 vars aren't so
# great, we go back and try other transformations / complex nonlinear stuff
# to try to build something better.

# setup -------------------------------------------------------------------
# load libraries
library(tidyverse)  # data hygiene
library(foreach)    # foreach() function
library(doParallel) # parallel computing
library(parallel)   # parallel computing
library(furrr)      # parallel computing
library(caret)      # cross-validation
library(randomForest) # random forests
library(MuMIn)      # model evaluation
library(Metrics)    # model evaluation

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 
set.seed(57812623)
options(scipen=999,digits=6)


# model 1: elastic net with 55 vars ---------------------------------------

# summary of transformations:
# included grand-mean-centered variables of all predictors, 2-way interactions
# between these centered variables, and log10 transformed versions of predictors
# elastic net reduced from 1400 variables to 55

# prepare data
load("train_data_transformations_V1.Rda")
variables_to_consider <- read_csv("output/ElasticNet_NOTmultilevel_ParameterWeights_Transformed_V1.csv") %>%
  filter(ParamWeight != 0)
train_data_trans_subset <- train_data_trans %>%
  select(Outgroup, bias, all_of(variables_to_consider$Variable))

# set up train control
train_control <- trainControl(
  method = "cv",          # cross-val
  number = 5,             # number of folds
  verboseIter = TRUE,     # trainin log
  savePredictions = TRUE, # save out-of-fold preds for best model
  classProbs = FALSE      # for classification; not needed
)

# define formula for model
formula_str <- paste("bias ~", paste(variables_to_consider$Variable, collapse = " + "))

# train the random forest model w caret
rf_model <- train(
  form = as.formula(formula_str),
  data = train_data_trans_subset,
  method = "rf",
  trControl = train_control,
  tuneLength = 5 # **CHANGEME** num of tuning parameters to try
)


print(rf_model)

# plot to see RMSE over different number of trees
plot(rf_model)

# evaluate model performance on training data
predictions <- predict(rf_model, newdata = train_data_trans_subset)
rmse <- sqrt(mean((predictions - train_data_trans_subset$bias)^2))
rsquared <- caret::R2(predictions, train_data_trans_subset$bias)

cat("RMSE:", rmse, "\n")
cat("R-squared:", rsquared, "\n")

# Variable importance
importance <- varImp(rf_model, scale = FALSE)
print(importance)
plot(importance)