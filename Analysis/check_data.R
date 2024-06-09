
library(dplyr)
library(tidyr)
library(ggplot2)
library(rlang)

# import training data
train_data <- read.csv("../TrainingData/train.csv") %>%
  select(Outgroup, bias, outgroup_att, everything())
# fix training data by recoding reversed values
train_data <- train_data %>%
  mutate(sThreat3 = 8 - sThreat3,
         Agreeable1 = 6 - Agreeable1)

# vars to exclude 
not_allowed_vars = c("WarmOG", "PositiveOG", "LikeOG", "WarmIG", "PositiveIG", "LikeIG", "diff_warm", "diff_pos", "diff_like")

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
raw_predictors_allowed <- setdiff(raw_predictors_key, not_allowed_vars)

transformed_vars_key <- c("diff_warm", # WarmIG - WarmOG
                          "diff_pos",  # PositiveIG - PositiveOG
                          "diff_like", # LikeIG - LikeOG
                          "contact_friendsz", # same as contact_friends but z-scored
                          "generalized_challdiff", # rThreatOG1 - rThreatIG1
                          "generalized_probdiff"  # rThreatOG2 - rThreatIG2
)
transformed_vars_allowed <- setdiff(transformed_vars_key, not_allowed_vars)

latent_vars_key <- c(
  "symbolic", "identification_sol", "identification_sat", "identification_cen",
  "identification_selfinvestment", "contact_quality",
  "b5a",
  "generalized", "disgust_p", "disgust_s", "disgust_r"
)
latent_vars_allowed <- setdiff(latent_vars_key, not_allowed_vars)
ggsave(filename = paste0('plots/bias_hist.png'))

# Plot bias distributions per group
ggplot(train_data, aes(x=bias)) +
  geom_histogram() +
  facet_wrap(~Outgroup)


# Plot correlation between bias & keys
for (k in latent_vars_allowed) {
  ggplot(train_data, aes(x = .data[[k]], y = .data[['bias']])) +
    geom_point() +
    facet_wrap(~Outgroup)
  ggsave(filename = paste0('plots/', k, '.png'))
}


ggplot(train_data, aes(x=bias, y=)) +
  geom_histogram() +
  facet_wrap(~Outgroup)






