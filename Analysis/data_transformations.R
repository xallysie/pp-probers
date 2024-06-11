
library(tidyr)
library(dplyr)
library(lavaan)

# import training data
train_data <- read.csv("../TrainingData/train.csv") %>%
  select(Outgroup, bias, outgroup_att, everything())
# fix training data by recoding reversed values
train_data <- train_data %>%
  mutate(sThreat3 = 8 - sThreat3,
         Agreeable1 = 6 - Agreeable1)

# vars to exclude 
not_allowed_vars = c("WarmOG", "PositiveOG", "LikeOG", "WarmIG", "PositiveIG", "LikeIG", "diff_warm", "diff_pos", "diff_like")

vars_bonan <- c(
  "generalized", "symbolic",
  "identification_selfinvestment",
  "identification_sol", "identification_sat", "identification_cen",
  "contact_quality", "contact_friendsz"
)
vars_bonan <- setdiff(vars_bonan, not_allowed_vars)


# do a sanity check
train_data_bn = train_data %>%
  mutate(generalized_bn = rThreatOG1 - rThreatIG1 + rThreatOG2 - rThreatIG2) %>%
  mutate(generalized_diff = generalized_bn - generalized) %>%
  mutate(diff_warm_bn=WarmIG - WarmOG - diff_warm) %>%
  mutate(diff_symbolic = sThreat1 + sThreat2 + sThreat3R + sThreat4 - symbolic)

train_data_bn %>% filter(diff_symbolic != 0)

# it's factor model!
symbolic<- 'fsymbolic =~ sThreat1 + sThreat2 + sThreat3R + sThreat4'
d = train_data %>% select(sThreat1, sThreat2, sThreat3R, sThreat4)
fitsymbolic<- cfa(symbolic, data=d, bootstrap=5000, missing="ml")
summary (fitsymbolic, fit.measures=TRUE, standardized=TRUE)
d$symbolic <- lavPredict(fitsymbolic, method="regression")
colnames(d) <- c('sThreat1', 'sThreat2', 'sThreat3R', 'sThreat4', 'symbolic')

train_data_bn$cfa_symbolic = d$symbolic
train_data_bn = train_data_bn %>% mutate(symbolic_dff = symbolic - cfa_symbolic)
train_data_bn %>% filter(symbolic_dff != 0)


# OK let's get some data transformation
# "bias ~ generalized + symbolic + contact_quality + contact_friendsz + identification_selfinvestment", 

# raise to powers
train_data_extended = train_data %>%
  mutate(
    generalized_squared = if_else(generalized < 0, -1*(generalized^2), generalized^2),
    generalized_cubid = generalized^3,
    
    symbolic_squared = if_else(symbolic < 0, -1*(symbolic^2), symbolic^2),
    symbolic_cubid = symbolic^3,
    
    identification_selfinvestment_squared = if_else(identification_selfinvestment < 0, -1*(identification_selfinvestment^2), identification_selfinvestment^2),
    identification_selfinvestment_cubid =  identification_selfinvestment^3,
    
    contact_quality_squared = if_else(contact_quality < 0, -1*(contact_quality^2), contact_quality^2),
    contact_quality_cubid = contact_quality*3,
    
    contact_friendsz_squared = if_else(contact_friendsz < 0, -1*(contact_friendsz^2), contact_friendsz^2),
    contact_friendsz_cubid = contact_friendsz^3,
    
    contact_total = contact_quality*contact_friendsz
  )
save(train_data_extended, file='../TrainingData/train_data_extended.Rdata')








