quick details about dataset:

**original dataset contained:**
2 outcomes (bias, outgroup_att)
36 raw measured variables 
11 latent variables
6 transformed variables
15 outgroups (clusters)

**transformations:**
- [X] grand-mean-centered versions of all predictors (raw measures, latent vars, transformed vars)
- [X] two-way interactions between grand-mean-centered predictors
- [X] log10-transformations of all valid predictors (log transformations that produced NaNs were omitted)

(version 2 of transformations, for interpretable mixed-effects models)
- [X] cluster-mean-centered versions of all predictors (each measure was converted into 2 variables: a "cluster mean" representing the average score of that measure for each Outgroup, and a "centered-within-cluster / cwc" value representing the original value minus cluster mean)
- [X] two-way interactions between all cluster means, and separately between all centered-within-clusters values
- [X] log10-transformations of all valid predictors (again, columns with NaNs were omitted)

- [ ] OTHER TRANSFORMATIONS TO TRY? ???!???? 

# exploratory stuff

- [ ] multilevel correlation analysis
- [ ] exploratory factor analysis 
- [ ] principal components analysis / dimension reduction
- [X] simple correlation analysis
- [X] if 2+ vars have significant interaction, create higher-order interaction terms

# search for parameter weights
OLS linear regression / multilevel model grid search for parameter weights
	goal: include ALL RAW/TRANSFORMED MEASURES and whittle down to fixed # of parameters and 
	identify parameter weights that perform the best


# benchmark with black box models
- [ ] TO-DO: create random forests or other black box solutions so we know what the "upper bound" of prediction is with this dataset (Kerem or Bonan help?)


# L1 regularization + elastic net
- [X] glmmLasso / elastic net regularization with all 36 raw measures + 11 latent variables + 6 transformed variables from ORIGINAL dataset (in competition materials)
- [X] glmmLasso / elastic net regularization with all transformed variables created 2024/06/05 
