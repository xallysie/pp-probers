# prejudice competition folder

In the Analysis directory, `PrejudiceModeling_Full_Script.R` contains wrapper functions to do basic model fitting, model evaluation, cross-validation, and model comparison/parameter search with parallelization. 

To submit jobs to Princeton's computing cluster, look at the `*.slurm` and `ClusterJob_*.R` files for an example - they're more compact and adapted for *Adroit*.

`Regularization.R` has code for elastic net (non multilevel) and L1/lasso (mulitlevel) regularization: I ran a basic version with the 36 raw measures that were included in the original training data, another version with the latent variables included in the original training data, and other versions with transformed variables (e.g., 2-way interaction terms, polynomials, log-transformed variables).

# To-Do

* think of other ways to transform our variables
* think of other models to try (e.g., random forest, bayesian mlm)? as long as they have interpretable parameter estimates, they can be submitted
* even if they don't have interpretable estimates, they can be used as a benchmark for what is possible (kind of an upper bound of what we can get out of our data)
* consider dimension reduction options to transform the raw variables

