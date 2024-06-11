# SETTINGS FOR SLURM PARALLEL JOB

args = commandArgs(trailingOnly=TRUE)
print(args)
idx <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(paste0("idx= ",idx))

# INSERT CODE, LIBRARIES, ETC HERE

source('run_cross_validation.R')

parameters <- seq(1, length(formula_strs))
formula_str <- parameters[idx + 1]         # array of test effects to estimate whatever parameter u want
print(paste0("test_effects: ",formula_str))

cluster_run(formula_str)
