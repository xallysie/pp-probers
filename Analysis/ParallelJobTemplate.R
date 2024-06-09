# SETTINGS FOR SLURM PARALLEL JOB

args = commandArgs(trailingOnly=TRUE)
print(args)
idx <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
print(paste0("idx= ",idx))
parameters <- seq(-.01, -.20, by=-.013) # sequence array from 0 to 14 
print(paste0("parameters: ",parameters))
test_effects <- parameters[idx + 1]         # array of test effects to estimate whatever parameter u want
print(paste0("test_effects: ",test_effects))

# INSERT CODE, LIBRARIES, ETC HERE