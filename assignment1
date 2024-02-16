#!/bin/bash

#======================================================
#
# Job script for running a parallel job on multiple cores across multiple nodes (shared) 
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=teaching
#
# Specify project account
#SBATCH --account=teaching
#
# Distribute processes in round-robin fashion
#SBATCH --distribution=cyclic
#
# No. of tasks required (max of 80), cores might be spread across various nodes (nodes will be shared)
#SBATCH --ntasks=80
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=00:20:00
#
# Job name
#SBATCH --job-name=SCMcG_Assignment_1
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

  module purge
  module load nvidia/sdk/21.3
  module load ansys/21.2
  module load anaconda/python-3.9.7/2021.11



#======================================================
# Prologue script to record job details
# Do not change the line below
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

# Modify the line below to run your program
python SCMcG_Assignment_1.py

#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------