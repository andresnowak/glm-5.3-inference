#!/bin/bash
# Download zai-org/GLM-5.3 into the scratch HF cache as a batch job.
#
# Why not the login node: two login-node attempts were killed silently mid-transfer
# (at ~245 GB and ~519 GB) with nothing in the log, which is what login-node resource
# policing looks like.
#
# Why not the xfer partition, the usual home for bulk data movement: nid001154/nid001306
# are x86_64, and this account's uv lives in ~/.local/aarch64/bin -- it dies there with
# "cannot execute binary file". Use a normal (GH200, aarch64) node instead.
#
# Resumable: hf skips blobs already complete, so re-submitting continues where it stopped.
#SBATCH --job-name=glm53-download
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=/iopsstor/scratch/cscs/anowak/tmp/glm53/logs/download-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/anowak/tmp/glm53/logs/download-%j.err

set -euo pipefail
exec /users/anowak/developer/glm-5.3/scripts/download.sh
