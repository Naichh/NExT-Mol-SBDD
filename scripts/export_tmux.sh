#!/bin/bash
set -x
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES='0,1'

# --- [ADD THIS LINE] ---
# Replace 'eth0' with the interface name you found in Step 1
export NCCL_SOCKET_IFNAME=eth0

# Activate your Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nextmol

# Ensure torchrun can automatically select a port
unset MASTER_PORT

echo "--- Starting Minimal Distributed Test with specified NCCL Interface ---"
torchrun --nproc_per_node=2 test_dist.py
echo "--- Minimal Distributed Test Finished ---"