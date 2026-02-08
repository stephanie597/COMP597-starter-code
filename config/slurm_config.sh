#!/bin/bash
# Custom SLURM configuration
# This overrides settings from default_slurm_config.sh

# Time limit: 10 minutes (for testing)
# For longer training, use "60:00" (1 hour) or "2:00:00" (2 hours)
export COMP597_SLURM_TIME_LIMIT="10:00"

# Optional: Increase memory if needed
# export COMP597_SLURM_MIN_MEM="16GB"