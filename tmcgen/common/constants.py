import os
from pathlib import Path

import torch


# ==============================================================================
# Training specific constants
# ==============================================================================

# Wandb
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# Directories
PROJECT_DIR = Path(__file__).parents[2]
EXP_DIR = f"{PROJECT_DIR}/experiments"

# Cluster
IS_CLUSTER = Path("/cluster").is_dir()
CLUSTER_GROUP_DIR = os.environ.get('CLUSTER_GROUP_DIR', None)
CLUSTER_PROJ_DIR = f"{CLUSTER_GROUP_DIR}/dockgame"
CLUSTER_EXP_DIR = f"{CLUSTER_PROJ_DIR}/experiments"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
