import os
import numpy as np
import torch
from typing import Union, Callable
from itertools import permutations

import tmcgen.common.constants as constants
from tmcgen.utils import geometry as geometry_ops


try:
    import tmtools
except Exception as e:
    pass


Tensor = torch.Tensor
Array = np.ndarray
ArrayOrTensor = Union[Tensor, Array]
RmsdOutput = tuple[ArrayOrTensor, dict[str, float]]

# ==============================================================================
# RMSD Metrics
# ==============================================================================


def rmsd(y_pred: ArrayOrTensor, y_true: ArrayOrTensor) -> ArrayOrTensor:
    #print('y_pred',y_pred)
    #print('y_true',y_true)
    se = (y_pred - y_true)**2
    try:
        mse = se.sum(dim=1).mean()
        return torch.sqrt(mse)
    except Exception:
        mse = se.sum(axis=1).mean()
        return np.sqrt(mse)

def permute_rmsd(y_pred: ArrayOrTensor, y_true: ArrayOrTensor) -> ArrayOrTensor:
    """Compute RMSD considering atom permutation but not Kabsch alignment."""
    num_atoms = y_true.shape[0]
    best_rmsd = float('inf')
    
    # Generate all permutations of the atom indices
    for perm in permutations(range(num_atoms)):
        permuted_y_pred = y_pred[list(perm), :]
        current_rmsd = rmsd(permuted_y_pred, y_true)
        
        # Check if this permutation gives a lower RMSD
        if current_rmsd < best_rmsd:
            best_rmsd = current_rmsd
    
    return best_rmsd

def compute_complex_rmsd_torch(
    complex_pred: Tensor, 
    complex_true: Tensor,
    rotation_only: bool=False,
) -> RmsdOutput:

    rot_mat, tr = geometry_ops.rigid_transform_kabsch_3D_torch(
        complex_pred.T, complex_true.T, rotation_only=rotation_only,
    )
    if rotation_only:
        complex_pred_aligned = ( (rot_mat @ complex_pred.T) ).T
    else:
        complex_pred_aligned = ( (rot_mat @ complex_pred.T) + tr ).T

    complex_rmsd = rmsd(complex_pred_aligned, complex_true)

    return complex_rmsd, {
        "complex_rmsd": np.round(complex_rmsd.item(), 4),
    }


def compute_complex_rmsd(
    complex_pred: Array, 
    complex_true: Array
) -> RmsdOutput:
    rot_mat, tr = geometry_ops.rigid_transform_kabsch_3D(
        complex_pred.T, complex_true.T
    )
    complex_pred_aligned = ( (rot_mat @ complex_pred.T) + tr ).T

    complex_rmsd = rmsd(complex_pred_aligned, complex_true)

    return complex_rmsd, {
        "complex_rmsd": np.round(complex_rmsd.item(), 4),
    }


def compute_pyrosetta_scores(self, 
                                 filenames: dict[str, str], 
                                 agent_keys: list[str] = None) -> float:
        return compute_pyrosetta_scores(
                filenames=filenames, agent_keys=agent_keys, 
                score_fn=self.score_fn
            )
# ==============================================================================
# Sphere 
# ==============================================================================


def compute_angles(positions):
    """Compute angles between all pairs of positions with the origin (0,0,0) as the middle point."""
    n_atoms = positions.shape[0]
    angles = []
    
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            vec1 = positions[i] 
            vec2 = positions[j]
            
            # Compute the cosine of the angle
            cos_theta = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
            # Ensure numerical stability
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angle = torch.acos(cos_theta)  # Angle in radians
            angles.append(angle.item())
    #print('angles',torch.tensor(angles, device=positions.device).shape, torch.tensor(angles, device=positions.device))
    return torch.tensor(angles, device=positions.device)

def compute_angles_pairwise(positions1, positions2):
    """
    Compute angles between vectors in `positions1` and `positions2` with the origin (0,0,0) as the middle point.

    Args:
        positions1 (torch.Tensor): Tensor of shape (N, 3) representing N vectors.
        positions2 (torch.Tensor): Tensor of shape (N, 3) representing N vectors.

    Returns:
        torch.Tensor: Tensor of shape (N,) with angles in radians between corresponding vectors in positions1 and positions2.
    """
    dot_products = torch.sum(positions1 * positions2, dim=-1)

    norms1 = torch.norm(positions1, dim=-1)
    norms2 = torch.norm(positions2, dim=-1)

    cos_theta = dot_products / (norms1 * norms2 + 1e-9) 

    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    angles = torch.acos(cos_theta)

    return angles
