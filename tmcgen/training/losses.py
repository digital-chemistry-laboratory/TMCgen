from functools import partial
import torch
from typing import Tuple, Optional, Sequence, Callable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from typing import Tuple, Optional, Sequence, Callable
from torch_geometric.data import HeteroData
from tmcgen.common.constants import DEVICE
import tmcgen.utils.so3 as so3
import tmcgen.utils.torus as torus
import tmcgen.utils.n_sphere_angle as n_sphere_angle
from tmcgen.common.constants import DEVICE


# Type aliases
Tensor = torch.Tensor


def score_matching_loss(
    scores_pred: Sequence[Tensor],
    scores_true: Sequence[Tensor],
    sigmas: Sequence[Tensor],
    distogram_pred: Tensor,
    data: HeteroData,
    w_tr: float = 1.0,
    w_rot: float = 1.0,
    w_tor: float = 1.0,
    w_dist: float = 1.0,
    w_sphere: float = 0.0,
    w_bl: float = 0.0,
    apply_mean: bool = True,
    no_torsion: bool = True, 
    num_bins: int = 32,
    use_distogram: bool = False,
    sphere_diffusion: bool = False,
    no_rot_first_lig: bool = False,
    no_sphere_first_lig: bool = False,
    restrict_rot_update: bool = False,


) -> Tuple[Tensor, dict[str, Tensor]]:  
    mean_dims = (0, 1) if apply_mean else 1


    tr_score_pred, rot_score_pred, tor_score_pred, sphere_score_pred, bl_score_pred = scores_pred
    tr_score_true, rot_score_true, tor_score_true, sphere_score_true, bl_score_true = scores_true
    tr_sigma, rot_sigma, tor_sigma, sphere_sigma, bl_sigma = sigmas

    rot_score_true = rot_score_true.squeeze(-1)
 
    
    # Translation Losses
    if sphere_diffusion:
        if no_sphere_first_lig:  
            if isinstance(data, list):
                mask_first_lig = torch.cat([d.mask_first_lig for d in data])
            else:
                mask_first_lig = data.mask_first_lig
            sphere_sigma_filtered = sphere_sigma[mask_first_lig == False]

            # Normalize score
            sphere_score_norm = n_sphere_angle.score_norm(sphere_sigma_filtered.cpu()).to(DEVICE)
        else:
            sphere_score_norm = n_sphere_angle.score_norm(sphere_sigma.cpu()).to(DEVICE)
            
        diff = sphere_score_pred - sphere_score_true
        sphere_loss = 3.0 * torch.mean(((diff) / (sphere_score_norm.unsqueeze(-1) + 1e-5))**2) 


        sphere_score_scalar = torch.linalg.norm(sphere_score_pred, dim=-1)
        sphere_base_loss =  3.0 * torch.mean((sphere_score_true / sphere_score_norm.unsqueeze(-1) )**2)

        if no_sphere_first_lig:
            count_nonzero = sum(torch.count_nonzero(sphere_score_true, dim=1)).float() / 3.0
            total_rows = sphere_score_true.shape[0]
            scaling_factor =  count_nonzero / total_rows
            if not torch.isclose(scaling_factor, torch.tensor(0.0)):
                sphere_loss = sphere_loss / (scaling_factor+1e-6)
                sphere_base_loss = sphere_base_loss / (scaling_factor+1e-6)

        bl_loss = (((bl_score_pred - bl_score_true) * bl_sigma.unsqueeze(-1)) ** 2).mean(dim=mean_dims)
        bl_base_loss = ((bl_score_true * bl_sigma.unsqueeze(-1)) ** 2).mean(dim=mean_dims).detach() 

        tr_loss, tr_base_loss = tr_score_pred, torch.zeros(1, dtype=torch.float, device=DEVICE)


    else:
        tr_sigma = tr_sigma.unsqueeze(-1)
        tr_loss = (((tr_score_pred - tr_score_true) * tr_sigma) ** 2).mean(dim=mean_dims)
        tr_base_loss = ((tr_score_true * tr_sigma) ** 2).mean(dim=mean_dims).detach()

        sphere_loss, sphere_base_loss = torch.zeros(1, dtype=torch.float, device=DEVICE), torch.zeros(1, dtype=torch.float, device=DEVICE)
        bl_loss, bl_base_loss = torch.zeros(1, dtype=torch.float, device=DEVICE), torch.zeros(1, dtype=torch.float, device=DEVICE)

    # Rotation losses

    
    if no_rot_first_lig:
        
        if isinstance(data, list):
            mask_first_lig = torch.cat([d.mask_first_lig for d in data])
            rot_sigma_filtered=rot_sigma[mask_first_lig == False]
        else:
            mask_first_lig = data.mask_first_lig
            rot_sigma_filtered = rot_sigma[data.mask_first_lig == False]

        rot_score_norm = so3.score_norm(rot_sigma_filtered.cpu()).unsqueeze(-1).to(DEVICE)
    else:
        rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1).to(DEVICE)

    if rot_score_true.numel() == 0:
        rot_loss = torch.tensor(0.0, device=DEVICE)
        rot_base_loss = torch.tensor(0.1, device=DEVICE)
    else:
        
        if restrict_rot_update:
            torch.set_printoptions(sci_mode=False, precision=6)
            tor_score_norm2 = torch.tensor(torus.score_norm(rot_sigma_filtered.cpu().numpy())).float().to(DEVICE)
            rot_loss = ((rot_score_pred - rot_score_true) ** 2 / tor_score_norm2)
            rot_base_loss = ((rot_score_true ** 2 / tor_score_norm2)).detach()
            rot_loss, rot_base_loss = rot_loss.mean() * torch.ones(1, dtype=torch.float, device=DEVICE), rot_base_loss.mean() * torch.ones(1, dtype=torch.float, device=DEVICE)

        else:
            rot_loss = (((rot_score_pred - rot_score_true) / (rot_score_norm + 1e-5)) ** 2).mean(dim=mean_dims)
            rot_base_loss = ((rot_score_true / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

        if no_rot_first_lig and not restrict_rot_update:
            count_nonzero = sum(torch.count_nonzero(rot_score_true, dim=1)).float() / 3.0
            total_rows = rot_score_true.shape[0]
            scaling_factor =  count_nonzero / total_rows
            if not torch.isclose(scaling_factor, torch.tensor(0.0)):
                rot_loss = rot_loss / (scaling_factor+1e-6)
                rot_base_loss = rot_base_loss / (scaling_factor+1e-6)
        
   

    if not no_torsion:
        if isinstance(data, list):
        

            tor_sigma_edge_valid = [e.flatten().to(DEVICE) 
                                for d in data 
                                for sublist in (d.tor_sigma_edge if isinstance(d.tor_sigma_edge, list) else [d.tor_sigma_edge]) 
                                for e in (sublist if isinstance(sublist, list) else [sublist]) 
                                if isinstance(e, torch.Tensor)]
        



            rot_score = torch.cat([d.rot_score for d in data], dim=0).to(DEVICE)
            tor_score = torch.cat([d.tor_score for d in data], dim=0).to(DEVICE)
        else:
            tor_sigma_edge_valid = [e.unsqueeze(0).to(DEVICE) for e in data.tor_sigma_edge if isinstance(e, torch.Tensor)]
            rot_score = data.rot_score.to(DEVICE)
            tor_score = data.tor_score.to(DEVICE)


        if not len(tor_sigma_edge_valid) == 0:
            edge_tor_sigma = torch.cat(tor_sigma_edge_valid).unsqueeze(0).to(DEVICE) if tor_sigma_edge_valid else torch.tensor([]).to(DEVICE)
            

            tor_score = torch.cat([e.unsqueeze(0) for e in tor_score_true if isinstance(e, torch.Tensor)] + 
                      [e.unsqueeze(0) for sublist in tor_score_true if isinstance(sublist, list) for e in sublist if isinstance(e, torch.Tensor)])


            tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float().to(DEVICE)
            tor_loss = ((tor_score_pred - tor_score) ** 2 / tor_score_norm2)
            tor_loss = torch.clamp(tor_loss, min=0, max=2)
            tor_base_loss = ((tor_score ** 2 / tor_score_norm2)).detach()
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float, device=DEVICE), tor_base_loss.mean() * torch.ones(1, dtype=torch.float, device=DEVICE)

        else:
            tor_loss, tor_base_loss = tor_score_pred, torch.zeros(1, dtype=torch.float, device=DEVICE)
        
    else:
        tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float, device=DEVICE), torch.zeros(1, dtype=torch.float, device=DEVICE)

   

    if w_bl == 0.0:
        bl_loss = torch.zeros(1, dtype=torch.float, device=DEVICE)
    if w_tr == 0.0:
        tr_loss = torch.zeros(1, dtype=torch.float, device=DEVICE)
    if w_tor == 0.0:
        tor_loss = torch.zeros(1, dtype=torch.float, device=DEVICE)


    loss = (tr_loss * w_tr + rot_loss * w_rot + tor_loss * w_tor + sphere_loss * w_sphere + bl_loss * w_bl ) 
    assert not torch.isnan(loss)
    
    loss_dict = {
        "loss": loss.item(),
        "tr_loss": tr_loss.item(),
        "rot_loss": rot_loss.item(),
        "tor_loss": tor_loss.item(),
        "sphere_loss": sphere_loss.item(),
        "bl_loss": bl_loss.item(),
        "tr_base_loss": tr_base_loss.item(),
        "rot_base_loss": rot_base_loss.item(),
        "tor_base_loss": tor_base_loss.item(),
        "sphere_base_loss": sphere_base_loss.item(),
        "bl_base_loss": bl_base_loss.item(),
        "rot_loss_ratio_to_base:": rot_loss.item() / (rot_base_loss.item()+ 1e-6), 
    }

    return loss, loss_dict


def loss_fn_from_args(args) -> Callable:

    
    
    
    loss_fn = partial(
        score_matching_loss,
        w_tr=args.w_tr,
        w_rot=args.w_rot,
        w_tor=args.w_tor,
        w_dist=args.w_dist,
        w_sphere=args.w_sphere,
        w_bl=args.w_bl,
        no_torsion=args.no_torsion,
        use_distogram=args.use_distogram,
        num_bins=args.num_bins,
        sphere_diffusion=args.sphere_diffusion,
        no_rot_first_lig=args.no_rot_first_lig,
        no_sphere_first_lig=args.no_sphere_first_lig,
        restrict_rot_update=args.restrict_rot_update,
    )
    return loss_fn


