import argparse
from collections import defaultdict
import copy
import traceback
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from tqdm import tqdm

from tmcgen.data import BaseDataset
from tmcgen.utils.ops import to_numpy
from tmcgen.common.constants import DEVICE
from tmcgen.models import ALLOWED_MODELS
from tmcgen.utils.setup import construct_log_dir
from tmcgen.game import DockingEngine, get_strategy_from_args
import random

from rdkit.Chem import GetPeriodicTable

ptable = GetPeriodicTable()

# Type aliases
Tensor = torch.Tensor
Loader = Union[DataLoader, DataListLoader]
StepFnOutputs = tuple[Tensor, dict[str, Tensor], Optional[dict[str, list]]]
Args = argparse.Namespace


class ProgressMonitor:

    def __init__(self, metric_names: Optional[list[str]] = None):
        if metric_names is not None:
            self.metric_names = metric_names
            self.metrics = {metric: 0.0 for metric in self.metric_names}
        self.count = 0

    def add(self, metric_dict: dict[str, float], batch_size: int = None):
        if not hasattr(self, 'metric_names'):
            self.metric_names = list(metric_dict.keys())
            self.metrics = {metric: 0.0 for metric in self.metric_names}

        self.count += (1 if batch_size is None else batch_size)

        for metric_name, metric_value in metric_dict.items():
            if metric_name not in metric_dict:
                self.metrics[metric_name] = 0.0
                self.metric_names.append(metric_name)
            if metric_value is None:
                continue
            self.metrics[metric_name] += metric_value * (1 if batch_size is None else batch_size)

    def summarize(self) -> dict[str,float]:
        return {k: np.round(v / self.count, 4) for k, v in self.metrics.items()}



def prepare_gt_outputs_score_model(
    data: Data, 
    t_to_sigma: Callable, 
    device: str = 'cpu'
) -> dict[str, tuple[Tensor, Tensor]]:

    tr_score = torch.cat(
        [d.tr_score for d in data], dim=0) \
            if isinstance(data, list) else data.tr_score
    rot_score = torch.cat([d.rot_score for d in data], dim=0) \
          if isinstance(data, list) else data.rot_score
    tor_score = torch.cat([d.tor_score for d in data], dim=0) \
          if isinstance(data, list) else data.tor_score
    sphere_score = torch.cat([d.sphere_score for d in data], dim=0) \
          if isinstance(data, list) else data.sphere_score
    bl_score = torch.cat([d.bl_score for d in data], dim=0) \
          if isinstance(data, list) else data.bl_score

    if isinstance(data, list):
        tr_sigma_list, rot_sigma_list, tor_sigma_list, sphere_sigma_list, bl_sigma_list = zip(*[t_to_sigma(d.t_tr, d.t_rot, d.t_tor, d.t_sphere, d.t_bl)    
                                              for d in data])
        tr_sigma = torch.cat(tr_sigma_list, dim=0)
        rot_sigma = torch.cat(rot_sigma_list, dim=0)
        tor_sigma = torch.cat(tor_sigma_list, dim=0)
        sphere_sigma = torch.cat(sphere_sigma_list, dim=0)
        bl_sigma = torch.cat(bl_sigma_list, dim=0)
    else:
        tr_sigma, rot_sigma, tor_sigma, sphere_sigma, bl_sigma = t_to_sigma(data.t_tr, data.t_rot, data.t_tor, data.t_sphere, data.t_bl)

    scores_true = (tr_score, rot_score,tor_score, sphere_score, bl_score)
    sigmas = (tr_sigma, rot_sigma, tor_sigma, sphere_sigma, bl_sigma)
    gt_outputs = {
        'scores_true': scores_true,
        'sigmas': sigmas
    }
    return gt_outputs


# ==============================================================================
# Custom step_fns for different models 
# ==============================================================================


def step_fn_score(
    model: torch.nn.Module,
    data: Data, 
    loss_fn: Callable,
    outputs=None,
    n_gpus: int = 0,
    **kwargs
) -> StepFnOutputs:  
    """Step function for the score model."""
    t_to_sigma_fn = kwargs['t_to_sigma_fn']

    if isinstance(data, list):
        data =  [d.to(DEVICE) for d in data]
    else:
        data = data.to(DEVICE)

    #torch._C._jit_set_bailout_depth(0)
    #torch._C._jit_set_fusion_strategy([("DYNAMIC", 0)])
    if kwargs["model_name"]=='confidence':
        confidence_pred =  model(data)
        confidence_true = data.confidence_true
        loss, loss_dict = loss_fn(
            confidence_pred,
            confidence_true
            )
    else:
        tr_pred, rot_pred, tor_pred, sphere_pred, bl_pred  = model(data)
        gt_outputs = \
            prepare_gt_outputs_score_model(
                data, t_to_sigma=t_to_sigma_fn, device=rot_pred.device)
        loss, loss_dict = loss_fn(
            data= data,
            scores_pred=(tr_pred, rot_pred, tor_pred, sphere_pred, bl_pred),
            scores_true=gt_outputs['scores_true'],
            distogram_pred=None,
            sigmas=gt_outputs['sigmas']
        )

    return loss, loss_dict, outputs


# ==============================================================================
# Train, validation and inference epoch fns
# ==============================================================================


def train_epoch(
    model: torch.nn.Module,
    loader: Loader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    ema_weights=None,
    model_name: str = 'dock_reward',
    grad_clip_value: float = 1.0,
    step_every: int = 1,
    n_gpus: int = 0,
    **kwargs) -> dict[str, float]:
    """Training epoch fn. Involves repeated calls to step_fn defined by model."""

    model.train()
    monitor = ProgressMonitor()
    grad_monitor = ProgressMonitor()

    optimizer.zero_grad()

    kwargs['model_name'] = model_name
    train_outputs = None

    for idx, data in enumerate(tqdm(loader)):
        if data is None:
            continue
        if not isinstance(data, list):
            data = data.to(DEVICE)

        train_step_fn = step_fn_score

        try:
            with torch.autograd.set_detect_anomaly(True):
                loss, loss_dict, train_outputs = train_step_fn(
                    model=model, data=data, loss_fn=loss_fn,
                    outputs=train_outputs, n_gpus=n_gpus,
                    **kwargs
                )

                if step_every > 0:
                    loss /= step_every
                    loss.backward()

            if idx % step_every == 0:
                if grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

                grads = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grads.append(param.grad.norm().item())
                grad_mean = torch.tensor(grads).mean()
                grad_monitor.add({'grads': grad_mean.item()})

                optimizer.step()
                optimizer.zero_grad()

            monitor.add(loss_dict)

            if ema_weights is not None:
                ema_weights.update(model.parameters())

        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch', flush=True)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch', flush=True)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                print(e, flush=True)
                traceback.print_exc()
                continue

    train_losses = monitor.summarize()
    grads_summary = grad_monitor.summarize()
    train_losses['grads'] = grads_summary['grads']

    return train_losses


def validation_epoch(
    model: torch.nn.Module,
    loader: Loader, 
    loss_fn: Callable,
    make_outputs: bool = False,
    model_name: str = 'dock_reward',
    n_gpus: int = 0,
    **kwargs,
) -> dict[str, float]:
    """Validation epoch. Involves repeated calls to step_fn defined by model."""

    model.eval()
    monitor = ProgressMonitor()

    val_outputs = None
    if make_outputs and model_name in ALLOWED_MODELS:
        val_outputs = defaultdict(list)
    
    kwargs["model_name"] = model_name
    for idx, data in enumerate(tqdm(loader, total=len(loader))):
        if data is None:
            continue
        try:
            with torch.no_grad():
                if (not isinstance(data, list)) or (not isinstance(data, dict)) :
                    data = data.to(DEVICE)

                if (model_name == 'score') or (model_name == 'confidence'):
                    val_step_fn = step_fn_score
                else:
                    raise NotImplementedError

                _, loss_dict, val_outputs = val_step_fn(
                    model=model, data=data, outputs=val_outputs, n_gpus=n_gpus,
                    loss_fn=loss_fn, **kwargs
                )
                monitor.add(loss_dict)
        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch', flush=True)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch', flush=True)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                print(e, flush=True)
                traceback.print_exc()
                continue
        
    try:
        val_losses = monitor.summarize()
    except:
        val_losses = None
    
    if make_outputs and val_outputs is not None:
        for key, output in val_outputs.items():
            val_outputs[key] = np.asarray(output)

        # Sort by true scores
        if model_name in ALLOWED_MODELS:
            sorted_indices = np.argsort(val_outputs["diff_true"], axis=None)

            for key, output in val_outputs.items():
                val_outputs[key] = output[sorted_indices]
    
        return val_losses, val_outputs
    return val_losses


import os


def inference_epoch(
    model: torch.nn.Module,
    dataset_orig: BaseDataset,
    args: Args
) -> dict[str, float]:
    """Inference epoch. Calls the DockingEngine to play the game and saves outputs."""
    log_dir = construct_log_dir(args=args)

    # Build id list
    id_list = dataset_orig.complexes_split
    if 'num_inference_complexes' in args:
        id_list = id_list[:args.num_inference_complexes]

    if 'inference_multiplicity' in args:
        if args.save_for_confidence:
            random.shuffle(id_list)
            id_list = [cid for cid in id_list for _ in range(args.inference_multiplicity)]
        else:
            id_list = id_list * args.inference_multiplicity
    else:
        id_list = id_list * 10 # default

    print("id_list first entries", id_list[:100], flush=True)

    # Setup strategy and docking engine
    strategy = get_strategy_from_args(
        strategy_type="langevin",
        model=model.module if DEVICE == 'cuda' and args.n_gpus > 1 else model,
        model_args=args,
        n_rounds=args.inference_steps,
        ode=args.ode if 'ode' in args else False,
        distance_penalty=args.distance_penalty if 'distance_penalty' in args else 0.0,
        device=DEVICE,
    )

    engine = DockingEngine(
        strategy=strategy,
        n_rounds=args.inference_steps,
        agent_type=args.agent_type,
        debug=args.debug,
        save_trajectory=args.save_trajectory if "save_trajectory" in args else False,
        log_every=None,
    )

    # Prepare output folders
    if args.save_for_confidence:
        output_folder = os.path.join(log_dir, "output_to_train_confidence")
        os.makedirs(output_folder, exist_ok=True)

    if args.save_xyz:
        xyz_folder = os.path.join(log_dir, "xyzs_gen")
        os.makedirs(xyz_folder, exist_ok=True)

    for idx, pdb_id in enumerate(tqdm(id_list, total=len(id_list))):
        try:
            data = torch.load(f"{dataset_orig.full_processed_dir}/{pdb_id}.pt")
            data.pdb_id = pdb_id
            complex_graph = copy.deepcopy(data).to(DEVICE)

            # Pick players based on perturbation strategy
            if args.pert_strategy == "all-but-one":
                players = data.agent_keys[:-1]
            elif args.pert_strategy == "two":
                players_all = data.agent_keys[:-1]
                if len(players_all) > 1:
                    players = data.agent_keys[0:2]
                else:
                    print("Skipping strategy 'two': Not enough agents available.")
                    raise NotImplementedError
            elif args.pert_strategy == "one":
                players_all = data.agent_keys[:-1]
                if len(players_all) > 1:
                    players = data.agent_keys[0:1]
                else:
                    print("Skipping strategy 'one': Not enough agents available.")
                    raise NotImplementedError

            protein_dict, _ = engine.play(
                data=complex_graph,
                agent_params=None,
                player_keys=players,
                monitor=None,
            )

            # Save XYZ file
            if args.save_xyz:
                xyz_filename = os.path.join(xyz_folder, f"{pdb_id}_{idx}.xyz")
                atomic_numbers = []
                positions = []
                for key in data.agent_keys:
                    if "atomic_numbers" in data[key]:
                        atomic_numbers.extend(data[key][key].atomic_numbers.cpu().numpy())
                        positions.extend(protein_dict[key].pos.cpu().numpy())

                with open(xyz_filename, "w") as xyz_file:
                    xyz_file.write(f"{len(atomic_numbers)}\n")
                    xyz_file.write(f"Generated for {pdb_id}\n")
                    for atom_num, atom_pos in zip(atomic_numbers, positions):
                        atom_symbol = ptable.GetElementSymbol(int(atom_num))
                        xyz_file.write(
                            f"{atom_symbol} {atom_pos[0]:.6f} {atom_pos[1]:.6f} {atom_pos[2]:.6f}\n"
                        )
                print(f"Saved XYZ file: {xyz_filename}", flush=True)

            # Save confidence training input
            if args.save_for_confidence:
                data_to_save = copy.deepcopy(data)
                for key in data_to_save.agent_keys:
                    if hasattr(data_to_save[key], 'pos'):
                        data_to_save[key].pos = protein_dict[key].pos.detach().cpu()

                output_path = os.path.join(output_folder, f"{pdb_id}_{idx}.pt")
                data_entry = {
                    "data": data_to_save,
                    "pdb_id": pdb_id,
                    "conformer_id": idx,
                }
                torch.save(data_entry, output_path)
                print(f"Saved confidence training input to: {output_path}", flush=True)

            del complex_graph

        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                print(e, flush=True)
                traceback.print_exc()
                continue

    return {}

