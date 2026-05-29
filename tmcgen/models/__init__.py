import os
import argparse
import yaml
from functools import partial
import time
from contextlib import contextmanager
import torch
from torch_geometric.nn.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
from tmcgen.models.score_model import ScoreModel
#from tmcgen.models.score_model_etflow import ScoreModelWithTorchMD
from tmcgen.utils.diffusion import get_timestep_embedding, t_to_sigma
from tmcgen.common.constants import DEVICE

from tmcgen.data.featurize import FEATURE_DIMENSIONS
import torch.distributed as dist

MODELS = {
    "score": ScoreModel,
    "confidence": ScoreModel,

    
}

ALLOWED_MODELS = list(MODELS.keys())






def build_score_model_from_args(args):

    if 'featurizer' not in args or args.featurizer is None:
        args.featurizer = "base"

    if 'node_encoder_type' not in args:
        args.node_encoder_type = "base"
    
    node_fdim = args.node_fdim

    model_cls = MODELS.get(args.model, None)
    if model_cls is None:
        raise ValueError(
            f"{args.model} is not a valid model name." +
            f"Allowed models: {list(MODELS.keys())}"
        )
    
    timestep_emb_fn = get_timestep_embedding(
        embedding_type=args.time_embed_type,
        embedding_dim=args.sigma_emb_dim,
        embedding_scale=args.time_embed_scale
    )
    
    t_to_sigma_fn = partial(
        t_to_sigma,
        tr_sigma_min=args.tr_sigma_min,
        tr_sigma_max=args.tr_sigma_max,
        rot_sigma_min=args.rot_sigma_min,
        rot_sigma_max=args.rot_sigma_max,
        tor_sigma_min=args.tor_sigma_min,
        tor_sigma_max=args.tor_sigma_max,
        sphere_sigma_min=args.sphere_sigma_min,
        sphere_sigma_max=args.sphere_sigma_max,
        bl_sigma_min=args.bl_sigma_min,
        bl_sigma_max=args.bl_sigma_max
    )
  
    params = {
        "architecture": args.architecture,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "num_rbf": args.num_rbf,
        "cutoff_lower": args.cutoff_lower,
        "cutoff_upper": args.cutoff_upper,
        "trainable_rbf": args.trainable_rbf,
        "node_fdim": node_fdim,
        "node_encoder_type": args.node_encoder_type,
        "edge_fdim": args.edge_fdim,
        "sh_lmax": args.sh_lmax,
        "n_s": args.n_s, "n_v": args.n_v,
        "n_conv_layers": args.n_conv_layers,
        "max_radius": args.max_radius,
        "cross_max_radius": args.cross_max_radius,
        "center_max_radius": args.center_max_radius,
        "distance_emb_dim": args.distance_emb_dim,
        "angle_emb_dim": args.angle_emb_dim if "angle_emb_dim" in args else 256,
        "cross_dist_emb_dim": args.cross_dist_emb_dim,
        "center_dist_emb_dim": args.center_dist_emb_dim,
        "timestep_emb_fn": timestep_emb_fn,
        "sigma_emb_dim": args.sigma_emb_dim,
        "dropout_p": args.dropout_p,
        "activation": args.activation,
        "scale_by_sigma": args.scale_by_sigma if "scale_by_sigma" in args else False,
        "t_to_sigma_fn": t_to_sigma_fn,
        "no_torsion": args.no_torsion,
        "use_distogram": args.use_distogram,
        "first_break": args.first_break,
        "last_break": args.last_break,
        "debug":args.debug,
        "num_bins": args.num_bins,
        "sphere_diffusion": args.sphere_diffusion,
        "no_rot_first_lig": args.no_rot_first_lig,
        "no_sphere_first_lig": args.no_sphere_first_lig if "no_sphere_first_lig" in args else False,
        "separate_rot_sphere_conv": args.separate_rot_sphere_conv,
        "sphere_projection": args.sphere_projection,
        "anchor_graph_sphere": args.anchor_graph_sphere,
        "batch_norm": args.batch_norm if "batch_norm" in args else False,
        "use_bl_metal_graph": args.use_bl_metal_graph if "use_bl_metal_graph" in args else False,
        "use_second_order_repr": args.use_second_order_repr if "use_second_order_repr" in args else False,
        "separate_intra_inter_agent_updates": args.separate_intra_inter_agent_updates if "separate_intra_inter_agent_updates" in args else True,
        "angle_encoding_repr_learning": args.angle_encoding_repr_learning if "angle_encoding_repr_learning" in args else False,
        "predict_x0_sphere": args.predict_x0_sphere if "predict_x0_sphere" in args else False,
        "restrict_rot_update": args.restrict_rot_update if "restrict_rot_update" in args else False,
        "confidence_mode": True if args.model=="confidence" else False,

    }

    
    print('Using device', DEVICE)
    if DEVICE == 'cuda': # and args.n_gpus > 1:
        if args.use_distributed:
            local_rank = int(os.getenv('LOCAL_RANK', 0))
            print(f'Using local rank: {local_rank}, GPU: {torch.cuda.get_device_name(local_rank)}')
            model = model_cls(**params).to(local_rank)
            
            
            print(f'Model moved to GPU: {local_rank}')
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            print('model has been setup with DistributedDataParallel')
        else:
            model = model_cls(**params)
    else:
        model = model_cls(**params)
    return model


def build_model_from_args(args, mode: str = "train") -> torch.nn.Module:
    if args.model in ["score", "score_torchmd", 'confidence']:
        print(f"Building {args.model} from args:")
    
        model = build_score_model_from_args(args=args)
    
    else:
        raise NotImplementedError
    print("Finished building model")
    model = model.to(DEVICE)
    return model


def load_model_from_args(
        args, mode: str = "train", 
        return_model_args: bool = False) -> dict[str, torch.Tensor]:
    model_ckpt = _fetch_pretrained_model_ckpt(args=args)
    if model_ckpt is None:
        raise ValueError("Unable to fetch model ckpt.")
     
    print(f'Model ckpt found at {model_ckpt}')
    ckpt_info = _load_pretrained_checkpoint(
        model_ckpt_file=model_ckpt, return_model_args=return_model_args
    )
    if args.use_distributed:
    #if 'n_gpus' in args and args.n_gpus > 1:
        model_state_dict = {}
        for key, value in ckpt_info['model_state'].items():
            model_state_dict[f'module.{key}'] = value
    else:
        model_state_dict = ckpt_info['model_state']

    if 'model_args' in ckpt_info:
        model_args = ckpt_info['model_args']

        print(f"Found model args in restored checkpoint.", flush=True)
        print(f"Restored checkpoint args: {model_args}", flush=True)
        print(flush=True)

        if not hasattr(model_args, 'sphere_projection'):
            model_args.sphere_projection = True  
        if not hasattr(model_args, 'separate_rot_sphere_conv'):
            model_args.separate_rot_sphere_conv = False 
        if not hasattr(model_args, 'use_lookup_bl'):
            model_args.sphere_projection = False 
        if not hasattr(model_args, 'predict_sphere_direction'):
            model_args.predict_sphere_direction = False  
        if not hasattr(model_args, 'use_rdkit_as_initial_guess'):
            model_args.use_rdkit_as_initial_guess = False  
        if not hasattr(model_args, 'use_rdkit_confs'):
            model_args.use_rdkit_confs = False 
        if not hasattr(model_args, 'align_multidentate_last_step'):
            model_args.align_multidentate_last_step = False 
            
    

        
        if not hasattr(model_args, 'align_multidentate'):
            model_args.align_multidentate = False 
            
        if not hasattr(model_args, 'debug'):
            model_args.debug = True  #TODO: CHANGE THIS


        model = build_model_from_args(model_args, mode=mode)
        
        if 'n_gpus' in args and args.n_gpus > 1:
            model_state_dict = {}
            for key, value in ckpt_info['model_state'].items():
                model_state_dict[f'module.{key}'] = value
        else:
            model_state_dict = ckpt_info['model_state']
        if 'debug_performance' in args and args.debug_performance:
            add_timing_hooks(model)
        model.load_state_dict(model_state_dict)
        return model, model_args
    
    return model_state_dict


def _fetch_pretrained_model_ckpt(args):
    if 'restore_from' in args:
        print("Found restore_from in args", flush=True)
        return args.restore_from

    if 'model_dir' in args:
        print("Found model_dir in args", flush=True)
        if 'model_name' not in args:
            print("No model name found in args. Using best_ema_model.pt")
            model_name = "best_ema_model.pt"
        else:
            model_name = args.model_name
        model_ckpt = f"{args.model_dir}/{model_name}"
        return model_ckpt

    if 'log_dir' in args and 'exp_name' in args:
        print("Found log_dir and exp_name in args", flush=True)
        model_ckpt = f"{args.log_dir}/{args.exp_name}/{args.model_name}"
        return model_ckpt
    
    return None


def _load_pretrained_checkpoint(
    model_ckpt_file: str,
    return_model_args: bool = False
) -> dict[str, dict[str, torch.Tensor]]:
    model_dict = torch.load(model_ckpt_file, map_location='cpu')
    model_dir = os.path.dirname(model_ckpt_file)

    if 'model' in model_dict.keys():
        model_dict = model_dict['model']

    with open(f'{model_dir}/config_train.yml') as f:
        model_args = argparse.Namespace(**yaml.full_load(f))

    ckpt_info = {}

    check_key = list(model_dict.keys())[0]
    if 'module.' in check_key: # Potential legacy models on multiple gpus
        model_dict_single_gpu = {}
        for key, value in model_dict.items():
            new_key = ".".join(key.split(".")[1:])
            model_dict_single_gpu[new_key] = value
    
        ckpt_info['model_state'] = model_dict_single_gpu
    else:
        ckpt_info['model_state'] = model_dict

    
    if return_model_args:
        ckpt_info['model_args'] = model_args

    return ckpt_info
    
def timer(name):
    
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{name} took {end_time - start_time:.6f} seconds")

# Function to recursively apply timers to the forward methods of a model
def add_timing_hooks(model):
    for name, module in model.named_children():
        # Wrap the forward method of each submodule to include timing
        if isinstance(module, nn.Module):
            original_forward = module.forward

            def timed_forward(*args, name=name, forward_func=original_forward):
                with timer(f"Layer {name}"):
                    return forward_func(*args)
            
            module.forward = timed_forward

        # Recursively apply the hooks to submodules
        add_timing_hooks(module)
