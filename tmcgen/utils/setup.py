import argparse
from argparse import FileType
import os
from typing import Callable

import torch
import wandb
import yaml

from tmcgen.common.constants import (
    WANDB_ENTITY, CLUSTER_EXP_DIR, IS_CLUSTER, EXP_DIR
)
from tmcgen.data import build_data_loader
from tmcgen.models import build_model_from_args, load_model_from_args
from tmcgen.training.updaters import get_ema, get_optimizer, get_scheduler


# Type aliases
Args = argparse.Namespace
ArgParser = argparse.ArgumentParser
Model = torch.nn.Module


def construct_log_dir(args: Args) -> str:
    if args.log_dir is not None:
        experiment_str = f"train_ds={args.train_dataset}"
        if args.val_dataset is not None:
            experiment_str += f"-val_ds={args.val_dataset}"        
        experiment_str += f"-model={args.model}-agent={args.agent_type}-feat={args.featurizer}"

        base_dir = f"{args.log_dir}/{experiment_str}"
        os.makedirs(base_dir, exist_ok=True)
        log_dir = f"{base_dir}/{args.run_name}"
        os.makedirs(log_dir, exist_ok=True)

        print(f"Logging experiments at directory: {log_dir}", flush=True)
        print(f"Experiment Name: {experiment_str}-{args.run_name}", flush=True)
        return log_dir
    return None


def wandb_setup(args: Args):
    DIR = CLUSTER_EXP_DIR if IS_CLUSTER else EXP_DIR
    print(f"Supplied experiment directory: {DIR}", flush=True)

    if not os.path.exists(DIR):
        os.makedirs(DIR)

    run_id = wandb.util.generate_id()

    if args.run_name is None:
        if args.group_name is not None:
            args.run_name = args.group_name + f"-{run_id}"
        else:
            args.run_name = run_id
    else:
        args.run_name = args.run_name + f"-{run_id}"
    
    print("Setting up wandb...", flush=True)
    wandb.init(
        id=run_id,
        project=args.project_name if 'project_name' in args else 'tmcgen',
        entity=args.wandb_entity,
        group=args.group_name,
        name=args.run_name,
        config=vars(args),
        dir=DIR,
        mode=args.wandb_mode,
        notes=args.notes,
        job_type=args.job_type
    )
    sweep_config = wandb.config  # Get the configuration passed by the W&B agent
    # Override argparse arguments with sweep parameters
    for key, value in sweep_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args = override_args_with_wandb_config(args, sweep_config)
    #print(f"Final args: {args}")
    print(f"Final configuration after W&B setup: {args}", flush=True)
    return args

def override_args_with_wandb_config(args, sweep_config):
    """
    Override argparse arguments with W&B sweep parameters, including handling nested dictionaries.
    """
    print("sweep_config", sweep_config)

    # Update args with flat parameters
    for key, value in sweep_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            # Handle nested dictionary overrides
            if key.startswith("temp_psi_"):
                sub_key = key.replace("temp_psi_", "")
                args.temp_psi[sub_key] = value
            elif key.startswith("temp_sampling_"):
                sub_key = key.replace("temp_sampling_", "")
                args.temp_sampling[sub_key] = value
            elif key.startswith("temp_sigma_data_"):
                sub_key = key.replace("temp_sigma_data_", "")
                args.temp_sigma_data[sub_key] = value
            elif key.startswith("use_temp_effects_"):
                sub_key = key.replace("use_temp_effects_", "")
                args.temp_sigma_data[sub_key] = value

    return args


def setup_parser() -> ArgParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Directory to save data to.")
    parser.add_argument("--log_dir", default=None, 
                        help="Directory to save local logs to.")
    parser.add_argument("--config", type=FileType(mode='r'), 
                        help="Config file to load args from. args will be overwritten")
    parser.add_argument("--train_complex_dir", type=str, default='complexes',
                        help="Directory where train complexes are located (e.g complexes)")
    parser.add_argument("--val_complex_dir", type=str, default=None,
                        help="Directory where val complexes are located")
    parser.add_argument("--train_complex_list_file", type=str, default='complexes.txt',
                        help="List of complexes train")
    parser.add_argument("--val_complex_list_file", type=str, default=None,
                        help="List of complexes train")
    
    parser.add_argument("--restore_from", default=None, 
                        help="Where to restore pretrained model from.")
    parser.add_argument("--use_distributed", action='store_true')
    parser.add_argument("--n_gpus", default=1, type=int)
    
    # wandb
    parser.add_argument("--project_name", default="tmcgen")
    parser.add_argument("--wandb_entity", default=WANDB_ENTITY)
    parser.add_argument("--group_name", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--wandb_mode", default="disabled")
    parser.add_argument("--job_type", default=None)
    parser.add_argument("--notes", default=None)

    # Data
    parser.add_argument("--train_dataset", default="db5", type=str)
    parser.add_argument("--val_dataset", default=None, type=str)
    parser.add_argument("--resolution", default="c_alpha", type=str)
    parser.add_argument("--agent_type", default="protein", type=str)
    parser.add_argument("--center_complex", action='store_true')
    parser.add_argument("--featurizer", default=None, 
                        choices=["base"])
    parser.add_argument("--train_size_sorted", action='store_true')
    parser.add_argument("--val_size_sorted", action='store_true')
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--train_bs", default=8, type=int)
    parser.add_argument("--val_bs", default=2, type=int)
    parser.add_argument("--max_radius", type=float, default=10.0)
    parser.add_argument("--max_neighbors", type=int, default=32)
    parser.add_argument("--cross_max_radius", type=float, default=40.0)
    parser.add_argument("--cross_max_neighbors", type=float, default=50)

    # Model
    parser.add_argument("--node_fdim", default=145, type=int)
    parser.add_argument("--edge_fdim", default=0, type=int)
    parser.add_argument("--n_s", type=int, default=20)
    parser.add_argument("--n_v", type=int, default=10)
    parser.add_argument("--activation", default='relu', type=str)
    parser.add_argument("--dropout_p", default=0.1, type=float)
    parser.add_argument("--n_conv_layers", default=1, type=int)
    parser.add_argument("--sh_lmax", type=int, default=2)
    parser.add_argument("--distance_emb_dim", type=int, default=32)
    parser.add_argument("--sphere_diffusion", action='store_true', help='Use sphere diffusion for translation')
    parser.add_argument('--predict_sphere_direction', action='store_true', help='Predict vector in target direction, not rotation vector')
    parser.add_argument("--no_rot_first_lig", action='store_true', help='Keep one ligand stationary for rotation')
    
    parser.add_argument("--no_rot_all_ligands", action='store_true', help='Keep all ligands stationary for rotation')
 
    parser.add_argument("--no_sphere_first_lig", action='store_true', help='Keep one ligand stationary for sphere update')
    parser.add_argument("--rot_center_anchor", action='store_true', help='Rotate around the anchor atom instead of the center')
    parser.add_argument("--keep_core_rigid", action='store_true', help="Keep core atoms stationary during torsional update")
    parser.add_argument("--sphere_projection", action='store_true', help="In model project onto tangent plane")
    parser.add_argument("--joint_rot_sphere_update", action='store_true', help="Rotate the ligands based to the sphere update")
    parser.add_argument("--anchor_graph_sphere",  action='store_true', help="Use output layer with anchors for sphere")
    parser.add_argument("--angle_encoding_repr_learning",   action='store_true', help="Use angle encoding")
    parser.add_argument("--predict_x0_sphere", action='store_true', help='x0 prediction for sphere updates')
    parser.add_argument("--restrict_rot_update", action='store_true', help='rotation only as torsion')


    parser.add_argument("--partially_rigid", action='store_true')

    parser.add_argument("--save_for_confidence_interval",type=int, default=20)
    parser.add_argument("--save_xyz",action='store_true')

    
    parser.add_argument("--save_for_confidence", action='store_true', help='save inference output for confidence model training')
    
    parser.add_argument("--scale_sphere", action='store_true', help="Scale Sphere continuously")
    parser.add_argument("--max_sphere_add_radius", default=0.0, type=float)


    # Training
    parser.add_argument("--n_epochs", default=10, type=int)

    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_performance", action='store_true')

    # Optimizer & Scheduler
    parser.add_argument("--optim_name", default='adam', type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--grad_clip_value", default=10.0, type=float, 
                        help="Gradient clipping max value")
    parser.add_argument("--scheduler", type=str, default="plateau")
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--scheduler_mode", type=str, default='min')  
    parser.add_argument("--ema_decay_rate", type=float, default=0.999)
    #parser.add_argument("--distributed", action="store_true", help="Use DistributedDataParallel for training")
    # Logging
    parser.add_argument("--log_every", default=1000, type=int, 
                        help="Logging frequency")
    parser.add_argument("--eval_every", default=10000, type=int, 
                        help="Evaluation frequency during training.")
    parser.add_argument("--step_every", default=1, type=int,
                        help="How quickly to accumulate gradient steps")
    parser.add_argument("--calc_permute_rmsd", action='store_true')

    # Val scheduler based stepping
    parser.add_argument("--lr_sched_metric", default="val_loss", type=str)
    parser.add_argument("--lr_sched_metric_goal", default="min", type=str)

    return parser



def parse_score_model_args() -> ArgParser:
    parser = setup_parser()

    parser.add_argument("--transform", type=str, default="ma_noise")
    parser.add_argument("--pert_strategy", type=str, default="all-but-one")
    parser.add_argument("--same_t_for_agent", action='store_true')
    parser.add_argument("--dynamic_max_cross", action='store_true', default=True)
    parser.add_argument("--timepoints_per_complex", type=int, default=1)
    parser.add_argument("--no_val_before_training", action='store_true')

    

    # Model
    parser.add_argument("--architecture", type=str, default="e3nn")
    parser.add_argument("--random_init_missing_keys", action='store_true')
    parser.add_argument("--model", type=str, default="score")
    parser.add_argument("--cross_cutoff_threshold", type=float, default=40.0)
    parser.add_argument("--center_max_radius", type=float, default=30.0)
    parser.add_argument("--cross_dist_emb_dim", type=int, default=32)
    parser.add_argument("--center_dist_emb_dim", type=int, default=32)
    parser.add_argument("--time_embed_type", type=str, default="sinusoidal")
    parser.add_argument("--time_embed_scale", type=int, default=10000)
    parser.add_argument("--sigma_emb_dim", type=int, default=32)

    parser.add_argument("--separate_intra_inter_agent_updates", type=bool, default=True, help="separate conv layers for intra and inter agent updates")
    
    parser.add_argument("--batch_norm", action='store_true')

    parser.add_argument("--tr_sigma_min", type=float, default=0.1)
    parser.add_argument("--tr_sigma_max", type=float, default=19.0)
    parser.add_argument("--rot_sigma_min", type=float, default=0.03)
    parser.add_argument("--rot_sigma_max", type=float, default=1.55)
    parser.add_argument('--tor_sigma_min', type=float, default=0.0314, help='Minimum sigma for torsional component')
    parser.add_argument('--tor_sigma_max', type=float, default=3.14, help='Maximum sigma for torsional component')
    parser.add_argument("--no_torsion", default=True, action='store_true')


    parser.add_argument("--use_rdkit_confs", action='store_true')
    
    parser.add_argument("--use_rdkit_as_initial_guess", action='store_true')

    #TOrchMD
    parser.add_argument("--hidden_channels", default=128, type=int, 
                    help="Number of hidden channels in the model.")
    parser.add_argument("--num_layers", default=6, type=int, 
                        help="Number of layers in the model.")
    parser.add_argument("--num_rbf", default=16, type=int, 
                        help="Number of radial basis functions.")
    parser.add_argument("--cutoff_lower", default=0.0, type=float, 
                        help="Lower cutoff for interactions.")
    parser.add_argument("--cutoff_upper", default=5.0, type=float, 
                        help="Upper cutoff for interactions.")
    parser.add_argument("--trainable_rbf", action='store_true', 
                        help="Enable trainable radial basis functions.")


    parser.add_argument("--scale_by_sigma", action='store_true', default=True)

    parser.add_argument("--w_tr", default=1.0)
    parser.add_argument("--w_tor", default=1.0)
    parser.add_argument("--w_rot", default=1.0)
    parser.add_argument("--w_dist", default=1.0)

    # temperature
    parser.add_argument("--temp_psi", type=str, default='', help="Temperature psi values as a dictionary (JSON string).")
    parser.add_argument("--temp_sampling", type=str, default='', help="Temperature sampling values as a dictionary (JSON string).")
    parser.add_argument("--temp_sigma_data", type=str, default='', help="Temperature sigma data as a dictionary (JSON string).")
    parser.add_argument("--use_temp_effects", action='store_true', help="Flag to enable temperature effects.")

    args = parser.parse_args()
    return args


def update_args_from_config(args: Args) -> Args:
    print("args.config",args.config)
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        args_dict = args.__dict__

        for key, value in config_dict.items():
            #print("key, value", key, value, flush=True)

            if isinstance(value, list):
                if key not in args_dict or args_dict[key] is None:
                    args_dict[key] = []  # ensure it's a list before appending

                for v in value:
                    args_dict[key].append(v)
            else:
                args_dict[key] = value
        
    return args


def parse_train_args(mode: str = 'score') -> Args:

    
    parse_fn = parse_score_model_args
    args = parse_fn()
    if args.log_dir is None:
        args.log_dir = CLUSTER_EXP_DIR if IS_CLUSTER else EXP_DIR
    args = update_args_from_config(args)
    return args


def parse_game_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir")
    parser.add_argument("--config", type=FileType(mode='r'), 
                        help="Config file to load args from. args will be overwritten")
    parser.add_argument("--out_dir", type=str, default="game_outputs/")

    parser.add_argument("--complex_dir", type=str, default='complexes')
    parser.add_argument("--complex_list_file", type=str, default='complexes.txt',
                        help="List of complexes to play game on")
    
   
    # wandb
    parser.add_argument("--wandb_entity", default=WANDB_ENTITY)
    parser.add_argument("--project_name", default="dockgame-inf")
    parser.add_argument("--group_name", default="gameplay")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--wandb_mode", default="disabled")
    parser.add_argument("--job_type", default=None)
    parser.add_argument("--notes", default=None)


    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--use_lookup_bl", action='store_true')
    parser.add_argument("--align_multidentate", action='store_true')
    parser.add_argument("--align_multidentate_last_step", action='store_true')

    
    # Common args for game
    parser.add_argument("--dataset", type=str, default="db5")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_rounds", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_vis", action='store_true')
    parser.add_argument("--save_trajectory", action='store_true')
    parser.add_argument("--agent_type", default="protein", 
                        choices=["protein", "chain"])
    parser.add_argument("--score_fn_name", default="dock_low_res", 
                        choices=["dock_low_res", "dock_high_res"])
    parser.add_argument("--n_equilibria", default=10, type=int)
    parser.add_argument("--strategy", default="langevin", 
                        choices=["langevin", "reward_grad"])

    # Langevin specific args
    parser.add_argument("--use_ode", action='store_true')


    

   
    

    args = parser.parse_args()
    args = update_args_from_config(args)

    args = override_args_with_wandb(args, sweep_config)
    return args


def count_parameters(model: Model, log_to_wandb: bool = False) -> int:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if log_to_wandb:
        wandb.log({'n_params': n_params})
    return n_params


def read_complexes_from_txt(filename: str) -> list[str]:
    with open(filename, "r") as f:
        print(f"Loading complex ids from {filename}", flush=True)
        pdb_ids = f.readlines()
        pdb_ids = [pdb_id.strip() for pdb_id in pdb_ids]
    return pdb_ids


def launch_experiment(train_fn: Callable, mode: str = 'score_matching') -> Model:
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4)
    #torch.manual_seed(1331323)

    # Load args from command line and replace values with those from config
    print(flush=True)
    args = parse_train_args(mode=mode)

    # Wandb setup
    wandb_setup(args)
    args.wandb_dir = os.path.dirname(wandb.run.dir)

    print(f"Args supplied for the experiment", flush=True)
    print(f"{args}", flush=True)


    log_dir = construct_log_dir(args=args)
    print(f"Run Name: {args.run_name}", flush=True)


    config_file = os.path.join(log_dir, "config_train.yml")
    yaml_dump = yaml.dump(args.__dict__)
    with open(config_file, "w") as f:
        f.write(yaml_dump)

    print(f"Saved model config to {config_file}", flush=True)

    print(f"Building data loaders...", flush=True)
    train_loader = build_data_loader(args=args, mode="train")
    val_loader = build_data_loader(args=args, mode="val")
    print(f"Built data loaders!", flush=True)


    print(f"Building model...", flush=True)
    model = build_model_from_args(args)
    print(f"Built model!", flush=True)
  

    if 'restore_from' in args and args.restore_from is not None:
        print(f"Loading pretrained model from {args.restore_from}", flush=True)

        model_dict = load_model_from_args(args=args, return_model_args=False)
        
        
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

        # Log warnings for missing or unexpected keys
        if missing_keys:
            if hasattr(args, "random_init_missing_keys") and args.random_init_missing_keys:
                print(f"Warning: Missing keys detected in state_dict: {missing_keys}. Initializing them randomly.", flush=True)
                with torch.no_grad():
                    for key in missing_keys:
                        param = model.state_dict()[key]
                        param.copy_(torch.randn_like(param))  # Random initialization
            else:
                raise RuntimeError(f"Missing keys detected in state_dict: {missing_keys}")

        if unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
       
    n_params = count_parameters(model=model, log_to_wandb=False and args.online)
    print(f"Model with {n_params / (10**6)}M parameters", flush=True)
    print(flush=True)



    # Optimizers
    optimizer = get_optimizer(model=model, optim_name=args.optim_name,
                              lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer=optimizer, scheduler_name=args.scheduler,
                              scheduler_mode=args.scheduler_mode, factor=0.7,
                              patience=args.scheduler_patience, min_lr=args.lr / 100,                              
                              first_cycle_steps=args.first_cycle_steps if args.scheduler == "warmup_cosine" else 0,
                              cycle_mult=args.cycle_mult if args.scheduler == "warmup_cosine" else None,
                              max_lr=args.max_lr if args.scheduler == "warmup_cosine" else 0.0,
                              warmup_steps=args.warmup_steps if args.scheduler == "warmup_cosine" else 0,
                              gamma=args.gamma if args.scheduler == "warmup_cosine" else 0.0)
    ema = get_ema(model=model, decay_rate=args.ema_decay_rate)
    print("train_loader",train_loader, flush=True)
    print("train_loader.dataset", train_loader.dataset, flush=True)

    train_fn(args=args, train_loader=train_loader, 
          val_loader=val_loader, model=model, optimizer=optimizer,
          scheduler=scheduler, ema_weights=ema, log_dir=log_dir)

    return model

