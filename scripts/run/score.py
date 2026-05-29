import os
import yaml
import torch
import wandb
import numpy as np
import copy
import datetime
import math
from tmcgen.training.epoch_fns import train_epoch, validation_epoch, inference_epoch
from tmcgen.training.losses import loss_fn_from_args
from tmcgen.utils.setup import launch_experiment
from tmcgen.common.constants import DEVICE
import torch.distributed as dist
import time
import socket
import traceback
import builtins
import argparse


original_print = builtins.print
_PRINT_FLUSH_ONLY = False

def custom_print(*args, **kwargs):
    if (not _PRINT_FLUSH_ONLY) or kwargs.get('flush', False):
        stack = traceback.extract_stack()
        filename, lineno, _, _ = stack[-2]
        filename = os.path.basename(filename)
        original_print(f'{filename}:{lineno}:', *args, **kwargs)

builtins.print = custom_print

def train(args, train_loader, val_loader, model, optimizer, scheduler, ema_weights=None, log_dir=None):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    logs = {}

    loss_fn = loss_fn_from_args(args)
    print("loss_fn", loss_fn)
    # Log initial validation and inference (if applicable)
    val_losses, inference_metrics = None, None

    if not args.no_val_before_training:
        val_losses = validation_epoch(
            model=model, loader=val_loader, loss_fn=loss_fn,
            make_outputs=False, model_name=args.model, n_gpus=args.n_gpus,
            t_to_sigma_fn=train_loader.dataset.transform.t_to_sigma if args.model!="confidence" else None,
        )
        print(f"Initial Validation Metrics: {val_losses}")

    if args.inference_every is not None and args.model!="confidence":
        inference_metrics = inference_epoch(
            model=model, dataset_orig=val_loader.dataset, args=args
        )
        print(f"Initial Inference Metrics: {inference_metrics}")
        


    # Initial WandB logging
    if args.wandb_mode == "online":
        log_init = {}
        if val_losses:
            log_init.update({'val_' + k: v for k, v in val_losses.items()})
        if inference_metrics:
            log_init.update({'val_inference_' + k: v for k, v in inference_metrics.items()})
        log_init['current_lr'] = optimizer.param_groups[0]['lr']
        log_init["step"] = 0
        wandb.log(log_init)

    print("==============================================")
    print(f"Starting training for {args.n_epochs} epochs.")
    print("==============================================")

    for epoch in range(args.n_epochs):
        if args.n_gpus > 1:
            train_loader.sampler.set_epoch(epoch)
            
        log_dict = {}
        # Train for one epoch
        start_time = time.time()
        train_losses = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
            ema_weights=ema_weights, grad_clip_value=args.grad_clip_value,
            model_name=args.model, step_every=args.step_every, n_gpus=args.n_gpus,
            t_to_sigma_fn=train_loader.dataset.transform.t_to_sigma if args.model!="confidence" else None,
        )
        train_time = time.time() - start_time
        print(f"Epoch {epoch + 1}: Train Loss: {train_losses}, Time: {train_time:.2f}s", flush=True)

        # Validation (if applicable)
        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            val_losses = validation_epoch(
                model=model, loader=val_loader, loss_fn=loss_fn,
                make_outputs=False, model_name=args.model, n_gpus=args.n_gpus,
                t_to_sigma_fn=train_loader.dataset.transform.t_to_sigma if args.model!="confidence" else None,
            )
            print(f"Epoch {epoch + 1}: Validation Metrics: {val_losses}")

        # Inference (if applicable)
        if args.inference_every and (epoch + 1) % args.inference_every == 0:
            if args.model!='confidence':
                inference_metrics = inference_epoch(
                    model=model, dataset_orig=val_loader.dataset, args=args
                )
                print(f"Epoch {epoch + 1}: Inference Metrics: {inference_metrics}")

        # Update best validation loss and inference metric
        if val_losses and val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch + 1
            if log_dir:
                save_model(model, log_dir, 'best_model.pt', ema_weights)
        
        #print('inference_metrics', inference_metrics)

        if args.wandb_mode == "online":
            # Logging metrics and losses
            log_dict.update({'train_' + k: v for k, v in train_losses.items()})
            try: 
                log_dict.update({'val_' + k: v for k, v in val_losses.items()})
            except:
                print('not updated')
            if args.inference_every is not None and args.model!='confidence': 
                if args.inference_every > 0 and (epoch + 1) % args.inference_every == 0:
                    
                    log_dict.update({'val_inference_' + k: v for k, v in inference_metrics.items()})
            log_dict['current_lr'] = optimizer.param_groups[0]['lr']
            log_dict["step"] = epoch + 1
            wandb.log(log_dict)

        '''
        if inference_metrics and (
            (args.inference_goal == 'min' and inference_metrics[args.inference_metric] < best_val_inference_value) or
            (args.inference_goal == 'max' and inference_metrics[args.inference_metric] > best_val_inference_value)
        ):
            best_val_inference_value = inference_metrics[args.inference_metric]
            best_val_inference_epoch = epoch + 1
            if log_dir:
                save_model(model, log_dir, 'best_inference_model.pt', ema_weights)
        '''
        # Scheduler step
        if scheduler:
            metric = logs.get(args.lr_sched_metric, logs.get("val_loss", best_val_loss))
            scheduler.step(metric)

        # Save the model state
        if log_dir:
            save_model(model, log_dir, 'last_model.pt', ema_weights, epoch)

        if dist.is_initialized():
            dist.barrier()

        print(f"Epoch {epoch + 1} complete. Time taken: {train_time:.2f}s")

    print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    if args.inference_every:
        print(f"Best Inference Metric: {best_val_inference_value:.4f} (Epoch {best_val_inference_epoch})")


def save_model(model, log_dir, filename, ema_weights=None, epoch=None):
    model_state = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
    save_dict = {'model': model_state}
    if ema_weights:
        save_dict['ema_weights'] = ema_weights.state_dict()
    if epoch is not None:
        save_dict['epoch'] = epoch
    torch.save(save_dict, os.path.join(log_dir, filename))
    print(f"Model saved to {os.path.join(log_dir, filename)}")

import sys
def main():
    global _PRINT_FLUSH_ONLY
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--print-all', dest='print_all', action='store_true',
                        help='Print everything (default: only flush=True prints).')
    cli_args, remaining = parser.parse_known_args()
    _PRINT_FLUSH_ONLY = not cli_args.print_all

    sys.argv = [sys.argv[0]] + remaining

    print("Number of GPUs:", torch.cuda.device_count())
    #if torch.cuda.device_count() > 1:
    if "LOCAL_RANK" in os.environ:

        dist.init_process_group(
            backend='nccl', init_method='env://',
            world_size=torch.cuda.device_count(),
            rank=int(os.getenv('LOCAL_RANK', 0)),
            timeout=datetime.timedelta(seconds=60)
        )
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        print(f"Initialized process group. Local rank: {local_rank}")

    trained_model = launch_experiment(train_fn=train, mode='score_matching')
    return trained_model


    

if __name__ == "__main__":
    main()
