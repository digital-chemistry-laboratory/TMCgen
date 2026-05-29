import numpy as np
import torch

# Helper functions

def calculate_sigma_data(sigma_min, sigma_max, temp_sigma_data):
    """
    Calculate adjusted sigma value based on temperature parameters.
    """
    return np.exp(temp_sigma_data * np.log(sigma_max) + (1 - temp_sigma_data) * np.log(sigma_min))

def compute_lambda(sigma, sigma_data, temp_sampling):
    """
    Compute lambda parameter for the temperature adjustment.
    """
    return (sigma_data + sigma) / (sigma_data + sigma / temp_sampling)

def calculate_deterministic_scale(sigma_data, sigma, temp_sampling, temp_psi):
    """
    Calculate the deterministic scale factor based on lambda and temperature psi.
    """
    lambda_value = compute_lambda(sigma, sigma_data, temp_sampling)
    return lambda_value + temp_sampling * temp_psi / 2

def calculate_temp_scales(score_updates, temp_sampling, temp_psi, use_temp_effects, tr_sigma_data,
                          rot_sigma_data, tor_sigma_data, sphere_sigma_data, bl_sigma_data):
    """
    Calculate scaling factors for each component (translation, rotation, torsion, sphere, bond length).
    Temperature effects can be toggled individually for each component.
    """

    # Initialize scales
    scales = {
        "tr_scale_deterministic": 1.0, "rot_scale_deterministic": 1.0, "tor_scale_deterministic": 1.0,
        "sphere_scale_deterministic": 1.0, "bl_scale_deterministic": 1.0,
        "tr_scale_stochastic": 1.0, "rot_scale_stochastic": 1.0, "tor_scale_stochastic": 1.0,
        "sphere_scale_stochastic": 1.0, "bl_scale_stochastic": 1.0,
    }
    if not use_temp_effects:
        return scales
        
    # Translation (tr)
    if use_temp_effects.get('tr', False):
        scales["tr_scale_deterministic"] = calculate_deterministic_scale(tr_sigma_data, score_updates['tr'], temp_sampling['tr'], temp_psi['tr'])
        scales["tr_scale_stochastic"] = 1 + temp_psi['tr']

    # Rotation (rot)
    if use_temp_effects.get('rot', False):
        scales["rot_scale_deterministic"] = calculate_deterministic_scale(rot_sigma_data, score_updates['rot'], temp_sampling['rot'], temp_psi['rot'])
        scales["rot_scale_stochastic"] = 1 + temp_psi['rot']

    # Torsion (tor)
    if use_temp_effects.get('tor', False):
        scales["tor_scale_deterministic"] = calculate_deterministic_scale(tor_sigma_data, score_updates['tor'], temp_sampling['tor'], temp_psi['tor'])
        scales["tor_scale_stochastic"] = 1 + temp_psi['tor']

    # Sphere (sphere)
    if use_temp_effects.get('sphere', False):
        scales["sphere_scale_deterministic"] = calculate_deterministic_scale(sphere_sigma_data, score_updates['sphere'], temp_sampling['sphere'], temp_psi['sphere'])
        scales["sphere_scale_stochastic"] = 1 + temp_psi['sphere']

    # Bond length (bl)
    if use_temp_effects.get('bl', False):
        scales["bl_scale_deterministic"] = calculate_deterministic_scale(bl_sigma_data, score_updates['bl'], temp_sampling['bl'], temp_psi['bl'])
        scales["bl_scale_stochastic"] = 1 + temp_psi['bl']

    return scales
