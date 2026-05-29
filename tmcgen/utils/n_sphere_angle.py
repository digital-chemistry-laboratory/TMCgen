import torch
import numpy as np
import os
from tqdm import tqdm
from scipy.special import gamma,gegenbauer
import time
import torch.nn.functional as F
import multiprocessing
from functools import partial
multiprocessing.set_start_method('spawn', force=True)
from itertools import islice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import gc
if torch.cuda.is_available():
    memory_threshold = 0.8 * torch.cuda.get_device_properties(0).total_memory



def legendre_polynomial(n, x):
    
    if n == 0:
        return torch.ones_like(x, dtype=torch.float64)
    elif n == 1:
        return x.to(torch.float64)
    else:
        P_n_minus_1 = x.to(torch.float64)
        P_n_minus_2 = torch.ones_like(x, dtype=torch.float64)
        for k in range(2, n + 1):
            P_n = ((2 * k - 1) * x * P_n_minus_1 - (k - 1) * P_n_minus_2) / k
            P_n_minus_2 = P_n_minus_1
            P_n_minus_1 = P_n
        P_n_minus_1 = torch.nan_to_num(P_n_minus_1)  # Replace NaNs with zero
    assert not torch.isnan(P_n_minus_1).any(), f"NaN values found in Legendre polynomial computation for n={n}, x={x}"

    return P_n_minus_1    


def stable_exponential(l, n, t):
    """
    Compute e^{-\ell(\ell + n - 2)t} in a numerically stable way.
    """

    # Compute the exponent
    exponent = -l * (l + n - 2) * t

    # Rescale the exponent for numerical stability
    max_exponent = torch.max(exponent)
    scaled_exp = torch.exp(exponent - max_exponent)

    # Recover the correct scaling
    result = scaled_exp * torch.exp(max_exponent)
    
    return result


def hyperspherical_heat_kernel(angles, t, n, num_terms, device='cpu', scale_t=True, spacing=1):
  
    if n>3:
        raise NotImplementedError
    # Surface area of Sn-1
    A_Sn_minus_1 = 2 * np.pi**(n / 2) / gamma(n / 2) #12.566370614359174
    
    angles = torch.tensor(angles, dtype=torch.float64, device=device, requires_grad=True).view(-1, 1)
    t = torch.tensor(t, dtype=torch.float64, device=device).view(1, -1)
    
    G_ext = torch.zeros((angles.size(0), t.size(1)), dtype=torch.float64, device=device)

    # Precompute terms that do not depend on l
    l_values = torch.arange(num_terms, device=device, dtype=torch.float64).view(-1, 1, 1)

    l_values = l_values.to(dtype=torch.float64)
    t = t.to(dtype=torch.float64)
    
    if scale_t:
        term1 = torch.exp(-l_values * (l_values + n - 2) * t**2 / 2.0)
        term1_stable = stable_exponential(l_values,n,t)
        assert torch.isclose(term1,term1_stable).all
        exponent = -l_values * (l_values + n - 2) * t**2 / 2.0

        
    else:
        term1 = torch.exp(-l_values * (l_values + n - 2) * t)
    term2 = (2 * l_values + n - 2) / (n - 2)
    

    chunk_size = 50  # Adjust this value based on available memory
    num_chunks = (num_terms + chunk_size - 1) // chunk_size  # Calculate the number of chunks

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    

    
    # Calculate the series expansion for G_ext
    for l in tqdm(range(num_terms), desc="Computing heat kernel series"):
        term3 = legendre_polynomial(l, torch.cos(angles))
        
        assert not torch.isnan(term3).any(), "term3" 
        additional_term = term1[l] * term2[l] * term3

        G_ext += additional_term

            
        
    assert not torch.isnan(G_ext).any(), "isnan"
    G_ext = torch.nan_to_num(G_ext,nan=0.0, neginf=0.0)  # Replace NaNs with zero
    G_ext /= A_Sn_minus_1

    

    G_ext = torch.maximum(G_ext, torch.tensor(0.0, device=device))
    threshold = 1e-5
    G_ext = torch.where(G_ext < threshold, torch.tensor(0.0, device=G_ext.device), G_ext)

    assert not torch.isnan(G_ext).any()

    #calculate log
    log_G_ext = torch.log(G_ext)
    assert not torch.isnan(log_G_ext).any()
    log_G_ext = torch.nan_to_num(log_G_ext,nan=0.0, neginf=0.0)
    
    log_grad_stacked = np.gradient(log_G_ext.cpu().detach().numpy(), spacing, axis=0)
    log_grad_stacked = np.nan_to_num(log_grad_stacked,nan=0.0, neginf=0.0)  # Replace NaNs with zero
    del log_G_ext
    
    log_grad_stacked = np.where(log_grad_stacked > 100, np.array(0.0), log_grad_stacked)
    log_grad_stacked[0] = log_grad_stacked[2]
    log_grad_stacked[1] = log_grad_stacked[2]


    #multiply with uniform distribution on sphere
    G_ext *= torch.sin(angles)


    #normalize to become probability density
    integrals = torch.trapz(G_ext, angles, axis=0)
    G_ext = G_ext / (integrals+1e-7)    #normalize G_ext

    return G_ext.cpu().detach().numpy(), log_grad_stacked

   


def sample_point_on_ring(device='cpu'):
    """
    Uniformly sample a point on the unit sphere in `dim` dimensions.
    """
    if device == 'cuda':
        vec = torch.randn(2, device=device)
        vec /= torch.norm(vec)
        vec_3d = torch.cat((vec, torch.tensor([0.0], device=device)))
    else:
        vec = np.random.normal(size=2)
        vec /= np.linalg.norm(vec)
        vec_3d = np.append(vec, 0)
    return vec_3d

def orthogonal_transformation(u, device='cpu'):
    """
    Construct an orthogonal transformation that sends the standard basis vector e_N
    to the normalized vector u/||u||.
    """
    if device == 'cuda':
        e3 = torch.tensor([0.0, 0.0, 1.0], device=device)
    else:
        e3 = np.array([0, 0, 1])

    v3 = u
    if device == 'cuda':
        if torch.abs(v3[0]) < torch.abs(v3[1]):
            v1 = torch.tensor([1.0, 0.0, 0.0], device=device)
        else:
            v1 = torch.tensor([0.0, 1.0, 0.0], device=device)
        
        v1 = v1 - torch.dot(v1, v3) * v3
        v1 = v1 / torch.norm(v1)
        v2 = torch.cross(v3, v1)
        v2 = v2 / torch.norm(v2)

        A = torch.stack((v1, v2, v3), dim=-1)
        
        assert(torch.allclose(torch.matmul(A.T, A), torch.eye(3, device=device)))
    else:
        if np.abs(v3[0]) < np.abs(v3[1]):
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.array([0, 1, 0])
        
        v1 = v1 - np.dot(v1, v3) * v3
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(v3, v1)
        v2 = v2 / np.linalg.norm(v2)

        A = np.column_stack((v1, v2, v3))
        assert(np.allclose(np.dot(A.T, A), np.eye(A.shape[0]), atol=1e-6))
    
    return A

def sample_point_given_angle(u, theta, dim, device='cpu'):
    """
    Sample a point on the unit sphere in `dim` dimensions such that its dot product with `u` is `dot_product`.
    """
    if device == 'cuda':
        #theta = torch.acos(dot_product)
        T = orthogonal_transformation(u, device=device)

        point_on_ring = sample_point_on_ring(device=device)
        e3 = torch.tensor([0.0, 0.0, 1.0], device=device)

        input_vec = torch.cos(theta) * e3 + torch.sin(theta) * point_on_ring
        output = torch.matmul(T, input_vec)
    else:
        #theta = np.arccos(dot_product)
        T = orthogonal_transformation(u, device=device)

        point_on_ring = sample_point_on_ring(device=device)
        e3 = np.array([0, 0, 1])

        input_vec = np.cos(theta) * e3 + np.sin(theta) * point_on_ring
        output = np.dot(T, input_vec)
    
    return output



# Precompute the heat kernel values and save them to disk
def precompute_heat_kernel(n, num_terms=400, x_n=2000, t_n=5000, x_min=0.0, x_max=torch.pi, t_min=1e-5, t_max=1.0, device='cpu', scale_t=True, num_workers=None):
    print(f"Starting precomputation of hyperspherical heat kernel (n={n}, num_terms={num_terms}, angles={x_n}, times={t_n})...")
    if num_workers is None:
        
        num_workers = min(multiprocessing.cpu_count(), 16)

    angles = np.linspace(x_min, x_max, x_n)
    t_values = np.logspace(np.log10(t_min), np.log10(t_max), t_n)

    spacing = (x_max - x_min) / (x_n - 1)
    assert spacing == (angles[1]-angles[0])

    start_time = time.time()

    angle_chunks = np.array_split(angles, num_workers)

    heat_kernel_values, score_values = hyperspherical_heat_kernel(angles, t_values, n, num_terms, device=device, scale_t=scale_t, spacing = spacing)
    end_time = time.time()
    total_time = (end_time - start_time)
    print(f"Precomputation completed in {total_time:.2f} seconds.")
    print(f"Total calculations: {x_n * t_n}, average time per calculation: {total_time / (x_n * t_n):.6f} seconds.")

    np.save('.angles_sphere.npy', angles)
    np.save('.t_values_sphere_angles.npy', t_values)
    np.save('.heat_kernel_sphere_angles.npy', heat_kernel_values)
    np.save('.score_values_sphere_angles.npy', score_values)
    print("Precomputed heat kernel values saved successfully.")

# Load the precomputed heat kernel values
def load_precomputed_heat_kernel():
    angles = np.load('.angles_sphere.npy')
    t_values = np.load('.t_values_sphere_angles.npy')
    heat_kernel_values = np.load('.heat_kernel_sphere_angles.npy')
    score_values = np.load('.score_values_sphere_angles.npy')
    return angles, t_values, heat_kernel_values, score_values

def load_precomputed_score_norm():
    score_norm_values = np.load('.score_norm_values_sphere.npy')
    return score_norm_values
import torch
import numpy as np


def sample(vector, t, n=3, num_samples=1, return_only_angles=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(vector, torch.Tensor):
        t_idx = (torch.abs(torch.tensor(t_values_linspace, dtype=torch.float64) - t)).argmin()
        G_ext_t = G_ext[:, t_idx]

        G_ext_t = torch.tensor(G_ext_t, dtype=torch.float64, device=device)
        probabilities = G_ext_t / torch.sum(G_ext_t)
        sampled_indices = torch.multinomial(probabilities, num_samples, replacement=True)
        sampled_angles = torch.tensor(angles_linspace, dtype=torch.float64, device=device)[sampled_indices]
    else:
        t_idx = (np.abs(t_values_linspace - t)).argmin()
        G_ext_t = G_ext[:, t_idx]

        probabilities = G_ext_t / np.sum(G_ext_t)
        sampled_angles = np.random.choice(angles_linspace, size=num_samples, p=probabilities)
    if return_only_angles:
        return sampled_angles
    sampled_points = [sample_point_given_angle(vector, torch.tensor(value.item()), len(vector)) for value in sampled_angles]
    sampled_points = np.array(sampled_points)

    return sampled_points, sampled_angles




def score(angle, t, n=3):
    """
    Returns the score_values_sphere corresponding to the closest t value in t_values_linspace 
    and closest angle/dot product in x_dot_y_linspace.
    
    Args:
        dot_product (float): The dot product (cosine of the angle) between two vectors.
        t (float): The time value.
        n (int): The dimension of the hypersphere (e.g., n=3 for a 2-sphere).
        
    Returns:
        float: The score value from score_values_sphere.
    """
    
    # Find the closest index for the dot product
    if isinstance(angle, torch.Tensor):
        angle_idx = torch.abs(torch.tensor(angles_linspace, dtype=torch.float64) - angle).argmin()
        t_idx = torch.abs(torch.tensor(t_values_linspace, dtype=torch.float64) - t).argmin()
    else:
        angle_idx = np.abs(angles_linspace - angle).argmin()
        t_idx = np.abs(t_values_linspace - t).argmin()

    
    # Retrieve the score value
    score_value = score_values[angle_idx, t_idx]
    
    return score_value

def score_norm(t):
    # Ensure the input is a torch tensor of type float64
    if isinstance(t, (float, np.ndarray)):
        t = torch.tensor(t, dtype=torch.float64)
    elif isinstance(t, torch.Tensor):
        t = t.to(torch.float64)
    else:
        raise TypeError("Input must be a float, numpy array, or torch tensor.")

    # Convert t_values_linspace to a torch tensor if it isn't already
    t_values_linspace_tensor = torch.tensor(t_values_linspace, dtype=torch.float64)

    # Find the index of the closest value in t_values_linspace
    t_idx = torch.abs(t_values_linspace_tensor.unsqueeze(0) - t.unsqueeze(1)).argmin(dim=1)

    score_norm_values_tensor = torch.tensor(score_norm_values, dtype=torch.float64)

    # Retrieve the corresponding score norm(s)
    score_norm_result = score_norm_values_tensor[t_idx]

    return score_norm_result.float()



def precompute_score_norm():
    G_ext_expanded = G_ext[:, :, np.newaxis].squeeze(-1) 

    
    score_norms = np.sqrt(np.sum(score_values**2 * G_ext_expanded, axis=0) / np.sum(G_ext_expanded, axis=0))
    print(' score_norms',score_norms.shape,score_norms)
    np.save('.score_norm_values_sphere.npy', score_norms)
    

def sample_from_normal_plane(pos: torch.Tensor, num_samples: int, device: str = 'cpu') -> torch.Tensor:
    """
    Samples vectors from a 2D Gaussian distribution on the plane normal to the position vectors.
    
    Args:
        pos (torch.Tensor): The position vectors (shape: [N, 3]), where N is the number of vectors.
        num_samples (int): The number of samples to draw (typically len(player_keys)).
        device (str): The device to run the computations on (e.g., 'cpu' or 'cuda').
    
    Returns:
        torch.Tensor: The sampled vectors from the plane normal to the input position vectors.
    """
    pos_normalized = F.normalize(pos, dim=-1)  # shape: (num_samples, 3)

    # Gram-Schmidt process
    random_vector = torch.randn_like(pos_normalized)
    # Compute first orthonormal vector by taking the cross product of pos_normalized and random_vector
    v1 = torch.cross(pos_normalized, random_vector, dim=-1)
    v1 = F.normalize(v1, dim=-1)
    v2 = torch.cross(pos_normalized, v1, dim=-1)
    v2 = F.normalize(v2, dim=-1)

    # Sample from a 2D Gaussian distribution
    z1 = torch.normal(mean=0, std=1, size=(num_samples, 1), device=device)
    z2 = torch.normal(mean=0, std=1, size=(num_samples, 1), device=device)

    # Construct the sphere_z by combining the components
    sphere_z = z1 * v1 + z2 * v2
    
    return sphere_z



# Example usage
t_n = 1000 
num_samples_score_norm = 2000 # 2000
t_min = 0.01 #0.035
t_max = 2 #10.0



if os.path.exists('.heat_kernel_sphere_angles.npy'):
    # Load the precomputed values if they exist
    angles_linspace, t_values_linspace, G_ext,score_values = load_precomputed_heat_kernel()
    score_norm_values = load_precomputed_score_norm()
   
else:
    scale_t = True
    
    # real calc
    precompute_heat_kernel(n=3, num_terms=500, x_n=2000, t_n=t_n, x_min=0, x_max=torch.pi, t_min=t_min, t_max=t_max, device='cuda' if torch.cuda.is_available() else 'cpu', scale_t=scale_t)
    angles_linspace, t_values_linspace, G_ext,score_values = load_precomputed_heat_kernel()
    precompute_score_norm()
    score_norm_values = load_precomputed_score_norm()




