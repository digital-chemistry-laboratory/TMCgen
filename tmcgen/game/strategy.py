import argparse
from functools import partial
from typing import Callable
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from tmcgen.data import transforms 
from tmcgen.utils.diffusion import t_to_sigma, get_t_schedule
from tmcgen.data.transforms import construct_score_transform
from tmcgen.game.agents import get_agent_cls, Agent, ActionDict
from scipy.spatial.transform import Rotation
from tmcgen.utils.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from tmcgen.utils.temperature import calculate_temp_scales, calculate_sigma_data
from tmcgen.common.constants import DEVICE
import tmcgen.utils.n_sphere_angle as n_sphere_angle
import copy
Tensor = torch.Tensor
AgentDict = dict[str, Agent]
Metrics = dict[str, float]
from tmcgen.analysis.metrics import (
    compute_complex_rmsd_torch, rmsd, permute_rmsd, compute_angles
)

# Orthonormal basis of SO(3) with shape [3, 3, 3]
basis = torch.tensor([
[[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]],
[[0.,0.,1.],[0.,0.,0.],[-1.,0.,0.]],
[[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]]])
# hat map from vector space Rˆ3 to Lie algebra so(3)
def hat(v): return torch.einsum("...i,ijk->...jk", v, basis.to(v.device))
# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(A): return torch.linalg.matrix_exp(A)
# Exponential map from tangent space at R0 to SO(3)
def expmap(R0, tangent):
    skew_sym = torch.einsum("...ij,...ik->...jk", R0, tangent)
    return torch.einsum("...ij,...jk->...ik", R0, exp(skew_sym))


class BaseStrategy:

    def __init__(
        self, n_rounds: int, 
        model: torch.nn.Module,
        transform: BaseTransform,
        device: str = 'cpu',
        **kwargs
    ):
        self.model = model
        self.transform = transform
        self.n_rounds = n_rounds
        self.device = device
        self.model.eval()

    def setup_game(self):
        
        pass
    
    def compute_actions(self, agent_dict, agent_keys, player_keys):
        raise NotImplementedError("Subclasses must implement for themselves")
    
    def gather_actions(self, updates, agent_keys, player_keys):
        raise NotImplementedError("Subclasses must implement for themselves")
    
    def apply_actions(self, agent_dict: AgentDict, action_dict: ActionDict, round_id) -> AgentDict:
        for key in action_dict:
            agent = agent_dict[key]
            agent.update_pose(action_dict[key],key,round_id )
        
        return agent_dict
    
    def play_round(self):
        raise NotImplementedError("Subclasses must implement for themselves")
    
    def to(self, device: str):
        self.model = self.model.to(device)

    def check_for_termination(self, running_logs, verbose: bool = False):
        return False

class ScoreMatching(BaseStrategy):

    def __init__(self,
        t_to_sigma: Callable,
        tr_sigmas: tuple[float],
        rot_sigmas: tuple[float],
        tor_sigmas: tuple[float],
        sphere_sigmas: tuple[float],
        bl_sigmas: tuple[float],
        t_schedule: list[float],
        ode: bool = False,
        no_final_noise: bool = True,
        no_torsion: bool = True,
        sphere_diffusion: bool = False,
        predict_sphere_direction: bool = False,
        no_rot_first_lig: bool = False,
        no_rot_all_ligands: bool = False,
        no_sphere_first_lig: bool = False,
        rot_center_anchor: bool = False,
        keep_core_rigid: bool = False,
        debug: bool = False,
        temp_psi: dict = None,
        temp_sampling: dict = None,
        temp_sigma_data: dict = None,
        use_temp_effects: bool = False,
        rotation_prior_guess: bool= False,
        predict_x0_sphere: bool=False,
        restrict_rot_update: bool=False,
        joint_rot_sphere_update: bool=False,
        partially_rigid: bool = False,
        use_lookup_bl: bool = False,
        align_multidentate: bool = False,
        align_multidentate_last_step: bool = False,
        use_rdkit_as_initial_guess: bool = False,
        use_rdkit_confs: bool = False,

        **kwargs,
    ):
        super().__init__(**kwargs)
        self.t_to_sigma_fn = t_to_sigma
        self.t_schedule = t_schedule
        self.ode = ode
        self.tr_sigma_min, self.tr_sigma_max = tr_sigmas
        self.rot_sigma_min, self.rot_sigma_max = rot_sigmas
        self.tor_sigma_min, self.tor_sigma_max = tor_sigmas
        self.sphere_sigma_min, self.sphere_sigma_max = sphere_sigmas
        self.bl_sigma_min, self.bl_sigma_max = bl_sigmas
        self.no_torsion = no_torsion
        self.sphere_diffusion = sphere_diffusion
        self.predict_sphere_direction = predict_sphere_direction
        self.no_final_noise = no_final_noise
        self.no_rot_first_lig = no_rot_first_lig
        self.no_rot_all_ligands = no_rot_all_ligands
        self.no_sphere_first_lig = no_sphere_first_lig
        self.rot_center_anchor = rot_center_anchor
        self.keep_core_rigid = keep_core_rigid
        self.debug = debug
        self.temp_psi = temp_psi
        self.temp_sampling = temp_sampling
        self.temp_sigma_data = temp_sigma_data
        self.use_temp_effects = use_temp_effects
        self.rotation_prior_guess = rotation_prior_guess
        self.predict_x0_sphere = predict_x0_sphere
        self.restrict_rot_update = restrict_rot_update
        self.joint_rot_sphere_update = joint_rot_sphere_update
        self.partially_rigid = partially_rigid
        self.use_lookup_bl= use_lookup_bl
        self.align_multidentate = align_multidentate
        self.align_multidentate_last_step = align_multidentate_last_step
        self.use_rdkit_as_initial_guess = use_rdkit_as_initial_guess
        self.use_rdkit_confs = use_rdkit_as_initial_guess



        #self.debug = True #Todo change!!

    def setup_game(self, 
                   data: HeteroData, 
                   agent_keys: list[str], 
                   player_keys: list[str], 
                   agent_params: dict[str, float] = None) -> AgentDict:
        agent_dict = {}
        agent_cls = get_agent_cls(cls_name='score')
        

        for agent_key in agent_keys:
            if self.partially_rigid:
                data[agent_key].mask_rotate = copy.deepcopy(data[agent_key].get('mask_rotate_part_rig', None))
                data[agent_key].edge_mask = copy.deepcopy(data[agent_key].get('edge_mask_part_rig', None))

            is_player = agent_key in player_keys
            #print("data", data)
            #print('agent_key',agent_key)
            print("data[agent_key] edge_mask!!",data[agent_key].get('edge_mask', None))
            print("data[agent_key].pos when loading", data[agent_key].pos)
            #print("bond",data[agent_key][agent_key,'bond',agent_key])
            #print('edge index',data[agent_key][agent_key,'bond',agent_key].edge_index)
            #print(data[agent_key].get((agent_key, agent_key), {}))
            #print("edge_attr" in data[agent_key][agent_key,'bond',agent_key])
            agent_info = agent_cls(
                #edge_attr=data[agent_key].get((agent_key, agent_key), {}).get('edge_attr', None),
                edge_attr=data[agent_key][agent_key,'bond',agent_key].get('edge_attr', None),
                edge_index=data[agent_key][agent_key,'bond',agent_key].get('edge_index', None),
                #edge_index=data[agent_key].get((agent_key, agent_key), {}).get('edge_index', None),
                edge_mask=data[agent_key].get('edge_mask', None),
                mask_rotate=data[agent_key].get('mask_rotate', None),
                edge_mask_part_rig=data[agent_key].get('edge_mask_part_rig', None),
                mask_rotate_part_rig=data[agent_key].get('mask_rotate_part_rig', None),
                name=agent_key,
                x=data[agent_key].x,
                pos=data[agent_key].pos,
                pos_original = None if not self.debug else copy.deepcopy(data[agent_key].pos_bound),
                mask_bound_atoms=data[agent_key].get('mask_bound_atoms', None),
                atomic_numbers=data[agent_key][agent_key].get('atomic_numbers',0),
                is_player=is_player,
                tr_sigma_max=self.tr_sigma_max,
                bl_sigma_max=self.bl_sigma_max,
                no_torsion=self.no_torsion,
                sphere_diffusion=self.sphere_diffusion,
                predict_sphere_direction=self.predict_sphere_direction,
                anchor_mask=data[agent_key].get('anchor_mask', None),
                #no_rot_first_lig=self.no_rot_first_lig  
                no_rot_first_lig=self.no_rot_first_lig,
                no_sphere_first_lig=self.no_sphere_first_lig,
                no_rot_all_ligands=self.no_rot_all_ligands,
                is_first_lig=data[agent_key].get('is_first_lig', False),   
                rot_center_anchor=self.rot_center_anchor,
                keep_core_rigid=self.keep_core_rigid,
                metal_center_element = data["ligand_1"]["ligand_1"].get('atomic_numbers',0),
                use_lookup_bl=self.use_lookup_bl,
                debug=self.debug,
                rotation_prior_guess=self.rotation_prior_guess,
                restrict_rot_update=self.restrict_rot_update,
                joint_rot_sphere_update=self.joint_rot_sphere_update,
                partially_rigid=self.partially_rigid,
                align_multidentate=self.align_multidentate,
                align_multidentate_last_step = self.align_multidentate_last_step,
                use_rdkit_as_initial_guess=self.use_rdkit_as_initial_guess,
                use_rdkit_confs=self.use_rdkit_as_initial_guess,
                n_rounds=self.n_rounds, 
                
                
            )
            print("before data[agent_key]", data[agent_key]["mask_rotate_part_rig"])
            print("agent_infoamask rotate", agent_info.mask_rotate_part_rig)
            agent_dict[agent_key] = agent_info

        return agent_dict
    
    def play_round(self, 
                   agent_dict: AgentDict, 
                   agent_keys: list[str], 
                   player_keys: list[str], 
                   round_id: int) -> tuple[AgentDict, dict[str, float]]:
        t_tr = self.t_schedule[round_id]
        t_rot = t_tr
        t_tor = t_tr
        t_sphere = t_tr
        t_bl = t_tr
        t_agents = {agent: (t_tr, t_rot, t_tor, t_sphere, t_bl) for agent in agent_keys}
        
        

        action_dict = self.compute_actions(
            agent_dict=agent_dict, agent_keys=agent_keys, 
            player_keys=player_keys, t_agents=t_agents,
            round_id=round_id
        )
        agent_dict = self.apply_actions(agent_dict=agent_dict, action_dict=action_dict, round_id=round_id)
        return agent_dict, {}

    def compute_actions(self, 
                        agent_dict: AgentDict, 
                        agent_keys: list[str], 
                        player_keys: list[str], 
                        t_agents: dict[str, tuple[float, float, float, float, float]],
                        round_id: int) -> ActionDict:
        #",agent_dict)
        #print("player_keys",player_keys)
        #print("agent_keys",agent_keys)
        complex_data = self.transform(
            complex_data=agent_dict, 
            t_agents=t_agents, 
            agents=agent_keys, 
            players=player_keys
        )
        #b = 1 # TODO CHECK complex_data.num_graphs
        #if self.partially_rigid:
        #    assert complex_data.mask_rotate_part_rig == complex_data.mask_rotate
        complex_data.to(self.device)

        #print("agent_dict",agent_dict)

        dt_tr = self.t_schedule[round_id] - self.t_schedule[round_id + 1] \
                if round_id < self.n_rounds - 1 else self.t_schedule[round_id]

        t_tr, t_rot, t_tor, t_sphere, t_bl = t_agents[agent_keys[0]]
        #print("time", t_tr)
        dt_rot = self.t_schedule[round_id] - self.t_schedule[round_id + 1] \
            if round_id < self.n_rounds- 1 else self.t_schedule[round_id]

        dt_tor = self.t_schedule[round_id] - self.t_schedule[round_id + 1] \
            if round_id < self.n_rounds- 1 else self.t_schedule[round_id]

        dt_sphere, dt_bl = dt_tor, dt_rot
        device = complex_data.x.device
        tr_sigma, rot_sigma, tor_sigma, sphere_sigma, bl_sigma = self.t_to_sigma_fn(t_tr, t_rot, t_tor, t_sphere, t_bl)

        

        with torch.no_grad():
            tr_score, rot_score, tor_score, sphere_score, bl_score = self.model(complex_data)
            print("sphere_score",sphere_score.shape, sphere_score)
            print("rot_score model output", rot_score)
            #print("bl_score", bl_score)

           
            
            #if False:
            if self.debug or self.predict_x0_sphere:
                if not self.predict_sphere_direction:
                    current_point_tensor = complex_data.pos[complex_data.anchor_mask]
                    sphere_score = sphere_score
                print('DEBUGGGING -------------------------------------')
                print('---round_id:', round_id)
                current_point_tensor = complex_data.pos[complex_data.anchor_mask]  # Shape: [N, 3]
                print("complex_data.pos", complex_data.pos)
                print("complex_data.pos_original", complex_data.pos_original)
                if self.debug:
                    target_point = complex_data.pos_original[complex_data.anchor_mask]  # Shape: [N, 3]
                elif self.predict_x0_sphere:
                    target_point = sphere_score
                #print("current_point_tensor",current_point_tensor)
                #print("target_point",target_point)

                # Normalize the vectors
                target_point_norm = torch.norm(target_point, dim=1, keepdim=True)
                current_point_tensor_norm = torch.norm(current_point_tensor, dim=1, keepdim=True)
                #print("target_point_norm", target_point_norm)
                target_point_normalized = target_point / target_point_norm
                current_point_tensor_normalized = current_point_tensor / current_point_tensor_norm
                #print("target_point_normalized", target_point_normalized)
                #print("current_point_tensor_normalized", current_point_tensor_normalized)
                # Compute the dot products for each pair of vectors
                dot_products = torch.sum(target_point_normalized * current_point_tensor_normalized, dim=1)
                #print("dot_products",dot_products)
                # Clamp the dot products to avoid numerical issues with arccos
                dot_products_clamped = torch.clamp(dot_products, -1.0, 1.0)

                # Compute the true angles for each vector pair
                true_angle = torch.tensor(torch.arccos(dot_products_clamped), device=device)  
                
                print("True_angle",true_angle)
                # Calculate the true score for each angle
                
                


                true_score = torch.tensor(
                    [ n_sphere_angle.score(angle.item(), sphere_sigma.cpu().numpy(), 3) for angle in true_angle], 
                    device=true_angle.device
                )
                vector_to_target = target_point_normalized - current_point_tensor_normalized

                # Project onto the tangent plane of the current point (orthogonal projection)
                projection_onto_tangent_plane = vector_to_target - torch.sum(vector_to_target * current_point_tensor_normalized, dim=1, keepdim=True) * current_point_tensor_normalized
                norm_projection = torch.linalg.vector_norm(projection_onto_tangent_plane, dim=1).unsqueeze(-1)
                
                true_axis_normalized = (projection_onto_tangent_plane / (norm_projection + 1e-7))

                # Adjust the score to reflect the predicted direction on the sphere
                #print("true_score scalar", true_score.shape, true_score)
                #print("true_axis_normalized", true_axis_normalized.shape, true_axis_normalized)
                sphere_score_true = (- true_axis_normalized * true_score.unsqueeze(-1))
                #print("true_axis_normalized", true_axis_normalized.shape, true_axis_normalized)
                # Ensure orthogonality of the true score with the current point
                dot_product_with_current_point = torch.sum(sphere_score_true * current_point_tensor_normalized.squeeze(1), dim=1, keepdim=True)
                #assert torch.isclose(dot_product_with_current_point, torch.tensor(0.0, device=dot_product_with_current_point.device, dtype=torch.double), atol=1e-3).all()

                # Ensure that the true scores point towards the target point
                dot_product_with_target = torch.sum(sphere_score_true * vector_to_target, dim=1, keepdim=True)
                #print('shape', dot_product_with_target.shape)
                #assert dot_product_with_target.all() > -0.05, f"true_scores do not point towards the target point. Dot product is {dot_product_with_target}"

                # Calculate the predicted score for each vector
                #predicted_score = true_axis_normalized * true_score.unsqueeze(1)  # Broadcasting true_score to [N, 1]

                #print("predicted_score", sphere_score)
                #print("true_score", sphere_score_true)
                
                #TODO Change
                #sphere_score = sphere_score_true
                if self.predict_x0_sphere:
                    sphere_score = sphere_score_true
                #diff = sphere_score - sphere_score_true

                

                #print('diff', diff)
                
                #now same for rotation
                #if self.use_true_rotation_score:
                #if False:
                    
                    
                #    rot_score = torch.tensor(so3.score_vec(vec=rot_update.cpu().numpy(),
                #                                    eps=rot_sigma), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                


                #print('shape',sphere_sigma.shape, sphere_sigma.unsqueeze(-1).shape)
                #sphere_score_norm = n_sphere_angle.score_norm(sphere_sigma.unsqueeze(-1).cpu()).to(DEVICE)
                
                #print('sphere_score_norm', sphere_score_norm)
                #sphere_loss = 3.0 * torch.mean(((diff) / (sphere_score_norm.unsqueeze(-1) + 1e-5))**2) 
                #sphere_base_loss =  3.0 * torch.mean((sphere_score_true / sphere_score_norm.unsqueeze(-1) )**2)
                #print("LOSS @ inference:" , sphere_loss.item(), "in round", round_id, ', base_loss', sphere_base_loss)

            
            
            #print('output',tr_score, rot_score, tor_score, _, sphere_score, bl_score)
            #print('rot score before noise',rot_score.shape,rot_score)
            if self.no_rot_first_lig: # and not self.restrict_rot_update:
                if self.restrict_rot_update:
                    rot_score = torch.cat((torch.zeros(1, device=DEVICE),rot_score),dim=0)
                else:
                    rot_score = torch.cat((torch.zeros(1, 3, device=DEVICE),rot_score),dim=0)
            if self.no_sphere_first_lig:
                sphere_score = torch.cat((torch.zeros(1, 3, device=DEVICE),sphere_score),dim=0)
            if self.sphere_diffusion:
                #assert sphere_score.size(0) == len(player_keys)
                assert bl_score.size(0) == len(player_keys)
            else:
                assert tr_score.size(0) == len(player_keys)
            #assert rot_score.size(0) == len(player_keys)
            if not self.no_torsion:
                #print("complex_data", complex_data, flush=True)
                #print("tor_score.size(0)", tor_score.size(0), flush=True)
                #print("complex_data.edge_mask.sum()", complex_data.edge_mask.sum(), flush=True)
                assert tor_score.size(0) == complex_data.edge_mask.sum()

        

        tr_g = tr_sigma * torch.sqrt(
            torch.tensor(
                2 * np.log(self.tr_sigma_max / self.tr_sigma_min
            ), device=device)
        )
        rot_g = rot_sigma * torch.sqrt(
            2 * torch.tensor(np.log(self.rot_sigma_max / self.rot_sigma_min
            ), device=device)
        )

        if self.debug:
            scale_sphere_g = torch.sqrt(
                2 * torch.tensor(np.log(self.sphere_sigma_max / self.sphere_sigma_min
                ), device=device))
            #scale_sphere_g = 4.5
            print("scale_sphere_g", scale_sphere_g)
            
        else:
            scale_sphere_g = torch.sqrt(
            2 * torch.tensor(np.log(self.sphere_sigma_max / self.sphere_sigma_min
            ), device=device))
        
        sphere_g = sphere_sigma * scale_sphere_g
        

        

        bl_g = bl_sigma * torch.sqrt(
            2 * torch.tensor(np.log(self.bl_sigma_max / self.bl_sigma_min
            ), device=device)
        )
        score_updates = {
            'tr': tr_score,
            'rot': rot_score,
            'tor': tor_score,
            'sphere': sphere_score,
            'bl': bl_score
        }
        #if self.temperature_sampling:
       #("temp_sigma_data", self.temp_sigma_data)
        if self.temp_sigma_data:
            tr_sigma_data = calculate_sigma_data(self.tr_sigma_min, self.tr_sigma_max, self.temp_sigma_data['tr'])
            rot_sigma_data = calculate_sigma_data(self.rot_sigma_min, self.rot_sigma_max, self.temp_sigma_data['rot'])
            tor_sigma_data = calculate_sigma_data(self.tor_sigma_min, self.tor_sigma_max, self.temp_sigma_data['tor'])
            sphere_sigma_data = calculate_sigma_data(self.sphere_sigma_min, self.sphere_sigma_max, self.temp_sigma_data['sphere'])
            bl_sigma_data  = calculate_sigma_data(self.bl_sigma_min, self.bl_sigma_max, self.temp_sigma_data['bl'])
        else:
            tr_sigma_data, rot_sigma_data, tor_sigma_data, sphere_sigma_data, bl_sigma_data = None, None, None, None, None
        
        temp_scales  = calculate_temp_scales(score_updates= score_updates, temp_sampling= self.temp_sampling, temp_psi = self.temp_psi, use_temp_effects= self.use_temp_effects,
            tr_sigma_data= tr_sigma_data,
            rot_sigma_data= rot_sigma_data,
            tor_sigma_data = tor_sigma_data,
            sphere_sigma_data = sphere_sigma_data,
            bl_sigma_data = bl_sigma_data)
        #print("temp_scales", temp_scales)
        #temp_scale_deterministic = #dictonary with each # (lambda_tr + temp_sampling[0] * temp_psi[0] / 2)
        #temp_scale_stochastic = # ictonary with each  1 + temp_psi for rot, bl, ...
        

        
        #print('relevant scales', sphere_g ** 2 * dt_sphere)
        #print('square', sphere_g ,sphere_g ** 2)
        #print('rot score shape',rot_score.shape)
        #print('sphere score shape',sphere_score.shape)
        #print('bl score shape',bl_score.shape)
        if self.ode:
            if self.sphere_diffusion:
                sphere_update= 0.5 * temp_scales["sphere_scale_deterministic"] * sphere_g ** 2 * dt_sphere * sphere_score 
                bl_update = 0.5 * temp_scales["bl_scale_deterministic"] * bl_g ** 2 * dt_bl * bl_score
                #print("ode update sphere", sphere_update, 'scale,' , 0.5 * sphere_g ** 2 * dt_sphere, )
                tr_update = torch.zeros(size=(len(player_keys), 3),device=device)
            else:
                tr_update = (0.5 * temp_scales["tr_scale_deterministic"] * tr_g ** 2 * dt_tr * tr_score)
                sphere_update =  torch.zeros(size=(len(player_keys), 3), device=device)
                bl_update =  torch.zeros(size=(len(player_keys), 1), device=device)
            rot_update = (0.5 * temp_scales["rot_scale_deterministic"] * rot_score * dt_rot * rot_g ** 2).unsqueeze(-1)
        else:
            if self.no_final_noise and round_id == self.n_rounds - 1:
                if self.restrict_rot_update:
                    rot_z = torch.zeros(size=(len(player_keys),), device=device)
                else:
                    rot_z = torch.zeros(size=(len(player_keys),3 ), device=device)

                if self.sphere_diffusion:
                    
                    sphere_z = torch.zeros(size=(len(player_keys), 3), device=device)
                    bl_z = torch.zeros(size=(len(player_keys), 1), device=device)
                else:
                    tr_z = torch.zeros(size=(len(player_keys), 3), device=device)
                
            else:
                if self.sphere_diffusion:
                    #TODO: sample from 2D gaussian on tangent plane
                    #sphere_z = torch.normal(mean=0, std=1, size=(len(player_keys), 3), device=device)
                    #print("complex_data", complex_data)
                    #print("complex_data.pos[complex_data.anchor_mask]", complex_data.pos[complex_data.anchor_mask])
                    anchor_pos = complex_data.pos[complex_data.anchor_mask][[agent_keys.index(player) for player in player_keys]]
                    #print('pos anchors', anchor_pos.shape)

                    sphere_z = n_sphere_angle.sample_from_normal_plane(anchor_pos.cpu(), num_samples=1).to(DEVICE)

                    dot_product_check = torch.sum(sphere_z * anchor_pos, dim=-1)
                    #assert torch.allclose(dot_product_check, torch.zeros_like(dot_product_check), atol=1e-3), "Predicted score is not orthogonal to the noisy sample!"


                    bl_z = torch.normal(mean=0, std=1, size=(len(player_keys), 1), device=device)
                else:
                    tr_z = torch.normal(mean=0, std=1, size=(len(player_keys), 3), device=device)
                print("self.restrict_rot_update", self.restrict_rot_update)
                if self.restrict_rot_update:
                    rot_z = torch.normal(mean=0, std=1, size=(len(player_keys),), device=device)        
                else:
                    rot_z = torch.normal(mean=0, std=1, size=(len(player_keys), 3), device=device)
            print("score vs rot_z", rot_score.shape, rot_z.shape)
            #if self.restrict_rot_update:
                #rot_score = rot_score.unsqueeze(-1)
            print('ashapes', (temp_scales["rot_scale_deterministic"] * rot_score * dt_rot * rot_g ** 2).shape)
            print("rotzshape",rot_z.shape)
            print("bshapes", (rot_g * torch.sqrt(dt_rot * temp_scales["rot_scale_stochastic"]) * rot_z).shape)
            
            rot_update = (temp_scales["rot_scale_deterministic"] * rot_score * dt_rot * rot_g ** 2 + rot_g * torch.sqrt(dt_rot * temp_scales["rot_scale_stochastic"]) * rot_z)
            print("rot_update shapeeee", rot_update)
            if self.sphere_diffusion:
                
                

                if self.debug:
                    print("deterministic update", sphere_g ** 2 * dt_sphere * sphere_score)
                    print('random update', sphere_g * torch.sqrt(dt_sphere) * sphere_z)
                    det_norms = torch.linalg.vector_norm(sphere_g ** 2 * dt_sphere * sphere_score, dim=1)
                    random_norms = torch.linalg.vector_norm(sphere_g * torch.sqrt(dt_sphere) * sphere_z, dim=1)
                    #ratios = det_norms / random_norms

                    # Compute the average ratio over all 3 vectors
                    #average_ratio = ratios.mean()
                    #print("det/rand ratio", average_ratio , "with sigma", sphere_sigma)
                #print("sphere_score, sphere_z",sphere_score, sphere_z)
                sphere_update= temp_scales["sphere_scale_deterministic"] * sphere_g ** 2 * dt_sphere * sphere_score + sphere_g * torch.sqrt(dt_sphere * temp_scales["sphere_scale_stochastic"]) * sphere_z
                #TODO: assert that we don't leave the tangent plane
                bl_update = (temp_scales["bl_scale_deterministic"] * bl_g ** 2 * dt_bl * bl_score + bl_g * torch.sqrt(dt_bl * temp_scales["bl_scale_stochastic"]) * bl_z)
                tr_update = torch.zeros(size=(len(player_keys), 3), device=device)
            else:
                tr_update = (temp_scales["tr_scale_deterministic"] * tr_g ** 2 * dt_tr * tr_score + tr_g * torch.sqrt(dt_tr * temp_scales["tr_scale_stochastic"]) * tr_z)
                sphere_update =  torch.zeros(size=(len(player_keys), 3), device=device)
                bl_update =  torch.zeros(size=(len(player_keys), 1), device=device)

        if self.debug:
            print("bl_update",bl_update)
            
            #rot_update =  torch.zeros(size=(len(player_keys), 3), device=device)
            bl_update =  torch.zeros(size=(len(player_keys), 1), device=device)
            print("DEBUG: no bl updates")
        
        if self.rot_center_anchor:
            #print(f'complex_data {complex_data}')
            #print('pos',complex_data.pos)
            #rot_center = [complex_data[key].pos[complex_data[key]['anchor_mask']].to(device) for key in agent_keys]
            rot_centers = complex_data.rot_center
            #print('rot_centers GAMEPLAY',rot_centers.shape,rot_centers)    
        else:
            rot_centers = None

        if self.restrict_rot_update:
                rot_axis_restrict = complex_data.pos[complex_data.anchor_mask]
                
                print("rot_update.shape",rot_update.shape, rot_update)
                rot_update = rot_update.unsqueeze(-1)
                rot_axis_restrict  /= torch.linalg.norm(rot_axis_restrict)
                print("rot_axis_restrict",rot_axis_restrict.shape, rot_axis_restrict)
                #rot_update = rot_axis_restrict.squeeze(0) * rot_score #.squeeze()  
                rot_update = rot_update * rot_axis_restrict  #.squeeze()  
                rot_update= torch.tensor(rot_update, dtype=torch.float)
                print("rot_update", rot_update.shape, rot_update)
        if not self.no_torsion:
            tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(self.tor_sigma_max / self.tor_sigma_min)))
            if self.ode:
                tor_update = (0.5 * tor_g ** 2 * dt_tor * tor_score)
            else:
                if self.no_final_noise and round_id == self.n_rounds - 1:
                    tor_z = torch.zeros(size=(tor_score.shape[0],), device=device)
                else:
                    tor_z = torch.normal(mean=0, std=1, size=(tor_score.shape[0],), device=device)                
                
                tor_update = (tor_g ** 2 * dt_tor * tor_score + tor_g * np.sqrt(dt_tor) * tor_z)
        else:
            tor_update = None        
        
        updates = (rot_update, tr_update, tor_update, sphere_update, bl_update)
        #print('.tor_update',tor_update, flush=True)
        print("rot_update, tr_update, tor_update, sphere_update, bl_update", rot_update.shape, tr_update.shape, sphere_update.shape)
        nr_torsion_angles = [agent_dict[key].mask_rotate.shape[0] for key in agent_keys] if not self.no_torsion else None


        action_updates = self.gather_actions(
            updates=updates, agent_keys=agent_keys, player_keys=player_keys, nr_torsion_angles=nr_torsion_angles, rot_centers=rot_centers
        )

        #if True:
        if self.debug:
            print('sphere_vec', round_id, sphere_update, torch.linalg.norm(sphere_update,dim=1))
            rmsd_val, _ = compute_complex_rmsd_torch(current_point_tensor, target_point)
            
            if round_id == 0:
                print('START position difference', torch.median(current_point_tensor - target_point))
                print('START RMSD', rmsd_val)
                print('START ANGLE RMSE', rmsd(compute_angles(current_point_tensor).unsqueeze(-1), compute_angles(target_point).unsqueeze(-1)))
            if round_id == self.n_rounds-1:
                print("END")
                print('END position difference', torch.median(current_point_tensor - target_point))
        
                print('END RMSD', rmsd_val)
                print('END ANGLE RMSE', rmsd(compute_angles(current_point_tensor).unsqueeze(-1), compute_angles(target_point).unsqueeze(-1)))
            
                
        return action_updates

    def gather_actions(self, 
                       updates: tuple[Tensor], 
                       agent_keys: list[str], 
                       player_keys: list[str],
                       rot_centers: Tensor,
                       nr_torsion_angles: list[int]) -> ActionDict:
        rot_update, tr_update, tor_update, sphere_update, bl_update = updates
        action_updates = {}
        idx = 0
        tor_idx = 0
        #print('updaaate',tr_update,tor_update)
        for key in agent_keys:
            if key not in player_keys: # Agent is stationary, no need to update
                action_updates[key] = {
                    'tr_vec': rot_update.new_zeros((1, 3)), 
                    'rot_vec': rot_update.new_zeros((1, 3)),
                    'tor_vec': rot_update.new_zeros((1, 0)), 
                    'sphere_vec': rot_update.new_zeros((1, 3)),
                    'bl_vec': rot_update.new_zeros((1, 1)),
                    'rot_center': None,
                }
            
            else:
                if not self.no_torsion:
                    torsions_per_molecule = nr_torsion_angles[idx]
                    tor_agent = tor_update[tor_idx:tor_idx + torsions_per_molecule].view(1, -1)
                    tor_idx += torsions_per_molecule
                else:
                    tor_agent = None
                tr_agent, rot_agent, sphere_agent, bl_agent = tr_update[idx:idx+1], rot_update[idx: idx+1], sphere_update[idx: idx+1], bl_update[idx: idx+1]
               # print('agend IDX at gather actions ',idx)
                #print('rot agent',rot_agent)
                #print('rot_update',rot_update)
                if self.no_rot_first_lig and idx == 0:
                    rot_agent = torch.zeros(1,3, device=DEVICE)
                if self.rot_center_anchor:
                    rot_center = rot_centers[idx,:]
                else:
                    rot_center = None
                action_updates[key] = {'tr_vec': tr_agent, 'rot_vec': rot_agent,'tor_vec': tor_agent, 'sphere_vec': sphere_agent, 'bl_vec': bl_agent, 'rot_center': rot_center}
                idx += 1

        return action_updates

def get_strategy_from_args(
        model: torch.nn.Module, 
        model_args: argparse.Namespace, 
        strategy_type: str, n_rounds: int,
        ode: bool = False,
        debug: bool = False,
        distance_penalty: float = 0.0,
        device: str = 'cpu'
    ) -> BaseStrategy:

    common_args = {
            'model': model, 'n_rounds': n_rounds, 'device': device
    }
    transform = construct_score_transform(
        args=model_args, mode='inference'
    )
    t_to_sigma_fn = partial(
        t_to_sigma,
        tr_sigma_min=model_args.tr_sigma_min,
        tr_sigma_max=model_args.tr_sigma_max,
        rot_sigma_min=model_args.rot_sigma_min,
        rot_sigma_max=model_args.rot_sigma_max,
        tor_sigma_min=model_args.tor_sigma_min,
        tor_sigma_max=model_args.tor_sigma_max,
        sphere_sigma_min=model_args.sphere_sigma_min,
        sphere_sigma_max=model_args.sphere_sigma_max,
        bl_sigma_min=model_args.bl_sigma_min,
        bl_sigma_max=model_args.bl_sigma_max,
        
    )
    t_schedule = get_t_schedule(inference_steps=n_rounds)


    strategy = ScoreMatching(
        t_to_sigma=t_to_sigma_fn,
        tr_sigmas=(model_args.tr_sigma_min, model_args.tr_sigma_max),
        rot_sigmas=(model_args.rot_sigma_min, model_args.rot_sigma_max),
        tor_sigmas=(model_args.tor_sigma_min, model_args.tor_sigma_max),
        sphere_sigmas=(model_args.sphere_sigma_min, model_args.sphere_sigma_max),
        bl_sigmas=(model_args.bl_sigma_min, model_args.bl_sigma_max),
        no_torsion=model_args.no_torsion,
        t_schedule=t_schedule,
        ode=ode,
        transform=transform,
        sphere_diffusion=model_args.sphere_diffusion,
        predict_sphere_direction=model_args.predict_sphere_direction,
        no_rot_first_lig=model_args.no_rot_first_lig,
        no_sphere_first_lig=model_args.no_sphere_first_lig,
        no_rot_all_ligands=model_args.no_rot_all_ligands if "no_rot_all_ligands" in model_args else False,
        rot_center_anchor=model_args.rot_center_anchor,
        keep_core_rigid=model_args.keep_core_rigid,
        debug = model_args.debug,
        temp_psi = model_args.temp_psi,
        temp_sampling = model_args.temp_sampling,
        temp_sigma_data = model_args.temp_sigma_data,
        use_temp_effects=model_args.use_temp_effects,
        joint_rot_sphere_update=model_args.joint_rot_sphere_update,
        rotation_prior_guess=model_args.rotation_prior_guess if "rotation_prior_guess" in model_args else False,
        predict_x0_sphere=model_args.predict_x0_sphere if "predict_x0_sphere" in model_args else False,
        restrict_rot_update=model_args.restrict_rot_update if "restrict_rot_update" in model_args else False,
        partially_rigid=model_args.partially_rigid,
        use_lookup_bl=model_args.use_lookup_bl if "use_lookup_bl" in model_args else True,
        align_multidentate=model_args.align_multidentate if "align_multidentate" in model_args else False,
        align_multidentate_last_step = model_args.align_multidentate_last_step  if "align_multidentate_last_step" in model_args else True,
        use_rdkit_as_initial_guess=model_args.use_rdkit_as_initial_guess  if "use_rdkit_as_initial_guess" in model_args else False,
        use_rdkit_confs=model_args.use_rdkit_as_initial_guess  if "use_rdkit_as_initial_guess" in model_args else False,
        **common_args
    )


    return strategy

