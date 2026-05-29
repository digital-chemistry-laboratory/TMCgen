from itertools import combinations
from functools import partial
import dataclasses
from typing import Callable, Any
import copy
import os
import torch
import numpy as np
from torch_cluster import radius, radius_graph
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor

from tmcgen.utils import geometry as geometry_ops
import tmcgen.utils.so3 as so3
import torch_geometric
from tmcgen.utils.diffusion import t_to_sigma
from tmcgen.common.constants import DEVICE
import tmcgen.analysis.metrics as metrics
from tmcgen.utils import torus
from tmcgen.utils import n_sphere_angle
from tmcgen.game.agents import ScoreGameAgent
from tmcgen.utils.utils import save_tensor_to_xyz
import tmcgen.data
import traceback

#import  tmcgen.data.transforms.MultiAgentData
# Type aliases
Tensor = torch.Tensor
Array = np.ndarray
import random
from rdkit import Chem


from collections import deque

# File to save the positions
SAVE_FILE = "pos_updates.pt"
MAX_ENTRIES = 10
from rdkit import Chem

def save_to_xyz(positions, atomic_numbers, file_path):
    periodic_table = Chem.GetPeriodicTable()

    with open(file_path, "w") as f:
        f.write(f"{len(positions)}\n\n")  # XYZ header
        for pos, z in zip(positions, atomic_numbers):
            atomic_num = z.item()  # Ensure atomic number is an integer
            symbol = periodic_table.GetElementSymbol(atomic_num)  # Get symbol from RDKit
            f.write(f"{symbol} {pos[0].item():.6f} {pos[1].item():.6f} {pos[2].item():.6f}\n")


@dataclasses.dataclass
class GameTransform(BaseTransform):
    """
    Base Transform class used in this project. This transform acts on game.agents
    or on data objects and induces appropriate constructions of self and cross
    graphs.
    """
    max_radius: float = 10.0
    max_neighbors: int = 24
    cross_max_radius: float = 10.0
    cross_max_neighbors: int = 24
    no_torsion: bool = True
    sphere_diffusion : bool = False
    predict_sphere_direction: bool = False
    no_rot_first_lig: bool = False
    no_sphere_first_lig: bool = False
    no_rot_all_ligands: bool = False
    rot_center_anchor: bool = False
    keep_core_rigid: bool = False
    debug: bool = False
    scale_sphere: bool = False
    max_sphere_add_radius: float = 0.0
    joint_rot_sphere_update: bool = False
    restrict_rot_update: bool = False
    predict_x0_sphere: bool = False
    partially_rigid: bool= False

    def build_self_graph(self, complex_data) -> Tensor:
        if isinstance(complex_data, HeteroData):
            agent_keys = complex_data.agent_keys
        elif isinstance(complex_data, dict):
            agent_keys = list(complex_data.keys())
        else:
            raise ValueError(f"Complex base type {type(complex_data)} is not supported.")
        

        assert self.max_radius is not None

        if self.max_neighbors is None:
            self.max_neighbors = 32

        edge_index = torch.zeros((2, 0), device=DEVICE).long()

        n_nodes = 0
        for agent_key in agent_keys:

            radius_edges = radius_graph(
                x=complex_data[agent_key].pos.to(DEVICE),
                r=self.max_radius,
                max_num_neighbors=self.max_neighbors
            ).to(DEVICE)

            edge_index = torch.cat([edge_index, radius_edges.long() + n_nodes], dim=1)
            n_nodes += complex_data[agent_key].x.size(0)
        
        return edge_index

    def build_cross_graph(self,
                          complex_data, 
                          cross_cutoff: float = None,
                          pos_attr: str = 'pos') -> Tensor:
        if isinstance(complex_data, HeteroData):
            agent_keys = complex_data.agent_keys
        elif isinstance(complex_data, dict):
            agent_keys = list(sorted(complex_data.keys()))
        else:
            raise ValueError(f"Complex base type {type(complex_data)} is not supported.")
        
        device = complex_data[agent_keys[0]].x.device

        if cross_cutoff is None:
            cross_cutoff = self.cross_max_radius
            
        if self.cross_max_neighbors is None:
            self.cross_max_neighbors = 32

        cross_edge_index = torch.zeros((2, 0), device=device).long()
        start_idxs, start_idx = [], 0

        for agent_key in agent_keys:
            start_idxs.append(start_idx)
            start_idx += complex_data[agent_key].x.size(0)

        comb = list(combinations(range(len(agent_keys)), r=2))

        for idx_a, idx_b in comb:
            if pos_attr == 'pos':
                agent_a_pos = complex_data[agent_keys[idx_a]].pos.to(DEVICE)
                agent_b_pos = complex_data[agent_keys[idx_b]].pos.to(DEVICE)
            
            elif pos_attr == 'ref':
                agent_a_pos = complex_data[agent_keys[idx_a]].pos_ref.to(DEVICE)
                agent_b_pos = complex_data[agent_keys[idx_b]].pos_ref.to(DEVICE)
            
            elif pos_attr == 'bound':
                agent_a_pos = complex_data[agent_keys[idx_a]].pos_bound.to(DEVICE)
                agent_b_pos = complex_data[agent_keys[idx_b]].pos_bound.to(DEVICE)
            
            else:
                raise ValueError(f"{pos_attr} pos type not supported.")

            cross_edges = radius(
                x=agent_a_pos,
                y=agent_b_pos,
                r=cross_cutoff,
                max_num_neighbors=self.cross_max_neighbors
            )
            
            src, dst = cross_edges
            src = src + start_idxs[idx_b]
            dst = dst + start_idxs[idx_a]

            cross_edges = torch.stack([src, dst], dim=0).to(DEVICE)
            cross_edge_index = torch.cat(
                [cross_edge_index.to(DEVICE), 
                    cross_edges, 
                    torch.flip(cross_edges, dims=[0])], 
                dim=1
            ).to(DEVICE)
            
        return cross_edge_index


# ==============================================================================
# Transforms and Data objects used for Reward gradient style gameplay
# ==============================================================================


@dataclasses.dataclass
class RewardTransform(GameTransform):

    def __call__(self, 
                 complex_data: HeteroData, 
                 agents: list[str] = None, 
                 players: list[str] = None) -> Data:
        
        if agents is None:
            if isinstance(complex_data, HeteroData) or isinstance(complex_data, torch_geometric.data.hetero_data.HeteroData):
                
                agents = complex_data.agent_keys
            elif isinstance(complex_data, dict):
                agents = list(sorted(complex_data.keys()))

        complex_out = Data()
        complex_out.x = torch.cat(
            [complex_data[key].x for key in agents], dim=0
        )
        complex_out.pos = torch.cat(
            [complex_data[key].pos for key in agents], dim=0
        )
        complex_out.pos_ref = torch.cat(
            [complex_data[key].pos_ref for key in agents], dim=0
        )

        self_edge_index = self.build_self_graph(complex_data=complex_data)
        complex_out.edge_index = self_edge_index
 
        cross_edge_index = self.build_cross_graph(complex_data=complex_data)
        complex_out.cross_edge_index = cross_edge_index

        ref_cross_edge_index = self.build_cross_graph(
            complex_data=complex_data, pos_attr='ref')
        complex_out.ref_cross_edge_index = ref_cross_edge_index

        # (TODO): This part is still experimental and untested.
        if isinstance(complex_data, HeteroData):

            complex_out.y = torch.tensor(complex_data.y).float()
            complex_out.y_ref = torch.tensor(complex_data.y_ref).float()
            complex_out.y_diff = torch.tensor(complex_data.y - complex_data.y_ref).float()
        
            # if self.norm_method is not None and "sqrt_diff" in self.norm_method:
            #     sign_diff = torch.sign(complex_out.y_diff)
            #     complex_out.y_diff = sign_diff * torch.sqrt(torch.abs(complex_out.y_diff))

            complex_out.pos_bound = torch.cat(
            [complex_data[key].pos_bound for key in agents], dim=0
            )

            complex_rmsd, _ = metrics.compute_complex_rmsd_torch(
                complex_pred=complex_out.pos, 
                complex_true=complex_out.pos_bound
            )
            complex_rmsd_ref, _ =  metrics.compute_complex_rmsd_torch(
                complex_pred=complex_out.pos_ref,
                complex_true=complex_out.pos_bound
            )
        
            complex_out.rmsd = complex_rmsd
            complex_out.rmsd_ref = complex_rmsd_ref
            complex_out.rmsd_diff = complex_rmsd - complex_rmsd_ref

        complex_out.agent_keys = agents
        complex_out.protein_keys = ["ligand", "receptor"] # Hardcoded for now!
        num_nodes = sum(complex_data[agent].x.size(0) for agent in agents)
        complex_out.batch = torch.zeros((num_nodes,), dtype=torch.long)
        return complex_out.to(DEVICE)


# ==============================================================================
# Transforms and Data objects used for Score-Matching style gameplay
# ==============================================================================


class MultiAgentData(Data):

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "batch" in key:
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        elif key == 'agent_membership':
            return self.num_agents
        elif key == 'agent_center_pos':
            return 0
        elif key == 'center_src':
            return self.num_nodes
        else:
            return 0
        
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or 'face' in key:
            return -1
        elif key == "agent_center_edges":
            return -1
        return 0


@dataclasses.dataclass
class ScoreMatchingTransform(GameTransform):
    """
    Transform class used on Data object during training and inference. This is 
    used for training and gameplay on the langevin dynamics based equilibria 
    computation method. 
    """
    t_to_sigma: Callable = None
    cross_cutoff_threshold: float = 40.0
    dynamic_max_cross: bool = False
    pert_strategy: str = "all-but-one"
    same_t_for_agent: bool = False

    def __call__(self, 
                 complex_data: HeteroData, 
                 t_agents: list[str] = None, 
                 agents: list[str] = None, 
                 players: list[str] = None) -> Data:
        
        if agents is None:
            if complex_data is None:
                print("\n[ERROR] complex_data is None\n", flush=True)
                return None
            if isinstance(complex_data, HeteroData) or isinstance(complex_data,torch_geometric.data.hetero_data.HeteroData):
                agents = complex_data.agent_keys
            if isinstance(complex_data, MultiAgentData):
                agents = complex_data.agent_keys

            elif isinstance(complex_data, dict):
                agents = list(sorted(complex_data.keys()))
            
        if self.pert_strategy is not None:
            if self.pert_strategy == "all-but-one":
                players = agents[:-1]
            elif self.pert_strategy == "one":
                players_all = agents[:-1]
                players = [np.random.choice(players_all)]
            elif self.pert_strategy == "two":
                players_all = agents[:-1]
                
                if len(players_all)>1:
                    players = agents[0:2]
                else: 
                    print("Skipping strategy 'two': Not enough agents available.")
                    #return None 
                    raise NotImplementedError
            elif self.pert_strategy == 'confidence':
                players = agents        
            elif self.pert_strategy == "all":
                players = agents
        if t_agents is None:
            if self.same_t_for_agent:
                t = np.random.uniform(low=1e-5, high=1.0)
                t_agents = {agent: (t, t, t, t, t) for agent in agents}
            else:
                ts =  {agent: np.random.uniform(low=1e-5, high=1.0) for agent in players}
                ts.update({agent: 0.0 for agent in agents if agent not in players})
                t_agents = {agent: (t, t, t, t, t) for agent, t in ts.items()}

        return self.apply_transform(complex_data, t_agents=t_agents, 
                                    agents=agents, players=players)

        
    def apply_transform(self, 
                        complex_data: HeteroData, 
                        t_agents: dict[str, tuple[float, float, float, float, float]], 
                        agents: list[str], 
                        players: list[str]) -> Data:
        if self.pert_strategy is not None:
            tr_score, rot_score, tor_score,tor_sigma_edge, sphere_score, bl_score,rot_score_restrict = [], [], [], [], [], [], []
        node_t_tr, node_t_rot,node_t_tor, node_t_sphere, node_t_bl = [], [], [],[],[]
        agent_t_tr, agent_t_rot, agent_t_tor, agent_t_sphere, agent_t_bl = [], [], [], [], []
        #print("APPLYING TRANSFORM")
        complex_out = MultiAgentData()
        complex_out.pos_original = copy.deepcopy(torch.cat([complex_data[key].pos.to(DEVICE) for key in agents], dim=0).to(DEVICE)).to(DEVICE)
        if self.pert_strategy=="confidence":
            #print("complex datqa in transform", complex_data)
            complex_out.confidence_true = complex_data["complex_angle_rmsd_anchor"]
            complex_out.pdb_id = complex_data["pdb_id"]
        
        center_src = []
        agent_membership = []
        rot_center = []
        
        t_max = max(t_agents[agent][0] for agent in players)
        
        if self.same_t_for_agent:
            assert all(t_max == t_agents[agent][0] for agent in players)
        n_nodes, agent_idx = 0, 0

        mask_last_lig = []
        for idx, agent in enumerate(agents):
            is_last_agent = (idx == len(agents) - 1)
            mask_last_lig.append(is_last_agent)

            if self.partially_rigid:
                
                    complex_data[agent].edge_mask = copy.deepcopy(complex_data[agent].edge_mask_part_rig)
                    
                    complex_data[agent].mask_rotate = copy.deepcopy(complex_data[agent].mask_rotate_part_rig)

            if self.no_torsion:
                ignore_torsion = True
            else:
                if complex_data[agent].mask_rotate==None:
                    ignore_torsion = True
                else:
                    ignore_torsion = complex_data[agent].mask_rotate.shape[0] == 0 

            t_tr, t_rot, t_tor, t_sphere, t_bl = t_agents[agent]

            
            node_t_tr.append((t_tr * torch.ones(complex_data[agent].num_nodes)).to(DEVICE))
            node_t_rot.append((t_rot * torch.ones(complex_data[agent].num_nodes)).to(DEVICE))
            node_t_tor.append((t_tor * torch.ones(complex_data[agent].num_nodes)).to(DEVICE))
            node_t_sphere.append((t_sphere * torch.ones(complex_data[agent].num_nodes)).to(DEVICE))
            node_t_bl.append((t_bl * torch.ones(complex_data[agent].num_nodes)).to(DEVICE))
            
            if agent in players:
                

                agent_t_tr.append(t_tr)
                agent_t_rot.append(t_rot)
                agent_t_tor.append(t_tor)
                agent_t_bl.append(t_bl)
                agent_t_sphere.append(t_sphere)

                center_src.append((torch.arange(complex_data[agent].num_nodes) + n_nodes).to(DEVICE))
                agent_membership.extend([agent_idx] * complex_data[agent].num_nodes)
            

                tr_sigma, rot_sigma, tor_sigma, sphere_sigma , bl_sigma = self.t_to_sigma(t_tr, t_rot, t_tor,t_sphere ,t_bl)
                
                if self.rot_center_anchor:
                    rot_center_player = complex_data[agent].pos[complex_data[agent].anchor_mask].to(DEVICE)
                else:
                    rot_center_player = None

                if (self.pert_strategy is not None) and (self.pert_strategy!='confidence'):
                    if self.sphere_diffusion:
                        
                        bl_original = torch.linalg.vector_norm(complex_data[agent].pos[complex_data[agent]['anchor_mask']],2).to(DEVICE)
                        normalized_anchor_pos = (complex_data[agent].pos[complex_data[agent]['anchor_mask']].to(DEVICE) / bl_original).to(DEVICE)
                        assert torch.isclose(torch.linalg.vector_norm(normalized_anchor_pos), torch.tensor(1.0, device=DEVICE))
                        if self.debug:
                            print("sphere_sigma",sphere_sigma)
                        sphere_vec_new, sampled_angles = n_sphere_angle.sample(normalized_anchor_pos.cpu().numpy().squeeze(),
                            t=sphere_sigma, 
                            n=3)
                        sphere_vec_new = torch.tensor(sphere_vec_new, dtype=torch.float, device=DEVICE)
                        bl_update = torch.normal(mean=0, std=bl_sigma, size=(1,1), dtype=torch.float, device=DEVICE)

                        if self.debug:
                            bl_update = torch.zeros(1,device=DEVICE)
                        assert torch.isclose(torch.linalg.vector_norm(sphere_vec_new),  torch.tensor(1.0,dtype=torch.float, device=DEVICE))

                        if self.scale_sphere:
                            sphere_add_radius =  self.max_sphere_add_radius * t_sphere    
                        else:
                            sphere_add_radius = 0.0

                        pos_anchor_new = (sphere_vec_new * (sphere_add_radius + bl_original + bl_update)).to(DEVICE)
                        tr_update = torch.tensor(pos_anchor_new - complex_data[agent].pos[complex_data[agent]['anchor_mask']].to(DEVICE),dtype=torch.float, device=DEVICE)
                        


                    else:
                        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3), dtype=torch.float, device=DEVICE)
                        sphere_vec = None
                        bl_update = None

                    if self.debug and self.restrict_rot_update:
                        print("tr_update",tr_update)
                    
                    
                    if self.no_rot_first_lig and agent_idx==0:
                        rot_update = torch.zeros(3,device=DEVICE)
                    elif self.no_rot_all_ligands: 
                        rot_update = torch.zeros(3,device=DEVICE)
                    else:
                        rot_update = torch.from_numpy(so3.sample_vec(eps=rot_sigma)).to(DEVICE)
                    
                    if self.restrict_rot_update:
                        if not(self.no_rot_first_lig and agent_idx==0):
                            rot_angle_restrict = torch.normal(mean=0.0, std=rot_sigma, size=(1,))
                            rot_axis_restrict = complex_data[agent].pos[complex_data[agent]['anchor_mask']] #TODO check this!
                            rot_axis_restrict  /= torch.linalg.norm(rot_axis_restrict)
                            
                            rot_update_restrict = rot_axis_restrict.squeeze(0) * rot_angle_restrict.squeeze()  
            
                            rot_score_restrict = torch.from_numpy(torus.score(rot_angle_restrict.cpu().numpy(), rot_sigma)).float().to(DEVICE)
                            

                            _modify_agent(complex_data=complex_data, 
                                    tr_update=torch.zeros(size=(1, 3), dtype=torch.float, device=DEVICE), 
                                    rot_update=rot_update_restrict,
                                    tor_update=None,
                                    agent_key=agent,
                                    rot_center=rot_center_player,
                                    anchor_mask=complex_data[agent].anchor_mask,
                                    keep_core_rigid=self.keep_core_rigid,
                                    debug=self.debug)
                            rot_update = torch.zeros(size=(1, 3), dtype=torch.float, device=DEVICE)
                            
                    
                    if self.no_sphere_first_lig and agent_idx==0:
                        sphere_update = torch.zeros(3,device=DEVICE)
                        tr_update = torch.zeros(3,device=DEVICE)

                    if self.joint_rot_sphere_update:
                        rotation_axis = torch.cross(normalized_anchor_pos, sphere_vec_new)
                        rotation_axis = rotation_axis / torch.linalg.norm(rotation_axis)
                        
                        sphere_angle = torch.acos(
                            torch.clamp(
                                torch.sum(normalized_anchor_pos * sphere_vec_new, dim=-1) /
                                (torch.linalg.norm(normalized_anchor_pos, dim=-1) * torch.linalg.norm(sphere_vec_new, dim=-1)),
                                min=-1.0,
                                max=1.0
                            )
                        )                        
                        print("sphere_angle", sphere_angle)
                        
                        if self.restrict_rot_update:
                            assert torch.allclose(rot_update, torch.zeros_like(rot_update)), "rot_update is not zero as expected"
                        
                        rot_update = rot_update + (rotation_axis * sphere_angle).squeeze(0)

                        if self.debug and self.restrict_rot_update:
                            rot_update = torch.zeros_like(rot_update)

                    
        
                    
                    if not ignore_torsion:
                        tor_update = np.random.normal(loc=0.0, scale=tor_sigma, size=complex_data[agent].edge_mask.sum()) 
                 
                    else:
                        tor_update = None 

                    old_pos = complex_data[agent].pos
                    
                    if self.debug and self.restrict_rot_update:
                        tr_update = torch.zeros_like(tr_update)
                        rot_update = torch.zeros_like(rot_update)

                    _modify_agent(complex_data=complex_data, 
                                tr_update=tr_update, 
                                rot_update=rot_update,
                                tor_update=tor_update,
                                agent_key=agent,
                                rot_center=rot_center_player,
                                anchor_mask=complex_data[agent].anchor_mask,
                                keep_core_rigid=self.keep_core_rigid,
                                debug=self.debug)
                    
                    
                    
                    if self.sphere_diffusion:
                        if not(self.no_sphere_first_lig and agent_idx==0):
                        
                        
                            angle = torch.acos(torch.dot(normalized_anchor_pos.squeeze(), sphere_vec_new.squeeze()))
                            if self.predict_x0_sphere:
                                sphere_score_player = sphere_vec_new
                            else:
                                sphere_score_player = torch.tensor(n_sphere_angle.score(angle.cpu(), sphere_sigma, n=3)).to(DEVICE)
                            
                                if self.predict_sphere_direction:
                                    vector_to_target =  normalized_anchor_pos.squeeze() - sphere_vec_new.squeeze() #target point - current noisy point
                                    projection_onto_tangent_plane = vector_to_target - torch.sum(vector_to_target * sphere_vec_new, dim=1, keepdim=True) * sphere_vec_new
                                    norm_projection = torch.linalg.vector_norm(projection_onto_tangent_plane,  dim=1).unsqueeze(-1)
                                    true_axis = (projection_onto_tangent_plane / (norm_projection+1e-7)).squeeze(1)
                                    
                                    sphere_score_player = (- true_axis * sphere_score_player.unsqueeze(-1))


                                    dot_product_with_current_point = torch.sum(sphere_score_player * sphere_vec_new.squeeze(1), dim=1, keepdim=True) 
                                    
                                else:    
                                    
                                    axis = torch.cross(sphere_vec_new.squeeze(), normalized_anchor_pos.squeeze())
                                    axis_norm = torch.linalg.vector_norm(axis)
                                    axis = axis / (axis_norm + 1e-6)
                                    sphere_score_player = (sphere_score_player * axis).unsqueeze(0)
                                    

                            if not(self.no_sphere_first_lig and agent_idx==0):
                                sphere_score.append(sphere_score_player)
                                
                        bl_score_player = (-bl_update / bl_sigma ** 2).to(DEVICE)
                        tr_score_player = None 
                    else:
                        tr_score_player = (-tr_update / tr_sigma ** 2).to(DEVICE)
                        bl_score_player = None
                        sphere_score_player = None
                        sphere_score.append(sphere_score_player)

                    


                    
                        
                    tor_score_player = None if ignore_torsion else torch.from_numpy(torus.score(tor_update, tor_sigma)).float().to(DEVICE)
                    tor_sigma_edge_player = None if ignore_torsion else torch.ones(complex_data[agent].edge_mask.sum(), device=DEVICE) * tor_sigma
                    
                    if not(self.no_rot_first_lig and agent_idx==0):
                        if self.restrict_rot_update:
                            print("rot_score_restrict",rot_score_restrict)
                            rot_score.append(rot_score_restrict)
                        else:
                            rot_score_player = torch.tensor(so3.score_vec(vec=rot_update.cpu().numpy(),
                                                        eps=rot_sigma), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                            rot_score.append(rot_score_player)
                    
                    bl_score.append(bl_score_player)
                    tr_score.append(tr_score_player)
                    
                    
                        
                    if tor_update is not None:
                        tor_score.append(tor_score_player)
                        tor_sigma_edge.append(tor_sigma_edge_player)
                rot_center.append(rot_center_player)
                agent_idx += 1

            n_nodes += complex_data[agent].num_nodes

        self_edges = self.build_self_graph(complex_data)
        
        

        # tr_sigma max of tr_sigma for all agents
        tr_sigma_max_agents, _ , _, _ , _ = self.t_to_sigma(t_max,t_max,t_max,t_max,t_max)
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma_max_agents * 3 + self.cross_cutoff_threshold)
        else:
            cross_cutoff = self.cross_max_radius

        cross_edges = self.build_cross_graph(complex_data, cross_cutoff=cross_cutoff) 

        if self.rot_center_anchor:
            center_pos = torch.cat([complex_data[agent].pos[complex_data[agent].anchor_mask].to(DEVICE) for agent in agents if agent in players], dim=0).to(DEVICE)
        else:
            center_pos = torch.cat([
                torch.mean(complex_data[agent].pos, dim=0, keepdim=True).to(DEVICE)
                for agent in agents if agent in players
            ], dim=0).to(DEVICE)

        complex_out.x = torch.cat([complex_data[key].x for key in agents], dim=0).to(DEVICE)
        complex_out.pos = torch.cat([complex_data[key].pos.to(DEVICE) for key in agents], dim=0).to(DEVICE)
        if not self.debug:
            complex_out.pos_original = copy.deepcopy(complex_out.pos).to(DEVICE)
        complex_out.atomic_numbers = torch.cat([complex_data[key].atomic_numbers.to(DEVICE).to(DEVICE)  if isinstance(complex_data[key],ScoreGameAgent) else complex_data[key][key].atomic_numbers.to(DEVICE) for key in agents]).to(DEVICE)
        
        complex_out.anchor_mask = torch.cat([complex_data[key].anchor_mask for key in agents], dim=0).squeeze().to(DEVICE)


        if self.debug:
            if t_sphere>0.7:
                random_index = random.randint(0, 10)
                save_to_xyz(complex_out.pos_original, complex_out.atomic_numbers, f"pos_original_{random_index}.xyz")
                save_to_xyz(complex_out.pos, complex_out.atomic_numbers, f"pos_updated_{random_index}.xyz")


        
        complex_out.num_agents = len(players) 
        if not self.no_torsion:
            
            edge_masks = [complex_data[key].edge_mask.to(DEVICE) for key in agents if complex_data[key].edge_mask.numel() > 0]
            complex_out.edge_mask = torch.cat(edge_masks).to(DEVICE) if edge_masks and not self.no_torsion else torch.empty(0,device=DEVICE,dtype=torch.bool)
            filtered_edge_index= [complex_data[key].edge_index.to(DEVICE)  if isinstance(complex_data[key],ScoreGameAgent) else complex_data[key][key,key].edge_index.to(DEVICE) for key in agents]
            lig_bonds_edge_index = self.adjust_indices(complex_data, agents,filtered_edge_index)
            lig_bonds_edge_index = [tensor for tensor in lig_bonds_edge_index if tensor.numel() > 0]
            complex_out.lig_bonds_edge_index = torch.cat(lig_bonds_edge_index, dim=1).to(DEVICE) if lig_bonds_edge_index else torch.empty(0,device=DEVICE,dtype=torch.int64)
        
        bond_index = [complex_data[key].edge_index.to(DEVICE)  if isinstance(complex_data[key],ScoreGameAgent) else complex_data[key][key,key].edge_index.to(DEVICE) for key in agents]
        
        complex_out.bond_index = torch.cat(bond_index, axis=1).to(DEVICE)
        
        bond_attr = [complex_data[key].edge_attr.to(DEVICE)  if isinstance(complex_data[key],ScoreGameAgent) else complex_data[key][key,key].edge_attr.to(DEVICE) for key in agents]
        
        complex_out.bond_attr = torch.cat(bond_attr).to(DEVICE)
        
        if self.no_rot_first_lig or self.no_sphere_first_lig:
             mask_first_lig = [complex_data[key].is_first_lig for key in agents if key in players]
        else:
            mask_first_lig = [False] * len(agents)
        last_non_player_agent = self.identify_last_non_player_agent(agents, players)
        last_non_player_mask = self.create_last_non_player_mask(complex_data, agents, last_non_player_agent)
        complex_out.last_non_player_mask = last_non_player_mask

        # Setting the time
        complex_out.node_t_tr = torch.cat(node_t_tr, dim=0).float().to(DEVICE)
        complex_out.node_t_rot = torch.cat(node_t_rot, dim=0).float().to(DEVICE)
        complex_out.node_t_tor = torch.cat(node_t_tor, dim=0).float().to(DEVICE)
        complex_out.node_t_bl = torch.cat(node_t_bl, dim=0).float().to(DEVICE)
        complex_out.node_t_sphere = torch.cat(node_t_sphere, dim=0).float().to(DEVICE)

        complex_out.t_tr = torch.tensor(agent_t_tr).float().to(DEVICE)
        complex_out.t_rot = torch.tensor(agent_t_rot).float().to(DEVICE)
        complex_out.t_tor = torch.tensor(agent_t_tor).float().to(DEVICE)
        complex_out.t_bl = torch.tensor(agent_t_bl).float().to(DEVICE)
        complex_out.t_sphere = torch.tensor(agent_t_sphere).float().to(DEVICE)

        # Adding edges and center position
        complex_out.edge_index = self_edges.to(DEVICE)
        complex_out.cross_edge_index = cross_edges.to(DEVICE)
        complex_out.agent_center_pos = center_pos.to(DEVICE)
        complex_out.agent_membership = torch.tensor(agent_membership).long().to(DEVICE)
        complex_out.center_src = torch.cat(center_src, dim=0).long().to(DEVICE)
        complex_out.mask_first_lig = torch.tensor(mask_first_lig).bool().to(DEVICE)
        complex_out.mask_last_lig = torch.tensor(mask_last_lig).bool().to(DEVICE)
        if self.debug:
            print("complex_out.mask_last_lig ",len(agents), len(players),complex_out.mask_last_lig )
        if self.rot_center_anchor:
            complex_out.rot_center = torch.cat(rot_center, dim=0).to(DEVICE)
        else:
            complex_out.rot_center = [None] * len(agents)

        # Computing the true scores
        if (self.pert_strategy is not None) and (self.pert_strategy!='confidence'):
            
            if rot_score:
                complex_out.rot_score = torch.cat(rot_score, dim=0).to(DEVICE)
            else:
                complex_out.rot_score = torch.tensor([], device=DEVICE) 

            if self.sphere_diffusion:
                if sphere_score:
                    complex_out.sphere_score = torch.cat(sphere_score, dim=0).to(DEVICE)
                else:
                    complex_out.sphere_score = torch.tensor([], device=DEVICE) 
                complex_out.bl_score = torch.cat(bl_score, dim=0).to(DEVICE)
                complex_out.tr_score = torch.tensor([]).to(DEVICE)
                assert complex_out.bl_score.size(0) == len(players)
            else:
                complex_out.tr_score = torch.cat(tr_score, dim=0).to(DEVICE)
                complex_out.sphere_score = torch.tensor([]).to(DEVICE)
                complex_out.bl_score = torch.tensor([]).to(DEVICE)
                assert complex_out.tr_score.size(0) == len(players)
            if not len(tor_score)==0:
                complex_out.tor_score =  torch.cat(tor_score).to(DEVICE)
                complex_out.tor_sigma_edge =  torch.cat(tor_sigma_edge, dim=0).to(DEVICE)
            else:
                complex_out.tor_score =  torch.tensor([]).to(DEVICE)
                complex_out.tor_sigma_edge =  torch.tensor([]).to(DEVICE)
 
        complex_out = complex_out.to(DEVICE)

      

        return complex_out
        
    def adjust_indices(self,complex_data, agents,tensors):
        pos_cumsum = torch.cumsum(torch.tensor([complex_data[key].pos.size(0) for key in agents]), dim=0)
        adjusted_tensors = []
        for i, tensor in enumerate(tensors):
            if i > 0:
                tensor = tensor + pos_cumsum[i-1]
            adjusted_tensors.append(tensor)
        return adjusted_tensors            
    @staticmethod
    def identify_last_non_player_agent(agents, players):
        """Identify the last agent that is not a player."""
        return agents[-1] if agents[-1] not in players else None

    @staticmethod
    def create_last_non_player_mask(complex_data, agents, last_non_player_agent):
        """Create a binary mask for nodes belonging to the last non-player agent."""
        return torch.cat([
            torch.ones(complex_data[agent].x.size(0), device=DEVICE) if agent == last_non_player_agent else
            torch.zeros(complex_data[agent].x.size(0), device=DEVICE)
            for agent in agents
        ])

# ==============================================================================
# Transform construction functions
# ==============================================================================

def _modify_agent( 
    complex_data: HeteroData, 
    tr_update: Tensor, 
    rot_update: Array, 
    tor_update: Tensor, # TODO CHECK?
    rot_center: Tensor,
    keep_core_rigid: bool,
    agent_key: str,
    anchor_mask: bool = False,
    debug: bool = False):


    pos_orig = complex_data[agent_key].pos.to(DEVICE)
    rot_vec = torch.tensor(rot_update, dtype=torch.float32)
    
    if tor_update is None or complex_data[agent_key].mask_rotate.shape[0]==0:

        
        pos_updated = geometry_ops.apply_rigid_transform(
            pos=pos_orig, rot_vec=rot_vec, tr_vec=tr_update, center=rot_center
        )

        if rot_center is not None:
            assert sum(anchor_mask) == 1
            
    else:
        if debug:
            print("apply_flexible_transform", tor_update, flush=True)
        
        anchor_mask = complex_data[agent_key].anchor_mask
  
        keep_anchor_fixed = True
        pos_updated = geometry_ops.apply_flexible_transform(
            complex_data[agent_key], key=agent_key, pos=pos_orig, rot_vec=rot_vec, tr_vec=tr_update, tor_vec=tor_update, center=rot_center, keep_core_rigid=keep_core_rigid,keep_anchor_fixed=keep_anchor_fixed, debug=debug) #, mask_core_rigid=mask_core_rigid)
        
        
        
    '''
    if debug and not torch.allclose(rot_vec, torch.zeros_like(rot_vec)):
        print("rot_update", rot_vec)
        print('pos original', pos_orig)
        print('pos updated', pos_updated)
        print(pos_orig.shape)
        if pos_orig.shape[0]>4:
            print("SAVING STRUCTURES")
            random_index = random.randint(0, 10)
    
            #save_to_xyz(pos_orig, complex_data[agent_key][agent_key].atomic_numbers, f"pos_original_{random_index}.xyz")
            #save_to_xyz(pos_updated, complex_data[agent_key][agent_key].atomic_numbers, f"pos_updated_{random_index}.xyz")
    '''

    complex_data[agent_key].pos = pos_updated
   


def construct_score_transform(args, mode: str = "train") -> ScoreMatchingTransform:  

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
            bl_sigma_max=args.bl_sigma_max,
        )
    
    # TODO: What should be done for same_t_for_agent
    if mode == "inference":
        pert_strategy = None
        same_t_for_agent = None
    else:
        pert_strategy = args.pert_strategy \
            if "pert_strategy" in args else "all-but-one"
        same_t_for_agent = args.same_t_for_agent \
            if "same_t_for_agent" in args else True
    
    if args.transform is None:
        print("Transform was given as none, using default ma_noise transform")

    if 'cross_cutoff_threshold' not in args:
        args.cross_cutoff_threshold = args.cross_max_radius

    if args.transform is None or args.transform == "ma_noise":
        print("DEBUGIIING in transform", args.debug)
        transform = ScoreMatchingTransform(
            t_to_sigma=t_to_sigma_fn,
            max_radius=args.max_radius,
            max_neighbors=args.max_neighbors,
            no_torsion = args.no_torsion,
            cross_cutoff_threshold=args.cross_cutoff_threshold,
            cross_max_radius=args.cross_max_radius,
            cross_max_neighbors=args.cross_max_neighbors,
            dynamic_max_cross=args.dynamic_max_cross,
            pert_strategy=pert_strategy,
            same_t_for_agent=same_t_for_agent,
            sphere_diffusion=args.sphere_diffusion,
            predict_sphere_direction=args.predict_sphere_direction,
            no_rot_first_lig=args.no_rot_first_lig,
            no_rot_all_ligands=args.no_rot_all_ligands,
            no_sphere_first_lig=args.no_sphere_first_lig,
            rot_center_anchor=args.rot_center_anchor,
            keep_core_rigid=args.keep_core_rigid,
            debug=args.debug,
            max_sphere_add_radius=args.max_sphere_add_radius,
            scale_sphere=args.scale_sphere,
            joint_rot_sphere_update=args.joint_rot_sphere_update,
            restrict_rot_update=args.restrict_rot_update,
            predict_x0_sphere=args.predict_x0_sphere,
            partially_rigid= args.partially_rigid if "partially_rigid" in args else False

            
        )
    else:
        raise ValueError(f"{args.transform} is not supported.")

    return transform


