import copy
import dataclasses
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
from rdkit.Chem import rdchem

from tmcgen.common.structure import Protein, Structure
from tmcgen.utils import geometry as geometry_ops
from tmcgen.utils.torsion import modify_conformer_torsion_angles
from tmcgen.common.constants import DEVICE
from tmcgen.data import transforms 
import random
import pandas as pd
import tmcgen.utils.so3 as so3
from tmcgen.utils.denticity_alignment import compute_rotation_angles, rotate_point

Tensor = torch.Tensor
Array = np.ndarray
ActionDict = dict[str, dict[str, Tensor]]


@dataclasses.dataclass
class Agent:

    name: str
    x: torch.Tensor = None
    edge_attr: torch.Tensor = None
    pos: torch.Tensor = None
    pos_original: torch.Tensor = None
    is_player: bool = False
    mask_bound_atoms: torch.Tensor = None 
    edge_mask: torch.Tensor = None 
    edge_index: torch.Tensor = None
    mask_rotate: torch.Tensor = None
    edge_mask_part_rig: torch.Tensor = None 
    mask_rotate_part_rig: torch.Tensor = None
    sphere_diffusion: bool = False
    predict_sphere_direction: bool = False
    anchor_mask: torch.Tensor = None
    no_rot_first_lig: bool = False
    no_rot_all_ligands: bool = False
    is_first_lig: bool = False
    rot_center_anchor: bool = False
    atomic_numbers: torch.Tensor = None
    debug: bool = False
    joint_rot_sphere_update: bool = False
    rotation_prior_guess: bool = False
    restrict_rot_update: bool = False
    partially_rigid: bool = False
    metal_center_element: torch.Tensor = None
    use_lookup_bl: bool = False
    align_multidentate: bool = False
    align_multidentate_last_step: bool = False
    no_sphere_first_lig: bool = False
    use_rdkit_as_initial_guess: bool = False
    use_rdkit_confs: bool = False
    n_rounds: int = -1
    
    def __post_init__(self):
        self.bl_stats_path = "data/raw/bl_lookup.csv"
        df = pd.read_csv(self.bl_stats_path)
        self.bl_lookup = df.set_index(['metal','ligand'])['mean'].to_dict()
        self.pt = rdchem.GetPeriodicTable()

        self.use_rdkit_as_initial_guess = self.use_rdkit_as_initial_guess if self.use_rdkit_confs else False

        if self.partially_rigid:
            #print("partially_rigid", flush=True)
            print('mask rotate comparison', self.mask_rotate_part_rig.sum(), self.mask_rotate.sum())
            print('edge_mask comparison', self.edge_mask_part_rig.sum(), self.edge_mask.sum())
            self.mask_rotate = self.mask_rotate_part_rig.clone()
            self.edge_mask = self.edge_mask_part_rig.clone()


        if self.is_player:
            self.pos, self.rot_init, self.tr_init, self.tor_init = self._randomize_pos(pos=self.pos)
        
        if self.debug:
            random_index = random.randint(0, 10)
            transforms.save_to_xyz(self.pos_original, self.atomic_numbers, f"pos_inference_original_{random_index}.xyz")
            transforms.save_to_xyz(self.pos, self.atomic_numbers, f"pos_inference_updated_{random_index}.xyz")
            print("SAVED RANDOMIZED POSITIONS GAMEPLAY")
        
        self.num_nodes = len(self.x)
        self.pos_init = self.pos.clone().detach()
        self.pos_ref = self.pos.clone().detach()
    def _rotate_multidentate(self, pos: torch.Tensor) -> torch.Tensor:
        """
        If there is more than one bound atom, rotate all of pos
        so that
        1) the first bound atom (i1) ends up exactly at its lookup bond-length,
        2) if there's a second bound atom (i2), we then rotate the whole
            structure around the *first bond* axis to place it at its lookup length.
        """
        n_bound = int(self.mask_bound_atoms.sum().item())
        if n_bound <= 1:
            return pos

        # 1. pivot (anchor)
        i0 = torch.nonzero(self.anchor_mask, as_tuple=False)[0,0].item()
        x0 = pos[i0]

        # 2. collect indices of bound atoms (excluding anchor if it's in there)
        candidates = torch.nonzero(self.mask_bound_atoms & ~self.anchor_mask,
                                as_tuple=False)[:,0].tolist()
        # --- First bound atom (i1) ---
        i1 = candidates[0]
        x1 = pos[i1]

        # 3. lookup target r1
        ligand_Z1 = int(self.atomic_numbers[i1])
        metal_Z   = int(self.metal_center_element[0])
        ligand_sym1 = self.pt.GetElementSymbol(ligand_Z1)
        metal_sym   = self.pt.GetElementSymbol(metal_Z)
        r_target1   = self.bl_lookup[(metal_sym, ligand_sym1)]

        # 4. compute axis & θ1 to bring x1 out to r_target1
        axis1 = torch.cross(x0, x1)
        θ1a, θ1b = compute_rotation_angles(x0, x1, torch.tensor(r_target1, device=pos.device))
        θ1 = θ1a if θ1a.abs() < θ1b.abs() else θ1b

        # 5. rotate everything around axis1 through pivot x0
        for idx in range(pos.shape[0]):
            pos[idx] = rotate_point(pos[idx], x0, axis1, θ1)

        
        # --- Now handle a second bound atom (i2) if present ---
        if n_bound > 2:
            i2 = candidates[1]
            x2 = pos[i2]
            if self.debug:
                print('bound2 before:', x2, x2.norm().item())

            # lookup target r2 for the second atom
            ligand_Z2   = int(self.atomic_numbers[i2])
            ligand_sym2 = self.pt.GetElementSymbol(ligand_Z2)
            r_target2   = self.bl_lookup.get((metal_sym, ligand_sym2), None)
            if r_target2 is None:
                #if self.debug:
                print(f"No lookup for {metal_sym}-{ligand_sym2}, skipping second alignment.")
            else:
                #if self.debug:
                print(f"r_target2 for {metal_sym}-{ligand_sym2}:", r_target2)

                # 6. define the *first‐bond* axis: the line through x0 in direction pos[i1]
                #    this keeps bond1 fixed during the second rotation
                axis2 = pos[i1] - x0
                if axis2.norm() < 1e-8:
                    #if self.debug:
                    print("Second rotation axis illdefined (first bond collapsed); skipping.")
                else:
                    # 7. compute θ2 to bring x2 out to r_target2
                    θ2a, θ2b = compute_rotation_angles(x0, x2, torch.tensor(r_target2, device=pos.device), axis=axis2)
                    θ2 = θ2a if θ2a.abs() < θ2b.abs() else θ2b

                    # 8. rotate everything around axis2 through pivot x0
                    for idx in range(pos.shape[0]):
                        pos[idx] = rotate_point(pos[idx], x0, axis2, θ2)

                    print("bound2 after:", pos[i2], pos[i2].norm().item())
                    print("bound1 recheck:", pos[i1], pos[i1].norm().item())
                    new_r2 = pos[i2].norm()
                    assert torch.isclose(new_r2, torch.tensor(r_target2, device=pos.device), atol=1e-3), \
                        f"Post-rotation norm of second bound atom ({new_r2.item():.4f}) != target {r_target2:.4f}"


        return pos
  
    def add_structure(self, structure: Structure):
        raise NotImplementedError
        

    def get_structure(self) -> Protein:
        return Protein(
            name=self.name,
            chain_ids=self.chain_ids,
            residue_types=self.residue_types,
            atom_positions=self.atom_positions,
            atom_mask=self.atom_mask,
            residue_index=self.residue_index
        )


@dataclasses.dataclass
class ScoreGameAgent(Agent):
    
    # Maximum translation noise used
    # This is used for sampling the random translation at the start of gameplay
    tr_sigma_max: float = 19.0
    bl_sigma_max: float = 0.0
    no_torsion: bool = True
    keep_core_rigid: bool = False
    def update_pose(self, action_dict: ActionDict,key, round_id):
        
        if not self.is_player:
            return
        
        assert action_dict is not None
        rot_vec = action_dict['rot_vec']
        tr_vec = action_dict['tr_vec']
        tor_vec = action_dict['tor_vec']
        sphere_vec = action_dict['sphere_vec']
        bl_vec = action_dict['bl_vec']
        rot_center = action_dict['rot_center']

        if not self.restrict_rot_update:
            rot_vec = rot_vec.squeeze(-1) 
        #print('rot vec in update_pose',rot_vec)
        #print('shapes',rot_vec.shape,tr_vec.shape,tor_vec.shape,sphere_vec.shape,bl_vec.shape)

        print('rot vec in update_pose',rot_vec)
        #print('tr_vec in update_pose',tr_vec)
        print('sphere_vec in update_pos',sphere_vec)
        #print('update_pose sphere_vec',sphere_vec)
        print('update_pose bl_vec',bl_vec)
        #print('update_pose rot_center',rot_center)
        print("anchor Z update pose", int(self.atomic_numbers[self.anchor_mask][0]))
        print('pos before anchor ',self.pos[self.anchor_mask],torch.linalg.vector_norm(self.pos[self.anchor_mask]))
        #print('pos before',self.pos)
        if self.debug:
            print('pos before',self.pos)
            #print("NO SPHERE UPDATE!!!")
            #tr_vec = torch.zeros_like(tr_vec)
            #sphere_vec = torch.zeros_like(sphere_vec)
            #print("SETTING BL VEC TO ZERO")
            #bl_vec = torch.zeros(1)
        
        if self.sphere_diffusion:
            original_pos = self.pos[self.anchor_mask].to(DEVICE)
            assert original_pos.shape[1] == 3
            original_bl = torch.linalg.vector_norm(original_pos, dim=1).to(DEVICE)
            #print("bl_vec in agents", bl_vec)
            print("original_bl in. apply action", int(self.atomic_numbers[self.anchor_mask][0]), self.is_first_lig, original_bl)
            new_bl = original_bl + bl_vec.to(DEVICE)

            orginial_pos_normalized = original_pos / original_bl.unsqueeze(1)

            if self.predict_sphere_direction:

                rot_vec_sphere = np.cross(orginial_pos_normalized.cpu().numpy(), sphere_vec.squeeze(0).cpu().numpy())[0,:] #TODO check the sign here!!
                if self.debug:

                    print("sphere rot_angle applied", np.linalg.norm(rot_vec_sphere))
                rot_mat = torch.tensor(Rotation.from_rotvec(rot_vec_sphere).as_matrix(), device=DEVICE, dtype=torch.float)
            else:
                #print('rot_vec', sphere_vec.squeeze(0).cpu().numpy())
                rot_mat = torch.tensor(Rotation.from_rotvec(- sphere_vec.squeeze(0).cpu().numpy()).as_matrix(), device=DEVICE, dtype=torch.float)
                #print('rot_mat',rot_mat)
            #print('rot mat', rot_mat)
            rotated_point = orginial_pos_normalized @ rot_mat.T
            #rotated_point_normalized = rotated_point / np.linalg.norm(rotated_point).to(DEVICE)
            rotated_point_normalized = rotated_point / torch.linalg.norm(rotated_point, dim=-1, keepdim=True).to(DEVICE)
            #print("rotated_point_normalized", rotated_point_normalized)
            new_pos = (rotated_point_normalized * new_bl.unsqueeze(1))
            #print('-----')
            tr_vec = (new_pos - original_pos).squeeze(0).float()
            #printing all sphere and bl vectors for debugging
            print('----- START')
            print("original pos", original_pos)
            print("new pos", new_pos)
            print("original bl", original_bl)
            print("new bl", new_bl)
            print("sphere_vec", sphere_vec)
            print("bl_vec", bl_vec)
            if self.debug:
                print('tr_vec', tr_vec)
            #print('cos', torch.dot(orginial_pos_normalized.squeeze(0), sphere_vec.squeeze(0)) / torch.linalg.vector_norm(sphere_vec, dim=1).to(DEVICE))
            print('----- END')
        if self.no_rot_first_lig and self.is_first_lig:
            #zero out rotation for first ligand
            #print("IN GAMEPLAY prediction KEEPING ROTATION VECTOR ZERO")
            rot_vec = torch.zeros(1,3, device=DEVICE)
        if self.debug and self.no_rot_all_ligands:
            print("AGGENT UPDATE no rotation update!")
            rot_vec = torch.zeros(1,3, device=DEVICE)

        if self.restrict_rot_update:
            #TODO CHANGE
            print("rot_vec applied for torsion", torch.linalg.vector_norm(rot_vec), rot_vec)
            print("self.pos before torsion", self.pos)
            self.pos = geometry_ops.apply_rigid_transform(
                pos=self.pos, rot_vec=rot_vec, tr_vec=torch.zeros_like(tr_vec), center=rot_center
            )
            print("self.pos after torsion", self.pos)
            rot_vec = torch.zeros_like(rot_vec)

        #print('pos before',self.pos)
        if self.joint_rot_sphere_update:
            print("joint_rot_sphere_update is True")
            print("rot_vec", rot_vec.shape, (rot_vec))
            #rot_vec_sphere = torch.tensor(rot_vec_sphere)
            print("rot_vec_sphere", rot_vec_sphere.shape, rot_vec_sphere)

            rot_vec = rot_vec + (torch.tensor(rot_vec_sphere,dtype=torch.float, device=DEVICE)) #.squeeze(0) # TODO: or minus


        if self.no_sphere_first_lig and self.is_first_lig:
            #zero out rotation for first ligand
            print("IN GAMEPLAY prediction KEEPING translation VECTOR ZERO")
            tr_vec = torch.zeros(1,3, device=DEVICE)

        #print("tor_vec ", tor_vec, flush=True)
        if tor_vec is None or tor_vec.numel()==0:
            #print(';shapes', self.pos.shape, rot_vec.shape, tr_vec.shape, rot_center.shape, rot_vec, tr_vec, rot_center)
            self.pos = geometry_ops.apply_rigid_transform(
                pos=self.pos, rot_vec=rot_vec, tr_vec=tr_vec, center=rot_center
            )
        else:
            #print('pos before',self.pos.shape,self.pos)
            #print("applying flexible,", flush=True)
            self.pos = geometry_ops.apply_flexible_transform(
                self, pos=self.pos,key=key, 
                rot_vec=rot_vec, 
                tr_vec=tr_vec, 
                tor_vec=tor_vec, 
                center=rot_center, 
                keep_core_rigid=self.keep_core_rigid,
                keep_anchor_fixed=True,
            )

        #if self.debug:
        print("round_id,n_rounds in agent",round_id, self.n_rounds)
        if (self.align_multidentate) or (self.align_multidentate_last_step and round_id==self.n_rounds-1):
            print("align_multidentate in round", round_id)
            print("self.mask_bound_atoms", self.mask_bound_atoms)
            if sum(self.mask_bound_atoms)>1:
                self.pos = self._rotate_multidentate(self.pos)
                
        print('pos after anchors',self.pos[self.anchor_mask],torch.linalg.vector_norm(self.pos[self.anchor_mask]))

             

    def _randomize_pos(self, pos):
        
        if self.use_rdkit_as_initial_guess:
            print("use_rdkit_as_initial_guess!@!!")
        #self.use_rdkit_as_initial_guess=True
        #if self.use_rdkit_as_initial_guess:
        #    return pos, None, None, None

        print("randomizing positions!!")
        print("original pos", pos, torch.linalg.vector_norm(pos))
        pos = pos.to(DEVICE)
        if self.debug:
            print('pos anchor before randomizing' , pos[self.anchor_mask])
        #print("pos before   ",pos.shape,pos)
        if not self.no_torsion and not self.use_rdkit_as_initial_guess:
            tor_vec = np.random.uniform(low=-np.pi, high=np.pi, size=self.edge_mask.sum())
            print("self.edge_mask.sum()",self.edge_mask.sum())
            #print("edge_index",self.edge_index)
            #print("pos",pos)
            if self.mask_rotate.shape[0]!=0:
                #print('pos old', pos)
                pos = \
                    modify_conformer_torsion_angles(pos,
                                                self.edge_index.T[ 
                                                self.edge_mask], 
                                                self.mask_rotate, #[0], 
                                                tor_vec).to(DEVICE) 
                #print('pos new', pos)
        
        else:
            tor_vec = None
        
        print(" pos after torsion", pos, torch.linalg.vector_norm(pos))
        print("self.no_rot_all_ligands", self.no_rot_all_ligands)
        if (self.no_rot_first_lig and self.is_first_lig) :
            #zero out rotation for first ligand
            #print("IN GAMEPLAY KEEPING ROTATION VECTOR ZERO")
            rot_vec = torch.zeros(1,3, device=DEVICE)
            
        elif self.no_rot_all_ligands:
            print("IN GAMEPLAY KEEPING ROTATION VECTOR ZERO")
            rot_vec = torch.zeros(1,3, device=DEVICE)
            
        else:
            rot_vec = Rotation.random(num=1).as_rotvec()
            rot_vec = pos.new_tensor(rot_vec, dtype=torch.float, device=DEVICE)
            
        if self.use_rdkit_as_initial_guess:
            rot_vec = torch.zeros(1,3, device=DEVICE)
            
        if self.restrict_rot_update:
            
            print("GAMEPLAY: ", self.restrict_rot_update)
            rot_axis_restrict =  pos[self.anchor_mask]
            rot_axis_restrict  /= torch.linalg.norm(rot_axis_restrict)
            rot_angle = np.random.uniform(low=-np.pi, high=np.pi)
            #if self.debug:
            #    rot_angle = np.pi / 2.0
            print("random rot angle", rot_angle)
            rot_vec = rot_axis_restrict * rot_angle
            print("rot vec restrict,", rot_vec)

            rot_center = pos[self.anchor_mask].to(DEVICE)
            print("self.pos Before torsion", self.pos)
            

            pos = geometry_ops.apply_rigid_transform(
                pos=pos, rot_vec=rot_vec, tr_vec=torch.zeros(size=(1,), device=DEVICE), center=rot_center
            )
            print("self.pos after torsion", self.pos)
            rot_vec=torch.zeros_like(rot_vec)

        #print('RANDOMIZING rot vec', rot_vec)
        #print('rot score from perturbation', torch.tensor(so3.score_vec(vec=rot_vec.cpu().numpy(),eps=1.65), dtype=torch.float32).unsqueeze(0).to(DEVICE))

        if self.sphere_diffusion:
            #print('sphere diffusion at randomization')
            original_pos = pos[self.anchor_mask].to(DEVICE)
            original_pos_normalized = original_pos / torch.linalg.vector_norm(original_pos).to(DEVICE)
            original_bl = torch.linalg.vector_norm(original_pos, dim=1).to(DEVICE)
            
            random_point_sphere = torch.randn(3, device=DEVICE)
            random_point_sphere /= torch.linalg.vector_norm(random_point_sphere).to(DEVICE)
            if self.debug:
                #print("original_bl", original_bl)
                bl_vec = torch.zeros(1,1, device=DEVICE)
                print("IN GAMEPLAY KEEPING Bond Length VECTOR ZERO when Randomizing")
            else:
                bl_vec = torch.normal(
                        mean=0, std=self.bl_sigma_max, size=(1, 1), device=DEVICE)

            if self.use_lookup_bl:
                print("using use_lookup_bl")
                anchor_atomic_nr = self.atomic_numbers[self.anchor_mask]
                metal_atomic_nr = self.metal_center_element[0]

                
                ligand_Z = int(self.atomic_numbers[self.anchor_mask][0])
                print("ligand_Z", ligand_Z)
                metal_Z  = int(self.metal_center_element)

                # map Z→symbol
                ligand_sym = self.pt.GetElementSymbol(ligand_Z)
                metal_sym  = self.pt.GetElementSymbol(metal_Z)
                new_bl = self.bl_lookup.get(
                    (metal_sym, ligand_sym))
                print("NEW BL",anchor_atomic_nr,metal_atomic_nr, new_bl)

            else:
                new_bl = original_bl + bl_vec



            if (self.no_sphere_first_lig and self.is_first_lig) :
                print("IN GAMEPLAY KEEPING FIRST Translation VECTOR ZERO")
                random_point_sphere = original_pos_normalized


            if self.use_rdkit_as_initial_guess:
                random_point_sphere = original_pos_normalized

                
            new_pos = (random_point_sphere * (new_bl)).to(DEVICE)
            
            tr_vec = (new_pos - original_pos).squeeze(0)

            #printing all sphere and bl vectors for debugging
            
            #print("RANDOMIZING sphere")
            #print("original pos", original_pos)
            print("anchor Z", int(self.atomic_numbers[self.anchor_mask][0]))
            print("anchor mask", self.anchor_mask)
            print("new pos", new_pos, torch.linalg.vector_norm(new_pos))
            print("original bl", original_bl)
            print("bl_vec", bl_vec)
            print("new bl", original_bl + bl_vec)
            #print("sphere_vec", random_point_sphere)
            print('tr_vec', tr_vec)
        else:
            tr_vec = torch.normal(
                mean=0, std=self.tr_sigma_max, size=(1, 3), device=DEVICE)  
        
        

        if self.joint_rot_sphere_update:
            print("original_pos_normalized", original_pos_normalized.shape)
            print("random_point_sphere", random_point_sphere.shape)
            rotation_axis = torch.cross(original_pos_normalized, random_point_sphere.unsqueeze(0))
            rotation_axis = rotation_axis / torch.linalg.norm(rotation_axis)
            #print('anchor pos', complex_data[agent].pos[complex_data[agent]['anchor_mask']])
            #print("normalized_anchor_pos", normalized_anchor_pos)
            #print("sphere_vec_new", sphere_vec_new)
            #print("norms", (torch.linalg.norm(normalized_anchor_pos) * torch.linalg.norm(sphere_vec_new)))
            sphere_angle = torch.acos(
                torch.clamp(
                    torch.sum(original_pos_normalized * random_point_sphere, dim=-1) /
                    (torch.linalg.norm(original_pos_normalized, dim=-1) * torch.linalg.norm(random_point_sphere, dim=-1)),
                    min=-1.0,
                    max=1.0
                )
            )                        
            print("sphere_angle", sphere_angle)

            print("rot_update before",rot_vec.shape, rot_vec)
            rot_vec = rot_vec + (rotation_axis * sphere_angle).squeeze(0)

        #if False:
        #if self.rotation_prior_guess:
        #    rot_vec_sphere = np.cross(orginial_pos_normalized.cpu().numpy(), sphere_vec.squeeze(0).cpu().numpy())[0,:]

        #    rotation_axis = so3.guess_rotation_axis(self.anchor_index, self.edge_index, self.pos)
        #    rotation_angle = np.random.uniform(-np.pi, np.pi)
        #    random_rotation_vec = rotation_axis * rotation_angle
        #    rot_vec = rot_vec_sphere + random_rotation_vec

        
        
        if self.rot_center_anchor:
            rot_center = pos[self.anchor_mask].to(DEVICE)
            #print("rot_center at GAMEPLAY",rot_center.shape,rot_center)
        else: 
            rot_center = None
        print("FINAL ROT VEC", rot_vec)
        
        randomized_pos = geometry_ops.apply_rigid_transform(
            pos=pos, rot_vec=rot_vec, tr_vec=tr_vec, center=rot_center
        )
        if self.rot_center_anchor:
            assert torch.isclose(randomized_pos[self.anchor_mask], pos[self.anchor_mask] + tr_vec).all()



        if self.align_multidentate or (self.align_multidentate_last_step):
            
            
            if sum(self.mask_bound_atoms)>1:
                print("align_multidentate after randomizing!!")
                randomized_pos = self._rotate_multidentate(randomized_pos)

        #print('pos + tr_vec',pos[self.anchor_mask] + tr_vec)
        


        #from tmcgen.analysis.metrics import rmsd
        #print("rmsd after randomization",rmsd(randomized_pos,pos))
        #print('pos before',pos)
        #print('pos randomized',randomized_pos)

        #from tmcgen.analysis.metrics import (compute_complex_rmsd_torch, rmsd, permute_rmsd)
        #permuted_rmsd = permute_rmsd(randomized_pos,pos)
        #print('permuted rmsd after randomization',permuted_rmsd)

        #if self.debug:
        #    print('randomized anchor pos',randomized_pos[self.anchor_mask])

        return randomized_pos, rot_vec, tr_vec, tor_vec


def get_agent_cls(cls_name: str):
    if cls_name == "score":
        return ScoreGameAgent
    elif cls_name == "reward":
        return RewardGameAgent
    else:
        raise ValueError(f"Agent cls of type {cls_name} is not supported")

