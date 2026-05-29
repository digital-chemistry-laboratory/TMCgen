import os
from itertools import product
from multiprocessing import Pool
import random
import pickle
import traceback
from typing import Union
from glob import glob
import traceback

import numpy as np
import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform
import torch_geometric
from tmcgen.data import featurize
from tmcgen.utils import geometry as geometry_ops

# Type aliases
Array = np.ndarray
DecoyInfo = dict[str, dict[str, Array]]
def contains_nan(data):
    """
    Check if any tensor in the data contains NaN values.
    Iterates through all agent keys and their associated attributes.
    """
        
    for key, hetero_data in data.items():
        if isinstance(hetero_data, torch.Tensor) and torch.isnan(hetero_data).any():
            print(f"NaN detected in '{key}'", flush=True)
            return True
        if isinstance(hetero_data, torch_geometric.data.HeteroData):
            #print("isinstance hetero")
            for attr_name, attr_value in hetero_data.items():
                if isinstance(attr_value, torch.Tensor):
                    #print( "isinstance tensor")
                    if torch.isnan(attr_value).any():
                        print(f"NaN detected in {key} -> {attr_name}")
                        return True
    return False

def pretty_print_pyg(data: HeteroData, key: str) -> str:
    attr_str = f"x={data[key].x.shape},"
    attr_str += f" pos={data[key].pos.shape},"
    attr_str += f" pos_bound={data[key].pos_bound.shape}"

    if 'edge_index' in data[key]:
        edge_index = data[key].edge_index
        if edge_index is not None:
            attr_str += f", edge_index={data[key].edge_index.shape}"
    return attr_str


def _construct_filenames(complex_id: str, dataset: str = 'db5') -> dict[str, str]:
    if dataset == 'db5':
        filenames = {
            "ligand": f"{complex_id}_l_b.pdb",
            "receptor": f"{complex_id}_r_b.pdb" 
        }

    elif dataset == "dips":
        # A simple fix checking for whether .dill is already part of the id
        if ".dill" in complex_id:
            filenames = {
                complex_id: complex_id
            }
        else:
            filenames = {
                complex_id: f"{complex_id}.dill"
            }
        
    elif dataset == "multimer":
        filenames = {
            complex_id: f"{complex_id}-assembly1.cif"
        }
    
    return filenames


# ==============================================================================
# Base Dataset class
# ==============================================================================


class BaseDataset(Dataset):

    def __init__(
        self,
        root: str,
        parser: None,
        featurizer: featurize.ProteinFeaturizer,
        complex_list_file: str,
        transform: BaseTransform = None,
        dataset: str = "db5",
        complex_dir: str = 'complexes',
        mode: str = "train",
        resolution: str = "c_alpha",
        agent_type: str = "protein",
        num_workers: int = 1,
        progress_every: int = None,
        esm_embeddings_path: str = None,
        center_complex: bool = False,
        size_sorted: bool = False,
        confidence_mode = False,
        node_fdim: int = 0,
        use_rdkit_confs: bool = False,
    ):
        super().__init__(root=root, transform=transform)

        # Base directory where splits and other metadata are stored
        
        self.raw_data_dir = os.path.join(self.raw_dir, dataset)
    
        if complex_list_file is not None:
            self.complex_list_file = os.path.join(
                self.raw_data_dir, complex_list_file)
        else:
            self.complex_list_file = None

        # Directory where complexes files (.pdb, .mmcif etc) are stored
        self.complex_dir = os.path.join(self.raw_data_dir, complex_dir)

        # Directory where processed files are stored
        self.processed_data_dir = os.path.join(self.processed_dir, dataset)

        self.dataset = dataset
        self.parser = None
        self.featurizer = featurizer

        self.mode = mode
        self.resolution = resolution
        self.agent_type = agent_type

        self.num_workers = num_workers
        self.progress_every = progress_every
        self.center_complex = center_complex
        self.size_sorted = size_sorted
        if use_rdkit_confs:
            print("USING RDKit local structures", flush=True)
            proceseed_arg_str = "preprocessed_rdlocal"
        else:
            proceseed_arg_str = "preprocessed_gt" 
        

        # Loading all complex ids
        
        self.full_processed_dir = os.path.join(
            self.processed_data_dir, proceseed_arg_str
        )
        print("Looking for data at:", self.full_processed_dir, flush=True)

        # --- read the complex list ---
        try:
            with open(self.complex_list_file, "r") as f:
                complex_ids_all = [c.strip() for c in f if c.strip()]
        except Exception as e:
            print(f"[dataset] FAILED to read complex_list_file: "
                f"{self.complex_list_file!r}", flush=True)
            traceback.print_exc()
            raise

        print(f"[dataset] read {len(complex_ids_all)} ids from {self.complex_list_file}",
            flush=True)
        print(f"[dataset] full_processed_dir = {self.full_processed_dir}", flush=True)

        # --- filter to existing .pt files ---
        self.complexes_split = [
            c for c in complex_ids_all
            if os.path.exists(f"{self.full_processed_dir}/{c}.pt")
        ]

        print(f"[dataset] Nr of pt files: {len(self.complexes_split)} "
            f"/ {len(complex_ids_all)}", flush=True)

        # --- no files found ---
        if len(self.complexes_split) == 0:
            raise FileNotFoundError(
                f"No .pt files found.\n"
                f"  list file : {self.complex_list_file} ({len(complex_ids_all)} ids)\n"
                f"  search dir: {self.full_processed_dir}\n"
                f"  exists?   : {os.path.isdir(self.full_processed_dir)}\n"
                f"  example   : {self.full_processed_dir}/"
                f"{complex_ids_all[0] if complex_ids_all else '<empty>'}.pt"
            )
        print("self.full_processed_dir", self.full_processed_dir, flush=True)
        print("Nr of pt files:", len(self.complexes_split), flush=True)
    
    def load_ids(self):
        raise NotImplementedError("Subclasses must implement for themselves")

    def len(self) -> int:
        return len(self.ids)
    
    def get(self):
        raise NotImplementedError("Subclasses must implement for themselves")

    def preprocess_complexes(self):
        os.makedirs(self.full_processed_dir, exist_ok=True)

        # Loading all complex ids
        with open(f"{self.complex_list_file}", "r") as f:
            complex_ids_all = f.readlines()
            complex_ids_all = [complex_id.strip() for complex_id in complex_ids_all]

        print(f"Preprocessing {len(complex_ids_all)} complexes.", flush=True)
        print(f"Loading from: {self.complex_dir}", flush=True)
        print(f"Saving to: {self.full_processed_dir}", flush=True)
        print(flush=True)

        failures = []

        if self.num_workers > 1:
            for i in range(len(complex_ids_all) // self.progress_every + 1):
                complex_ids_batch = complex_ids_all[
                    self.progress_every * i : self.progress_every * (i + 1)
                ]

                p = Pool(self.num_workers, maxtasksperchild=1)
                map_fn = p.imap_unordered
                for (complex, complex_id) in map_fn(
                    self.preprocess_complex, complex_ids_batch):
                    if complex is not None:
                        complex_file = f"{self.full_processed_dir}/{complex_id}.pt"

                        #pdb_ids have a / in their name which can get confused
                        if self.dataset.lower() == "dips": 
                            dirname = os.path.dirname(complex_file)
                            os.makedirs(dirname, exist_ok=True)

                        print(f"Saving {complex_id} to {complex_file}", flush=True)
                        torch.save(complex, f"{complex_file}")
                        print(flush=True)
                    else:
                        failures.append(complex_id)
                        print(flush=True)
                p.__exit__(None, None, None)

        else:
            for (complex, complex_id) in map(
                self.preprocess_complex, complex_ids_all):
                if complex is not None:
                    complex_file = f"{self.full_processed_dir}/{complex_id}.pt"

                    #pdb_ids have a / in their name which can get confusing
                    if self.dataset.lower() == "dips": 
                        dirname = os.path.dirname(complex_file)
                        os.makedirs(dirname, exist_ok=True)

                    print(f"Saving {complex_id} to {complex_file}", flush=True)

                    torch.save(complex, f"{complex_file}")
                    print(flush=True)
                else:
                    failures.append(complex_id)
                    print(flush=True)

        print("Finished preprocessing complexes", flush=True)
        print(f"Failures: {failures}", flush=True)

    def preprocess_complex(self, complex_id: str) -> Union[HeteroData, str]:
        filenames = _construct_filenames(
            complex_id=complex_id, 
            dataset=self.dataset
        )
        for key, filename in filenames.items():
            filenames[key] = f"{self.complex_dir}/{filename}"

        try:
            structures = self.parser.parse(filenames=filenames)
            base_complex = HeteroData()

            if self.featurizer:
                base_complex = self.featurizer.featurize(
                    structures=structures, 
                    graph=base_complex
                )
            #print(base_complex)
            if self.size_sorted:
                sizes = [base_complex[agent].x.size(0) for agent in base_complex.agent_keys]
                sorted_idxs = sorted(range(len(sizes)), key=lambda x: sizes[x])
                agent_keys = [base_complex.agent_keys[idx] for idx in sorted_idxs]
                base_complex.agent_keys = agent_keys
  
            if self.center_complex:
                base_complex = BaseDataset._center_complex(complex_data=base_complex)
        
            for agent in base_complex.agent_keys:
                attr_str_agent = pretty_print_pyg(base_complex, agent)
                print(f"{complex_id}: Prepared {agent} graph - {attr_str_agent}", flush=True)

            #print(base_complex.agent_keys, flush=True)
            
            return base_complex, complex_id
        except Exception as e:
            print(f"Failed to process {complex_id} because of {e}", flush=True)
            traceback.print_exc()
            return None, complex_id
    
    @staticmethod
    def _center_complex(complex_data: HeteroData) -> HeteroData:
        stationary_agent = complex_data.agent_keys[-1]
        agent_center = complex_data[stationary_agent].pos.mean(dim=0, keepdims=True)

        for agent in complex_data.agent_keys:
            complex_data[agent].pos -= agent_center
            complex_data[agent].pos_bound -= agent_center
            #print(agent, complex_data[agent].pos.mean(dim=0))

        assert torch.allclose(
            complex_data[stationary_agent].pos.mean(dim=0, keepdims=True),
            torch.zeros(1, 3), atol=1e-3
        )
        return complex_data

# ==============================================================================
# Score Matching Dataset
# ==============================================================================


class DockScoreDataset(BaseDataset):

    def __init__(
        self,
        timepoints_per_complex: int = 1,
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.timepoints_per_complex = timepoints_per_complex
        self.load_ids()

    def load_ids(self):
        if self.timepoints_per_complex is not None:
            self.sample_ids = list(range(self.timepoints_per_complex))
            self.ids = list(product(self.complexes_split, self.sample_ids))

        else:
            self.ids = self.complexes_split
        print(f"Number of {self.dataset} {self.mode} complexes: {len(self.ids)}", flush=True)
        if len(self.ids) == 0:
            raise ValueError(
                f"Loaded dataset is empty (dataset={self.dataset}, mode={self.mode}). "
                f"Looked for processed files in {self.full_processed_dir} "
                f"using complex list {self.complex_list_file}."
            )
        random.shuffle(self.ids)

    def get(self, idx: int) -> HeteroData:
        complex_id, _ = self.ids[idx]
        complex_file = f"{self.full_processed_dir}/{complex_id}.pt"
        if not os.path.exists(complex_file):
            return None

        complex_base = torch.load(complex_file, map_location="cpu", weights_only=False)
        return complex_base.clone()


