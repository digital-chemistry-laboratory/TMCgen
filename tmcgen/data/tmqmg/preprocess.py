import argparse
import multiprocessing as mp
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from tmcgen.data.featurize import featurize_mol
from tmcgen.data.process_mols import get_lig_graph
from tmcgen.data.tmqmg.graph import (
    find_metal_node,
    translate_atoms_to_metal_center,
    disconnect_metal_node,
    extract_subgraphs,
    clean_and_reorder_graphs,
)
from tmcgen.data.tmqmg.utils import (
    save_hetero_data,
    choose_ligand,
    move_chosen_ligand_to_first_position,
    convert_nx_to_rdkit,
    convert_rdkit_to_nx,
)
from tmcgen.utils.torsion import get_transformation_mask



def apply_sphere_diffusion(separate_graphs):
    for graph in separate_graphs:
        connected_to_metal_nodes = [node for node in graph.nodes() if graph.nodes[node]['is_connected_to_metal'] == 1]
        if not connected_to_metal_nodes:
            continue

        center = np.mean([graph.nodes[node]['node_position'] for node in connected_to_metal_nodes], axis=0)
        closest_node = min(
            connected_to_metal_nodes,
            key=lambda node: np.linalg.norm(np.array(graph.nodes[node]['node_position']) - center)
        )
        graph.nodes[closest_node]['is_anchor'] = True


def process_and_save_graphs(separate_graphs, G, output_path, file_path):
    ligand = choose_ligand(separate_graphs[:-1])
    if ligand:
        move_chosen_ligand_to_first_position(ligand, separate_graphs)
    if len(separate_graphs) == 1:
        return
    hetero_data = prepare_hetero_data(separate_graphs, G)
    save_hetero_data(hetero_data, output_path, file_path)

def prepare_hetero_data(separate_graphs, G):
    hetero_data = HeteroData()
    hetero_data["agent_keys"] = []
    for i, sg in enumerate(separate_graphs):
        if sg.number_of_nodes() == 0:
            continue

        node_type = 'ligand_1' if i == len(separate_graphs) - 1 else f'receptor_{i}'
        x, pos, complex_graph = prepare_graph_tensors(sg, node_type,i)
        hetero_data[node_type] = complex_graph
        

        edge_mask_part_rig, mask_rotate_part_rig = get_transformation_mask(hetero_data[node_type], node_type, keep_ligating_rigid= True)
        edge_mask, mask_rotate = get_transformation_mask(hetero_data[node_type], node_type, keep_ligating_rigid= False)
        
        assert (edge_mask_part_rig[edge_mask == 0] == 0).all(), "Assertion failed: edge_mask_part_rig is True where edge_mask is False"

        if edge_mask.sum() == 0:
            mask_core_rigid = torch.zeros(complex_graph.num_nodes_dict[node_type], dtype=torch.bool)
        else:
            mask_core_rigid = torch.zeros(complex_graph.num_nodes_dict[node_type], dtype=torch.bool)
            
            anchor_index = (complex_graph.anchor_mask == True).nonzero(as_tuple=True)[0]

            mask_core_rigid[anchor_index] = True  # Anchor atom
            for edge in complex_graph[node_type, 'bond', node_type].edge_index.t():
                if edge[0] == anchor_index:
                    mask_core_rigid[edge[1]] = True
                elif edge[1] == anchor_index:
                    mask_core_rigid[edge[0]] = True

            if mask_core_rigid.sum() <= 2:

                direct_neighbors = mask_core_rigid.nonzero(as_tuple=True)[0]
                for neighbor in direct_neighbors:
                    for edge in complex_graph[node_type, 'bond', node_type].edge_index.t():
                        if edge[0] == neighbor and not mask_core_rigid[edge[1]]:
                            mask_core_rigid[edge[1]] = True
                        elif edge[1] == neighbor and not mask_core_rigid[edge[0]]:
                            mask_core_rigid[edge[0]] = True
            
            assert mask_core_rigid.sum() > 2
        
        hetero_data[node_type].mask_core_rigid = mask_core_rigid



        assign_graph_attributes(hetero_data[node_type], x, pos, edge_mask, mask_rotate, i, len(separate_graphs), edge_mask_part_rig, mask_rotate_part_rig)
        hetero_data["agent_keys"].append(node_type)
    return hetero_data

def prepare_graph_tensors(sg, node_type, i):
    i_one_hot = F.one_hot(torch.tensor(i), num_classes=max_num_agents).float()

    x = torch.tensor([
        [
            *i_one_hot.tolist(),
            node['is_anchor'],
            node['is_connected_to_metal'],
            node['feature_covalent_radius'],
            node['feature_electronegativity'],
            *node['featurized_x'],
        ]
        for _, node in sg.nodes(data=True)
    ], dtype=torch.float32)


    pos = torch.tensor([[
        node['node_position'][0],
        node['node_position'][1],
        node['node_position'][2]
    ] for _, node in sg.nodes(data=True)], dtype=torch.float32)

    assert pos.shape[1] == 3

    complex_graph = HeteroData()
    
    get_lig_graph(convert_nx_to_rdkit(sg),complex_graph,node_type)
    
    atomic_numbers = [node['feature_atomic_number'] for _, node in sg.nodes(data=True)]
    complex_graph[node_type].atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int)
    complex_graph.num_nodes_dict = {node_type: sg.number_of_nodes()}
    complex_graph[node_type].num_nodes = sg.number_of_nodes()

    complex_graph.mask_bound_atoms = torch.tensor(
            [node['is_bound_atom'] for _, node in sg.nodes(data=True)], dtype=torch.bool
        )
    complex_graph.anchor_mask = torch.tensor(
            [data['is_anchor'] for _, data in sg.nodes(data=True)], dtype=torch.bool
        )

    return x, pos, complex_graph

def assign_graph_attributes(graph, x, pos, edge_mask, mask_rotate, index, total_graphs, edge_mask_part_rig=None, mask_rotate_part_rig=None):
    graph.edge_mask = torch.tensor(edge_mask)
    graph.mask_rotate = torch.tensor(mask_rotate)
    if edge_mask_part_rig is not None:
        graph.edge_mask_part_rig = torch.tensor(edge_mask_part_rig)
        graph.mask_rotate_part_rig = torch.tensor(mask_rotate_part_rig)
    graph.x = x
    graph.pos = pos
    graph.pos_bound = pos.clone()
    graph.is_first_lig = (index == 0)


def create_rdkit_molecule_from_dat(file_path):
    atoms = []
    positions = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        in_geometry = False
        
        for line in lines:
            if line.startswith("F011::Equilibrium Geometry"):
                in_geometry = True
                continue
            
            if in_geometry:
                parts = line.split()
                if len(parts) != 5 or not parts[2].replace('.', '', 1).lstrip('+-').isdigit():
                    break
                
                try:
                    atom = parts[1]
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    atoms.append(atom)
                    positions.append((x, y, z))
                except ValueError:
                    continue
    
    if not atoms or not positions:
        raise ValueError(f"No valid geometry data found in {file_path}")

    mol = Chem.RWMol()
    conformer = Chem.Conformer(len(atoms))
    
    for idx, (atom, pos) in enumerate(zip(atoms, positions)):
        rdkit_atom = Chem.Atom(atom)
        mol_idx = mol.AddAtom(rdkit_atom)
        conformer.SetAtomPosition(mol_idx, pos)

    mol.AddConformer(conformer)
    rdDetermineBonds.DetermineConnectivity(mol, covFactor=1.0)
    return mol

def process_graph_file(file_path, output_path, sphere_diffusion=True, from_xyz=False, from_dat=False, rdkit_as_gt=False):
    if from_xyz:
        raw_mol = Chem.rdmolfiles.MolFromXYZFile(file_path)
        mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineConnectivity(mol, covFactor=1.0)
        G = convert_rdkit_to_nx(mol)
    elif from_dat:
        mol = create_rdkit_molecule_from_dat(file_path)
        G = convert_rdkit_to_nx(mol)
        types = {'Al': 0, 'Cl': 1, 'Co': 2, 'Cr': 3, 'Cu': 4, 'F': 5, 'Fe': 6, 'Mn': 7, 'Ni': 8, 'O': 9, 'Sc': 10, 'Ti': 11, 'V': 12, 'Zn': 13}
    elif rdkit_as_gt:
        csv_file = "INSERT PATH TO xyz2mol_tm/SMILES_csvs/tmqmg_smiles.csv"
        df = pd.read_csv(csv_file)
        df['mol'] = df['smiles_CSD_fix'].apply(lambda smi: Chem.MolFromSmiles(smi))
    else:
        G = nx.read_gml(file_path)
        mol = convert_nx_to_rdkit(G)
        types = {e: i for i, e in enumerate(['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'I', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'])}

    
    data = featurize_mol(mol, types)
    if data is None:
        return

    mol_atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    G_atoms = [G.nodes[node]['feature_atomic_number'] for node in G.nodes()]
    
    assert mol_atoms == G_atoms, f"Atom order mismatch between Mol and G in file {file_path}"


    # Add featurized data as node attributes in G
    for i, node in enumerate(G.nodes):
        G.nodes[node]['featurized_x'] = data.x[i].tolist()

    metal_center_element = G.graph['meta_data']['metal_center_element']

    node_to_remove, metal_position = find_metal_node(G, metal_center_element)
    if metal_position is None:
        raise ValueError("Metal atom not found in the graph.")

    translate_atoms_to_metal_center(G, metal_position)
    disconnect_metal_node(G, node_to_remove)
    
    separate_graphs = extract_subgraphs(G, node_to_remove)
    separate_graphs = clean_and_reorder_graphs(separate_graphs, node_to_remove)
    if sphere_diffusion:
        apply_sphere_diffusion(separate_graphs)
    process_and_save_graphs(separate_graphs,G, output_path, file_path)



def process_graph_files_parallel(file_paths, output_path, sphere_diffusion, from_xyz=False, rdkit_as_gt=False):
    for file_path in file_paths:
        process_graph_file(file_path, output_path, sphere_diffusion, from_xyz, rdkit_as_gt)

def main(input_folder, output_path, nr_files=10**6, num_workers=None, from_xyz=False, from_dat=False, rdkit_as_gt=False, sphere_diffusion=True):
    os.makedirs(output_path, exist_ok=True)
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.gml') or f.endswith('.xyz') or f.endswith(".dat")]
    selected_files = random.sample(all_files, min(len(all_files), nr_files))
    file_paths = [os.path.join(input_folder, filename) for filename in selected_files]

    # Determine the number of workers (use all available cores if num_workers is None)
    if num_workers is None:
        num_workers = mp.cpu_count()
    if num_workers == 1:
        for file_path in file_paths:
            process_graph_file(file_path, output_path, sphere_diffusion, from_xyz, from_dat, rdkit_as_gt)
    else:
        # Split the file paths into chunks, one for each worker
        chunk_size = len(file_paths) // num_workers
        chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]

        # Create a pool of workers
        with mp.Pool(processes=num_workers) as pool:
            pool.starmap(process_graph_files_parallel, [(chunk, output_path, sphere_diffusion, from_xyz, rdkit_as_gt) for chunk in chunks])




max_num_agents = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some files.')
    
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    from_dat = False
    rdkit_as_gt = False
    if args.dataset == 'tmqmg':
        nr_files=60799
        from_xyz=False
        num_workers=None
        input_folder = "<INPUT_FOLDER>"
        output_path = "<OUTPUT_PATH>"
    elif args.dataset == 'tmqmg_rdkit':
        nr_files = 100
        from_xyz = False
        num_workers = None
        input_folder = "<INPUT_FOLDER>"
        output_path = "<OUTPUT_PATH>"
    else: 
        raise NotImplementedError
    main(input_folder, output_path,nr_files=nr_files, from_xyz=from_xyz, from_dat=from_dat,rdkit_as_gt=rdkit_as_gt, num_workers=num_workers)
