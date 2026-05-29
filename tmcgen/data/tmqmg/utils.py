import os
import torch
from itertools import combinations
import numpy as np
import networkx as nx
import rdkit
from rdkit import Chem

def move_chosen_ligand_to_first_position(ligand, separate_graphs):
    ligand_index = separate_graphs.index(ligand)
    separate_graphs.insert(0, separate_graphs.pop(ligand_index))

def save_hetero_data(hetero_data, output_path, file_path):
    if os.path.basename(file_path).endswith('.xyz'):
        pt_path = os.path.basename(file_path).replace('.xyz', '.pt')
    elif os.path.basename(file_path).endswith('.gml'):
        pt_path = os.path.basename(file_path).replace('.gml', '.pt')
    elif os.path.basename(file_path).endswith('.dat'):
        pt_path = os.path.basename(file_path).replace('.dat', '.pt')
    savepath = os.path.join(output_path, pt_path)
    torch.save(hetero_data, savepath)

def is_collinear(positions):
    if len(positions) < 3:
        return False
    for comb in combinations(positions, 3):
        vec1 = np.array([comb[1][i] - comb[0][i] for i in range(3)])
        vec2 = np.array([comb[2][i] - comb[0][i] for i in range(3)])
        area = np.linalg.norm(np.cross(vec1, vec2))
        if area > 1e-3:
            return False
    return True

def count_heavy_atoms(graph):
    return sum(1 for _, data in graph.nodes(data=True) if data['feature_atomic_number'] > 1)

def count_different_elements(graph):
    return len(set(data['feature_atomic_number'] for _, data in graph.nodes(data=True) if data['feature_atomic_number'] > 1))

def has_non_isomorphic_subgraph(graph, heavy_atoms):
    """Check if there is at least one non-isomorphic subgraph formed by heavy atoms."""
    subgraphs = [graph.subgraph(nodes) for nodes in combinations(heavy_atoms, 3)]
    #print("subgraphs",subgraphs)
    for i, sg1 in enumerate(subgraphs):
        for sg2 in subgraphs[i + 1:]:
            if not nx.is_isomorphic(sg1, sg2):
                return True
    return False

def choose_ligand(graphs):
    candidates = []
    max_heavy_atoms_graph = None
    max_heavy_atoms_count = 0
    
    for graph in graphs:
        heavy_atoms = [node for node, data in graph.nodes(data=True) if data['feature_atomic_number'] > 1]
        heavy_atoms_count = len(heavy_atoms)
        
        # Track the graph with the most heavy atoms
        if heavy_atoms_count > max_heavy_atoms_count:
            max_heavy_atoms_count = heavy_atoms_count
            max_heavy_atoms_graph = graph

        if heavy_atoms_count < 3:
            continue
        
        if has_non_isomorphic_subgraph(graph, heavy_atoms):
            positions = [graph.nodes[node]['node_position'] for node in heavy_atoms[:3]]
            if not is_collinear(positions):
                candidates.append((graph, heavy_atoms_count))
        

    if candidates:
        return max(candidates, key=lambda x: x[1])[0]
    
    # Return the graph with the most heavy atoms if no suitable ligand is found
    print("No suitable ligand found, returning the graph with the most heavy atoms")
    return max_heavy_atoms_graph if max_heavy_atoms_graph else None


def convert_nx_to_rdkit(G):
    # Create a new RDKit molecule
    rdkit_mol = Chem.RWMol()

    # Add atoms to the molecule
    atom_indices = {}
    for node_idx, node_data in G.nodes(data=True):
        atomic_num = node_data.get('feature_atomic_number', 6)  # Default to carbon if not specified
        atom = Chem.Atom(atomic_num)
        mol_idx = rdkit_mol.AddAtom(atom)
        atom_indices[node_idx] = mol_idx

    # Add bonds to the molecule based on Wiberg bond order
    for start, end, edge_data in G.edges(data=True):
        wiberg_bond_order = edge_data.get('feature_wiberg_bond_order_int', 1)  # Default to single bond
        bond_type = get_bond_type(wiberg_bond_order)
        rdkit_mol.AddBond(atom_indices[start], atom_indices[end], bond_type)

    # Get the final molecule
    final_mol = rdkit_mol.GetMol()
    #Chem.SanitizeMol(final_mol)  # Sanitize the molecule
    return final_mol


def get_bond_type(wiberg_bond_order):
    if wiberg_bond_order == 1:
        return Chem.BondType.SINGLE
    elif wiberg_bond_order == 2:
        return Chem.BondType.DOUBLE
    elif wiberg_bond_order == 3:
        return Chem.BondType.TRIPLE
    else:
        raise ValueError


from rdkit.Chem import rdMolTransforms
import networkx as nx

def is_transition_metal(atomic_num):
    transition_metals = list(range(13,14)) + list(range(21, 31)) + list(range(39, 49)) + list(range(57, 81)) + list(range(89, 113))
    return atomic_num in transition_metals

def convert_rdkit_to_nx(mol):
    G = nx.Graph()

    conf = mol.GetConformer() 
    metal_center_element = None
    # Add nodes (atoms) with attributes
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)


        atomic_num = atom.GetAtomicNum()



        connectivity = len(atom.GetNeighbors())
        if is_transition_metal(atomic_num): #$and connectivity > 3:
            metal_center_element = atom.GetSymbol()

        G.add_node(
            idx,
            feature_atomic_number=atom.GetAtomicNum(),
            is_aromatic=atom.GetIsAromatic(),
            node_label=atom.GetSymbol(),
            feature_covalent_radius=Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()),
            feature_electronegativity=0.0,
            node_position=(pos.x, pos.y, pos.z)
        )

    # Add edges (bonds) with attributes
    if mol.GetNumBonds() == 0:
        print("Warning: Molecule has no bonds.")

    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bond_order = bond.GetBondTypeAsDouble()
        
        
        G.add_edge(
            end_idx,
            start_idx,
            feature_wiberg_bond_order_int=bond_order,
        )
    G.graph['meta_data'] = {
        'n_atoms': mol.GetNumAtoms(),
        'metal_center_element': metal_center_element,

    }
    return G
