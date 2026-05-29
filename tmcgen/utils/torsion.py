import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data



def get_transformation_mask(pyg_data,node_type, keep_ligating_rigid= True):
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    num_atoms=len(G.nodes())
    
    # Convert mask_bound_atoms boolean tensor to a set of node indices where mask_bound_atoms is True
    mask_bound_atoms = set(torch.where(pyg_data.mask_bound_atoms)[0].cpu().numpy())
    nx.set_node_attributes(G, {node: (node in mask_bound_atoms) for node in G.nodes()}, "mask_bound")

    to_rotate = []
    edges = pyg_data[node_type, node_type].edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])

        

        if not nx.is_connected(G2):
            #keep_ligating_rigid=T
            if keep_ligating_rigid:
                print("keep_ligating_rigid")
                connected_components = list(nx.connected_components(G2))
                assert len(connected_components)==2
                
                list_nr_bound_atoms = []
                for component in connected_components:
                    mask_bound_count = sum(G.nodes[node]["mask_bound"] for node in component)
                    list_nr_bound_atoms.append(mask_bound_count)
                if not 0 in list_nr_bound_atoms:
                    print("Skipping this bond (multidentate)")
                    to_rotate.append([])
                    to_rotate.append([])
                    continue 



            l = list(sorted(nx.connected_components(G2), key=len)[0])

            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)

    idx = 0
    for i in range(min(edges.shape[0], num_atoms)):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1
    return mask_edges, mask_rotate



"""
    Preprocessing and computation for torsional updates to conformers
"""
def get_transformation_mask_OLD(pyg_data,node_type):
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    edges = pyg_data[node_type, node_type].edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(min(edges.shape[0], len(G.edges()))):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False, invert_transform=False):

    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    
    if type(mask_rotate) == list: mask_rotate = mask_rotate[0]
    
    if isinstance(torsion_updates,torch.Tensor):
        torsion_updates = torsion_updates.tolist()[0]
    else:
        torsion_updates = torsion_updates.tolist()

    edge_index_np = edge_index.cpu().numpy()
    mask_rotate_np = mask_rotate.cpu().numpy() if isinstance(mask_rotate, torch.Tensor) else mask_rotate


    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        if mask_rotate_np[idx_edge, u] or (not mask_rotate_np[idx_edge, v]):
            print("mask rotate exception")
        
        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()
        
        pos[mask_rotate_np[idx_edge]] = (pos[mask_rotate_np[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]
        


    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def modify_conformer_torsion_angles_batch(pos, edge_index, mask_rotate, torsion_updates):
    from tmcgen.utils.geometry import axis_angle_to_matrix

    pos = pos + 0
    for idx_edge, e in enumerate(edge_index):
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[:, u] - pos[:, v]  # convention: positive rotation if pointing inwards
        rot_mat = axis_angle_to_matrix(
            rot_vec / torch.linalg.norm(rot_vec, dim=-1, keepdims=True) * torsion_updates[:, idx_edge:idx_edge + 1])

        pos[:, mask_rotate[idx_edge]] = torch.bmm(pos[:, mask_rotate[idx_edge]] - pos[:, v:v + 1], torch.transpose(rot_mat, 1, 2)) + pos[:, v:v + 1]

    return pos


def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    if type(data) is Data:
        return modify_conformer_torsion_angles(data.pos,
                                               data.edge_index.T[data.edge_mask],
                                               data.mask_rotate, torsion_updates)
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]]
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer_torsion_angles(pos, edges, mask_rotate, torsion_update)
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new


def get_dihedrals(data_list):
    edge_index, edge_mask = data_list[0]['ligand', 'ligand'].edge_index, data_list[0]['ligand'].edge_mask
    edge_list = [[] for _ in range(torch.max(edge_index) + 1)]

    for p in edge_index.T:
        edge_list[p[0]].append(p[1])

    rot_bonds = [(p[0], p[1]) for i, p in enumerate(edge_index.T) if edge_mask[i]]

    dihedral = []
    for a, b in rot_bonds:
        c = edge_list[a][0] if edge_list[a][0] != b else edge_list[a][1]
        d = edge_list[b][0] if edge_list[b][0] != a else edge_list[b][1]
        dihedral.append((c.item(), a.item(), b.item(), d.item()))
    # dihedral_numpy = np.asarray(dihedral)
    # print(dihedral_numpy.shape)
    dihedral = torch.tensor(dihedral)
    return dihedral
