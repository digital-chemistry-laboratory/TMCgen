from typing import Callable
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_add
from e3nn import o3
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps
from torch_cluster import radius, radius_graph
import tmcgen.utils.so3 as so3
from tmcgen.utils import torus
from tmcgen.utils import geometry as geometry_ops
from tmcgen.common.constants import DEVICE
from tmcgen.utils import n_sphere_angle
from tmcgen.analysis.metrics import compute_angles_pairwise

torch._C._jit_set_bailout_depth(0)


def get_activation_layer(activation):
    
    if activation == "relu":
        return nn.ReLU()
    
    elif activation == "silu":
        return nn.SiLU()

    elif activation == "leaky_relu":
        return nn.LeakyReLU()


class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        mu = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (mu[1] - mu[0]).item() ** 2
        self.register_buffer('mu', mu)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.mu.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    

class NodeEmbedding(nn.Module):

    def __init__(self, n_residues: int, residue_feats: int, sigma_embed_dim, n_s: int):
        super().__init__()
        self.residue_type_embed = nn.Embedding(n_residues, n_s)
        self.residue_feats_embed = nn.Linear(residue_feats, n_s)
        self.sigma_embed = nn.Linear(sigma_embed_dim, n_s)

        self.n_residues = n_residues
        self.residue_feats = residue_feats
        self.sigma_embed_dim = sigma_embed_dim

    def forward(self, x):
        x_restype = torch.argmax(x[:, :self.n_residues], dim=1)
        res_type_emb = self.residue_type_embed(x_restype)

        x_resfeats = x[:, self.n_residues: self.n_residues + self.residue_feats]
        res_feats_emb = self.residue_feats_embed(x_resfeats)

        x_sigma = x[:, self.n_residues + self.residue_feats:]
        sigma_emb = self.sigma_embed(x_sigma)

        node_emb = res_feats_emb + res_type_emb + sigma_emb
        return node_emb
    

class TensorProductConvLayer(nn.Module):

    def __init__(self, in_irreps, sh_irreps, out_irreps, edge_fdim, batch_norm=False, residual=True, dropout=0.0,
                 h_dim=None, rot_mode = None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        self.rot_mode = rot_mode
        if h_dim is None:
            h_dim = edge_fdim

        
        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc_net = nn.Sequential(
            nn.Linear(edge_fdim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, x, edge_index, edge_attr, edge_sh, out_nodes=None, aggr='mean'):
        edge_src, edge_dst = edge_index
        

        edge_sh = edge_sh.float()
        edge_attr =edge_attr.float()
        x = x.float()
        

        tp_out = self.tp(x[edge_src], edge_sh, self.fc_net(edge_attr))

        out_nodes = out_nodes or x.shape[0]
    
        #print("out_nodes", out_nodes)
        out = scatter(src=tp_out, index=edge_dst, dim=0, dim_size=out_nodes, reduce=aggr)
        if self.batch_norm:
                out = self.batch_norm(out)
                
        if self.residual:
            padded = F.pad(x, (0, out.shape[-1] - x.shape[-1]))
            out = out + padded

        return out    


class ScoreModel(nn.Module):

    def __init__(self, 
            node_fdim: int, 
            edge_fdim: int, 
            sh_lmax: int = 2,
            n_s: int = 16, 
            n_v: int = 16, 
            n_conv_layers: int = 2, 
            max_radius: float = 10.0, 
            cross_max_radius: float = 10.0,
            center_max_radius: float = 10.0,
            distance_emb_dim: int = 32,
            angle_emb_dim: int = 256,
            cross_dist_emb_dim: int = 32,
            center_dist_emb_dim: int = 32,
            timestep_emb_fn=None,
            sigma_emb_dim: int = 32,
            dropout_p: float = 0.2,
            activation: str = "relu",
            t_to_sigma_fn: Callable = None,
            scale_by_sigma: bool = False,
            node_encoder_type: str = 'base',
            no_torsion: bool = True,
            use_distogram: bool = False,
            first_break: float = 0.2, 
            last_break: float = 8.0, 
            num_bins: int = 32,
            sphere_diffusion: bool = False,
            no_rot_first_lig: bool = False,
            no_sphere_first_lig: bool = False,
            separate_rot_sphere_conv: bool = False,
            sphere_projection: bool = False,
            anchor_graph_sphere: bool = False,
            batch_norm: bool = True,
            debug: bool = False,
            architecture: str = 'e3nn',
            hidden_channels=160,
            num_layers=20,
            num_rbf=64,
            cutoff_lower=0.0,
            cutoff_upper=10.0,
            node_attr_dim=97,
            trainable_rbf=True,
            use_bl_metal_graph=False,
            use_second_order_repr=False,
            separate_intra_inter_agent_updates: bool= True,
            angle_encoding_repr_learning: bool=False,
            predict_x0_sphere: bool=False,
            confidence_mode: bool=False,
            confidence_dropout: bool=0.1,
            restrict_rot_update: bool=False,
            **kwargs
        ):

        super().__init__(**kwargs)

        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.predict_x0_sphere = predict_x0_sphere
        self.restrict_rot_update = restrict_rot_update


        self.n_s, self.n_v = n_s, n_v
        self.n_conv_layers = n_conv_layers

        self.max_radius = max_radius
        self.cross_max_radius = cross_max_radius
        self.timestep_emb_fn = timestep_emb_fn

        self.scale_by_sigma = scale_by_sigma
        self.t_to_sigma_fn = t_to_sigma_fn

        self.no_torsion = no_torsion
        self.use_distogram = use_distogram
        self.angle_encoding_repr_learning = angle_encoding_repr_learning
        self.first_break = first_break
        self.last_break = last_break
        self.num_bins = num_bins
        self.sphere_diffusion = sphere_diffusion
        self.no_rot_first_lig = no_rot_first_lig
        self.no_sphere_first_lig = no_sphere_first_lig
        self.separate_rot_sphere_conv = separate_rot_sphere_conv
        self.anchor_graph_sphere = anchor_graph_sphere
        self.sphere_projection = sphere_projection
        self.use_bl_metal_graph = use_bl_metal_graph
        self.debug = debug
        self.separate_intra_inter_agent_updates = separate_intra_inter_agent_updates

        self.architecture = architecture
        

        act_layer = get_activation_layer(activation)

        
        
         
        
        
        if node_encoder_type == 'base':
            self.node_embedding = nn.Sequential(
                nn.Linear(node_fdim + sigma_emb_dim, n_s),
                act_layer,
                nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
                nn.Linear(n_s, n_s)
            )
        else:
            self.node_embedding = NodeEmbedding(
                n_residues=len(amino_acid_types),
                residue_feats=node_fdim - len(amino_acid_types),
                sigma_embed_dim=sigma_emb_dim,
                n_s=n_s
            )

        if cross_dist_emb_dim is None:
            cross_dist_emb_dim = distance_emb_dim

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_fdim + distance_emb_dim + sigma_emb_dim, n_s),
            act_layer,
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
            nn.Linear(n_s, n_s)
        )
        if angle_encoding_repr_learning:
            
            self.cross_angle_embedding = nn.Sequential(
            nn.Linear(angle_emb_dim, n_s),
            act_layer,
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
            nn.Linear(n_s, n_s)
        )
            
        
        self.cross_edge_embedding = nn.Sequential(
            nn.Linear(cross_dist_emb_dim + sigma_emb_dim, n_s),
            act_layer,
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
            nn.Linear(n_s, n_s)
        )

        self.dist_expansion = GaussianSmearing(start=0.0, stop=min(max_radius,5.0), num_gaussians=distance_emb_dim)
        if angle_encoding_repr_learning:
            self.angular_smearing=GaussianSmearing(start=0.0, stop=torch.pi, num_gaussians=angle_emb_dim)
        if use_bl_metal_graph:
            self.bl_dist_expansion = GaussianSmearing(start=1.4, stop=2.6, num_gaussians=distance_emb_dim)

        self.cross_dist_expansion = GaussianSmearing(start=0.0, stop=min(cross_max_radius,5.0), 
                                                        num_gaussians=cross_dist_emb_dim)
        self.center_dist_expansion = GaussianSmearing(start=0.0, stop=min(center_max_radius,5.0),
                                                      num_gaussians=center_dist_emb_dim)
        self.confidence_mode = confidence_mode
        self.confidence_dropout = confidence_dropout


        if self.architecture == 'e3nn':    
            if use_second_order_repr:       
                irrep_seq = [
                    f'{n_s}x0e',
                    f'{n_s}x0e + {n_v}x1o + {n_v}x2e',
                    f'{n_s}x0e + {n_v}x1o + {n_v}x2e + {n_v}x1e + {n_v}x2o',
                    f'{n_s}x0e + {n_v}x1o + {n_v}x2e + {n_v}x1e + {n_v}x2o + {n_s}x0o'
                ]        

            else:    
                irrep_seq = [
                    f"{n_s}x0e",
                    f"{n_s}x0e + {n_v}x1o",
                    f"{n_s}x0e + {n_v}x1o + {n_v}x1e",
                    f"{n_s}x0e + {n_v}x1o + {n_v}x1e + {n_s}x0o"
                ]

            
            conv_layers, cross_conv_layers = [], []

            for i in range(n_conv_layers):
                in_irreps = irrep_seq[min(i, len(irrep_seq)-1)]
                out_irreps = irrep_seq[min(i+1, len(irrep_seq)-1)]

                parameters = {
                    "in_irreps": in_irreps,
                    "sh_irreps": self.sh_irreps,
                    "out_irreps": out_irreps,
                    "edge_fdim": 3 * n_s,
                    "h_dim": 3 * n_s,
                    "residual": False,
                    "dropout": dropout_p,
                }

                conv_layer = TensorProductConvLayer(**parameters)
                

                conv_layers.append(conv_layer)
                if self.separate_intra_inter_agent_updates:
                    cross_conv_layer = TensorProductConvLayer(**parameters)
                    cross_conv_layers.append(cross_conv_layer)

            self.conv_layers = nn.ModuleList(conv_layers)
            if self.separate_intra_inter_agent_updates:
                self.cross_conv_layers = nn.ModuleList(cross_conv_layers)

        
     

        if self.separate_rot_sphere_conv and self.sphere_diffusion: 
            dim_input = center_dist_emb_dim + sigma_emb_dim if not anchor_graph_sphere else center_dist_emb_dim + sigma_emb_dim + angle_emb_dim
            self.center_edge_embedding_sphere = nn.Sequential(
                    nn.Linear(dim_input, n_s),
                    act_layer,
                    nn.Dropout(dropout_p),
                    nn.Linear(n_s, n_s),
                )
            self.center_edge_embedding_rot = nn.Sequential(
                    nn.Linear(center_dist_emb_dim + sigma_emb_dim, n_s),
                    act_layer,
                    nn.Dropout(dropout_p),
                    nn.Linear(n_s, n_s),
                )
            self.center_edge_embedding = nn.Sequential(
                    nn.Linear(center_dist_emb_dim + sigma_emb_dim, n_s),
                    act_layer,
                    nn.Dropout(dropout_p),
                    nn.Linear(n_s, n_s),
                )

            self.final_conv_sphere = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps if self.architecture=='e3nn' else f"{n_s}x0e+{n_v}x1o",
                sh_irreps=self.sh_irreps,
                out_irreps=f'1o + 1e',
                edge_fdim=2 * n_s ,
                h_dim=2 * n_s,
                residual=False,
                dropout=dropout_p,
                batch_norm=batch_norm,
            )

            self.final_conv_rot = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps  if self.architecture=='e3nn' else f"{n_s}x0e+{n_v}x1o",
                sh_irreps=self.sh_irreps,
                out_irreps=f'1o + 1e',
                edge_fdim=2 * n_s,
                h_dim=2 * n_s,
                residual=False,
                dropout=dropout_p,
                batch_norm=batch_norm,
                rot_mode = 'rot'
            )

        else:

            self.final_conv = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps  if self.architecture=='e3nn' else f"{n_s}x0e+{n_v}x1o",
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                edge_fdim=2 * n_s,
                h_dim=2 * n_s,
                residual=False,
                dropout=dropout_p,
                batch_norm=batch_norm,
            )

        self.tr_final_layer = nn.Sequential(
            nn.Linear(1 + sigma_emb_dim, n_s),
            nn.Dropout(dropout_p),
            act_layer,
            nn.Linear(n_s, 1),
        )
        if self.restrict_rot_update or not no_torsion:
           
            self.final_edge_embedding = nn.Sequential(
                nn.Linear(center_dist_emb_dim, n_s),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s)
            )
        if self.restrict_rot_update:
            self.final_tp_tor_anchors = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bond_conv_anchors = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps  if self.architecture=='e3nn' else  f"{n_s}x0e+{n_v}x1o",
                sh_irreps=self.final_tp_tor_anchors.irreps_out,
                out_irreps=f'{n_s}x0o + {n_s}x0e',
                edge_fdim=3 * n_s,
                residual=False,
                dropout=dropout_p,
                batch_norm=batch_norm
            )
            self.tor_final_layer_anchors = nn.Sequential(
                nn.Linear(2 * n_s, n_s, bias=False),
                nn.Tanh(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, 1, bias=False)
                )
        else:
            self.rot_final_layer = nn.Sequential(
                nn.Linear(1 + sigma_emb_dim, n_s),
                nn.Dropout(dropout_p),
                act_layer,
                nn.Linear(n_s, 1),
            )

        if not no_torsion:
            # torsion angles components
            self.final_edge_embedding = nn.Sequential(
                nn.Linear(center_dist_emb_dim, n_s),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s)
            )
            self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bond_conv = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps  if self.architecture=='e3nn' else  f"{n_s}x0e+{n_v}x1o",
                sh_irreps=self.final_tp_tor.irreps_out,
                out_irreps=f'{n_s}x0o + {n_s}x0e',
                edge_fdim=3 * n_s,
                residual=False,
                dropout=dropout_p,
                batch_norm=batch_norm
            )
            self.tor_final_layer = nn.Sequential(
                nn.Linear(2 * n_s, n_s, bias=False),
                nn.Tanh(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, 1, bias=False)
                )

        if self.sphere_diffusion:
            self.final_conv_radial = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps  if self.architecture=='e3nn' else f"{n_s}x0e+{n_v}x1o",
                sh_irreps=self.sh_irreps, 
                out_irreps=f'0o + 0e', #TODO change this?
                edge_fdim=2 * n_s,
                h_dim=2 * n_s,
                residual=False,
                dropout=dropout_p,
                batch_norm=batch_norm,
            )
            self.bl_final_layer = nn.Sequential(
            nn.Linear(1 + sigma_emb_dim, n_s),
            nn.Dropout(dropout_p),
            act_layer,
            nn.Linear(n_s, 1),
        )
        
        
    def forward(self, data):
        if self.architecture == 'e3nn':
            if self.separate_intra_inter_agent_updates:
                x, edge_index, edge_attr, edge_sh = self.setup_graph(data)
                edge_sh = edge_sh.float()
                src, dst = edge_index
                x = self.node_embedding(x)
                edge_attr = self.edge_embedding(edge_attr)

                cross_edge_index, cross_edge_attr, cross_edge_sh, angle_emb = self.setup_cross_graph(data)
                
                cross_src, cross_dst = cross_edge_index
                cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

                if angle_emb is not None:
                    angle_emb = self.cross_angle_embedding(angle_emb)
                    cross_edge_attr = cross_edge_attr + angle_emb


                for i in range(self.n_conv_layers):
                    edge_attr_ = torch.cat([edge_attr, x[src, :self.n_s], x[dst, :self.n_s]], dim=-1)
                    x_intra_update = self.conv_layers[i](x, edge_index, edge_attr_, edge_sh)

                    cross_edge_attr_ = torch.cat([cross_edge_attr, x[cross_src, :self.n_s], x[cross_dst, :self.n_s]], dim=-1)
                    x_inter_update = self.cross_conv_layers[i](x, cross_edge_index, cross_edge_attr_, cross_edge_sh)

                    x = F.pad(x, (0, x_intra_update.shape[-1] - x.shape[-1]))
                    x = x + x_intra_update + x_inter_update
                    
            else:
                data.node_sigma_emb = self.timestep_emb_fn(data.node_t_tr)
                x, edge_index, edge_attr, edge_sh = self.setup_distance_graph(data)
                edge_sh = edge_sh.float()
                src, dst = edge_index
                x = self.node_embedding(x)
                edge_attr = self.edge_embedding(edge_attr)

                for i in range(self.n_conv_layers):
                    edge_attr_ = torch.cat([edge_attr, x[src, :self.n_s], x[dst, :self.n_s]], dim=-1)
                    x_update = self.conv_layers[i](x, edge_index, edge_attr_, edge_sh)

                    

                    x = F.pad(x, (0, x_update.shape[-1] - x.shape[-1]))
                    x = x + x_update
        

        else:
            raise NotImplementedError


        if self.confidence_mode:
            graph_representation = scatter(x[:, :self.n_s], data.batch, dim=0, reduce="mean")
            confidence = self.confidence_predictor(graph_representation).squeeze(-1)
            return confidence
            
        
        center_edge_index, center_edge_attr, center_sh = self.setup_center_graph(data)
        center_src, _ = center_edge_index
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr_ = torch.cat([center_edge_attr, x[center_src, :self.n_s]], dim=-1)
        
        if self.separate_rot_sphere_conv and self.sphere_diffusion and self.no_rot_first_lig:
            if self.no_sphere_first_lig:
                graph_type_sphere = 'rot'
            else: 
                graph_type_sphere = None
            if self.anchor_graph_sphere:
                center_edge_index_sphere, center_edge_attr_sphere, center_sh_sphere = self.setup_anchor_graph(data,  graph_type=graph_type_sphere)
            else:
                center_edge_index_sphere, center_edge_attr_sphere, center_sh_sphere = self.setup_center_graph(data, graph_type=graph_type_sphere)
            center_src_sphere, _ = center_edge_index_sphere
            center_edge_index_rot, center_edge_attr_rot, center_sh_rot = self.setup_center_graph(data, graph_type='rot')
            try:
                center_edge_attr_sphere = self.center_edge_embedding_sphere(center_edge_attr_sphere)
                center_edge_attr_rot = self.center_edge_embedding_rot(center_edge_attr_rot)
            except:
                center_edge_attr_sphere = self.center_edge_embedding(center_edge_attr_sphere)
                center_edge_attr_rot = self.center_edge_embedding(center_edge_attr_rot)

            center_edge_attr_sphere_ = torch.cat([center_edge_attr_sphere, x[center_src_sphere, :self.n_s]], dim=-1)

            
            center_src_rot, _ = center_edge_index_rot
            
            center_edge_attr_rot_ = torch.cat([center_edge_attr_rot, x[center_src_rot, :self.n_s]], dim=-1)
            out_nodes_sphere = (data.mask_first_lig == False).sum().item() if self.no_sphere_first_lig else data.agent_center_pos.size(0)
            global_pred_sphere = self.final_conv_sphere(x, center_edge_index_sphere, center_edge_attr_sphere_, center_sh_sphere,
                                        out_nodes=out_nodes_sphere)

    
            
            global_pred_rot = self.final_conv_rot(x, center_edge_index_rot, center_edge_attr_rot_, center_sh_rot,
                                        out_nodes= (data.mask_first_lig == False).sum().item() )
            

            rot_pred = global_pred_rot[:, :3] + global_pred_rot[:, 3:6]


            if self.restrict_rot_update:
                _, tor_edge_index_anchors, tor_edge_attr_anchors, tor_edge_sh_anchors = self.build_bond_conv_graph(data, use_anchors=True)
                if self.debug:
                    print("tor_edge_index_anchors", tor_edge_index_anchors.shape, tor_edge_index_anchors)
                    print("tor_edge_attr_anchors", tor_edge_attr_anchors.shape, tor_edge_attr_anchors)
                tor_bond_vec_anchors = (data.pos[data.anchor_mask][data.mask_first_lig == False] ) / 2
                tor_bond_attr_anchors =  x[data.anchor_mask][data.mask_first_lig == False] #TODO x[tor_bonds[1]]

                tor_bonds_sh_anchors = o3.spherical_harmonics("2e", tor_bond_vec_anchors, normalize=True, normalization='component')
                tor_edge_sh_anchors = self.final_tp_tor_anchors(tor_edge_sh_anchors.to(DEVICE).float(), tor_bonds_sh_anchors[tor_edge_index_anchors[0]].to(DEVICE).float())
                tor_edge_attr_anchors = torch.cat([tor_edge_attr_anchors, x[tor_edge_index_anchors[1], :self.n_s],
                                    tor_bond_attr_anchors[tor_edge_index_anchors[0], :self.n_s]], -1)
                rot_pred = self.tor_bond_conv_anchors(x, 
                                        tor_edge_index_anchors[[1, 0], :], 
                                        tor_edge_attr_anchors, 
                                        tor_edge_sh_anchors, 
                                    out_nodes= (data.mask_first_lig == False).sum().item(), #data.anchor_mask.sum(),
                                    aggr='mean')
                rot_pred = self.tor_final_layer_anchors(rot_pred).squeeze(1)

            if int((data.mask_first_lig == False).sum().item())==0:  
                rot_pred=torch.empty(1,3).to(DEVICE)
                
            sphere_new_pred = global_pred_sphere[:, :3] + global_pred_sphere[:, 3:6]
           
            if self.sphere_projection:
                
                if self.no_sphere_first_lig:
                    mask = (data.mask_first_lig == False)  
                    data_agent_center_pos = data.agent_center_pos[mask]
                else:
                    data_agent_center_pos = data.agent_center_pos

                pred_norm_original = torch.norm(sphere_new_pred, dim=-1)
                agent_center_pos_normalized = data_agent_center_pos / (torch.norm(data_agent_center_pos, dim=1, keepdim=True) + 1e-7)

                # Step 1: Project sphere_new_pred onto the plane perpendicular to agent_center_pos
                dot_product = torch.sum(sphere_new_pred * agent_center_pos_normalized, dim=1, keepdim=True)
                sphere_new_pred = sphere_new_pred - dot_product * agent_center_pos_normalized

                dot_products = torch.sum(sphere_new_pred * data_agent_center_pos, dim=1)
                
            

            tr_pred =  torch.zeros(1, dtype=torch.float, device=DEVICE)
        
        else:
            center_edge_index, center_edge_attr, center_sh = self.setup_center_graph(data)
            center_src, _ = center_edge_index
            center_edge_attr = self.center_edge_embedding(center_edge_attr)
            center_edge_attr_ = torch.cat([center_edge_attr, x[center_src, :self.n_s]], dim=-1)
            global_pred = self.final_conv(x, center_edge_index, center_edge_attr_, center_sh,
                                      out_nodes=data.agent_center_pos.size(0))
            if self.sphere_diffusion:
                sphere_new_pred = global_pred[:, :3] + global_pred[:, 6:9]
                tr_pred = None
            else:
                tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
                sphere_new_pred = None
            
            rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]


        if self.sphere_diffusion:
            if self.use_bl_metal_graph:
                
                center_edge_index, center_edge_attr, center_sh = self.setup_metal_bl_graph(data)

            bl_global_pred = self.final_conv_radial(x, center_edge_index, center_edge_attr_, center_sh,
                                      out_nodes=data.agent_center_pos.size(0))
            bl_pred = bl_global_pred[:,0] + bl_global_pred[:,1]
        else:
            bl_pred = None

            

        agent_sigma_emb = self.timestep_emb_fn(data.t_tr)
        


        
        
        if not self.restrict_rot_update:
            rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
            if (data.mask_first_lig == False).sum().item() > 0:
                rot_scale = self.rot_final_layer(torch.cat([rot_norm, agent_sigma_emb[data.mask_first_lig == False]], dim=1)) if self.no_rot_first_lig else self.rot_final_layer(torch.cat([rot_norm, agent_sigma_emb], dim=1))
            else:
                rot_scale = torch.zeros_like(rot_norm) 
            rot_pred = (rot_pred / (rot_norm+10e-6)) * rot_scale
        
        
        if self.sphere_diffusion:
            
            bl_pred = self.bl_final_layer(torch.cat([bl_pred.unsqueeze(1), agent_sigma_emb], dim=1))
            sphere_norm = torch.linalg.vector_norm(sphere_new_pred, dim=1).unsqueeze(1)
        
            if self.no_sphere_first_lig:
               
                if (data.mask_first_lig == False).sum().item() > 0:
                    sphere_scale =  self.tr_final_layer(torch.cat([sphere_norm, agent_sigma_emb[data.mask_first_lig == False]], dim=1))

                else:
                    sphere_scale = torch.zeros_like(sphere_norm) 

            else:
                sphere_scale =  self.tr_final_layer(torch.cat([sphere_norm, agent_sigma_emb], dim=1))


            sphere_new_pred = (sphere_new_pred / (sphere_norm+1e-6)) * sphere_scale
            
        else:
            tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
            tr_scale = self.tr_final_layer(torch.cat([tr_norm, agent_sigma_emb], dim=1))
            tr_pred = (tr_pred / tr_norm) * tr_scale
    
        

           
        if self.scale_by_sigma:
            tr_sigma, rot_sigma, tor_sigma, sphere_sigma, bl_sigma = self.t_to_sigma_fn(data.t_tr, data.t_rot, data.t_tor, data.t_sphere, data.t_bl)

            if self.sphere_diffusion:
                bl_pred = bl_pred / bl_sigma.unsqueeze(1)
                if self.predict_x0_sphere:
                    sphere_new_pred = sphere_new_pred / torch.linalg.norm(sphere_new_pred, dim=-1, keepdim=True)
                else:
                    sphere_new_pred = sphere_new_pred * n_sphere_angle.score_norm(sphere_sigma[data.mask_first_lig == False].cpu()).to(data.agent_center_pos.device).unsqueeze(1) if self.no_sphere_first_lig else  sphere_new_pred * n_sphere_angle.score_norm(sphere_sigma.cpu()).to(data.agent_center_pos.device).unsqueeze(1)
                
            else:
                tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            if self.restrict_rot_update:
                rot_pred = rot_pred * torch.sqrt(torch.tensor(torus.score_norm(rot_sigma[data.mask_first_lig == False].cpu().numpy())).float()
                                             .to(data.agent_center_pos.device))
                
            else:
                rot_pred = rot_pred * so3.score_norm(rot_sigma[data.mask_first_lig == False].cpu()).unsqueeze(1).to(DEVICE) if self.no_rot_first_lig else rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(DEVICE)


        

        if self.no_torsion or data.edge_mask.sum()==0:
            return tr_pred, rot_pred, torch.zeros(1, dtype=torch.float, device=DEVICE), sphere_new_pred, bl_pred
  



        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data) 
        tor_bond_vec = data.pos[tor_bonds[1]] - data.pos[tor_bonds[0]]
        tor_bond_attr = x[tor_bonds[0]] + x[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh.to(DEVICE).float(), tor_bonds_sh[tor_edge_index[0]].to(DEVICE).float())

        tor_edge_attr = torch.cat([tor_edge_attr, x[tor_edge_index[1], :self.n_s],
                                   tor_bond_attr[tor_edge_index[0], :self.n_s]], -1)

        tor_pred = self.tor_bond_conv(x, tor_edge_index[[1, 0], :], tor_edge_attr, tor_edge_sh, #TODO check the [[1, 0], :]
                                  out_nodes=data.edge_mask.sum(), 
                                  aggr='mean')
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        batch = torch.zeros(data.pos.size(0), dtype=torch.long, device=DEVICE) if data.batch is None else data.batch
        
        edge_sigma = tor_sigma[batch][data.lig_bonds_edge_index[0]][data.edge_mask] 

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data.agent_center_pos.device))
        
 

        return tr_pred, rot_pred, tor_pred, sphere_new_pred, bl_pred

 

    def setup_graph(self, data):
        data.node_sigma_emb = self.timestep_emb_fn(data.node_t_tr)
        x = torch.cat([data.x, data.node_sigma_emb], dim=-1).to(DEVICE)
        if self.edge_fdim>0:
            edge_index = torch.cat([data.edge_index, data.bond_index], dim=1)
        else: 
            edge_index = data.edge_index
        src, dst = edge_index
        edge_vec = data.pos[src.long()] - data.pos[dst.long()]

        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data.node_sigma_emb[dst.long()]
        if self.edge_fdim>0:
            bond_attr = torch.cat([
                data.bond_attr,
                torch.zeros(data.edge_index.shape[-1], self.edge_fdim, device=data.x.device)
            ], 0)
            edge_attr = torch.cat([bond_attr, edge_sigma_emb, edge_length_emb], dim=-1).to(DEVICE)
        else: 
            edge_attr = torch.cat([ edge_sigma_emb, edge_length_emb], dim=-1).to(DEVICE)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalization='component', normalize=True)

        return x, edge_index, edge_attr, edge_sh

    def setup_cross_graph(self, data):
        cross_edge_index = data.cross_edge_index

        cross_src, cross_dst = cross_edge_index
        edge_vec = data.pos[cross_src.long()] - data.pos[cross_dst.long()]

        edge_length_emb = self.cross_dist_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data.node_sigma_emb[cross_dst.long()]

        
        
        if self.angle_encoding_repr_learning:
            angle_emb = torch.zeros(edge_length_emb.size(0), self.angular_smearing.mu.size(0), device=edge_length_emb.device)
            is_anchor = data.anchor_mask[cross_src.long()] 
            if is_anchor.any():
                angles = compute_angles_pairwise(data.pos[cross_src[is_anchor].long()], data.pos[cross_dst[is_anchor].long()])
                angle_emb[is_anchor] = self.angular_smearing(angles)
                
            
        else:
            angle_emb = None
        cross_edge_attr = torch.cat([edge_length_emb, edge_sigma_emb], dim=-1).to(DEVICE)
        cross_edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalization='component', normalize=True)

        return cross_edge_index, cross_edge_attr, cross_edge_sh, angle_emb
    
    def setup_center_graph(self, data,  graph_type=None):
        if hasattr(data, 'center_src'):
            center_src = data.center_src.unsqueeze(0)
        else:
            raise ValueError("center_src must have been computed before.")
        center_dst = data.agent_membership.unsqueeze(0)
       
        if graph_type == 'rot':
            valid_mask = torch.where(data.mask_first_lig == False)[0]
            matches = (center_dst.unsqueeze(-1) == valid_mask).any(-1)
            valid_indices = torch.nonzero(matches, as_tuple=True)[1]
            center_src = center_src[:, valid_indices]
            center_dst = center_dst[:, valid_indices]
            unique_values, inverse_indices = torch.unique(center_dst, return_inverse=True)
            center_dst = inverse_indices
        edge_index = torch.cat([center_src, center_dst], dim=0)
        src, dst = edge_index

        edge_vec = data.pos[src.long()] - data.agent_center_pos[dst.long()]
        edge_length_emb = self.center_dist_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data.node_sigma_emb[src.long()]

        edge_attr = torch.cat([edge_length_emb, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def setup_metal_bl_graph(self, data, graph_type=None):
        """
        Sets up the BL graph by connecting the center (last non-player agent)
        to every unique batch index without duplicates, and includes self-connections.
        """
        # Compute the source indices for the last non-player agent
        last_non_player_mask = data.last_non_player_mask.bool()
        
        source = torch.nonzero(last_non_player_mask, as_tuple=True)[0]
        # Unique batch indices
        unique_batches = torch.unique(data.batch)

        # Generate center_src and center_dst with no duplicates
        center_src, center_dst = [], []
        for batch_idx in unique_batches:
            for src_idx in source:
                center_src.append(src_idx.item())
                center_dst.append(batch_idx.item())

        # Convert to tensors
        center_src = torch.tensor(center_src, dtype=torch.long, device=data.pos.device)
        center_dst = torch.tensor(center_dst, dtype=torch.long, device=data.pos.device)

        # Add self-connections for unique `center_dst`
        self_connections_src = center_dst
        self_connections_dst = center_dst

        # Combine connections
        edge_index = torch.stack([center_src,center_dst], dim=0)

        # Compute edge vectors
        src, dst = edge_index
        edge_vec = data.pos[src.long()] - data.agent_center_pos[dst.long()]

        # Encode edge lengths using the adjusted Gaussian range
        edge_length_emb = self.bl_dist_expansion(edge_vec.norm(dim=-1))

        # Encode edge attributes
        edge_sigma_emb = data.node_sigma_emb[src.long()]

        # Combine embeddings
        edge_attr = torch.cat([edge_length_emb, edge_sigma_emb], dim=1)

        # Compute spherical harmonics for the edge vectors
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization='component'
        )

        return edge_index, edge_attr, edge_sh


    def setup_anchor_graph(self, data, graph_type=None):
        # Ensure anchor indices are properly defined
        anchor_indices = torch.where(data.anchor_mask)[0].to(data.pos.device)  # Move to same device as data.pos
        if anchor_indices.numel() == 0:
            raise ValueError("No anchors found in anchor_mask.")

        
            
        num_anchors = anchor_indices.size(0)

        # Create edges for anchors
        src = anchor_indices.repeat(num_anchors)
        dst = torch.arange(num_anchors, device=data.pos.device).repeat_interleave(num_anchors)

        # Add center_src and center_dst
        center_src = data.center_src.unsqueeze(0)
        center_dst = data.agent_membership.unsqueeze(0)

        if graph_type == 'rot':
            raise NotImplementedError
            

        combined_src = torch.cat([src, center_src.squeeze()])
        combined_dst = torch.cat([dst, center_dst.squeeze()])

        # Stack into edge_index
        edge_index = torch.stack([combined_src, combined_dst], dim=0).to(data.pos.device)

        # Compute edge vectors
        edge_vec = data.pos[combined_src.long()] - data.agent_center_pos[combined_dst.long()]

        # Filter edges where the source is from `src` (not `center_src`)
        is_anchor_src = combined_src.unsqueeze(1) == src.unsqueeze(0)
        is_anchor_src = is_anchor_src.any(dim=1)  # True for edges originating from `src`

        angular_smearing_output_dim = self.angular_smearing.mu.size(0)


        angle_emb = torch.zeros(edge_vec.size(0), angular_smearing_output_dim, device=data.pos.device) 
        angle_emb[is_anchor_src] = self.angular_smearing(
            compute_angles_pairwise(data.pos[combined_src[is_anchor_src].long()], 
                                    data.agent_center_pos[combined_dst[is_anchor_src].long()])
        )
        
        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))
        # Retrieve sigma embedding
        edge_sigma_emb = data.node_sigma_emb[combined_src.long()]
        # Combine edge attributes
        edge_attr = torch.cat([angle_emb, edge_length_emb, edge_sigma_emb], dim=-1).to(data.pos.device)

        # Compute spherical harmonics
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh






    def build_bond_conv_graph(self, data, use_anchors=False):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        
        if use_anchors:
            anchor_pos = data.pos[data.anchor_mask][data.mask_first_lig == False] #if no_rot_first_lig else data.anchor_mask]
            
            
            origin = torch.zeros_like(anchor_pos[0], device=anchor_pos.device)
            bond_pos = (anchor_pos + origin) / 2
            bond_batch = torch.zeros(anchor_pos.size(0), dtype=torch.long, device=anchor_pos.device) if data.batch is None else data.batch[data.anchor_mask][data.mask_first_lig == False]

            batch_x = torch.zeros(data.pos.size(0), dtype=torch.long, device=anchor_pos.device) if data.batch is None else data.batch
            bonds=None

        else:
            bonds = data.lig_bonds_edge_index[:, data.edge_mask].long()
            bond_pos = (data.pos[bonds[0]] + data.pos[bonds[1]]) / 2
        

            bond_batch = torch.zeros_like(bonds[0], dtype=torch.long, device=DEVICE) if data.batch is None else data.batch[bonds[0]]
            batch_x = torch.zeros(data.pos.size(0), dtype=torch.long, device=DEVICE) if data.batch is None else data.batch
    
        edge_index = radius(data.pos, bond_pos, self.max_radius, batch_x=batch_x, batch_y=bond_batch)
        edge_vec = data.pos[edge_index[1]] - bond_pos[edge_index[0]]

        edge_attr = self.dist_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        print("use_anchors",use_anchors)
        print("edge_index", edge_index.shape)
        print("edge_vec", edge_vec.shape)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh, 
    

    def setup_distance_graph(self, data):
        """
        Build a radius-based graph for the input data.
        If separate_intra_inter_agent_updates is False, the graph combines all nodes
        without differentiating between intra- and inter-agent updates.
        """

        # Extract node positions and compute the radius graph
        pos = data.pos  # Combined node positions
        edge_index_rad = radius(
            pos,  # Source positions
            pos,  # Destination positions
            r=self.max_radius,  # Maximum radius threshold
            batch_x=data.batch, 
            batch_y=data.batch 
        )
        if self.edge_fdim>0:
            edge_index = torch.cat([edge_index_rad, data.bond_index], dim=1)
        else: 
            edge_index = edge_index_rad
        # Compute edge vectors
        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]

        # Embed distances using Gaussian smearing
        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))

        # Embed sigma (timestep) features for edges
        node_sigma_emb = data.node_sigma_emb
        edge_sigma_emb = node_sigma_emb[dst]  # Sigma embedding for destination nodes

        # Combine edge attributes
        if self.edge_fdim>0:
            bond_attr = torch.cat([
                data.bond_attr,
                torch.zeros(edge_index_rad.shape[-1], self.edge_fdim, device=data.x.device)
            ], 0)
            edge_attr = torch.cat([bond_attr, edge_sigma_emb, edge_length_emb], dim=-1).to(DEVICE)
        else: 
            edge_attr = torch.cat([ edge_sigma_emb, edge_length_emb], dim=-1).to(DEVICE)
        #edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], dim=-1).to(data.pos.device)

        # Compute spherical harmonics for edge vectors
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalization='component', normalize=True
        )

        # Combine node features with sigma embeddings
        x = torch.cat([data.x, node_sigma_emb], dim=-1).to(data.pos.device)

        return x, edge_index, edge_attr, edge_sh
