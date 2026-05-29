import numpy as np
import networkx as nx

def find_metal_node(G, metal_center_element):
    print(metal_center_element)
    for node, data in G.nodes(data=True):
        if data.get('node_label') == metal_center_element:
            G.nodes[node]['is_bound_atom'] = True
            return node, np.array(G.nodes[node]['node_position'])
    return None, None

def translate_atoms_to_metal_center(G, metal_position):
    for node, data in G.nodes(data=True):
        current_position = np.array(data['node_position'])
        G.nodes[node]['node_position'] = (current_position - metal_position).tolist()

def disconnect_metal_node(G, node_to_remove):
    connected_nodes = list(G.neighbors(node_to_remove))
    G.remove_edges_from(list(G.edges(node_to_remove)))

    nx.set_node_attributes(G, 0, 'is_connected_to_metal')
    nx.set_node_attributes(G, False, 'is_anchor')
    nx.set_node_attributes(G, False, 'is_bound_atom')

    for node in connected_nodes:
        G.nodes[node]['is_connected_to_metal'] = 1
        G.nodes[node]['is_bound_atom'] = True

    return 

def extract_subgraphs(G, node_to_remove):
    connected_components = list(nx.connected_components(G))
    return [G.subgraph(component).copy() for component in connected_components]

def clean_and_reorder_graphs(separate_graphs, node_to_remove):
    metal_graph_index = next((i for i, graph in enumerate(separate_graphs) if node_to_remove in graph), None)

    if metal_graph_index is not None and metal_graph_index != len(separate_graphs) - 1:
        separate_graphs.append(separate_graphs.pop(metal_graph_index))
    subgraphs_to_remove = [
        graph for graph in separate_graphs[:-1]
        if not any(data['is_bound_atom'] == True for _, data in graph.nodes(data=True))
    ]
    for sg in subgraphs_to_remove:
        separate_graphs.remove(sg)

    assert separate_graphs[-1].number_of_nodes() == 1


    return separate_graphs