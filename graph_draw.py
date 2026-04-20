import networkx as nx

from rdkit import Chem
from rdkit.Chem import rdchem

# mapping file
from utils.mapping_conf import ATOM, EDGE

NODE_COLOR = {
    "C": "grey",
    "O": "red",
    "N": "blue",
    "Br": "brown",
    "Cl": "green",
    "S": "yellow",
    "P": "orange",
    "F": "lightgreen",
    "I": "purple"
}


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    G = nx.Graph()

    # nodes = atoms
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetSymbol())

    # edges = bonds
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            label=str(bond.GetBondType())
        )

    # layout
    pos = nx.kamada_kawai_layout(G)

    # colors
    node_colors = [
        NODE_COLOR.get(G.nodes[v]['label'], "gray")
        for v in G.nodes
    ]

    # draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(
        G,
        pos,
        labels={v: G.nodes[v]['label'] for v in G.nodes}
    )

    return G

def atomic_num_to_symbol(num):
    return Chem.PeriodicTable.GetElementSymbol(
        Chem.GetPeriodicTable(), num
    )
def decode_atom(dataset_name, feat_idx):
    atomic_num = ATOM[dataset_name][feat_idx]
    return atomic_num_to_symbol(atomic_num)
def decode_bond(dataset_name, edge_idx):
    return EDGE[dataset_name].get(edge_idx, rdchem.BondType.SINGLE)

def pyg_to_nx(data, dataset_name="Mutagenicity"):

    G = nx.Graph()

    for i in range(data.x.shape[0]):
        feat_idx = data.x[i].argmax().item()
        atomic_num = ATOM[dataset_name][feat_idx]
        atom_symbol = atomic_num_to_symbol(atomic_num)

        G.add_node(i, label=atom_symbol)

    for src, dst in data.edge_index.t().tolist():
        G.add_edge(src, dst)

    return G