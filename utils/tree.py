
import torch
from torch.utils.data import Dataset


def _undirected_edges(edge_index: torch.Tensor):
    """Return unique undirected edges from a COO edge_index tensor."""
    seen = set()
    edges = []
    if edge_index.numel() == 0:
        return edges
    for src, dst in edge_index.t().tolist():
        if src == dst:
            continue
        key = (src, dst) if src < dst else (dst, src)
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)
    return edges


def _node_signature(data, node_idx: int):
    """Build a compact, hashable motif key for a node-centric motif."""
    feat = data.x[node_idx]
    if feat.dim() == 0:
        feat_id = int(feat.item())
    elif feat.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        feat_id = int(feat.view(-1)[0].item())
    else:
        feat_id = int(torch.argmax(feat).item())

    degree = 0
    if data.edge_index.numel() > 0:
        degree = int((data.edge_index[0] == node_idx).sum().item())
    return (feat_id, degree)


class Tree:
    """Generic graph-to-tree adaptor.

    Each node is treated as a motif, and tree edges mirror graph connectivity.
    This avoids any chemistry-specific dependencies while keeping MAGE's API.
    """

    def __init__(self, data, data_name, add_H=False):
        self.data = data
        self.data_name = data_name
        self.add_H = add_H
        self.y = data.y

    def transform(self):
        num_nodes = int(self.data.num_nodes)
        self.fragments = [_node_signature(self.data, i) for i in range(num_nodes)]
        self.atom_list = [[i] for i in range(num_nodes)]
        self.bond_list = _undirected_edges(self.data.edge_index)

class TreeDataset(Dataset):
    def __init__(self, trees):
        self.trees = trees

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):
        tree = self.trees[idx]
        # Process the tree data as needed, e.g., converting to tensor
        idx = torch.tensor([idx], dtype=torch.long)
        x = tree.x
        edge_index = tree.edge_index
        return x, edge_index, idx