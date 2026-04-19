import os
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


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
    return G

def draw_molecule(data, title=None, save_path=None):
    mol = pyg_to_mol(data)

    img = Draw.MolToImage(mol)

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def pyg_to_mol(data):
    mol = Chem.RWMol()

    # ---- 1. Add atoms ----
    # NOTE: assumes x is one-hot or categorical
    for i in range(data.x.shape[0]):
        atom_type = int(data.x[i].argmax().item())

        # fallback mapping (VERY common in datasets like yours)
        # adjust if you know exact encoding
        atomic_num = atom_type if atom_type > 0 else 6  # default carbon

        mol.AddAtom(Chem.Atom(atomic_num))

    # ---- 2. Add bonds ----
    bond_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC
    }

    added = set()

    for i in range(data.edge_index.shape[1]):
        src = int(data.edge_index[0, i])
        dst = int(data.edge_index[1, i])

        # avoid duplicate edges (undirected graph)
        if (dst, src) in added:
            continue
        added.add((src, dst))

        bond_type_id = int(data.edge_attr[i].argmax().item())
        bond_type = bond_map.get(bond_type_id, Chem.BondType.SINGLE)

        try:
            mol.AddBond(src, dst, bond_type)
        except:
            pass

    return mol.GetMol()