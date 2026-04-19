# In this file, we will import a pretrained GNN and use MAGE class to train an explanation model.

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from utils.model import GCN
from utils.utils import sanitize_smiles, to_smiles
from torch_geometric.data import DataLoader
from new_mage import MAGE
import argparse
import numpy as np

#=============================
import os
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
#=============================

# COULD REMOVE ALL OF THIS AND SIMPLY USE THE SMILES FILES THAT GENERATE FROM sample.py

# Create an argument parser
parser = argparse.ArgumentParser(description='Train target model')
parser.add_argument('--data_name', type=str, default='Mutagenicity', help='Name of the dataset')
parser.add_argument('--input_channels', type=int, default=14, help='Number of input channels')
parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
parser.add_argument('--output_channels', type=int, default=2, help='Number of output channels')
parser.add_argument('--target_model', type=str, default='checkpoints/models/Mutagenicity_model.pth', help='Path to the pretrained GNN model')
parser.add_argument('--dataset', type=str, default='checkpoints/datasets/Mutagenicity.pt', help='Path to the dataset')
parser.add_argument('--label', type=int, default=0, help='Label of the data')



# Parse the arguments
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# Initialize the GNN model
model = GCN(input_channels=args.input_channels, hidden_channels=args.hidden_channels, output_channels=args.output_channels).to(device)

# Load the pretrained GNN model
model.load_state_dict(torch.load(args.target_model))
model.eval()

# Load the dataset from checkpoints
dataset = torch.load(args.dataset)

#==============================================
print(type(dataset))
print(dataset[0])

from graph_SMILES import draw_molecule, pyg_to_mol

#draw_molecule(dataset[5], title="chem from dataset")
#==============================================


new_dataset = []
count = 0
prob = 0
smiles_set = []
for data in dataset:
    batch = torch.zeros(data.num_nodes, dtype=torch.long).to(device)
    pred = model(data.x.to(device), data.edge_index.to(device), batch=batch)
    smiles = to_smiles(data, data_name=args.data_name)
    smiles = sanitize_smiles(smiles)
    if not smiles:
        continue
    smiles_set.append(smiles)
    if pred.argmax().item() == args.label:
        if pred.softmax(1)[0][args.label].item() > 0.9:
            new_dataset.append(data)
            count += 1
            prob += pred.softmax(1)[0][args.label].item()

# Initialize the Mage class
mage = MAGE(gnn=model, model=model, dataset=new_dataset, whole_dataset=dataset, smiles_set=smiles_set, data_name=args.data_name, add_H=False, label=args.label, hidden_channels=args.hidden_channels, output_channels=args.output_channels, device=device)

path_dict = {
    'T_encoder': f'checkpoints/models/{args.data_name}_label_{args.label}_T_encoder.pth', 
    'pred_node_topo': f'checkpoints/models/{args.data_name}_label_{args.label}_pred_node_topo.pth', 
    'pred_node_label': f'checkpoints/models/{args.data_name}_label_{args.label}_pred_node_label.pth', 
    'linear_topo': f'checkpoints/models/{args.data_name}_label_{args.label}_linear_topo.pth', 
    'linear_label': f'checkpoints/models/{args.data_name}_label_{args.label}_linear_label.pth', 
    'T_mean': f'checkpoints/models/{args.data_name}_label_{args.label}_T_mean.pth', 
    'T_var': f'checkpoints/models/{args.data_name}_label_{args.label}_T_var.pth'}

mage.load(path_dict)

sampled_data, pred_prob, invalid_count, tree_acc = mage.sample(100, max_iter=5)

mean = np.mean(pred_prob)
std = np.std(pred_prob)

#============================================================================
SMILES_path = f'sampled_data/{args.data_name}_label_{args.label}_SMILES.txt'
os.makedirs(os.path.dirname(SMILES_path), exist_ok=True)

with open(SMILES_path, 'w') as f:
    for data in sampled_data:
        f.write(f'{data}\n')

prob_path = f'sampled_data/{args.data_name}_label_{args.label}_prob.txt'
os.makedirs(os.path.dirname(SMILES_path), exist_ok= True)

with open(prob_path, 'w') as f:
    for data in pred_prob:
        f.write(f'{data}\n')


from graph_SMILES import smiles_to_graph

graphs = []
index = 0
for data in sampled_data:
    graph = smiles_to_graph(data)
    graphs.append(graph)

    if graph is None:
        continue

    graphs_path = f"sampled_data/graphs/{args.data_name}_label_{args.label}_graph_{index}.png"

    os.makedirs(os.path.dirname(graphs_path), exist_ok=True)

    # draw graph
    pos = nx.spring_layout(graph)
    labels = nx.get_node_attributes(graph, "label")

    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=500)

    # save image
    plt.savefig(graphs_path, bbox_inches="tight")
    plt.close()

    index +=1

    if index == 20:
        break

#============================================================================================