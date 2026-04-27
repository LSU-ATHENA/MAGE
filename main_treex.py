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
import sys
from pathlib import Path


TREEX_REPO = Path("/home/kgmills/TreeX").resolve()
if str(TREEX_REPO) not in sys.path:
    sys.path.insert(0, str(TREEX_REPO))

from Utils.utils import load_model, check_task, detect_exp_setting, detect_motif_nodes
from Utils.datasets import get_dataset


# Create an argument parser
parser = argparse.ArgumentParser(description='Train target model')
parser.add_argument('--data_name', type=str, default='PTC_FM', help='Name of the dataset')
parser.add_argument('--input_channels', type=int, default=18, help='Number of input channels')
parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
parser.add_argument('--output_channels', type=int, default=2, help='Number of output channels')
parser.add_argument('--target_model', type=str, default='checkpoints/models/PTC_FM_model.pth', help='Path to the pretrained GNN model')
parser.add_argument('--dataset', type=str, default='checkpoints/datasets/PTC_FM.pt', help='Path to the dataset')
parser.add_argument('--label', type=int, default=1, help='Label of the data')
parser.add_argument('--gnn', type=str, default="gin", help='Type of GNN')
parser.add_argument('--clusters', type=int, default=3)
parser.add_argument('--into_st', type=int, default=6)
parser.add_argument('--local_cluster', type=str, default='kmeans', choices=['kmeans', 'em'])
parser.add_argument('--lmda', type=float, default=1)

# Parse the arguments
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# Initialize the GNN model
#model = GCN(input_channels=args.input_channels, hidden_channels=args.hidden_channels, output_channels=args.output_channels).to(device)

# Load the pretrained GNN model
#model.load_state_dict(torch.load(args.target_model))
#model.eval()

# Load the dataset from checkpoints
#dataset = torch.load(args.dataset)

dataname = args.data_name
print(dataname)
task_type = check_task(dataname)
dataset = get_dataset(dataname)
try:dataset.print_summary()
except AttributeError: pass

try:n_fea, n_cls = dataset.num_features, dataset.num_classes 
except AttributeError: n_fea, n_cls = dataset.num_features, 2
explain_ids = detect_exp_setting(dataname, dataset)
motif_nodes_number = detect_motif_nodes(dataname)
gnn_model = load_model(dataname, args.gnn, n_fea, n_cls)
gnn_model.eval()
print(f"GNN Model Loaded. {dataname}, {task_type}. \nsize of Motif: {motif_nodes_number}. num of samples to explain: {len(explain_ids)}")
print(f'Dataset={dataname}-{args.gnn}-lclus{args.clusters}-gclus{args.into_st}-{args.local_cluster}-{args.lmda}')

model = gnn_model
model.forward = model.forward_mage

count = 0
prob = 0
new_dataset = []
smiles_set = []
for data in dataset:
    batch = torch.zeros(data.num_nodes, dtype=torch.long).to(device)
    pred = model(data.x.to(device), data.edge_index.to(device), batch=batch)
    smiles = to_smiles(data, data_name=args.data_name)
    smiles = sanitize_smiles(smiles)
    smiles_set.append(smiles)
    if pred.softmax(1)[0][args.label].item() > 0.9:
        new_dataset.append(data)
        
        count += 1
        prob += pred.softmax(1)[0][args.label].item()

# Initialize the Mage class
mage = MAGE(gnn=model, model=model, dataset=new_dataset, whole_dataset=dataset, smiles_set=smiles_set, data_name=args.data_name, add_H=False, label=args.label, hidden_channels=args.hidden_channels, output_channels=args.output_channels, device=device)

# Train a teacher encoder
#mage.train_t_encoder(epochs=300, lr=0.0001, batch_size=32, save_path=f'checkpoints/models/{args.data_name}_label_{args.label}_T_encoder.pth')
mage.train_t_encoder(epochs=300, lr=0.0001, batch_size=32, save_path=f'checkpoints/models/treex_{args.data_name}_label_{args.label}_T_encoder.pth')

path_dict = {
    'T_encoder': f'checkpoints/models/treex_{args.data_name}_label_{args.label}_T_encoder.pth', 
    'pred_node_topo': f'checkpoints/models/treex_{args.data_name}_label_{args.label}_pred_node_topo.pth', 
    'pred_node_label': f'checkpoints/models/treex_{args.data_name}_label_{args.label}_pred_node_label.pth', 
    'linear_topo': f'checkpoints/models/treex_{args.data_name}_label_{args.label}_linear_topo.pth', 
    'linear_label': f'checkpoints/models/treex_{args.data_name}_label_{args.label}_linear_label.pth', 
    'T_mean': f'checkpoints/models/treex_{args.data_name}_label_{args.label}_T_mean.pth', 
    'T_var': f'checkpoints/models/treex_{args.data_name}_label_{args.label}_T_var.pth'}

#mage.train(epochs=100, batch_size=4, lr=0.001, max_iter=5, path_dict=path_dict, t_encoder_path=f'checkpoints/models/{args.data_name}_label_{args.label}_T_encoder.pth')
mage.train(epochs=100, batch_size=4, lr=0.001, max_iter=5, path_dict=path_dict, t_encoder_path=f'checkpoints/models/treex_{args.data_name}_label_{args.label}_T_encoder.pth')
