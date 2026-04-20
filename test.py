import torch
import matplotlib.pyplot as plt
import networkx as nx
from graph_draw import smiles_to_graph, pyg_to_nx

#dataset test
print("======dataset test======")
data = torch.load("checkpoints/datasets/Mutagenicity.pt")
print(type(data))
print("Entry Example = ", data[0])
G = pyg_to_nx(data[0])

print(G)

plt.figure(figsize=(8, 8))

pos = nx.kamada_kawai_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=300)
nx.draw_networkx_edges(G, pos, width=1)

labels = {n: G.nodes[n]['label'] for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=labels)

plt.axis("off")
plt.show()