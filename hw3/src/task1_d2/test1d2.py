import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GINConv
from torch_geometric.data import Data
from torch_geometric.utils import degree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    OneCycleLR
)
import time

def load_data(path):
    node_features = np.load(f"{path}/node_feat.npy")
    edges = np.load(f"{path}/edges.npy")
    return node_features, edges


# --------------------
# 2) MODEL DEFINITION
# --------------------
class EnhancedGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.4):
        super().__init__()
        self.dropout = dropout
        self.layers  = torch.nn.ModuleList()
        self.layers.append(SAGEConv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            if i % 2 == 1:
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dims[i], hidden_dims[i])
                )
                self.layers.append(GINConv(mlp))
            else:
                self.layers.append(SAGEConv(hidden_dims[i-1], hidden_dims[i]))
        self.final = SAGEConv(hidden_dims[-1], output_dim)
        self.batch_norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(d) for d in hidden_dims]
        )

    def forward(self, data):
        x, edge_idx = data.x, data.edge_index
        for i, layer in enumerate(self.layers):
            x_res = x
            x = layer(x, edge_idx)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0 and x.shape == x_res.shape:
                x = x + x_res
        x = self.final(x, edge_idx)
        return F.log_softmax(x, dim=1)
    

def inference_on_unseen_users(test_graph_path, model_path, out_file_path):
    start = time.time()
    # --------------------
    # 1) SETUP
    # --------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    node_features, edges = load_data(test_graph_path)
    unique = 5

    edge_index = torch.tensor(edges.T, dtype=torch.long)
    x_raw      = torch.tensor(node_features, dtype=torch.float)
    row, col   = edge_index
    deg        = degree(col, x_raw.size(0), dtype=x_raw.dtype)
    deg_norm   = (deg - deg.mean()) / (deg.std() + 1e-8)
    x          = torch.cat([x_raw, deg_norm.unsqueeze(1)], dim=1)

    data = Data(x=x, edge_index=edge_index).to(device)


    input_dim  = x.shape[1]
    output_dim = 18
    epochs     = 1000
    patience   = 100

    model_configs = [
        {
            # Model 1: SAGE+GIN, Adam
            "hidden_dims": [128, 256, 256, 128],
            "optimizer": torch.optim.Adam,
            "optim_kwargs": {"lr": 5e-3, "weight_decay": 5e-4},
            "scheduler": ReduceLROnPlateau,
            "sched_kwargs": {"mode": "min", "factor": 0.5, "patience": 15}
        },
        {
            # Model 3: SAGE+GIN, AdamW + One-Cycle
            "hidden_dims": [64, 128, 256, 128],
            "optimizer": torch.optim.AdamW,
            "optim_kwargs": {"lr": 1e-3, "weight_decay": 1e-4},
            "scheduler": OneCycleLR,
            "sched_kwargs": {
                "max_lr": 1e-2,
                "total_steps": epochs,
                "pct_start": 0.1,
                "anneal_strategy": "cos"
            }
        }
    ]

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    states = ckpt['model_states']

    loaded_models = []
    for cfg, sd in zip(model_configs, states):
        m = EnhancedGNN(input_dim, cfg["hidden_dims"], output_dim).to(device)
        m.load_state_dict(sd)
        m.eval()
        loaded_models.append(m)

    with torch.no_grad():
        ensemble_logits = sum(m(data).exp() for m in loaded_models) / len(loaded_models)
        ensemble_probs  = ensemble_logits.cpu().numpy()
    
    ensemble_probs = np.exp(ensemble_probs)
    ensemble_pred  = ensemble_probs.argmax(axis=1).astype(int)

    # Save full‐graph probabilities
    np.savetxt(out_file_path, ensemble_pred, fmt="%d", delimiter=',')
    print(f"Predictions saved to {out_file_path}")
    print("Saved predictions. Inference time: {:.1f}s".format(time.time()-start))


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process test graph, model, and output file paths.")
    parser.add_argument('--testGraphPath', type=str, required=True, help="Path to the test graph file")
    parser.add_argument('--modelPath', type=str, required=True, help="Path to the model file")
    parser.add_argument('--outFile', type=str, required=True, help="Path to the output file")

    # Parse arguments
    args = parser.parse_args()

    # Store the parsed arguments in variables
    test_graph_path = args.testGraphPath
    model_path = args.modelPath
    out_file_path = args.outFile

    # Print the paths
    print(f"Inputs are :")
    print(f"Test Graph Path: <{test_graph_path}>")
    print(f"Model Path: <{model_path}>")
    print(f"Output File Path: <{out_file_path}>")

    inference_on_unseen_users(test_graph_path, model_path, out_file_path)
