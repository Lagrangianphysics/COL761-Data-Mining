import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GINConv
from torch_geometric.data import Data
from torch_geometric.utils import degree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    OneCycleLR
)
import time




def load_data(path):
    node_features = np.load(f"{path}/node_feat.npy")
    edges         = np.load(f"{path}/edges.npy")
    labels        = np.load(f"{path}/label.npy")
    train_mask    = ~np.isnan(labels)
    return node_features, edges, labels, train_mask

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
        self.final      = SAGEConv(hidden_dims[-1], output_dim)
        self.batch_norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(d) for d in hidden_dims]
        )

    def forward(self, data):
        x, edge_idx = data.x, data.edge_index
        for i, layer in enumerate(self.layers):
            x_res = x
            x     = layer(x, edge_idx)
            x     = self.batch_norms[i](x)
            x     = F.relu(x)
            x     = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0 and x.shape == x_res.shape:
                x = x + x_res
        x = self.final(x, edge_idx)
        return F.log_softmax(x, dim=1)


def get_model(train_graph_path , out_model_path):
    
    # --------------------
    # 1) SETUP
    # --------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    node_features, edges, labels, train_mask_np = load_data(train_graph_path)
    unique, counts = np.unique(labels[train_mask_np], return_counts=True)
    print(f"Classes: {list(zip(unique, counts))}")

    edge_index = torch.tensor(edges.T, dtype=torch.long)
    x_raw      = torch.tensor(node_features, dtype=torch.float)
    row, col   = edge_index
    deg        = degree(col, x_raw.size(0), dtype=x_raw.dtype)
    deg_norm   = (deg - deg.mean()) / (deg.std() + 1e-8)
    x          = torch.cat([x_raw, deg_norm.unsqueeze(1)], dim=1)

    y               = torch.tensor(labels, dtype=torch.float)
    y[torch.isnan(y)] = -1
    train_mask      = torch.tensor(train_mask_np, dtype=torch.bool)

    all_train_idx = train_mask.nonzero(as_tuple=True)[0]
    train_idx, val_idx = train_test_split(
        all_train_idx.cpu().numpy(),
        test_size=0.15,
        random_state=42
    )
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx   = torch.tensor(val_idx, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y).to(device)

    # --------------------
    # 3) ENSEMBLE CONFIG
    # --------------------
    input_dim  = x.shape[1]
    output_dim = len(unique)
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

    models, optimizers, schedulers = [], [], []
    for cfg in model_configs:
        m = EnhancedGNN(input_dim, cfg["hidden_dims"], output_dim).to(device)
        opt = cfg["optimizer"](m.parameters(), **cfg["optim_kwargs"])
        sch = cfg["scheduler"](opt, **cfg["sched_kwargs"])
        models.append(m)
        optimizers.append(opt)
        schedulers.append(sch)


    # --------------------
    # 4) TRAIN ENSEMBLE
    # --------------------
    criterion = torch.nn.CrossEntropyLoss()
    print("→ Training ensemble …")
    t0 = time.time()

    for idx, (model, opt, sch) in enumerate(zip(models, optimizers, schedulers), 1):
        best_acc = 0.0
        best_sd  = None
        wait     = 0

        for epoch in range(1, epochs+1):
            model.train()
            opt.zero_grad()
            out = model(data)
            loss = criterion(out[train_idx], data.y[train_idx].long())
            loss.backward()
            opt.step()

            # scheduler step
            if isinstance(sch, ReduceLROnPlateau):
                sch.step(loss)
            else:
                sch.step()

            # validation
            model.eval()
            with torch.no_grad():
                val_probs = out[val_idx].exp()
                val_pred = val_probs.argmax(dim=1)
                val_acc = accuracy_score(data.y[val_idx].long().cpu(), val_pred.cpu())

            if val_acc > best_acc:
                best_acc = val_acc
                best_sd  = model.state_dict()
                wait     = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"  • Model {idx}: early stop @ epoch {epoch}, best ACC={best_acc:.4f}")
                    break

            if epoch % 10 == 0:
                print(f"  • Model {idx} | Epoch {epoch} | Loss {loss:.4f} | Val ACC {val_acc:.4f}")

        # restore best
        model.load_state_dict(best_sd)

    print(f"Training completed in {(time.time()-t0):.1f}s")


    # --------------------
    # 5) SAVE ENSEMBLE
    # --------------------
    torch.save({
        'model_states': [m.state_dict() for m in models]
    }, out_model_path)
    print(f"✅ Ensemble saved to {out_model_path}")

    # --------------------
    # Compute ensemble AUC on validation
    # --------------------
    ckpt = torch.load(out_model_path, map_location=device, weights_only=True)
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

    # 1. Compute val_probs and val_pred with NumPy
    val_probs = np.exp(ensemble_probs[val_idx])
    val_pred  = val_probs.argmax(axis=1)
    y_true = data.y[val_idx].long().cpu().numpy()
    
    # 3. Compute accuracy
    final_acc = accuracy_score(y_true, val_pred)
    print(f"Ensemble Validation ACC: {final_acc:.4f}")
    


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a model using the provided training graph.")
    parser.add_argument('--trainGraphPath', type=str, required=True, help="Path to the training graph file.")
    parser.add_argument('--outModel_path', type=str, required=True, help="Path to save the output model file.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Extract arguments
    train_graph_path = args.trainGraphPath
    out_model_path = args.outModel_path
    
    print(f"Inputs are :")
    print(f"Training graph path: <{train_graph_path}>")
    print(f"Output model path: <{out_model_path}>")

    get_model(train_graph_path , out_model_path)
