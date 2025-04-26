import argparse

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def inference_on_unseen_users(test_graph_path, model_path, out_file_path):

    start = time.time()
    
    print(f"Loading Data ...")

   # -------------------- Data Loading --------------------
    user_feats    = np.load(f'{test_graph_path}/user_features.npy')    # (m, d1)
    product_feats = np.load(f'{test_graph_path}/product_features.npy') # (p, d2)
    edges         = np.load(f'{test_graph_path}/user_product.npy').astype(int)  # (e,2)

    # reindex products if needed
    prod_offset = int(edges[:,1].min())
    edges[:,1] -= prod_offset

    device = torch.device('cuda')

    X_u = torch.from_numpy(user_feats).float().to(device)    # (m, d1)
    X_p = torch.from_numpy(product_feats).float().to(device) # (p, d2)

    m, d1 = X_u.shape
    p, d2 = X_p.shape

    users_idx = torch.tensor(edges[:,0], dtype=torch.long, device=device)
    prods_idx = torch.tensor(edges[:,1], dtype=torch.long, device=device)

    # BGCN normalization
    deg_u = torch.zeros(m, device=device)
    deg_p = torch.zeros(p, device=device)
    for u,pid in edges:
        deg_u[u]   += 1
        deg_p[pid] += 1
    deg_u.clamp_(min=1); deg_p.clamp_(min=1)
    norm = 1.0 / torch.sqrt(deg_u[users_idx] * deg_p[prods_idx])  # (e,)
        
    # -------------------- Helper: segment_softmax --------------------
    def segment_softmax(src, idx, num_nodes, eps=1e-6):
        """
        src: (E, H) unnormalized scores
        idx: (E,)   destination node indices in [0..num_nodes)
        returns:    (E, H) softmaxed per-group
        """
        E, H = src.shape

        # 1) max per group
        # initialize to -inf so reduce “amax” works
        max_vals = torch.full((num_nodes, H), float('-inf'), device=src.device)
        # scatter_reduce_ in place
        max_vals.scatter_reduce_(0,
            idx.unsqueeze(1).expand(-1, H),
            src,
            reduce='amax',
            include_self=False
        )
        # 2) subtract max, exp
        src_exp = torch.exp(src - max_vals[idx])

        # 3) sum per group
        sum_vals = torch.zeros_like(max_vals)
        sum_vals.scatter_reduce_(0,
            idx.unsqueeze(1).expand(-1, H),
            src_exp,
            reduce='sum',
            include_self=False
        )
        # 4) normalize
        return src_exp / (sum_vals[idx] + eps)

    # -------------------- Multi‐Head Bipartite GAT Layer --------------------
    class BGCNGATLayer(nn.Module):
        def __init__(self, in_u, in_p, out, heads=4, dropout=0.2):
            super().__init__()
            assert out % heads == 0, "out must be divisible by heads"
            self.heads   = heads
            self.out_h   = out // heads
            # linear projections
            self.W_u = nn.Linear(in_u, heads*self.out_h, bias=False)
            self.W_p = nn.Linear(in_p, heads*self.out_h, bias=False)
            # attention parameters: one vector per head
            self.a   = nn.Parameter(torch.Tensor(heads, 2*self.out_h))
            # residual projections
            self.res_u = nn.Linear(in_u, heads*self.out_h, bias=False)
            self.res_p = nn.Linear(in_p, heads*self.out_h, bias=False)
            # layer norms
            self.ln_u  = nn.LayerNorm(heads*self.out_h)
            self.ln_p  = nn.LayerNorm(heads*self.out_h)
            self.leaky = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(dropout)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.W_u.weight)
            nn.init.xavier_uniform_(self.W_p.weight)
            nn.init.xavier_uniform_(self.a.view(self.heads, -1))
            nn.init.xavier_uniform_(self.res_u.weight)
            nn.init.xavier_uniform_(self.res_p.weight)

        def forward(self, X_u, X_p, u_idx, p_idx, norm):
            # 1) project and reshape for heads
            h_u = self.W_u(X_u).view(-1, self.heads, self.out_h)  # (m,H,Ho)
            h_p = self.W_p(X_p).view(-1, self.heads, self.out_h)  # (p,H,Ho)

            # 2) gather per-edge (e,H,Ho)
            hu = h_u[u_idx]  # from-users
            hp = h_p[p_idx]  # from-products

            # 3) compute raw attention
            cat = torch.cat([hu, hp], dim=-1)                   # (e,H,2*Ho)
            e   = self.leaky((cat * self.a.unsqueeze(0)).sum(dim=-1))  # (e,H)

            # 4) softmax per destination
            alpha_u = segment_softmax(e, u_idx, m)  # for p→u
            alpha_p = segment_softmax(e, p_idx, p)  # for u→p

            # 5) compute messages
            # expand norm to (E, 1, 1) so it broadcasts over heads and features
            norm3 = norm.view(-1, 1, 1)  # shape (E,1,1)

            m_p = hp * alpha_u.unsqueeze(-1) * norm3  # now (E,H,Ho)
            m_u = hu * alpha_p.unsqueeze(-1) * norm3


            # 6) aggregate
            H_u = torch.zeros(m, self.heads, self.out_h, device=X_u.device)
            H_p = torch.zeros(p, self.heads, self.out_h, device=X_p.device)
            H_u = H_u.index_add(0, u_idx, m_p)
            H_p = H_p.index_add(0, p_idx, m_u)

            # 7) combine heads
            H_u = H_u.view(-1, self.heads*self.out_h)
            H_p = H_p.view(-1, self.heads*self.out_h)

            # 8) residual + LayerNorm + activation + dropout
            res_u = self.res_u(X_u)
            res_p = self.res_p(X_p)
            H_u = self.ln_u(H_u + res_u)
            H_p = self.ln_p(H_p + res_p)
            H_u = self.dropout(F.relu(H_u))
            H_p = self.dropout(F.relu(H_p))

            return H_u, H_p

    # -------------------- Full Model --------------------
    class BGCNGAT(nn.Module):
        def __init__(self, d1, d2, hidden, num_classes,
                    heads=4, num_layers=2, dropout=0.2):
            super().__init__()
            layers = []
            # first: raw dims → hidden
            layers.append(BGCNGATLayer(d1, d2, hidden, heads, dropout))
            # hidden → hidden
            for _ in range(num_layers-1):
                layers.append(BGCNGATLayer(hidden, hidden, hidden, heads, dropout))
            self.layers = nn.ModuleList(layers)
            self.classifier = nn.Linear(hidden, num_classes)

        def forward(self, X_u, X_p, u_idx, p_idx, norm):
            h_u, h_p = X_u, X_p
            for layer in self.layers:
                h_u, h_p = layer(h_u, h_p, u_idx, p_idx, norm)
            return self.classifier(h_u)

    print(f"Loading model...")
    # -------------------- Load the model --------------------  
    # 1) Load state dict with weights_only to suppress warning
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    # 2) Reinstantiate and load
    model = BGCNGAT(d1, d2,
                    hidden=96,
                    num_classes=9,
                    heads=4,
                    num_layers=3,
                    dropout=0.3).to(device)
    model.load_state_dict(state_dict)
     
    print(f"Inference...")
   # -------------------- Inference & Save --------------------
    model.eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(
            model(X_u, X_p, users_idx, prods_idx, norm)
        ).cpu().numpy()

    final_preds = (final_probs >= 0.5).astype(int)

    np.savetxt(out_file_path, final_preds, fmt='%d', delimiter=',')
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
