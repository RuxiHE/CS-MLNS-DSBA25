# Need to check the output path


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score, accuracy_score
import networkx as nx
from tqdm import tqdm
import csv
import os

class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='GCN'):
        super(GNNLinkPredictor, self).__init__()
        
        # Select the GNN layer type
        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, out_channels)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Link prediction layers
        self.link_predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        # Get node features for the edges
        src, dst = edge_index
        
        # Combine the node embeddings for each edge
        return self.link_predictor(torch.cat([z[src], z[dst]], dim=1))
    
    def forward(self, x, edge_index, predict_edges=None):
        # Encode node features
        z = self.encode(x, edge_index)
        
        # If we're given specific edges to predict on, use those
        if predict_edges is not None:
            return self.decode(z, predict_edges)
        
        # Otherwise predict on the input edges
        return self.decode(z, edge_index)

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    node_info_path = os.path.join(base_path, "node_information.csv")
    train_path = os.path.join(base_path, "train.txt")
    test_path = os.path.join(base_path, "test.txt")

    # Load node features
    node_features = {}
    with open(node_info_path, "r") as f:
        for line in f:
            values = line.strip().split(',') 
            node_id = int(values[0])
            features = np.array([float(x) for x in values[1:]])
            node_features[node_id] = features
    
    # Find the maximum node ID to determine feature matrix size
    max_node_id = max(node_features.keys())
    feature_size = len(next(iter(node_features.values())))
    
    # Create feature matrix
    X = np.zeros((max_node_id + 1, feature_size))
    for node_id, features in node_features.items():
        X[node_id] = features
    
    # Load training edges
    train_edges = []
    train_labels = []
    with open(train_path , "r") as f:
        for line in f:
            values = line.strip().split()
            node1 = int(values[0])
            node2 = int(values[1])
            label = int(values[2])
            train_edges.append((node1, node2))
            train_labels.append(label)
    
    # Load test edges
    test_edges = []
    with open(test_path , "r") as f:
        for line in f:
            values = line.strip().split()
            node1 = int(values[0])
            node2 = int(values[1])
            test_edges.append((node1, node2))
    
    return X, train_edges, train_labels, test_edges, max_node_id

def prepare_pyg_data(X, train_edges, train_labels, max_node_id):
    # Convert to torch tensors
    x = torch.FloatTensor(X)
    
    # Create positive and negative edge indices
    pos_edge_index = []
    neg_edge_index = []
    
    for edge, label in zip(train_edges, train_labels):
        if label == 1:
            pos_edge_index.append(edge)
        else:
            neg_edge_index.append(edge)
    
    pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long).t()
    neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long).t()
    
    # Create a PyG Data object
    data = Data(x=x, num_nodes=max_node_id+1)
    
    # Split positive edges for training and validation
    data.train_pos_edge_index = pos_edge_index
    data.train_neg_edge_index = neg_edge_index
    
    # Create a combined edge index for message passing
    edge_index = torch.cat([pos_edge_index, pos_edge_index.flip(0)], dim=1)  # Make it undirected
    data.edge_index = edge_index
    
    return data

def train(model, data, optimizer, batch_size=64):
    model.train()
    
    # Create positive and negative examples
    pos_edge_index = data.train_pos_edge_index
    neg_edge_index = data.train_neg_edge_index
    
    # Shuffle edges
    pos_perm = torch.randperm(pos_edge_index.size(1))
    neg_perm = torch.randperm(neg_edge_index.size(1))
    
    pos_edge_index = pos_edge_index[:, pos_perm]
    neg_edge_index = neg_edge_index[:, neg_perm]
    
    # Limit to the smaller set's size
    size = min(pos_edge_index.size(1), neg_edge_index.size(1))
    pos_edge_index = pos_edge_index[:, :size]
    neg_edge_index = neg_edge_index[:, :size]
    
    # Process in batches to save memory
    total_loss = 0
    num_batches = (size + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        optimizer.zero_grad()
        
        # Get batch slice
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, size)
        
        pos_batch = pos_edge_index[:, start:end]
        neg_batch = neg_edge_index[:, start:end]
        
        # Create labels: 1 for positive edges, 0 for negative edges
        pos_y = torch.ones(pos_batch.size(1), 1, device=pos_batch.device)
        neg_y = torch.zeros(neg_batch.size(1), 1, device=neg_batch.device)
        
        # Get predictions
        pos_pred = model(data.x, data.edge_index, pos_batch)
        neg_pred = model(data.x, data.edge_index, neg_batch)
        
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_pred, neg_pred], dim=0),
            torch.cat([pos_y, neg_y], dim=0)
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / num_batches

def evaluate(model, data, edge_index, labels):
    model.eval()
    
    with torch.no_grad():
        # Make predictions
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, edge_index)
        pred = torch.sigmoid(out).cpu().numpy()
        
        # Calculate metrics
        auc = roc_auc_score(labels, pred)
        
        # Get binary predictions
        pred_labels = (pred > 0.5).astype(int)
        acc = accuracy_score(labels, pred_labels)
        
    return auc, acc, pred

def predict_on_test(model, data, test_edges, output_file="gnn_predictions.csv"):
    model.eval()

    if output_file is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(base_path, "gnn_predictions.csv")
    
    # Convert test edges to tensor
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t()
    
    with torch.no_grad():
        # Make predictions
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, test_edge_index)
        pred_probs = torch.sigmoid(out).cpu().numpy().flatten()
        
        # Get binary predictions
        predictions = (pred_probs > 0.5).astype(int)
        
        # Save predictions
        with open(output_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Predicted"])
            for i, pred in enumerate(predictions):
                writer.writerow([i, int(pred)])
        
        print(f"Predictions saved to {output_file}")

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    X, train_edges, train_labels, test_edges, max_node_id = load_data()
    
    print("Preparing PyG data...")
    data = prepare_pyg_data(X, train_edges, train_labels, max_node_id)
    data = data.to(device)
    
    # Split train data into train and validation
    train_ratio = 0.8
    train_size = int(len(train_labels) * train_ratio)
    
    # Create validation set
    val_edges = train_edges[train_size:]
    val_labels = train_labels[train_size:]
    val_edge_index = torch.tensor(val_edges, dtype=torch.long).t().to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.float).to(device)
    
    # Model parameters
    in_channels = data.num_features
    hidden_channels = 128
    out_channels = 64
    
    # Initialize model
    model_type = 'SAGE'  # Options: 'GCN', 'SAGE', 'GAT'
    model = GNNLinkPredictor(in_channels, hidden_channels, out_channels, model_type=model_type).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Training loop
    best_val_auc = 0
    best_epoch = 0
    patience = 10
    counter = 0
    

    # 設定模型儲存路徑
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "best_gnn_model.pt")
    

    print(f"Training {model_type} model...")
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data, optimizer)
        
        # Validation
        val_auc, val_acc, _ = evaluate(model, data, val_edge_index, val_labels.cpu().numpy())
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"Best model at epoch {best_epoch} with validation AUC: {best_val_auc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(model_path))
    
    # Make predictions on test data
    print("Making predictions on test data...")
    test_edge_tensor = torch.tensor(test_edges, dtype=torch.long).to(device)
    predict_on_test(model, data, test_edge_tensor)
    
    print("Done!")

if __name__ == "__main__":
    main()