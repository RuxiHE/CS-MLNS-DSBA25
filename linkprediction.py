import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import csv
import os

# Load data
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    node_info_path = os.path.join(base_path, "node_information.csv")
    train_path = os.path.join(base_path, "train.txt")
    test_path = os.path.join(base_path, "test.txt")
    
    # Load node features
    node_features = {}
    df = pd.read_csv(node_info_path, header=None)
    for index, row in df.iterrows():
        node_id = int(row[0])
        features = row[1:].values.astype(float)
        node_features[node_id] = features
    
    # Load training edges
    train_edges = []
    train_labels = []
    with open(train_path, "r") as f:
        for line in f:
            values = line.strip().split()
            node1 = int(values[0])
            node2 = int(values[1])
            label = int(values[2])
            train_edges.append((node1, node2))
            train_labels.append(label)
    
    # Load test edges
    test_edges = []
    with open(test_path, "r") as f:
        for line in f:
            values = line.strip().split()
            node1 = int(values[0])
            node2 = int(values[1])
            test_edges.append((node1, node2))
    
    return node_features, train_edges, train_labels, test_edges

# Construct graph from training edges
def build_graph(train_edges, train_labels):
    G = nx.Graph()
    
    # Add all nodes that appear in training data
    nodes = set()
    for edge in train_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    G.add_nodes_from(nodes)
    
    # Add edges with positive labels only
    for edge, label in zip(train_edges, train_labels):
        if label == 1:
            G.add_edge(edge[0], edge[1])
    
    return G

# Feature engineering
def extract_features(G, node_features, edge_list):
    features = []
    
    print("Extracting features for each edge...")
    for src, dst in tqdm(edge_list):
        feature_vector = []
        
        # Node feature similarity (cosine similarity)
        if src in node_features and dst in node_features:
            src_feat = node_features[src]
            dst_feat = node_features[dst]
            dot_product = np.dot(src_feat, dst_feat)
            norm_src = np.linalg.norm(src_feat)
            norm_dst = np.linalg.norm(dst_feat)
            
            if norm_src > 0 and norm_dst > 0:
                cosine_sim = dot_product / (norm_src * norm_dst)
            else:
                cosine_sim = 0
                
            feature_vector.append(cosine_sim)
            
            # Jaccard similarity of features (treating them as binary)
            src_binary = (src_feat > 0).astype(int)
            dst_binary = (dst_feat > 0).astype(int)
            
            intersection = np.sum(np.logical_and(src_binary, dst_binary))
            union = np.sum(np.logical_or(src_binary, dst_binary))
            
            jaccard_sim = intersection / union if union > 0 else 0
            feature_vector.append(jaccard_sim)
        else:
            feature_vector.extend([0, 0])  # Default if node features are missing
        
        # Graph topological features
        if src in G and dst in G:
            # Common neighbors
            common_neighbors = list(nx.common_neighbors(G, src, dst))
            cn_count = len(common_neighbors)
            feature_vector.append(cn_count)
            
            # 取得鄰居列表
            src_neighbors = set(G.neighbors(src))
            dst_neighbors = set(G.neighbors(dst))
            src_degree = len(src_neighbors)
            dst_degree = len(dst_neighbors)
            
            # Jaccard coefficient
            jaccard_coef = 0
            if src_degree > 0 and dst_degree > 0:
                jaccard_coef = cn_count / (src_degree + dst_degree - cn_count)
            feature_vector.append(jaccard_coef)
            
            # Preferential attachment
            pref_attachment = src_degree * dst_degree
            feature_vector.append(pref_attachment)
            
            # Adamic-Adar index
            aa_index = 0
            for neighbor in common_neighbors:
                neighbor_degree = len(list(G.neighbors(neighbor)))
                if neighbor_degree > 1:  # Avoid log(1) = 0 division
                    aa_index += 1.0 / np.log(neighbor_degree)
            feature_vector.append(aa_index)
            
            # Resource allocation index
            ra_index = 0
            for neighbor in common_neighbors:
                neighbor_degree = len(list(G.neighbors(neighbor)))
                if neighbor_degree > 0:
                    ra_index += 1.0 / neighbor_degree
            feature_vector.append(ra_index)
            
            # Try to compute shortest path
            try:
                shortest_path = nx.shortest_path_length(G, src, dst)
                feature_vector.append(shortest_path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                feature_vector.append(-1)
        else:
            # Default values if nodes aren't in the graph
            feature_vector.extend([-1, -1, -1, -1, -1, -1])
            
        # Node degree features
        src_degree = len(list(G.neighbors(src))) if src in G else 0
        dst_degree = len(list(G.neighbors(dst))) if dst in G else 0
        feature_vector.append(src_degree)
        feature_vector.append(dst_degree)
        feature_vector.append(abs(src_degree - dst_degree))
        feature_vector.append(src_degree + dst_degree)
        
        features.append(feature_vector)
    
    return np.array(features)

def train_model(train_features, train_labels):
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model...")
    model.fit(train_features_scaled, train_labels)
    
    # Evaluate on training data
    train_preds = model.predict(train_features_scaled)
    accuracy = accuracy_score(train_labels, train_preds)
    precision = precision_score(train_labels, train_preds)
    recall = recall_score(train_labels, train_preds)
    f1 = f1_score(train_labels, train_preds)
    
    print(f"Training - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return model, scaler

def predict_and_save(model, scaler, test_features, output_file="predictions.csv"):
    # Scale test features
    test_features_scaled = scaler.transform(test_features)
    
    # Make predictions
    print("Making predictions on test data...")
    test_preds = model.predict(test_features_scaled)
    
    # Save predictions
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Predicted"])
        for i, pred in enumerate(test_preds):
            writer.writerow([i, int(pred)])
    
    print(f"Predictions saved to {output_file}")

def main():
    print("Loading data...")
    node_features, train_edges, train_labels, test_edges = load_data()
    
    print("Building graph...")
    G = build_graph(train_edges, train_labels)
    
    print("Extracting features...")
    train_features = extract_features(G, node_features, train_edges)
    test_features = extract_features(G, node_features, test_edges)
    
    print("Training model...")
    model, scaler = train_model(train_features, train_labels)
    
    print("Making predictions...")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.csv")
    predict_and_save(model, scaler, test_features, output_file=output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()