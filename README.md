# Predicting Missing Links in an Actor Co-Occurrence Network

This repository contains our approach to the **MLNS Kaggle Challenge**, where we predict missing links in an actor co-occurrence network. The dataset consists of a graph where **nodes represent actors** and **edges indicate co-occurrence** on the same Wikipedia page. In addition to the graph structure, **each actor is associated with processed textual features** extracted from their Wikipedia page.

Our goal is to **reconstruct the original network** by leveraging both graph-theoretical and node feature information.

## 📂 Project Structure

- **`gnn.ipynb`** – Experiments with Graph Neural Networks (GNNs), including GCN, GAT, and TransformerConv.
- **`rf.ipynb`** – Implementation of the Random Forest model, which ultimately achieved the **best AUC score**.
- **`Model_combination.ipynb`** – Exploration of model fusion techniques, including meta-classifiers and weighted averaging.

## 🚀 Approach & Findings

### **1️⃣ Graph Neural Networks (GNNs)**
We initially experimented with **GCN, GAT, and TransformerConv**. While GCN performed the best among them, deeper GNNs suffered from **over-smoothing** in sparse graphs, and attention-based models struggled due to weak node feature signals.

### **2️⃣ Random Forest (Best Model)**
Given the high-dimensional textual features and sparse nature of the graph, **Random Forest** proved to be the most effective. It efficiently captured non-linear relationships between actors, achieving an **AUC of 0.7702**.

### **3️⃣ Model Combination**
We explored methods to combine **GCN and Random Forest**, including:
- **Meta-classifiers** (logistic regression on model outputs)
- **Confidence-weighted averaging**
- **Rule-based merging**
- **Simple averaging**

The **meta-classifier approach** achieved an **AUC of 0.7677**, but **Random Forest alone remained the most effective model**.

## 📈 Results Summary

| Model | AUC Score |
|--------|----------|
| **Random Forest (Best Model)** | **0.7702** |
| Meta-classifier (GCN + RF) | 0.7677 |
| GCN | 0.71 |
| GraphSAGE | 0.68 |
| GAT | 0.65 |

## 📌 Key Takeaways
✔️ **Graph-based heuristics (Jaccard, Adamic-Adar) were ineffective** due to the scale-free nature of the network.  
✔️ **Text features alone were not strong predictors** but complemented structural features.  
✔️ **GNNs struggled with sparsity**, making classical machine learning models like Random Forest more effective.  
✔️ **The best-performing model was Random Forest (AUC 0.7702).**

## 📌 Future Work
- Explore **graph augmentation** techniques to improve GNN performance.
- Investigate **more advanced fusion models** combining GNNs and classical ML.
- Experiment with **graph embeddings** for better representation learning.

## 👥 Team
- **Ruxi He**
- **Piangpim Chancharunee**
- **Chiao-Kai Chiang**
- **I-Hsun Lu**
