import os
import sys
import networkx as nx
import torch_geometric
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, mean_squared_error, f1_score, fbeta_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_recall_fscore_support, fbeta_score
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import fbeta_score, average_precision_score
from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def read_data(file_path, criteria):
    data = pd.read_excel(file_path)
    user_id = data['User_ID']
    item_id = data['Items_ID']
    user_id_map = {uid: i for i, uid in enumerate(user_id.unique())}
    item_id_map = {mid: i for i, mid in enumerate(item_id.unique())}
    num_users = len(user_id_map)
    num_items = len(item_id_map)
    num_criteria = len(criteria)
    base_ground_truth_ratings = np.zeros((num_users, num_items, num_criteria), dtype=np.int32)

    for i, row in data.iterrows():
        uid = row['User_ID']
        mid = row['Items_ID']
        criterion_ratings = [row[criterion] for criterion in criteria]
        if uid in user_id_map and mid in item_id_map:
            user_idx = user_id_map[uid]
            item_idx = item_id_map[mid]
            base_ground_truth_ratings[user_idx, item_idx] = criterion_ratings

    return user_id_map, item_id_map, base_ground_truth_ratings

def create_bipartite_graph(file_path, criteria):
    data = pd.read_excel(file_path)
    G = nx.MultiGraph()

    users = set()
    items = set()

    for uid in data['User_ID']:
        G.add_node(uid, bipartite=0)
        users.add(uid)

    for mid in data['Items_ID']:
        G.add_node(mid, bipartite=1)
        items.add(mid)

    for i in range(len(data)):
        uid = data['User_ID'][i]
        mid = data['Items_ID'][i]

        for criterion in criteria:
            rating = data[criterion][i]

            if rating > 0:
                G.add_edge(uid, mid, criterion=criterion, weight=rating)

    print(f"Number of user nodes: {len(users)}")
    print(f"Number of item nodes: {len(items)}")

    user_item_edges = [(u, v, data) for u, v, data in G.edges(data=True) if u in users and v in items]
    print(f"Number of edges between user and item nodes: {len(user_item_edges)}")

    for u, v, data in G.edges(data=True):
        if u in users and v in items and 'criterion' in data and 'weight' in data:
            user_id = u
            item_id = v
            criterion = data['criterion']
            rating = data['weight']  # Use the correct attribute name

            # print(f"Edge between User_ID {user_id} and Items_ID {item_id} (Criterion: {criterion}):")
            # print(f"  Weight (Rating): {rating}")

    return G

def create_subgraphs(file_path, criteria):
    graph_data = pd.read_excel(file_path)
    subgraphs = []

    for criterion in criteria:
        subgraph = nx.Graph()
        subgraphs.append(subgraph)

    for i in range(len(graph_data)):
        uid = graph_data['User_ID'][i]
        mid = graph_data['Items_ID'][i]

        for criterion, subgraph in zip(criteria, subgraphs):
            rating = graph_data[criterion][i]

            if rating > 0:
                subgraph.add_node(uid, bipartite=0)
                subgraph.add_node(mid, bipartite=1)
                subgraph.add_edge(uid, mid, weight=rating)

    for criterion, subgraph in zip(criteria, subgraphs):
        # print(f"\nSubgraph for Criterion {criterion}:")
        is_bipartite = nx.is_bipartite(subgraph)
        # print(f"Is bipartite: {is_bipartite}")

        user_nodes = [node for node in subgraph.nodes() if subgraph.nodes[node]['bipartite'] == 0]
        item_nodes = [node for node in subgraph.nodes() if subgraph.nodes[node]['bipartite'] == 1]
        # print(f"Number of user nodes: {len(user_nodes)}")
        # print(f"Number of item nodes: {len(item_nodes)}")

        subgraph_edges = [(u, v, data) for u, v, data in subgraph.edges(data=True)]
        # print(f"Number of edges in subgraph: {len(subgraph_edges)}")

def create_and_normalize_adjacency_matrices(file_path, criteria, user_ids, item_ids):
    graph_data = pd.read_excel(file_path)
    bgnn_matrices = []  # Initialize a list to store the BGNN matrices for each criterion

    user_id_to_index = {}
    user_index_to_id = {}
    item_id_to_index = {}
    item_index_to_id = {}

    for criterion in criteria:
        subgraph = nx.Graph()

        for i in range(len(graph_data)):
            uid = graph_data['User_ID'][i]
            mid = graph_data['Items_ID'][i]

            rating = graph_data[criterion][i]

            if rating > 0:
                subgraph.add_node(uid, bipartite=0)
                subgraph.add_node(mid, bipartite=1)
                subgraph.add_edge(uid, mid, weight=rating)

        n_nodes = len(subgraph.nodes())
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int32)

        for uid, mid, data in subgraph.edges(data=True):
            uid_idx = list(subgraph.nodes()).index(uid)
            mid_idx = list(subgraph.nodes()).index(mid)
            adj_matrix[uid_idx][mid_idx] = data['weight']
            adj_matrix[mid_idx][uid_idx] = data['weight']

        # # Print the matrix
        # print(f"\nMatrix for criterion '{criterion}':")
        # print(adj_matrix)

        # # Count zero and non-zero cells in the matrix and print the results
        # zero_cells = np.sum(adj_matrix == 0)
        # non_zero_cells = np.sum(adj_matrix != 0)
        # print(f"\nMatrix for criterion '{criterion}' has {zero_cells} zero cells and {non_zero_cells} non-zero cells.")

        # Calculate the degree matrices DC_uv and DC_vu as before
        DC_uv = np.diag(np.sum(adj_matrix, axis=1))
        DC_vu = np.diag(np.sum(adj_matrix, axis=0))

        # Normalize the matrix using the degree matrices
        BC_uv_norm = np.linalg.pinv(DC_uv) @ adj_matrix
        BC_vu_norm = adj_matrix @ np.linalg.pinv(DC_vu)

        # Average the two normalized matrices to get a single normalized matrix
        normalized_matrix = (BC_uv_norm + BC_vu_norm) / 2.0

        # Convert the normalized matrix to the format of BGNN (Block Graph Neural Network)
        n = normalized_matrix.shape[0] // 2  # Assuming the matrix is of the form (0 Bu; Bv 0)
        Bu = normalized_matrix[:n, n:]
        Bv = normalized_matrix[n:, :n]

        # Ensure Bu and Bv have the same dimensions along axis 1
        min_cols = min(Bu.shape[1], Bv.shape[1])
        Bu = Bu[:, :min_cols]
        Bv = Bv[:, :min_cols]

        bgnn_matrix = np.block([[np.zeros_like(Bu), Bu], [Bv, np.zeros_like(Bv)]])
        bgnn_matrices.append(bgnn_matrix)

    # Create mappings from IDs to indices and vice versa for users and items
    for idx, user_id in enumerate(user_ids):
        user_id_to_index[user_id] = idx
        user_index_to_id[idx] = user_id

    for idx, item_id in enumerate(item_ids):
        item_id_to_index[item_id] = idx
        item_index_to_id[idx] = item_id

    return bgnn_matrices, user_id_to_index, user_index_to_id, item_id_to_index, item_index_to_id

def L_BGNN(file_path, criteria, user_ids, item_ids):
    graph_data = pd.read_excel(file_path)
    matrices = []  # Initialize a list to store the normalized matrices for each criterion

    user_id_to_index = {}
    user_index_to_id = {}
    item_id_to_index = {}
    item_index_to_id = {}

    for criterion in criteria:
        subgraph = nx.Graph()

        for i in range(len(graph_data)):
            uid = graph_data['User_ID'][i]
            mid = graph_data['Items_ID'][i]

            rating = graph_data[criterion][i]

            if rating > 0:
                subgraph.add_node(uid, bipartite=0)
                subgraph.add_node(mid, bipartite=1)
                subgraph.add_edge(uid, mid, weight=rating)

        n_nodes = len(subgraph.nodes())
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int32)

        for uid, mid, data in subgraph.edges(data=True):
            uid_idx = list(subgraph.nodes()).index(uid)
            mid_idx = list(subgraph.nodes()).index(mid)
            adj_matrix[uid_idx][mid_idx] = data['weight']
            adj_matrix[mid_idx][uid_idx] = data['weight']

        # Calculate the degree matrices DC_uv and DC_vu as before
        DC_uv = np.diag(np.sum(adj_matrix, axis=1))
        DC_vu = np.diag(np.sum(adj_matrix, axis=0))

        # Normalize the matrix using the degree matrices
        BC_uv_norm = np.linalg.pinv(DC_uv) @ adj_matrix
        BC_vu_norm = adj_matrix @ np.linalg.pinv(DC_vu)

        # Average the two normalized matrices to get a single normalized matrix
        normalized_matrix = (BC_uv_norm + BC_vu_norm) / 2.0

        matrices.append(normalized_matrix)
        
        # # Print the normalized matrix
        # print(f"\nNormalized Matrix for criterion '{criterion}':")
        # print(normalized_matrix)

    # Create mappings from IDs to indices and vice versa for users and items
    for idx, user_id in enumerate(user_ids):
        user_id_to_index[user_id] = idx
        user_index_to_id[idx] = user_id

    for idx, item_id in enumerate(item_ids):
        item_id_to_index[item_id] = idx
        item_index_to_id[idx] = item_id

    return tuple(matrices), user_id_to_index, user_index_to_id, 

def resize_matrices(matrices):
    # Find the maximum size among all matrices
    max_size = max(matrix.shape[0] for matrix in matrices)

    # Initialize a list to store resized matrices
    resized_matrices = []

    # Resize each matrix to the maximum size
    for matrix in matrices:
        if matrix.shape[0] < max_size:
            # Pad the matrix with zeros if it's smaller than the maximum size
            padded_matrix = np.pad(matrix, ((0, max_size - matrix.shape[0]), (0, max_size - matrix.shape[1])), mode='constant')
            resized_matrices.append(padded_matrix)
        elif matrix.shape[0] > max_size:
            # Truncate the matrix if it's larger than the maximum size
            truncated_matrix = matrix[:max_size, :max_size]
            resized_matrices.append(truncated_matrix)
        else:
            # If the matrix is already of the maximum size, no need to resize
            resized_matrices.append(matrix)

    # # Print the resized matrices and their shapes
    # for i, matrix in enumerate(resized_matrices):
    #     print(f"\nResized Matrix {i + 1}:")
    #     print(matrix)
    #     print(f"Shape: {matrix.shape}")

    #     # Count the number of zero and non-zero elements
    #     num_zeros = np.count_nonzero(matrix == 0)
    #     num_non_zeros = np.count_nonzero(matrix != 0)

    #     print(f"Number of zeros: {num_zeros}")
    #     print(f"Number of non-zeros: {num_non_zeros}")

    return resized_matrices

# ------------------------ Define the GAT model
class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.conv_layers = nn.ModuleList([
            GATConv(in_channels, self.head_dim, heads=1) for _ in range(num_heads)
        ])
        self.fc = nn.Linear(num_heads * self.head_dim, out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.global_fc = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)

        # Local Attention
        head_outs = [conv(x, edge_index) for conv in self.conv_layers]
        x_local = torch.cat(head_outs, dim=-1)

        # Self-Attention within each head
        self_attention = F.leaky_relu(self.fc(x_local))
        self_attention = F.softmax(self_attention, dim=-1)

        # Multiply each element in x_local by the corresponding element in self_attention
        x_local = x_local * self_attention

        # Apply LeakyReLU activation
        x_local = self.leakyrelu(x_local)

        # Apply Fully Connected Layer
        x_local = self.fc(x_local)

        # Apply Layer Normalization
        x_local = self.layer_norm(x_local)

        # Apply L2 normalization along dimension 1
        x_local = F.normalize(x_local, p=2, dim=1)

        # Global Attention
        x_global = torch.mean(x_local, dim=0)  # Aggregate information globally
        # x_global = torch.sum(x_local, dim=0)  # Aggregate information globally using sum
        
        global_attention = F.relu(self.global_fc(x_global))
        global_attention = F.softmax(global_attention, dim=-1)

        # Multiply each element in x_local by the corresponding element in global_attention
        x = x_local * global_attention

        return x

    def Multi_Embd(self, matrices, user_ids, item_ids, num_epochs=100, learning_rate=0.01):
        resized_matrices = resize_matrices(matrices)  # Use resize_matrices function here
        dataset_list = []

        for normalized_matrix in resized_matrices:
            edges = torch.tensor(np.array(np.where(normalized_matrix)).T, dtype=torch.long).t().contiguous().clone().detach()
            edge_attr = torch.tensor(normalized_matrix[edges[0], edges[1]], dtype=torch.float).clone().detach()
            x = torch.randn(normalized_matrix.shape[0], 16)  # Assuming in_channels=16 for the GAT model
            dataset = Data(x=x, edge_index=edges, edge_attr=edge_attr)
            dataset_list.append(dataset)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        embeddings_list = []
        for i, dataset in enumerate(dataset_list):
            print(f'Training GAT graph A_C{i + 1}')
            for epoch in range(num_epochs):
                # Implement train_GAT function and call it here
                loss = self.train_GAT(optimizer, dataset, embeddings_list)
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss {loss:.4f}')

            with torch.no_grad():
                embeddings = self(dataset.x, dataset.edge_index)

            embeddings_list.append(embeddings)

        fused_embeddings = self.fusion_embeddings_vectors(embeddings_list, user_ids, item_ids)

        return fused_embeddings
    
    def fusion_embeddings_vectors(self, embeddings_list, user_ids, item_ids):
        max_size = max([embedding.size(0) for embedding in embeddings_list])
        
        # Pad embeddings to the maximum size
        padded_embeddings = [F.pad(embedding, (0, 0, 0, max_size - embedding.size(0))) for embedding in embeddings_list]
        
        # Concatenate the padded embeddings along the second dimension (axis 1)
        fused_embeddings = torch.cat(padded_embeddings, dim=1)
        
        return fused_embeddings
           
    def mse_loss(self, inputs, embeddings_list, attention_weights):
        num_views = len(embeddings_list)
        total_loss = torch.tensor(0.0)

        if not embeddings_list:
            return total_loss

        for i in range(num_views):
            anchor_embeddings = embeddings_list[i]
            predicted_ratings = torch.matmul(inputs, anchor_embeddings.t())

            for j in range(num_views):
                if i != j:
                    negative_embeddings = embeddings_list[j]
                    neg_predicted_ratings = torch.matmul(inputs, negative_embeddings.t())

                    # Use MSE loss between positive and negative pairs
                    mse_loss = F.mse_loss(predicted_ratings, neg_predicted_ratings)
                    total_loss += attention_weights[i] * mse_loss.mean()

        return total_loss
    
    def l2_regularization(self, l2_weight=0.1):
        l2_reg = torch.norm(torch.stack([torch.norm(param, p=2) for param in self.parameters()]), p=2)
        return l2_weight * l2_reg

    def global_similarity_loss(self, embeddings_list):
        global_similarity_loss_value = torch.tensor(0.0)
        num_views = len(embeddings_list)

        # Compute global similarity loss between all pairs of views
        for i in range(num_views):
            for j in range(num_views):
                if i != j:
                    view_i_embeddings = embeddings_list[i]
                    view_j_embeddings = embeddings_list[j]

                    min_size = min(view_i_embeddings.size(0), view_j_embeddings.size(0))
                    view_i_embeddings = view_i_embeddings[:min_size, :]
                    view_j_embeddings = view_j_embeddings[:min_size, :]

                    global_similarity_loss_value += torch.mean(torch.abs(view_i_embeddings - view_j_embeddings))

        return global_similarity_loss_value

    def train_GAT(self, optimizer, data, embeddings_list, alpha=0.5, beta=0.5, gamma=0.1):
        self.train()
        optimizer.zero_grad()
        outputs = self(data.x, data.edge_index)
        embeddings = outputs

        num_views = len(embeddings_list)
        attention_weights = torch.nn.Parameter(torch.ones(num_views))

        mse_loss = self.mse_loss(embeddings, embeddings_list, attention_weights)
        global_similarity_loss_value = self.global_similarity_loss(embeddings_list)
        l2_reg = self.l2_regularization()

        # Combine MSE loss, Similarity loss, and L2 regularization
        total_loss = (alpha * mse_loss + beta * global_similarity_loss_value) + gamma * l2_reg

        # Update attention weights based on MSE loss and Similarity loss
        attention_weights.grad = None  # Clear previous gradients
        total_loss.backward()
        optimizer.step()

        return total_loss
    
# -------------Recommendation Section -------------------------

def create_ground_truth_ratings(file_path, criteria):  
    data = pd.read_excel(file_path)

    # Create a mapping from user/item IDs to unique integer indices
    user_id_map = {uid: i for i, uid in enumerate(data['User_ID'].unique())}
    item_id_map = {mid: i for i, mid in enumerate(data['Items_ID'].unique())}

    num_users = len(user_id_map)
    num_items = len(item_id_map)
    ground_truth_ratings_matrix = np.zeros((num_users, num_items, 1), dtype=np.float32)
    
    for _, row in data.iterrows():
        uid = row['User_ID']
        mid = row['Items_ID']
        overall_rating = row['Overall_Rating']

        if uid in user_id_map and mid in item_id_map:
            user_idx = user_id_map[uid]
            item_idx = item_id_map[mid]
            ground_truth_ratings_matrix[user_idx, item_idx, 0] = overall_rating
        
    return data, ground_truth_ratings_matrix, user_id_map, item_id_map

def Recommendation_items_Top_k(fused_embeddings, file_path, criteria, threshold_func=None, threshold=None, top_k=10):
    data, ground_truth_ratings_matrix, user_id_map, item_id_map = create_ground_truth_ratings(file_path, criteria)
    recommendations_f_items = {}

    # Convert fused_embeddings Tensor to dictionary
    fused_embeddings_dict = {user_id: embedding for user_id, embedding in zip(user_id_map.keys(), fused_embeddings)}

    # Compute similarities between embeddings
    similarities = {}
    for user_id_1, embedding_1 in fused_embeddings_dict.items():
        similarities[user_id_1] = {}
        for user_id_2, embedding_2 in fused_embeddings_dict.items():
            # Reshape embeddings to 2D arrays
            embedding_1_2d = embedding_1.reshape(1, -1)
            embedding_2_2d = embedding_2.reshape(1, -1)
            similarity_score = cosine_similarity(embedding_1_2d, embedding_2_2d)[0][0]
            similarities[user_id_1][user_id_2] = similarity_score

    # Iterate over all users
    for user_id, embedding in fused_embeddings_dict.items():
        # Determine threshold value
        if threshold_func is not None:
            threshold_A = threshold_func(embedding).item()
        elif threshold is not None:
            threshold_A = threshold
        else:
            raise ValueError("Either threshold_func or threshold must be provided.")

        # Find similar users based on cosine similarity and dynamic threshold
        similar_users = {user_id_2: sim for user_id_2, sim in similarities[user_id].items() if sim.item() >= threshold_A}

        # Sort similar users by similarity score and select top_k users
        similar_users = sorted(similar_users.items(), key=lambda x: x[1].item(), reverse=True)[:top_k]

        # Get overall rating of the current user
        current_user_rating = data[data['User_ID'] == user_id]['Overall_Rating'].values[0]
        rated_items = data[data['User_ID'] == user_id]['Overall_Rating'].values
        avg_rating = sum(rated_items) / len(rated_items) if len(rated_items) > 0 else 0

        # Initialize recommended items list for the current user
        recommended_items = []

        # Get recommended items for the user
        for user_id_2, _ in similar_users:
            for _, row in data[data['User_ID'] == user_id_2].iterrows():
                item_id = row['Items_ID']
                overall_rating = row['Overall_Rating']
                
                # Check if overall rating is similar to the current user's rating
                if abs(overall_rating - current_user_rating) <= threshold_A:  
                    recommended_items.append({'item_id': item_id, 'Overall_Rating': overall_rating})

        # Sort recommended items by overall rating
        recommended_items = sorted(recommended_items, key=lambda x: x['Overall_Rating'], reverse=True)[:top_k]

        # Add recommended items for the current user to the dictionary
        recommendations_f_items[user_id] = recommended_items
            
    return recommendations_f_items

def evaluate_recommendation_model(fused_embeddings, file_path, criteria, threshold_A=0.7, top_k=10, test_size=0.3):
    # Create ground truth ratings matrix and obtain necessary data
    data, ground_truth_ratings_matrix, user_id_map, item_id_map = create_ground_truth_ratings(file_path, criteria)
    
    # Calculate the number of users and items
    num_users = len(user_id_map)
    num_items = len(item_id_map)

    # Calculate the number of users and items for the test set
    num_test_users = int(len(user_id_map) * test_size)
    num_test_items = int(len(item_id_map) * test_size)

    # Select the first num_test_users users and items as the test set
    test_user_ids = list(user_id_map.keys())[:num_test_users]
    test_item_ids = list(item_id_map.keys())[:num_test_items]

    # The remaining users and items are used for the training set
    train_user_ids = list(user_id_map.keys())[num_test_users:]
    train_item_ids = list(item_id_map.keys())[num_test_items:]

    # Print the number of users and items in train and test sets
    print("Recommendation Model 4:")
    print("Number of users in train set:", len(train_user_ids))
    print("Number of items in train set:", len(train_item_ids))
    print("Number of users in test set:", len(test_user_ids))
    print("Number of items in test set:", len(test_item_ids))
    
    # Ensure that fused_embeddings contain embeddings for all user and item IDs
    fused_embeddings_with_ids = {}
    for user_id, embedding in zip(user_id_map.keys(), fused_embeddings):
        fused_embeddings_with_ids[user_id] = embedding

    # Split fused_embeddings based on test data
    train_fused_embeddings = {user_id: fused_embeddings_with_ids[user_id] for user_id in train_user_ids}

    # Generate recommendations for train data
    # train_recommendations = Recommendation_items_Top_k(train_fused_embeddings, file_path, criteria, threshold_A, top_k)
    train_recommendations = Recommendation_items_Top_k(train_fused_embeddings, file_path, criteria, threshold=threshold_A, top_k=10)


    # Initialize lists to store differences between predicted and actual ratings for train data
    train_rating_diffs = []

    # Calculate differences between predicted and actual ratings for train data
    for user_id, recommended_items in train_recommendations.items():
        # Retrieve ground truth ratings for the user
        ground_truth_ratings = data[data['User_ID'] == user_id].set_index('Items_ID')['Overall_Rating']
        # Calculate differences for the user's recommended items
        for item in recommended_items:
            item_id = item['item_id']
            overall_rating = item['Overall_Rating']
            if item_id in ground_truth_ratings.index:
                train_rating_diffs.append(overall_rating - ground_truth_ratings[item_id])

    # Calculate MAE and RMSE for train data
    train_mae = mean_absolute_error(train_rating_diffs, [0]*len(train_rating_diffs))
    train_rmse = mean_squared_error(train_rating_diffs, [0]*len(train_rating_diffs), squared=False)

    # Print MAE and RMSE for train data
    print("MAE for train data (Recommendation Model):", train_mae)
    print("RMSE for train data (Recommendation Model):", train_rmse)
  
    # Split fused_embeddings based on test data
    test_fused_embeddings = {user_id: fused_embeddings_with_ids[user_id] for user_id in test_user_ids}

    # Generate recommendations for test data
    test_recommendations = Recommendation_items_Top_k(test_fused_embeddings, file_path, criteria, threshold=threshold_A, top_k=10)

    # Initialize lists to store differences between predicted and actual ratings for test data
    test_rating_diffs = []

    # Calculate differences between predicted and actual ratings for test data
    for user_id, recommended_items in test_recommendations.items():
        # Retrieve ground truth ratings for the user
        ground_truth_ratings = data[data['User_ID'] == user_id].set_index('Items_ID')['Overall_Rating']
        # Calculate differences for the user's recommended items
        for item in recommended_items:
            item_id = item['item_id']
            overall_rating = item['Overall_Rating']
            if item_id in ground_truth_ratings.index:
                test_rating_diffs.append(overall_rating - ground_truth_ratings[item_id])

    # Calculate MAE and RMSE for test data
    test_mae = mean_absolute_error(test_rating_diffs, [0]*len(test_rating_diffs))
    test_rmse = mean_squared_error(test_rating_diffs, [0]*len(test_rating_diffs), squared=False)

    # Print MAE and RMSE for test data
    print("MAE for test data (Recommendation Model):", test_mae)
    print("RMSE for test data (Recommendation Model):", test_rmse)

    return train_mae, train_rmse, test_mae, test_rmse


# --------------------------------------------------------------------------------------
# Main Function ---------------------------
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Define your file paths for different datasets in Katana Server
    file_paths = {
        'Movies_Original': '/home/z5318340/MCRS4/MoviesDatasetYahoo.xlsx',
        'Movies_Modified': '/home/z5318340/MCRS4/Movies_Modified_Rating_Scores.xlsx',
        'BeerAdvocate': '/home/z5318340/MCRS4/BeerAdvocate.xlsx',
        'TripAdvisor': '/home/z5318340/MCRS4/new_Trip_filtered_dataset.xlsx'
    }
    
    # Define your file paths for different datasets in local Server
    file_paths = {
        'Movies_Original': 'C://Yahoo//Global//Movies.xlsx',
        # 'Movies_Modified': 'C://Yahoo//Global//Movies_Modified.xlsx',
        'BeerAdvocate': 'C://Yahoo//Global//Modified_BeerAdvocate.xlsx',
        'TripAdvisor': 'C://Yahoo//Global//TripAdvisor.xlsx'
    }
    
    # Define criteria for different datasets
    criteria_mapping = {
        'Movies_Original': ['C1', 'C2', 'C3', 'C4'],
        'Movies_Modified': ['C1', 'C2', 'C3', 'C4'],
        'BeerAdvocate': ['C1', 'C2', 'C3', 'C4'],
        'TripAdvisor': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    }

    # Define the dataset to run
    dataset_to_run = 'BeerAdvocate'

    # Read data for the selected dataset
    file_path = file_paths[dataset_to_run]
    criteria = criteria_mapping[dataset_to_run]
    user_id_map, item_id_map, base_ground_truth_ratings = read_data(file_path, criteria)

    # Call other functions
    create_bipartite_graph(file_path, criteria)
    create_subgraphs(file_path, criteria)

    # Read data from the Excel file and create ID mappings
    user_ids = list(user_id_map.keys())
    item_ids = list(item_id_map.keys())

    # Call the function to create and normalize adjacency matrices
    result = create_and_normalize_adjacency_matrices(file_path, criteria, user_ids, item_ids)
    
    matrices, user_id_to_index, user_index_to_id = L_BGNN(file_path, criteria, user_ids, item_ids)
    resized_matrices = resize_matrices(matrices)
    
    # Combine user_ids and item_ids into a single list to build a unique mapping
    combined_ids = np.concatenate((user_ids, item_ids))

    # Convert all elements in combined_ids to strings to ensure uniform type
    combined_ids_str = combined_ids.astype(str)

    # Create a mapping of unique IDs to unique integer values
    unique_ids_str = np.unique(combined_ids_str)
    id_to_int = {id_: i for i, id_ in enumerate(unique_ids_str)}

    # Convert user_ids and item_ids to integers using the mapping
    user_ids_int = np.array([id_to_int[str(user_id)] for user_id in user_ids])
    item_ids_int = np.array([id_to_int[str(item_id)] for item_id in item_ids])

    # Check for any invalid mappings (IDs not found in the dictionary)
    if -1 in user_ids_int or -1 in item_ids_int:
        print("Invalid ID found in user_ids or item_ids.")
        
    # Convert user_ids_int and item_ids_int to PyTorch tensors
    user_ids_tensor = torch.tensor(user_ids_int).clone().detach()
    item_ids_tensor = torch.tensor(item_ids_int).clone().detach()

    #---Attention Embedding------
    model = GAT(in_channels=16, out_channels=256)
    result = model.Multi_Embd(resized_matrices, user_ids_tensor, item_ids_tensor, num_epochs=100, learning_rate=0.01)
    fused_embeddings_with_ids = result  # unpack the values you need
    print("Fused Embeddings:")
    print(fused_embeddings_with_ids)
    
    # Recommendation section
    num_samples = fused_embeddings_with_ids.shape[0]

    # Create an instance of the MultiCriteriaRecommender class
    output_dim = fused_embeddings_with_ids.shape[1]  # Set output_dim to the number of criteria

    # Convert fused_embeddings_with_ids to a torch tensor
    fused_embeddings_tensor = fused_embeddings_with_ids.clone().detach().to(torch.float32)

    # Reshape fused_embeddings_tensor to match the expected shape
    num_samples, num_features = fused_embeddings_tensor.shape
    num_users = len(user_ids)
    num_criteria = len(criteria)
    num_items = len(item_ids)

    # Calculate the total number of features per criterion
    num_features_per_criterion = num_features // num_criteria
        
    # Create a DataFrame with user and item identifiers as MultiIndex
    df_users_items = pd.DataFrame(index=pd.MultiIndex.from_tuples([(user_id, item_id) for user_id in user_id_map.keys() for item_id in item_id_map.keys()]))
    
    # Call the create_real_ratings function
    data, ground_truth_ratings_matrix, user_id_map, item_id_map = create_ground_truth_ratings(file_path, criteria)
    # Call the P_Recommendation_item function

    # Call the function with the defined threshold function   
    def threshold_func(fused_embeddings_with_ids):
        threshold = 0.7  # Example threshold value
        threshold_tensor = torch.tensor(threshold)  
        return threshold_tensor

    # Call the function with the defined threshold function
    recommendations = Recommendation_items_Top_k(fused_embeddings_with_ids, file_path, criteria, threshold_func=threshold_func, top_k=10)

    train_mae, train_rmse,test_mae, test_rmse = evaluate_recommendation_model(fused_embeddings_with_ids, file_path, criteria, threshold_A=0.7, top_k=10, test_size=0.2)
