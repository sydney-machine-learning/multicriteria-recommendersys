
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
    
    # def attention_fusion(self, embeddings_list, user_ids, item_ids):
    #     attention_weights = torch.nn.Parameter(torch.ones(len(embeddings_list)))
    #     attention_weights = F.softmax(attention_weights, dim=0)

    #     # Apply attention weights to embeddings
    #     weighted_embeddings = [weight * embedding for weight, embedding in zip(attention_weights, embeddings_list)]

    #     # Sum the weighted embeddings
    #     fused_embeddings = torch.sum(torch.stack(weighted_embeddings), dim=0)

    #     return fused_embeddings
        
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

    def train_GAT(self, optimizer, data, embeddings_list, alpha=0.1, beta=0.2, gamma=0.5):
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
    user_id = data['User_ID']
    item_id = data['Items_ID']

    # Create a mapping from user/item IDs to unique integer indices
    user_id_map = {uid: i for i, uid in enumerate(user_id.unique())}
    item_id_map = {mid: i for i, mid in enumerate(item_id.unique())}

    num_users = len(user_id_map)
    num_items = len(item_id_map)
    num_criteria = len(criteria)
    ground_truth_ratings_matrix = np.zeros((num_users, num_items, num_criteria), dtype=np.int16)
    
    # Additional columns
    data['Overal_Rating'] = 0
    data['item_id'] = ''  # Add the 'item_id' column
    data['Number_Rated_Items'] = 0  # Add the 'Number_Rated_Items' column

    for i, row in data.iterrows():
        uid = row['User_ID']
        mid = row['Items_ID']
        criterion_ratings = [row[criterion] for criterion in criteria]
        
        # Calculate average rating for criteria with a rating greater than 0
        non_zero_ratings = [rating for rating in criterion_ratings if rating > 0]
        Overal_Rating = np.mean(non_zero_ratings) if non_zero_ratings else 0

        # Assign values to additional columns
        data.at[i, 'Overal_Rating'] = Overal_Rating
        data.at[i, 'item_id'] = mid

        # Calculate and assign the number of rated items by each user
        num_rated_items_by_user = np.sum(data[data['User_ID'] == uid][criteria].apply(lambda x: (x > 0).any(), axis=1))
        data.at[i, 'Number_Rated_Items'] = num_rated_items_by_user
        
        if uid in user_id_map and mid in item_id_map:
            user_idx = user_id_map[uid]
            item_idx = item_id_map[mid]
            ground_truth_ratings_matrix[user_idx, item_idx] = criterion_ratings

    return data, ground_truth_ratings_matrix

def normalize_hadamard_embeddings(fused_embeddings):
    # Detach PyTorch tensors
    fused_embeddings = fused_embeddings.detach().numpy()

    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()

    # Fit the scaler on the summed_embeddings and transform them
    normalized_embeddings = scaler.fit_transform(fused_embeddings)

    return normalized_embeddings
#----------------------------------------

def Recommendation_items_Fixed_TopK(normalized_embeddings, file_path, criteria, threshold_A=0.9, top_k=1):
    data, _ = create_ground_truth_ratings(file_path, criteria)
    recommendations_f_items = {}

    num_users_actual, _ = normalized_embeddings.shape
    normalized_embeddings_2d = normalized_embeddings.reshape((num_users_actual, -1))
    similarities = cosine_similarity(normalized_embeddings_2d)

    # Counter variable to limit the number of printed users
    printed_users_count = 0

    for i in range(num_users_actual):
        similar_user_index = np.argsort(similarities[i])[::-1][:top_k]

        similar_user_items = data.iloc[similar_user_index]

        similar_user_rated_items = similar_user_items.groupby(['User_ID', 'Items_ID'])['Overal_Rating'].mean().reset_index()
        similar_user_rated_items = similar_user_rated_items.sort_values(by='Overal_Rating', ascending=False)

        # Apply the threshold_A to filter out low-rated recommendations
        similar_user_rated_items = similar_user_rated_items[similar_user_rated_items['Overal_Rating'] >= threshold_A]

        # Take the top-K recommendations after applying the threshold_A
        similar_user_rated_items = similar_user_rated_items.head(top_k)

        # Create the recommendation
        recommended_items = similar_user_rated_items.to_dict(orient='records')

        # Add 'item_id' to each item dictionary
        for item in recommended_items:
            item['item_id'] = item['Items_ID']

        recommendations_f_items[data.iloc[i]['User_ID']] = {
            'User_ID': data.iloc[i]['User_ID'],
            'recommended_items': recommended_items,
            'item_id': data.iloc[i]['Items_ID'],
            'Overal_Rating': float(data.iloc[i]['Overal_Rating'])  
        }

        # # Print recommendations for the specified number of users
        # if printed_users_count < 5:
        #     print(f"Recommendations for User {data.iloc[i]['User_ID']}: {recommendations_f_items[data.iloc[i]['User_ID']]}")
        #     printed_users_count += 1
        # else:
        #     break  # Break out of the loop once 10 users are printed

    return recommendations_f_items

def evaluate_recommendations_Prediction_Fixed_TopK(ground_truth_real_matrix, recommendations_f_items, user_id_map, item_id_map):
    predicted_ratings = np.zeros_like(ground_truth_real_matrix, dtype=np.float32)
    actual_ratings = []
    indices = []

    for user_id, recommendation in recommendations_f_items.items():
        items = recommendation['recommended_items']
        user_idx = user_id_map[recommendation['User_ID']]
        
        if len(items) > 0:
            # Calculate the average rating of recommended items
            avg_rating = np.mean([item['Overal_Rating'] for item in items])

            for item in items:
                item_idx = item_id_map[item['item_id']]
                # Assign the average rating to the predicted rating matrix
                predicted_ratings[user_idx, item_idx] = avg_rating
                
                # Check if the user actually rated the item and store the actual rating and index
                actual_rating = ground_truth_real_matrix[user_idx, item_idx]
                if np.any(actual_rating != 0):
                    actual_ratings.append(actual_rating)
                    indices.append((user_idx, item_idx))

    actual_ratings = np.array(actual_ratings)
    indices = np.array(indices)

    # Split the indices into training and testing sets
    train_indices, test_indices, _, _ = train_test_split(indices, actual_ratings, test_size=0.2)

    # Extract corresponding values from the predicted ratings matrix
    actual_train = ground_truth_real_matrix[train_indices[:, 0], train_indices[:, 1]]
    actual_test = ground_truth_real_matrix[test_indices[:, 0], test_indices[:, 1]]
    predicted_train = predicted_ratings[train_indices[:, 0], train_indices[:, 1]]
    predicted_test = predicted_ratings[test_indices[:, 0], test_indices[:, 1]]

    mae = mean_absolute_error(actual_test, predicted_test)
    rmse = np.sqrt(mean_squared_error(actual_test, predicted_test))
    
    # Print the results
    print(f"\nMAE based Fixed Test/train Size: {mae}")
    print(f"RMSE based Fixed  Test/train Size: {rmse}")
 
    return mae, rmse

def evaluate_crossfold_recommendations_Prediction_Fixed_TopK(ground_truth_real_matrix, recommendations_f_items, user_id_map, item_id_map):
    predicted_ratings = np.zeros_like(ground_truth_real_matrix, dtype=np.float32)
    actual_ratings = []
    indices = []

    mae_values = []
    rmse_values = []

    for user_id, recommendation in recommendations_f_items.items():
        items = recommendation['recommended_items']
        user_idx = user_id_map[recommendation['User_ID']]
        
        if len(items) > 0:
            # Calculate the average rating of recommended items
            avg_rating = np.mean([item['Overal_Rating'] for item in items])

            for item in items:
                item_idx = item_id_map[item['item_id']]
                # Assign the average rating to the predicted rating matrix
                predicted_ratings[user_idx, item_idx] = avg_rating
                
                # Check if the user actually rated the item and store the actual rating and index
                actual_rating = ground_truth_real_matrix[user_idx, item_idx]
                if np.any(actual_rating != 0):
                    actual_ratings.append(actual_rating)
                    indices.append((user_idx, item_idx))

    actual_ratings = np.array(actual_ratings)
    indices = np.array(indices)

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(indices):
        train_indices, test_indices = indices[train_index], indices[test_index]

        # Extract corresponding values from the predicted ratings matrix
        actual_train = ground_truth_real_matrix[train_indices[:, 0], train_indices[:, 1]]
        actual_test = ground_truth_real_matrix[test_indices[:, 0], test_indices[:, 1]]
        predicted_train = predicted_ratings[train_indices[:, 0], train_indices[:, 1]]
        predicted_test = predicted_ratings[test_indices[:, 0], test_indices[:, 1]]

        mae = mean_absolute_error(actual_test, predicted_test)
        rmse = np.sqrt(mean_squared_error(actual_test, predicted_test))

        # Print the results for each fold
        # print(f"\nFold - MAE: {mae}, RMSE: {rmse}")

        # Store MAE and RMSE values for each fold
        mae_values.append(mae)
        rmse_values.append(rmse)

    # Calculate and print the average MAE and RMSE across all folds
    average_mae = np.mean(mae_values)
    average_rmse = np.mean(rmse_values)
    print(f"\nAverage MAE based on the Cross Validation: {average_mae}")
    print(f"Average RMSE based on the Cross Validation: {average_rmse}")

    return average_mae, average_rmse


#----------------------------------------------
#--------------------------------------------------

def Recommendation_items_Dynamic_TopK(normalized_embeddings, file_path, criteria, threshold=0.9):
    
    data, _ = create_ground_truth_ratings(file_path, criteria)
    recommendations_items = {}

    num_users_actual, _ = normalized_embeddings.shape
    normalized_embeddings_2d = normalized_embeddings.reshape((num_users_actual, -1))
    similarities = cosine_similarity(normalized_embeddings_2d)

    for user_index in range(num_users_actual):
        user_id = data.iloc[user_index]['User_ID']

        # Dynamically set top_k based on the number of items the user has rated
        user_data = data[data['User_ID'] == user_id]
        top_k_user = min(len(user_data), num_users_actual)

        similar_user_index = np.argsort(similarities[user_index])[::-1]

        # Get the top-K similar users
        similar_user_index = similar_user_index[:top_k_user]

        similar_user_items = data.iloc[similar_user_index]

        similar_user_rated_items = similar_user_items.groupby(['User_ID', 'Items_ID'])['Overal_Rating'].mean().reset_index()
        similar_user_rated_items = similar_user_rated_items.sort_values(by='Overal_Rating', ascending=False)

        # Apply the threshold to filter out low-rated recommendations
        similar_user_rated_items = similar_user_rated_items[similar_user_rated_items['Overal_Rating'] >= threshold]

        # Take the top-K recommendations after applying the threshold
        similar_user_rated_items = similar_user_rated_items.head(top_k_user)

        # Create the recommendation
        recommended_items = similar_user_rated_items.to_dict(orient='records')

        # Add 'item_id' to each item dictionary
        for item in recommended_items:
            item['item_id'] = item['Items_ID']

        recommendations_items[user_id] = {
            'User_ID': user_id,
            'recommended_items': recommended_items,
            'item_id': data.iloc[user_index]['Items_ID'],
            'Overal_Rating': float(data.iloc[user_index]['Overal_Rating'])  
        }

    # # Print the number of recommendations for each user outside the loop
    # for user_id, recommendation in recommendations_items.items():
    #     print(f"User {user_id} has {len(recommendation['recommended_items'])} recommendations.")

    return recommendations_items

def Evaluate_RS_ManualMetrics_Dynamic_Topk(ground_truth_real_matrix, recommendations_items, user_id_map, item_id_map):
    actual_ratings = []
    indices = []

    total_tp = 0  # Total True Positives
    total_fp = 0  # Total False Positives
    total_fn = 0  # Total False Negatives

    for user_id, recommendation in recommendations_items.items():
        tp = 0  # Reset True Positives for each user
        fp = 0  # Reset False Positives for each user
        fn = 0  # Reset False Negatives for each user

        items = recommendation['recommended_items']
        user_idx = user_id_map[recommendation['User_ID']]

        if len(items) > 0:
            # Extract actual ratings and indices for the current user
            user_actual_ratings = ground_truth_real_matrix[user_idx, :]
            actual_ratings.extend(user_actual_ratings)
            indices.extend([(user_idx, i) for i in range(len(user_actual_ratings))])

            recommended_items_count = len(items)

            # Calculate True Positives, False Positives, and False Negatives for the current user
            for i in range(len(user_actual_ratings)):
                if np.any(user_actual_ratings[i] > 0):
                    if i in [item_id_map[item['item_id']] for item in items]:
                        tp += 1
                    else:
                        fn += 1
                elif i in [item_id_map[item['item_id']] for item in items]:
                    fp += 1

            # Print information for each user, including the total number of rated items
            # print(f"User ID: {user_id}, True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

        total_tp += tp
        total_fp += fp
        total_fn += fn

    actual_ratings = np.array(actual_ratings)
    indices = np.array(indices)

    # Split the indices into training and testing sets
    train_indices, test_indices, _, _ = train_test_split(indices, actual_ratings, test_size=0.2, random_state=42)

    # Extract corresponding values from the predicted ratings matrix for the test set
    actual_test = ground_truth_real_matrix[test_indices[:, 0], test_indices[:, 1]]

    # Calculate precision and recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    
    # Print the results in a different table format
    results_table = [
        ["Total number of indices", len(indices)],
        ["Number of training indices", len(train_indices)],
        ["Number of testing indices", len(test_indices)],
        ["True Positives (tp)", total_tp],
        ["False Positives (fp) ", total_fp],
        ["False Negatives (fn)", total_fn],
        ["Precision", precision],
        ["Recall", recall],
        ["F1", f1],
        ["F2", f2],
    ]

    print(tabulate(results_table, headers=["Manual Metrics", "Score"], tablefmt="grid"))

    return precision, recall, f1, f2


# ---------------------------------------------------------------------------------------
# Main Function ---------------------------
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Define the file path and criteria
    # file_path = '/home/z5318340/MCRS4/MoviesDatasetYahoo.xlsx'
    
    
    # Movies and BeerAdvocate datasets
    
    file_path = 'C://Yahoo//Global//Movies.xlsx'
    # file_path = 'C://Yahoo//Global//Movies_Modified.xlsx'
    # file_path = 'C://Yahoo//Global//BeerAdvocate.xlsx'
    criteria = ['C1', 'C2', 'C3', 'C4'] 
    
    # TripAdvisor dataset
    # file_path = 'C://Yahoo//Global//TripAdvisor.xlsx'
    # criteria = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    # Call the read_data function to get user_id_map and item_id_map
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
    
    # Movies and BeerAdvocate datasets
    matrix1, matrix2, matrix3, matrix4 = matrices
    # TripAdvisor dataset
    # matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7 = matrices
                      
    # Combine user_ids and item_ids into a single list to build a unique mapping
    combined_ids = np.concatenate((user_ids, item_ids))

    # Create a mapping of unique IDs to unique integer values
    unique_ids = np.unique(combined_ids)
    id_to_int = {id_: i for i, id_ in enumerate(unique_ids)}

    # Convert user_ids and item_ids to integers using the mapping
    user_ids_int = np.array([id_to_int.get(user_id, -1) for user_id in user_ids])
    item_ids_int = np.array([id_to_int.get(item_id, -1) for item_id in item_ids])

    # Check for any invalid mappings (IDs not found in the dictionary)
    if -1 in user_ids_int or -1 in item_ids_int:
        print("Invalid ID found in user_ids or item_ids.")
        
    # Convert user_ids_int and item_ids_int to PyTorch tensors
    user_ids_tensor = torch.tensor(user_ids_int).clone().detach()
    item_ids_tensor = torch.tensor(item_ids_int).clone().detach()

    
    #---Attention Embedding------
    # GAT and Fusion Embeddings
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
        
    # Call the create_ground_truth_ratings function
    ground_truth_ratings = create_ground_truth_ratings(file_path, criteria)

    # Create a DataFrame with user and item identifiers as MultiIndex
    df_users_items = pd.DataFrame(index=pd.MultiIndex.from_tuples([(user_id, item_id) for user_id in user_id_map.keys() for item_id in item_id_map.keys()]))
    
    # normalized_Embeddings vectors of summed_embeddings
    normalized_H_F_embeddings = normalize_hadamard_embeddings(fused_embeddings_with_ids)
    # print("Normalize:",normalized_H_F_embeddings)
    
    # Call the create_real_ratings function
    data, ground_truth_real_matrix = create_ground_truth_ratings(file_path, criteria)
    # Call the P_Recommendation_item function
    recommendations_f_items = Recommendation_items_Fixed_TopK(normalized_H_F_embeddings, file_path, criteria, threshold_A=0.9, top_k=10)
    recommendations_items = Recommendation_items_Dynamic_TopK(normalized_H_F_embeddings, file_path, criteria, threshold=0.9)
    
    mae, rmse = evaluate_recommendations_Prediction_Fixed_TopK(ground_truth_real_matrix, recommendations_f_items, user_id_map, item_id_map)
    average_mae, average_rmse = evaluate_crossfold_recommendations_Prediction_Fixed_TopK(ground_truth_real_matrix, recommendations_f_items, user_id_map, item_id_map)
    precision, recall, f1, f2 = Evaluate_RS_ManualMetrics_Dynamic_Topk(ground_truth_real_matrix, recommendations_items, user_id_map, item_id_map)

    