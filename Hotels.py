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


def read_data(file_path, criteria):
    data = pd.read_excel(file_path)
    user_id = data['User_ID']
    hotel_id = data['Hotel_ID']

    user_id_map = {uid: i for i, uid in enumerate(user_id.unique())}
    hotel_id_map = {mid: i for i, mid in enumerate(hotel_id.unique())}

    num_users = len(user_id_map)
    num_hotels = len(hotel_id_map)
    num_criteria = len(criteria)
    base_ground_truth_ratings = np.zeros((num_users, num_hotels, num_criteria), dtype=np.int32)

    for i, row in data.iterrows():
        uid = row['User_ID']
        hid = row['Hotel_ID']
        criterion_ratings = [row[criterion] for criterion in criteria]
        if uid in user_id_map and hid in hotel_id_map:
            user_idx = user_id_map[uid]
            hotel_idx = hotel_id_map[hid]
            base_ground_truth_ratings[user_idx, hotel_idx] = criterion_ratings

    return user_id_map, hotel_id_map, base_ground_truth_ratings

def create_graph(file_path, criteria):
    data = pd.read_excel(file_path)
    G = nx.MultiGraph()

    users = set()
    hotels = set()

    for uid in data['User_ID']:
        G.add_node(uid, bipartite=0)
        users.add(uid)

    for hid in data['Hotel_ID']:
        G.add_node(hid, bipartite=1)
        hotels.add(hid)

    for i in range(len(data)):
        uid = data['User_ID'][i]
        hid = data['Hotel_ID'][i]

        for criterion in criteria:
            rating = data[criterion][i]

            if rating > 0:
                G.add_edge(uid, hid, criterion=criterion, weight=rating)

    user_nodes = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 0]
    hotel_nodes = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 1]

    is_bipartite = nx.is_bipartite(G)

    if is_bipartite:
        print("The graph is bipartite.")
    else:
        print("The graph is not bipartite.")

    print(f"Number of user nodes: {len(user_nodes)}")
    print(f"Number of hotel nodes: {len(hotel_nodes)}")

    user_hotel_edges = [(u, v, data) for u, v, data in G.edges(data=True) if u in user_nodes and v in hotel_nodes]
    print(f"Number of edges between user and hotel nodes: {len(user_hotel_edges)}")

    for u, v, data in G.edges(data=True):
        if u in users and v in hotels and 'criterion' in data and 'rating' in data:
            user_id = u
            hotel_id = v
            criterion = data['criterion']
            rating = data['rating']

            # print(f"Edge between User_ID {str(user_id)} and Hotel_ID {str(hotel_id)} (Criterion: {str(criterion)}):")
            # print(f"  Weight (Rating): {str(rating).encode('utf-8', 'replace').decode('utf-8')}")

    for u, v, data in G.edges(data=True):
        weight = data['weight']
        # print(f"Edge between {u} and {v} has weight: {weight}")

def create_subgraphs(file_path, criteria):
    graph_data = pd.read_excel(file_path)
    subgraphs = []

    for criterion in criteria:
        subgraph = nx.Graph()
        subgraphs.append(subgraph)

    for i in range(len(graph_data)):
        uid = graph_data['User_ID'][i]
        hid = graph_data['Hotel_ID'][i]

        for criterion, subgraph in zip(criteria, subgraphs):
            rating = graph_data[criterion][i]

            if rating > 0:
                subgraph.add_node(uid, bipartite=0)
                subgraph.add_node(hid, bipartite=1)
                subgraph.add_edge(uid, hid, weight=rating)

    for criterion, subgraph in zip(criteria, subgraphs):
        # print(f"\nSubgraph for Criterion {criterion}:")
        is_bipartite = nx.is_bipartite(subgraph)
        # print(f"Is bipartite: {is_bipartite}")

        user_nodes = [node for node in subgraph.nodes() if subgraph.nodes[node]['bipartite'] == 0]
        hotel_nodes = [node for node in subgraph.nodes() if subgraph.nodes[node]['bipartite'] == 1]
        # print(f"Number of user nodes: {len(user_nodes)}")
        # print(f"Number of hotel nodes: {len(hotel_nodes)}")

        subgraph_edges = [(u, v, data) for u, v, data in subgraph.edges(data=True)]
        # print(f"Number of edges in subgraph: {len(subgraph_edges)}")

def create_and_normalize_adjacency_matrices(file_path, criteria, user_ids, hotel_ids):
    graph_data = pd.read_excel(file_path)
    bgnn_matrices = []  # Initialize a list to store the BGNN matrices for each criterion

    user_id_to_index = {}
    user_index_to_id = {}
    hotel_id_to_index = {}
    hotel_index_to_id = {}

    for criterion in criteria:
        subgraph = nx.Graph()

        for i in range(len(graph_data)):
            uid = graph_data['User_ID'][i]
            hid = graph_data['Hotel_ID'][i]

            rating = graph_data[criterion][i]

            if rating > 0:
                subgraph.add_node(uid, bipartite=0)
                subgraph.add_node(hid, bipartite=1)
                subgraph.add_edge(uid, hid, weight=rating)

        n_nodes = len(subgraph.nodes())
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int32)

        for uid, hid, data in subgraph.edges(data=True):
            uid_idx = list(subgraph.nodes()).index(uid)
            hid_idx = list(subgraph.nodes()).index(hid)
            adj_matrix[uid_idx][hid_idx] = data['weight']
            adj_matrix[hid_idx][uid_idx] = data['weight']

        # Print the matrix
        print(f"\nMatrix for criterion '{criterion}':")
        print(adj_matrix)

        # Count zero and non-zero cells in the matrix and print the results
        zero_cells = np.sum(adj_matrix == 0)
        non_zero_cells = np.sum(adj_matrix != 0)
        print(f"\nMatrix for criterion '{criterion}' has {zero_cells} zero cells and {non_zero_cells} non-zero cells.")

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

    # Create mappings from IDs to indices and vice versa for users and hotels
    for idx, user_id in enumerate(user_ids):
        user_id_to_index[user_id] = idx
        user_index_to_id[idx] = user_id

    for idx, hotel_id in enumerate(hotel_ids):
        hotel_id_to_index[hotel_id] = idx
        hotel_index_to_id[idx] = hotel_id

    return bgnn_matrices, user_id_to_index, user_index_to_id, hotel_id_to_index, hotel_index_to_id

def L_BGNN(file_path, criteria, user_ids, hotel_ids):
    graph_data = pd.read_excel(file_path)
    matrices = []  # Initialize a list to store the normalized matrices for each criterion

    user_id_to_index = {}
    user_index_to_id = {}
    hotel_id_to_index = {}
    hotel_index_to_id = {}

    for criterion in criteria:
        subgraph = nx.Graph()

        for i in range(len(graph_data)):
            uid = graph_data['User_ID'][i]
            hid = graph_data['Hotel_ID'][i]

            rating = graph_data[criterion][i]

            if rating > 0:
                subgraph.add_node(uid, bipartite=0)
                subgraph.add_node(hid, bipartite=1)
                subgraph.add_edge(uid, hid, weight=rating)

        n_nodes = len(subgraph.nodes())
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int32)

        for uid, hid, data in subgraph.edges(data=True):
            uid_idx = list(subgraph.nodes()).index(uid)
            hid_idx = list(subgraph.nodes()).index(hid)
            adj_matrix[uid_idx][hid_idx] = data['weight']
            adj_matrix[hid_idx][uid_idx] = data['weight']

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

    # Create mappings from IDs to indices and vice versa for users and hotels
    for idx, user_id in enumerate(user_ids):
        user_id_to_index[user_id] = idx
        user_index_to_id[idx] = user_id

    for idx, hotel_id in enumerate(hotel_ids):
        hotel_id_to_index[hotel_id] = idx
        hotel_index_to_id[idx] = hotel_id

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

    # Print the resized matrices and their shapes
    for i, matrix in enumerate(resized_matrices):
        print(f"\nResized Matrix {i + 1}:")
        print(matrix)
        print(f"Shape: {matrix.shape}")

        # Count the number of zero and non-zero elements
        num_zeros = np.count_nonzero(matrix == 0)
        num_non_zeros = np.count_nonzero(matrix != 0)

        print(f"Number of zeros: {num_zeros}")
        print(f"Number of non-zeros: {num_non_zeros}")

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
        x = torch.cat(head_outs, dim=-1)

        # Self-Attention within each head
        self_attention = F.leaky_relu(self.fc(x))
        self_attention = F.softmax(self_attention, dim=-1)

        # Multiply each element in x by the corresponding element in self_attention
        x = x * self_attention

        # Apply LeakyReLU activation
        x = self.leakyrelu(x)

        # Apply Fully Connected Layer
        x = self.fc(x)

        # Apply Layer Normalization
        x = self.layer_norm(x)

        # Apply L2 normalization along dimension 1
        x = F.normalize(x, p=2, dim=1)

        # Global Attention
        global_attention = F.relu(self.global_fc(x))

        # Apply softmax to global_attention
        global_attention = F.softmax(global_attention, dim=-1)

        # Multiply each element in x by the corresponding element in global_attention
        x = x * global_attention

        return x

    def Multi_Embd(self, matrices, user_ids, hotel_ids, num_epochs=100, learning_rate=0.01):
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

        fused_embeddings = self.fusion_embeddings_vectors(embeddings_list, user_ids, hotel_ids)

        return fused_embeddings
    
    def fusion_embeddings_vectors(self, embeddings_list, user_ids, hotel_ids):
        max_size = max([embedding.size(0) for embedding in embeddings_list])
        
        # Pad embeddings to the maximum size
        padded_embeddings = [F.pad(embedding, (0, 0, 0, max_size - embedding.size(0))) for embedding in embeddings_list]
        
        # Concatenate the padded embeddings along the second dimension (axis 1)
        fused_embeddings = torch.cat(padded_embeddings, dim=1)
        
        return fused_embeddings

    def train_GAT(self, optimizer, data, embeddings_list, margin_triplet=0.1, margin_contrastive=0.1):
        self.train()
        optimizer.zero_grad()
        outputs = self(data.x, data.edge_index)
        embeddings = outputs

        # Triplet margin loss
        triplet_loss = self.triplet_margin_loss(embeddings, embeddings_list, margin_triplet)

        # Contrastive loss
        contrastive_loss = self.contrastive_loss(embeddings, embeddings_list, margin_contrastive)

        # Add L2 regularization
        l2_reg = self.l2_regularization()

        # Total loss
        total_loss = triplet_loss + contrastive_loss + l2_reg

        total_loss.backward()
        optimizer.step()

        return total_loss.item()
    
    def triplet_margin_loss(self, inputs, embeddings_list, margin=0.1):
        num_views = len(embeddings_list)
        total_loss = torch.tensor(0.0)

        # Check if the embeddings_list is not empty
        if not embeddings_list:
            return total_loss

        for i in range(num_views):
            anchor_embeddings = embeddings_list[i]

            for j in range(num_views):
                if i != j:
                    negative_embeddings = embeddings_list[j]

                    # Pairwise distances
                    pos_distance = F.pairwise_distance(inputs, anchor_embeddings)
                    neg_distance = F.pairwise_distance(inputs, negative_embeddings)

                    # Triplet margin loss
                    triplet_loss = F.relu(pos_distance - neg_distance + margin)

                    total_loss += triplet_loss.mean()

        return total_loss

    def l2_regularization(self, l2_weight=0.1):
        l2_reg = torch.tensor(0.0)

        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)

        return l2_weight * l2_reg

    def contrastive_loss(self, inputs, embeddings_list, margin=0.1):
        num_views = len(embeddings_list)
        total_loss = torch.tensor(0.0)

        # Check if the embeddings_list is not empty
        if not embeddings_list:
            return total_loss

        for i in range(num_views):
            anchor_embeddings = embeddings_list[i]

            for j in range(num_views):
                if i != j:
                    negative_embeddings = embeddings_list[j]

                    # Pairwise distances
                    pos_distance = F.pairwise_distance(inputs, anchor_embeddings)
                    neg_distance = F.pairwise_distance(inputs, negative_embeddings)

                    # Contrastive loss
                    contrastive_loss = F.relu(margin - pos_distance + neg_distance)

                    total_loss += contrastive_loss.mean()

        return total_loss
#---------------------------------

class MultiCriteriaRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(MultiCriteriaRecommender, self).__init__()

        # Define your neural network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input tensor to (num_samples, num_features) before passing through linear layers
        x = x.view(x.size(0), -1)
        
        # Continue with the rest of the neural network layers as before
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)  # Apply sigmoid activation
        return x
    
    def sigmoid_prediction(self, fused_embeddings):
        with torch.no_grad():  
            # Generate predictions using the model on the fused_embeddings
            sigmoid_output = self(fused_embeddings)

            # Apply sigmoid activation to the sigmoid_output
            sigmoid_output = 1 / (1 + torch.exp(-sigmoid_output))

        return fused_embeddings, sigmoid_output  # Return both original embeddings and sigmoid predictions
    
    def hadamard_product(self, fused_embeddings, sigmoid_predictions):
        # Ensure that sigmoid_predictions has the same shape as fused_embeddings
        sigmoid_predictions = sigmoid_predictions.view(fused_embeddings.shape)

        # Element-wise multiplication (Hadamard product) between fused_embeddings and sigmoid_predictions
        fused_embeddings_hadamard = fused_embeddings * sigmoid_predictions

        return fused_embeddings_hadamard

def mse_loss(inputs, reconstructions):
    mse = nn.MSELoss(reduction='mean')
    return mse(reconstructions, inputs)

def mse_with_l2_loss(inputs, reconstructions, model, l2_weight=0.01):
    mse = mse_loss(inputs, reconstructions)
    
    l2_reg = torch.tensor(0., device=inputs.device).clone().detach()
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)
    total_loss = mse + l2_weight * l2_reg
    return total_loss

def train_model(model, train_data, num_epochs, learning_rate=0.01, l2_weight=0.01):
    # Ensure consistent device placement for tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples, num_features = train_data.shape
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        train_data_tensor = train_data.clone().detach()
        outputs = model(train_data_tensor)

        # Reshape the outputs tensor to match the shape of train_data_tensor
        outputs_reshaped = outputs.view(train_data_tensor.shape)

        # Use sigmoid_prediction function
        fused_embeddings, sigmoid_predictions = model.sigmoid_prediction(train_data_tensor)
        fused_embeddings_hadamard = model.hadamard_product(fused_embeddings, sigmoid_predictions)

        loss = mse_with_l2_loss(train_data_tensor, outputs_reshaped, model, l2_weight)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    print("MLP Training completed.")

# -------------Recommendation Section -------------------------

def create_ground_truth_ratings(file_path, criteria):  
    data = pd.read_excel(file_path)
    user_id = data['User_ID']
    hotel_id = data['Hotel_ID']

    # Create a mapping from user/hotel IDs to unique integer indices
    user_id_map = {uid: i for i, uid in enumerate(user_id.unique())}
    hotel_id_map = {mid: i for i, mid in enumerate(hotel_id.unique())}

    num_users = len(user_id_map)
    num_hotels = len(hotel_id_map)
    num_criteria = len(criteria)
    ground_truth_ratings_matrix = np.zeros((num_users, num_hotels, num_criteria), dtype=np.int16)
    
    # Additional columns
    data['Overal_Rating'] = 0
    data['hotel_id'] = ''  # Add the 'hotel_id' column

    for i, row in data.iterrows():
        uid = row['User_ID']
        hid = row['Hotel_ID']
        criterion_ratings = [row[criterion] for criterion in criteria]
        
        # Calculate average rating for criteria with a rating greater than 0
        non_zero_ratings = [rating for rating in criterion_ratings if rating > 0]
        Overal_Rating = np.mean(non_zero_ratings) if non_zero_ratings else 0

        # Assign values to additional columns
        data.at[i, 'Overal_Rating'] = Overal_Rating
        data.at[i, 'hotel_id'] = hid
        
        if uid in user_id_map and hid in hotel_id_map:
            user_idx = user_id_map[uid]
            hotel_idx = hotel_id_map[hid]
            ground_truth_ratings_matrix[user_idx, hotel_idx] = criterion_ratings

    # # Print the resulting ground_truth_ratings_matrix
    # print("Ground Truth Ratings Matrix:")
    # print(ground_truth_ratings_matrix)

    # # Print the DataFrame with additional columns
    # print("\nDataFrame with Additional Columns:")
    # print(data)

    return data, ground_truth_ratings_matrix

# ------------------------------------------------------------------------------------------
# ---------------------------------Evaluate MCRS based-Prediction ---------------------------
# ---------------------------------------------------------------------------------------


#-------------Real Rating-----------------

# def P_Recommendation_item_TopK(fused_embeddings_hadamard, similarity_threshold, top_k_Pre, file_path, criteria):
#     data, _ = create_ground_truth_ratings(file_path, criteria)
#     recommendations_items = {}

#     num_users_actual, _ = fused_embeddings_hadamard.shape
#     fused_embeddings_hadamard_2d = fused_embeddings_hadamard.reshape((num_users_actual, -1))
#     similarities = cosine_similarity(fused_embeddings_hadamard_2d)

#     for i in range(num_users_actual):
#         # Filter out users with no positive Overal_Rating
#         if data.loc[data['User_ID'] == data.iloc[i]['User_ID'], 'Overal_Rating'].max() <= 0:
#             continue
#         similar_users_indices = [
#             j for j, similarity_score in enumerate(similarities[i])
#             if j != i and similarity_score >= similarity_threshold
#         ]

#         similar_user_hotels = data.iloc[similar_users_indices]
#         similar_user_rated_hotels = similar_user_hotels.groupby(['User_ID', 'hotel_id'])['Overal_Rating'].mean().reset_index()
#         similar_user_rated_hotels = similar_user_rated_hotels.sort_values(by='Overal_Rating', ascending=False)

#         k = top_k_Pre[i] if i < len(top_k_Pre) else top_k_Pre[-1]
#         recommended_hotels = similar_user_rated_hotels.head(k).to_dict(orient='records')

#         recommendations_items[data.iloc[i]['User_ID']] = {
#             'User_ID': data.iloc[i]['User_ID'],
#             'recommended_hotels': recommended_hotels,
#             'hotel_id': data.iloc[i]['hotel_id'],
#             'Overal_Rating': float(data.iloc[i]['Overal_Rating'])  # Convert to float
#         }
#     print("Recommendations:", recommendations_items)
    
#     return recommendations_items

# def P_Recommendation_item(fused_embeddings_hadamard, similarity_threshold, file_path, criteria):
#     data, _ = create_ground_truth_ratings(file_path, criteria)
#     recommendations_items = {}

#     num_users_actual, _ = fused_embeddings_hadamard.shape
#     fused_embeddings_hadamard_2d = fused_embeddings_hadamard.reshape((num_users_actual, -1))
#     similarities = cosine_similarity(fused_embeddings_hadamard_2d)

#     for i in range(num_users_actual):
#         # Filter out users with no positive Overal_Rating
#         if data.loc[data['User_ID'] == data.iloc[i]['User_ID'], 'Overal_Rating'].max() <= 0:
#             continue
#         similar_users_indices = [
#             j for j, similarity_score in enumerate(similarities[i])
#             if j != i and similarity_score >= similarity_threshold
#         ]

#         # Filter DataFrame based on similar_users_indices
#         similar_user_hotels = data.iloc[similar_users_indices]
#         similar_user_rated_hotels = similar_user_hotels.groupby(['User_ID', 'hotel_id'])['Overal_Rating'].mean().reset_index()
#         similar_user_rated_hotels = similar_user_rated_hotels.sort_values(by='Overal_Rating', ascending=False)

#         # Create the recommendation
#         recommended_hotels = similar_user_rated_hotels.to_dict(orient='records')

#         recommendations_items[data.iloc[i]['User_ID']] = {
#             'User_ID': data.iloc[i]['User_ID'],
#             'recommended_hotels': recommended_hotels,
#             'hotel_id': data.iloc[i]['hotel_id'],
#             'Overal_Rating': float(data.iloc[i]['Overal_Rating'])  # Convert to float
#         }
#     print("Recommendations:", recommendations_items)

#     return recommendations_items

def P_Recommendation_item_simplified(fused_embeddings_hadamard, file_path, criteria):
    data, _ = create_ground_truth_ratings(file_path, criteria)
    recommendations_items = {}

    num_users_actual, _ = fused_embeddings_hadamard.shape
    fused_embeddings_hadamard_2d = fused_embeddings_hadamard.reshape((num_users_actual, -1))
    similarities = cosine_similarity(fused_embeddings_hadamard_2d)

    # Counter variable to limit the number of printed users
    printed_users_count = 0

    for i in range(num_users_actual):
        similar_user_index = np.argmax(similarities[i])

        # Use iloc to select a row and pass it as a DataFrame
        similar_user_hotels = data.iloc[[similar_user_index]]

        # Assuming 'User_ID' is a column in data
        similar_user_rated_hotels = similar_user_hotels.groupby(['User_ID', 'Hotel_ID'])['Overal_Rating'].mean().reset_index()
        similar_user_rated_hotels = similar_user_rated_hotels.sort_values(by='Overal_Rating', ascending=False)

        # Create the recommendation
        recommended_hotels = similar_user_rated_hotels.to_dict(orient='records')

        # Add 'hotel_id' to each hotel dictionary
        for hotel in recommended_hotels:
            hotel['hotel_id'] = hotel['Hotel_ID']

        recommendations_items[data.iloc[i]['User_ID']] = {
            'User_ID': data.iloc[i]['User_ID'],
            'recommended_hotels': recommended_hotels,
            'hotel_id': data.iloc[i]['Hotel_ID'],
            'Overal_Rating': float(data.iloc[i]['Overal_Rating'])  # Convert to float
        }

        # Print recommendations for the specified number of users
        if printed_users_count < 10:
            print(f"Recommendations for User {data.iloc[i]['User_ID']}: {recommendations_items[data.iloc[i]['User_ID']]}")
            printed_users_count += 1
        else:
            break  # Break out of the loop once 10 users are printed

    return recommendations_items


def evaluate_recommendations_Prediction_Normalize(ground_truth_real_matrix, recommendations_items, user_id_map, hotel_id_map):
    predicted_ratings = np.zeros_like(ground_truth_real_matrix, dtype=np.float32)

    for user_id, recommendation in recommendations_items.items():
        hotels = recommendation['recommended_hotels']
        user_idx = user_id_map[recommendation['User_ID']]
        
        if len(hotels) > 0:
            # Calculate the average rating of recommended hotels
            avg_rating = np.mean([hotel['Overal_Rating'] for hotel in hotels])
            
            # Assign the average rating to all recommended hotels for the current user
            for hotel in hotels:
                hotel_idx = hotel_id_map[hotel['hotel_id']]
                predicted_ratings[user_idx, hotel_idx] = avg_rating

    actual_ratings = ground_truth_real_matrix.flatten()

    # Normalize ratings to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    actual_ratings_normalized = scaler.fit_transform(actual_ratings.reshape(-1, 1))
    predicted_ratings_normalized = scaler.transform(predicted_ratings.reshape(-1, 1))

    # Split the data into training and testing sets
    actual_train, actual_test, predicted_train, predicted_test = train_test_split(
        actual_ratings_normalized, predicted_ratings_normalized, test_size=0.3, random_state=42)

    mae = mean_absolute_error(actual_test, predicted_test)
    rmse = np.sqrt(mean_squared_error(actual_test, predicted_test))
    
    # Print the results
    print(f"\nMAE: {mae}")
    print(f"RMSE: {rmse}")
 
    return mae, rmse

def evaluate_recommendations_Prediction_Unnormalize(ground_truth_real_matrix, recommendations_items, user_id_map, hotel_id_map):
    predicted_ratings = np.zeros_like(ground_truth_real_matrix, dtype=np.float32)

    for user_id, recommendation in recommendations_items.items():
        hotels = recommendation['recommended_hotels']
        user_idx = user_id_map[recommendation['User_ID']]
        
        if len(hotels) > 0:
            # Calculate the average rating of recommended hotels
            avg_rating = np.mean([hotel['Overal_Rating'] for hotel in hotels])
            
            # Assign the average rating to all recommended hotels for the current user
            for hotel in hotels:
                hotel_idx = hotel_id_map[hotel['hotel_id']]
                predicted_ratings[user_idx, hotel_idx] = avg_rating

    actual_ratings = ground_truth_real_matrix.flatten()

    # Reshape predicted_ratings to have the same number of elements as actual_ratings
    predicted_ratings = np.reshape(predicted_ratings, actual_ratings.shape)

    # Split the data into training and testing sets
    actual_train, actual_test, predicted_train, predicted_test = train_test_split(
        actual_ratings, predicted_ratings, test_size=0.3, random_state=42)

    mae = mean_absolute_error(actual_test, predicted_test)
    rmse = np.sqrt(mean_squared_error(actual_test, predicted_test))
    
    # Print the results
    print(f"\nMAE: {mae}")
    print(f"RMSE: {rmse}")
 
    return mae, rmse




#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
#-------------------------------------------



def evaluate_top_k_recommendations(recommendations_items, ground_truth_real_matrix, top_k_values, user_id_map, hotel_id_map, test_size=0.3, random_state=42):
    results = {}

    # Split the data into training and testing sets
    actual_ratings = ground_truth_real_matrix.flatten()
    users = actual_ratings

    train_users, test_users = train_test_split(users, test_size=test_size, random_state=random_state)

    for k in top_k_values:
        # Initialize lists to store precision, recall, and f1 scores for each user
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Initialize dictionaries to store tp, fp, and fn for this value of 'k'
        tp_dict = {}
        fp_dict = {}
        fn_dict = {}

        for user_id in test_users:
            # Convert user_id to the appropriate type (int or str)
            user_id = int(user_id)  # or str(user_id), depending on the type of user_id

            recommendation = recommendations_items[user_id]
            true_positives = 0
            false_positives_count = 0
            false_negatives_count = 0

            user_idx = user_id_map[recommendation['User_ID']]
            user_actual_ratings = ground_truth_real_matrix[user_idx]

            # Skip users with no positive Overal_Rating
            if np.sum(user_actual_ratings) <= 0:
                continue

            recommended_hotels = [hotel['hotel_id'] for hotel in recommendation['recommended_hotels']]

            # Calculate true positives
            for hotel_id in recommended_hotels:
                hotel_idx = hotel_id_map[hotel_id]
                if user_actual_ratings[hotel_idx] > 0:
                    true_positives += 1

            # If a user has at least one true positive, add them to the list
            if true_positives >= 1:
                # Calculate false positives (recommended but not rated)
                false_positives_count = len(recommended_hotels) - true_positives

                # Calculate false negatives (rated but not recommended)
                false_negatives_count = np.sum(user_actual_ratings) - true_positives

                tp_dict[user_id] = true_positives
                fp_dict[user_id] = false_positives_count
                fn_dict[user_id] = false_negatives_count

                # Calculate precision
                if true_positives + false_positives_count > 0:
                    precision = true_positives / (true_positives + false_positives_count)
                else:
                    precision = 0
                precision_scores.append(precision)

                # Calculate recall
                if true_positives + false_negatives_count > 0:
                    recall = true_positives / (true_positives + false_negatives_count)
                else:
                    recall = 0
                recall_scores.append(recall)

                # Calculate F1-score
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0
                f1_scores.append(f1)

        # Calculate average precision, recall, and F1 score
        average_precision = sum(precision_scores) / len(precision_scores) if len(precision_scores) > 0 else 0
        average_recall = sum(recall_scores) / len(recall_scores) if len(recall_scores) > 0 else 0
        average_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0

        # Print metrics for each iteration
        print(f"For k={k}:")
        print("Precision:", average_precision)
        print("Recall:", average_recall)
        print("F1:", average_f1)

        results[k] = {
            'Precision': average_precision,
            'Recall': average_recall,
            'F1-score': average_f1
        }

    # Additional information
    total_users = len(actual_ratings)
    count_users_with_true_positives = len([user_id for user_id, tp_count in tp_dict.items() if tp_count >= 1])
    count_users_with_false_negatives = len([user_id for user_id, fn_count in fn_dict.items() if fn_count >= 1])

    print(f"Number of Users: {total_users}")
    print(f"Number of Users in Test Set: {len(test_users)}")
    print(f"Number of Users with True Positives >= 1: {count_users_with_true_positives}")
    print(f"Number of Users with False Negatives >= 1: {count_users_with_false_negatives}")

    return results


# ------------------------------------------------------------------------------------------
# ---------------------------------Evaluate MCRS based-Recommendation ---------------------------
# ---------------------------------------------------------------------------------------

# def create_ground_truth_ratings_TopK(file_path, criteria):  
#     data = pd.read_excel(file_path)
#     user_id = data['User_ID']
#     hotel_id = data['Hotel_ID']

#     # Create a mapping from user/hotel IDs to unique integer indices
#     user_id_map = {uid: i for i, uid in enumerate(user_id.unique())}
#     hotel_id_map = {mid: i for i, mid in enumerate(hotel_id.unique())}

#     num_users = len(user_id_map)
#     num_hotels = len(hotel_id_map)
#     num_criteria = len(criteria)
#     ground_truth_ratings_matrix = np.zeros((num_users, num_hotels, num_criteria), dtype=np.int16)
    
#     # Additional columns
#     data['Overal_Rating'] = 0
#     data['hotel_id'] = ''  # Add the 'hotel_id' column

#     for i, row in data.iterrows():
#         uid = row['User_ID']
#         hid = row['Hotel_ID']
#         criterion_ratings = [row[criterion] for criterion in criteria]
        
#         # Calculate average rating for criteria with a rating greater than 0
#         non_zero_ratings = [rating for rating in criterion_ratings if rating > 0]
#         Overal_Rating = np.mean(non_zero_ratings) if non_zero_ratings else 0

#         # Assign values to additional columns
#         data.at[i, 'Overal_Rating'] = Overal_Rating
#         data.at[i, 'hotel_id'] = hid
        
#         if uid in user_id_map and hid in hotel_id_map:
#             user_idx = user_id_map[uid]
#             hotel_idx = hotel_id_map[hid]
#             ground_truth_ratings_matrix[user_idx, hotel_idx] = criterion_ratings

#     return data, ground_truth_ratings_matrix

# def Recommendation_items_TopK(fused_embeddings_hadamard, similarity_threshold, top_k_values, file_path, criteria, ground_truth_ratings_matrix):
#     data, ground_truth_ratings_matrix = create_ground_truth_ratings(file_path, criteria)
#     recommendations = {}
#     actual_ratings = {}

#     num_users_actual, _ = fused_embeddings_hadamard.shape
#     fused_embeddings_hadamard_2d = fused_embeddings_hadamard.reshape((num_users_actual, -1))
#     similarities = cosine_similarity(fused_embeddings_hadamard_2d)

#     for i in range(num_users_actual):
#         # Filter out users with no positive Overal_Rating
#         if data.loc[data['User_ID'] == data.iloc[i]['User_ID'], 'Overal_Rating'].max() <= 0:
#             continue
#         similar_users_indices = [
#             j for j, similarity_score in enumerate(similarities[i])
#             if j != i and similarity_score >= similarity_threshold
#         ]

#         similar_users_ids = data.iloc[similar_users_indices]['User_ID'].tolist()

#         user_actual_ratings = data.iloc[i]['hotel_id']
#         actual_ratings[i] = user_actual_ratings if user_actual_ratings != '' else 'No ratings available'

#         similar_user_hotels = data[data['User_ID'].isin(similar_users_ids)]
#         similar_user_rated_hotels = similar_user_hotels.groupby(['User_ID', 'hotel_id'])['Overal_Rating'].mean().reset_index()
#         similar_user_rated_hotels = similar_user_rated_hotels.sort_values(by='Overal_Rating', ascending=False)

#         k = top_k_values[i] if i < len(top_k_values) else top_k_values[-1]
#         recommended_hotels = similar_user_rated_hotels.head(k).to_dict(orient='records')

#         # Collect User_IDs similar to this user
#         similar_user_ids_to_user = data.iloc[similar_users_indices]['User_ID'].tolist()

#         recommendations[i] = {
#             'User_ID': data.iloc[i]['User_ID'],
#             'similar_user_ids': similar_user_ids_to_user,
#             'recommended_hotels': recommended_hotels,
#             'hotel_id': data.iloc[i]['hotel_id'],
#             'Overal_Rating': data.iloc[i]['Overal_Rating']
#         }
        
#     return recommendations, actual_ratings

# def evaluate_top_k_recommendations(recommendations_items, actual_ratings, top_k_values, test_size=0.3, random_state=42):
#     results = {}

#     # Split the data into training and testing sets
#     users = list(actual_ratings.keys())
#     train_users, test_users = train_test_split(users, test_size=test_size, random_state=random_state)

#     for k in top_k_values:
#         # Initialize lists to store precision, recall, and f1 scores for each user
#         precision_scores = []
#         recall_scores = []
#         f1_scores = []

#         # Initialize dictionaries to store tp, fp, and fn for this value of 'k'
#         tp_dict = {}
#         fp_dict = {}
#         fn_dict = {}

#         for user_id in test_users:
#             recommendation = recommendations_items[user_id]
#             true_positives = 0
#             false_positives_count = 0
#             false_negatives_count = 0

#             user_actual_ratings = actual_ratings[user_id]

#             # Skip users with no positive Overal_Rating
#             if user_actual_ratings == 'No ratings available':
#                 continue

#             recommended_hotels = [hotel['hotel_id'] for hotel in recommendation['recommended_hotels']]

#             # Calculate true positives
#             for hotel_id in recommended_hotels:
#                 if hotel_id in user_actual_ratings:
#                     true_positives += 1

#             # If a user has at least one true positive, add them to the list
#             if true_positives >= 1:
#                 # Calculate false positives (recommended but not rated)
#                 false_positives_count = len(recommended_hotels) - len(user_actual_ratings)

#                 # Calculate false negatives (rated but not recommended)
#                 false_negatives_count = len(user_actual_ratings) - true_positives

#                 tp_dict[user_id] = true_positives
#                 fp_dict[user_id] = false_positives_count
#                 fn_dict[user_id] = false_negatives_count

#                 # Calculate precision
#                 if true_positives + false_positives_count > 0:
#                     precision = true_positives / (true_positives + false_positives_count)
#                 else:
#                     precision = 0
#                 precision_scores.append(precision)

#                 # Calculate recall
#                 if true_positives + false_negatives_count > 0:
#                     recall = true_positives / (true_positives + false_negatives_count)
#                 else:
#                     recall = 0
#                 recall_scores.append(recall)

#                 # Calculate F1-score
#                 if precision + recall > 0:
#                     f1 = 2 * (precision * recall) / (precision + recall)
#                 else:
#                     f1 = 0
#                 f1_scores.append(f1)

#         # Calculate average precision, recall, and F1 score
#         average_precision = sum(precision_scores) / len(precision_scores) if len(precision_scores) > 0 else 0
#         average_recall = sum(recall_scores) / len(recall_scores) if len(recall_scores) > 0 else 0
#         average_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0

#         results[k] = {
#             'Precision': average_precision,
#             'Recall': average_recall,
#             'F1-score': average_f1
#         }

#     # Calculate the count of users with true positives >= 1
#     count_users_with_true_positives = len([user_id for user_id, tp_count in tp_dict.items() if tp_count >= 1])
#     # Calculate the count of users with false negatives >= 1
#     count_users_with_false_negatives = len([user_id for user_id, fn_count in fn_dict.items() if fn_count >= 1])
#     total_users = len(actual_ratings)

#     print(f"Number of Users: {total_users}")
#     # Print the number of users in the test set
#     print(f"Number of Users in Test Set: {len(test_users)}")
#     print(f"Number of Users with True Positives >= 1: {count_users_with_true_positives}")
#     print(f"Number of Users with False Negatives >= 1: {count_users_with_false_negatives}")
#     # Print metrics for each iteration
#     print(f"For k={k}:")
#     print("Precision:", average_precision)
#     print("Recall:", average_recall)
#     print("F1:", average_f1)

#     return results


# ---------------------------------------------------------------------------------------
# Main Function ---------------------------
# ---------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    # Define the file path and criteria
    file_path = 'C://Yahoo//TripAdvisor.xlsx'
    criteria = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    # Call the read_data function to get user_id_map and hotel_id_map
    user_id_map, hotel_id_map, base_ground_truth_ratings = read_data(file_path, criteria)

    # Call other functions
    create_graph(file_path, criteria)
    print("**************************")
    create_subgraphs(file_path, criteria)

    # Read data from the Excel file and create ID mappings
    user_ids = list(user_id_map.keys())
    hotel_ids = list(hotel_id_map.keys())

    # Call the function to create and normalize adjacency matrices
    result = create_and_normalize_adjacency_matrices(file_path, criteria, user_ids, hotel_ids)

    # Print or use the 'result' variable as needed
    # print(result)
    
    matrices, _, _ = L_BGNN(file_path, criteria, user_ids, hotel_ids)
    matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7 = matrices
    
    matrices, user_id_to_index, user_index_to_id = L_BGNN(file_path, criteria, user_ids, hotel_ids)
    resized_matrices = resize_matrices(matrices)
                
    # Combine user_ids and hotel_ids into a single list to build a unique mapping
    combined_ids = np.concatenate((user_ids, hotel_ids))

    # Create a mapping of unique IDs to unique integer values
    unique_ids = np.unique(combined_ids)
    id_to_int = {id_: i for i, id_ in enumerate(unique_ids)}

    # Convert user_ids and hotel_ids to integers using the mapping
    user_ids_int = np.array([id_to_int[user_id] for user_id in user_ids])
    hotel_ids_int = np.array([id_to_int[hotel_id] for hotel_id in hotel_ids])
    
    # Convert user_ids_int and hotel_ids_int to PyTorch tensors
    user_ids_tensor = torch.tensor(user_ids_int).clone().detach()
    hotel_ids_tensor = torch.tensor(hotel_ids_int).clone().detach()
    
    #---Attention Embedding------
    # GAT and Fusion Embeddings
    model = GAT(in_channels=16, out_channels=128)
    result = model.Multi_Embd(resized_matrices, user_ids_tensor, hotel_ids_tensor, num_epochs=100, learning_rate=0.01)
    print(result)
    fused_embeddings_with_ids = result  # unpack the values you need
    print("Fused Embeddings:")
    print(fused_embeddings_with_ids)
    print("*********************")
    print("*********************")
    #---MLP model------
    # Recommendation section
    num_samples = fused_embeddings_with_ids.shape[0]

    # Create an instance of the MultiCriteriaRecommender class
    output_dim = fused_embeddings_with_ids.shape[1]  # Set output_dim to the number of criteria
    model = MultiCriteriaRecommender(fused_embeddings_with_ids.shape[1], 32, output_dim)

    # Convert fused_embeddings_with_ids to a torch tensor
    fused_embeddings_tensor = fused_embeddings_with_ids.clone().detach().to(torch.float32)

    # Reshape fused_embeddings_tensor to match the expected shape
    num_samples, num_features = fused_embeddings_tensor.shape
    num_users = len(user_ids)
    num_criteria = len(criteria)
    num_hotels = len(hotel_ids)

    # Calculate the total number of features per criterion
    num_features_per_criterion = num_features // num_criteria
    
    # Train the model
    train_model(model, fused_embeddings_tensor, num_epochs=100, learning_rate=0.01, l2_weight=0.01)

    # After training, you can call the methods for prediction
    fused_embeddings, sigmoid_predictions = model.sigmoid_prediction(fused_embeddings_tensor)
    fused_embeddings_hadamard = model.hadamard_product(fused_embeddings, sigmoid_predictions)
    print("Shape of fused_embeddings_hadamard:", fused_embeddings_hadamard.shape)
    
    # Call the create_ground_truth_ratings function
    ground_truth_ratings = create_ground_truth_ratings(file_path, criteria)

    # Create a DataFrame with user and hotel identifiers as MultiIndex
    df_users_hotels = pd.DataFrame(index=pd.MultiIndex.from_tuples([(user_id, hotel_id) for user_id in user_id_map.keys() for hotel_id in hotel_id_map.keys()]))

    # Set parameters
    similarity_threshold = 0.5  # Adjust as needed
    top_k_values = [10] * len(user_id_map)  # Set the default value of k for each user
    
    
    top_k_Pre=[1]
    # Call the create_real_ratings function
    data, ground_truth_real_matrix = create_ground_truth_ratings(file_path, criteria)
    # Call the P_Recommendation_item function
    # recommendations_items = P_Recommendation_item(fused_embeddings_hadamard, similarity_threshold, top_k_Pre, file_path, criteria)
    recommendations_items = P_Recommendation_item_simplified(fused_embeddings_hadamard, file_path, criteria)
        
    mae,rmse=evaluate_recommendations_Prediction_Normalize(ground_truth_real_matrix, recommendations_items, user_id_map, hotel_id_map)
    # Call this function after calling evaluate_recommendations_Prediction
    mae, rmse = evaluate_recommendations_Prediction_Unnormalize(ground_truth_real_matrix, recommendations_items, user_id_map, hotel_id_map)

    #--------------------------------
    # # Generate ground truth ratings matrix using the provided file_path and criteria
    # data, ground_truth_ratings_matrix = create_ground_truth_ratings_TopK(file_path, criteria)

    # # Call the Recommendation_items function
    # recommendations, actual_ratings = Recommendation_items_TopK(fused_embeddings_hadamard, similarity_threshold, top_k_values, file_path, criteria, ground_truth_ratings_matrix)

    # # Call the evaluate_top_k_recommendations function
    # results = evaluate_top_k_recommendations(recommendations, actual_ratings, top_k_values, test_size=0.3, random_state=42)
    # # Example usage:
    # results = evaluate_top_k_recommendations(recommendations_items, ground_truth_real_matrix, top_k_values, user_id_map, hotel_id_map)




