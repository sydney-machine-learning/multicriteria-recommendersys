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



def read_data(file_path, criteria):
    data = pd.read_excel(file_path)
    user_id = data['User_ID']
    movie_id = data['Movies_ID']
    user_id_map = {uid: i for i, uid in enumerate(user_id.unique())}
    movie_id_map = {mid: i for i, mid in enumerate(movie_id.unique())}
    num_users = len(user_id_map)
    num_movies = len(movie_id_map)
    num_criteria = len(criteria)
    base_ground_truth_ratings = np.zeros((num_users, num_movies, num_criteria), dtype=np.int32)

    for i, row in data.iterrows():
        uid = row['User_ID']
        mid = row['Movies_ID']
        criterion_ratings = [row[criterion] for criterion in criteria]
        if uid in user_id_map and mid in movie_id_map:
            user_idx = user_id_map[uid]
            movie_idx = movie_id_map[mid]
            base_ground_truth_ratings[user_idx, movie_idx] = criterion_ratings

    return user_id_map, movie_id_map, base_ground_truth_ratings

def create_bipartite_graph(file_path, criteria):
    data = pd.read_excel(file_path)
    G = nx.MultiGraph()

    users = set()
    movies = set()

    for uid in data['User_ID']:
        G.add_node(uid, bipartite=0)
        users.add(uid)

    for mid in data['Movies_ID']:
        G.add_node(mid, bipartite=1)
        movies.add(mid)

    for i in range(len(data)):
        uid = data['User_ID'][i]
        mid = data['Movies_ID'][i]

        for criterion in criteria:
            rating = data[criterion][i]

            if rating > 0:
                G.add_edge(uid, mid, criterion=criterion, weight=rating)

    print(f"Number of user nodes: {len(users)}")
    print(f"Number of movie nodes: {len(movies)}")

    user_movie_edges = [(u, v, data) for u, v, data in G.edges(data=True) if u in users and v in movies]
    print(f"Number of edges between user and movie nodes: {len(user_movie_edges)}")

    for u, v, data in G.edges(data=True):
        if u in users and v in movies and 'criterion' in data and 'weight' in data:
            user_id = u
            movie_id = v
            criterion = data['criterion']
            rating = data['weight']  # Use the correct attribute name

            # print(f"Edge between User_ID {user_id} and Movies_ID {movie_id} (Criterion: {criterion}):")
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
        mid = graph_data['Movies_ID'][i]

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
        movie_nodes = [node for node in subgraph.nodes() if subgraph.nodes[node]['bipartite'] == 1]
        # print(f"Number of user nodes: {len(user_nodes)}")
        # print(f"Number of movie nodes: {len(movie_nodes)}")

        subgraph_edges = [(u, v, data) for u, v, data in subgraph.edges(data=True)]
        # print(f"Number of edges in subgraph: {len(subgraph_edges)}")

def create_and_normalize_adjacency_matrices(file_path, criteria, user_ids, movie_ids):
    graph_data = pd.read_excel(file_path)
    bgnn_matrices = []  # Initialize a list to store the BGNN matrices for each criterion

    user_id_to_index = {}
    user_index_to_id = {}
    movie_id_to_index = {}
    movie_index_to_id = {}

    for criterion in criteria:
        subgraph = nx.Graph()

        for i in range(len(graph_data)):
            uid = graph_data['User_ID'][i]
            mid = graph_data['Movies_ID'][i]

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

    # Create mappings from IDs to indices and vice versa for users and movies
    for idx, user_id in enumerate(user_ids):
        user_id_to_index[user_id] = idx
        user_index_to_id[idx] = user_id

    for idx, movie_id in enumerate(movie_ids):
        movie_id_to_index[movie_id] = idx
        movie_index_to_id[idx] = movie_id

    return bgnn_matrices, user_id_to_index, user_index_to_id, movie_id_to_index, movie_index_to_id

def L_BGNN(file_path, criteria, user_ids, movie_ids):
    graph_data = pd.read_excel(file_path)
    matrices = []  # Initialize a list to store the normalized matrices for each criterion

    user_id_to_index = {}
    user_index_to_id = {}
    movie_id_to_index = {}
    movie_index_to_id = {}

    for criterion in criteria:
        subgraph = nx.Graph()

        for i in range(len(graph_data)):
            uid = graph_data['User_ID'][i]
            mid = graph_data['Movies_ID'][i]

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

    # Create mappings from IDs to indices and vice versa for users and movies
    for idx, user_id in enumerate(user_ids):
        user_id_to_index[user_id] = idx
        user_index_to_id[idx] = user_id

    for idx, movie_id in enumerate(movie_ids):
        movie_id_to_index[movie_id] = idx
        movie_index_to_id[idx] = movie_id

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

    def Multi_Embd(self, matrices, user_ids, movie_ids, num_epochs=100, learning_rate=0.01):
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

        fused_embeddings = self.fusion_embeddings_vectors(embeddings_list, user_ids, movie_ids)

        return fused_embeddings
    
    def fusion_embeddings_vectors(self, embeddings_list, user_ids, movie_ids):
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
    movie_id = data['Movies_ID']

    # Create a mapping from user/movie IDs to unique integer indices
    user_id_map = {uid: i for i, uid in enumerate(user_id.unique())}
    movie_id_map = {mid: i for i, mid in enumerate(movie_id.unique())}

    num_users = len(user_id_map)
    num_movies = len(movie_id_map)
    num_criteria = len(criteria)
    ground_truth_ratings_matrix = np.zeros((num_users, num_movies, num_criteria), dtype=np.int16)
    
    # Additional columns
    data['Overal_Rating'] = 0
    data['movie_id'] = ''  # Add the 'movie_id' column
    data['Number_Rated_Items'] = 0  # Add the 'Number_Rated_Items' column

    for i, row in data.iterrows():
        uid = row['User_ID']
        mid = row['Movies_ID']
        criterion_ratings = [row[criterion] for criterion in criteria]
        
        # Calculate average rating for criteria with a rating greater than 0
        non_zero_ratings = [rating for rating in criterion_ratings if rating > 0]
        Overal_Rating = np.mean(non_zero_ratings) if non_zero_ratings else 0

        # Assign values to additional columns
        data.at[i, 'Overal_Rating'] = Overal_Rating
        data.at[i, 'movie_id'] = mid

        # Calculate and assign the number of rated items by each user
        num_rated_items_by_user = np.sum(data[data['User_ID'] == uid][criteria].apply(lambda x: (x > 0).any(), axis=1))
        data.at[i, 'Number_Rated_Items'] = num_rated_items_by_user
        
        if uid in user_id_map and mid in movie_id_map:
            user_idx = user_id_map[uid]
            movie_idx = movie_id_map[mid]
            ground_truth_ratings_matrix[user_idx, movie_idx] = criterion_ratings

    return data, ground_truth_ratings_matrix

def normalize_hadamard_embeddings(fused_embeddings_hadamard):
    # Detach PyTorch tensors
    fused_embeddings_hadamard = fused_embeddings_hadamard.detach().numpy()

    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()

    # Fit the scaler on the summed_embeddings and transform them
    normalized_embeddings = scaler.fit_transform(fused_embeddings_hadamard)

    return normalized_embeddings

def Generate_Recommendation(normalized_embeddings, file_path, criteria, threshold=0.9):
    
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

        similar_user_movies = data.iloc[similar_user_index]

        similar_user_rated_movies = similar_user_movies.groupby(['User_ID', 'Movies_ID'])['Overal_Rating'].mean().reset_index()
        similar_user_rated_movies = similar_user_rated_movies.sort_values(by='Overal_Rating', ascending=False)

        # Apply the threshold to filter out low-rated recommendations
        similar_user_rated_movies = similar_user_rated_movies[similar_user_rated_movies['Overal_Rating'] >= threshold]

        # Take the top-K recommendations after applying the threshold
        similar_user_rated_movies = similar_user_rated_movies.head(top_k_user)

        # Create the recommendation
        recommended_movies = similar_user_rated_movies.to_dict(orient='records')

        # Add 'movie_id' to each movie dictionary
        for movie in recommended_movies:
            movie['movie_id'] = movie['Movies_ID']

        recommendations_items[user_id] = {
            'User_ID': user_id,
            'recommended_movies': recommended_movies,
            'movie_id': data.iloc[user_index]['Movies_ID'],
            'Overal_Rating': float(data.iloc[user_index]['Overal_Rating'])  
        }

    # Print the number of recommendations for each user outside the loop
    for user_id, recommendation in recommendations_items.items():
        print(f"User {user_id} has {len(recommendation['recommended_movies'])} recommendations.")

    return recommendations_items

def evaluate_Recommendations_Prediction(ground_truth_real_matrix, recommendations_items, user_id_map, movie_id_map, data):
    predicted_ratings = np.zeros_like(ground_truth_real_matrix, dtype=np.float32)
    actual_ratings = []
    indices = []

    for user_id, recommendation in recommendations_items.items():
        movies = recommendation['recommended_movies']
        user_idx = user_id_map[recommendation['User_ID']]
        
        if len(movies) > 0:
            # Calculate the average rating of recommended movies
            avg_rating = np.mean([movie['Overal_Rating'] for movie in movies])

            for movie in movies:
                movie_idx = movie_id_map[movie['movie_id']]
                # Assign the average rating to the predicted rating matrix
                predicted_ratings[user_idx, movie_idx] = avg_rating
                
                # Check if the user actually rated the movie and store the actual rating and index
                actual_rating = ground_truth_real_matrix[user_idx, movie_idx]
                if np.any(actual_rating != 0):
                    actual_ratings.append(actual_rating)
                    indices.append((user_idx, movie_idx))

    actual_ratings = np.array(actual_ratings)
    indices = np.array(indices)
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=42)
    # Split the indices into training and testing sets
    train_indices, test_indices, _, _ = train_test_split(indices, actual_ratings, test_size=0.3, random_state=42)

    # Extract corresponding values from the predicted ratings matrix
    actual_train = ground_truth_real_matrix[train_indices[:, 0], train_indices[:, 1]]
    actual_test = ground_truth_real_matrix[test_indices[:, 0], test_indices[:, 1]]
    predicted_train = predicted_ratings[train_indices[:, 0], train_indices[:, 1]]
    predicted_test = predicted_ratings[test_indices[:, 0], test_indices[:, 1]]

    mae = mean_absolute_error(actual_test, predicted_test)
    rmse = np.sqrt(mean_squared_error(actual_test, predicted_test))
    
    # Normalize actual and predicted ratings between 0 and 1
    scaler = MinMaxScaler()
    actual_test_normalized = scaler.fit_transform(actual_test.reshape(-1, 1)).flatten()
    predicted_test_normalized = scaler.transform(predicted_test.reshape(-1, 1)).flatten()

    mae_normalized = mean_absolute_error(actual_test_normalized, predicted_test_normalized)
    rmse_normalized = np.sqrt(mean_squared_error(actual_test_normalized, predicted_test_normalized))

    # Print the results
    print(f"\nMAE: {mae}")
    print(f"RMSE: {rmse}")
    # Print the results
    print(f"\nMAE (Normalized): {mae_normalized}")
    print(f"RMSE (Normalized): {rmse_normalized}")
 
    return mae, rmse, mae_normalized, rmse_normalized

#********************************************************************************
def Evaluate_RS_ManualMetrics(ground_truth_real_matrix, recommendations_items, user_id_map, movie_id_map):
    actual_ratings = []
    indices = []

    total_tp = 0  # Total True Positives
    total_fp = 0  # Total False Positives
    total_fn = 0  # Total False Negatives

    for user_id, recommendation in recommendations_items.items():
        tp = 0  # Reset True Positives for each user
        fp = 0  # Reset False Positives for each user
        fn = 0  # Reset False Negatives for each user

        movies = recommendation['recommended_movies']
        user_idx = user_id_map[recommendation['User_ID']]

        if len(movies) > 0:
            # Extract actual ratings and indices for the current user
            user_actual_ratings = ground_truth_real_matrix[user_idx, :]
            actual_ratings.extend(user_actual_ratings)
            indices.extend([(user_idx, i) for i in range(len(user_actual_ratings))])

            recommended_movies_count = len(movies)

            # Calculate True Positives, False Positives, and False Negatives for the current user
            for i in range(len(user_actual_ratings)):
                if np.any(user_actual_ratings[i] > 0):
                    if i in [movie_id_map[movie['movie_id']] for movie in movies]:
                        tp += 1
                    else:
                        fn += 1
                elif i in [movie_id_map[movie['movie_id']] for movie in movies]:
                    fp += 1

            # Print information for each user, including the total number of rated items
            print(f"User ID: {user_id}, True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

        total_tp += tp
        total_fp += fp
        total_fn += fn

    actual_ratings = np.array(actual_ratings)
    indices = np.array(indices)

    # Split the indices into training and testing sets
    train_indices, test_indices, _, _ = train_test_split(indices, actual_ratings, test_size=0.3, random_state=42)

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
        ["False Positives (fp)", total_fp],
        ["False Negatives (fn)", total_fn],
        ["Precision", precision],
        ["Recall", recall],
        ["F1", f1],
        ["F2", f2],
    ]

    print(tabulate(results_table, headers=["Manual Metrics", "Score"], tablefmt="grid"))

    return precision, recall, f1, f2


#**************************************************************************************

def reciprocal_rank(y_true, y_score):
    # Flatten y_true and y_score
    y_true_flat = np.ravel(y_true)
    y_score_flat = np.ravel(y_score)

    order = np.argsort(y_score_flat)[::-1]
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)

    # Find the index of the first relevant item
    first_rank_index = np.argmax(y_true_flat[order])

    # Check if there is at least one relevant item
    if y_true_flat[order[first_rank_index]] == 0:
        return 0.0

    return 1.0 / ranks[first_rank_index]

def calculate_map(y_true, y_score):
    # Calculate Average Precision (AP)
    ap_score = average_precision_score(y_true, y_score)

    return ap_score

def Evaluate_RS_LibraryMetrics(ground_truth_real_matrix, recommendations_items, user_id_map, movie_id_map):
    predicted_ratings = np.zeros_like(ground_truth_real_matrix, dtype=np.float32)
    actual_ratings = []
    indices = []

    for user_id, recommendation in recommendations_items.items():
        movies = recommendation['recommended_movies']
        user_idx = user_id_map[recommendation['User_ID']]

        if len(movies) > 0:
            # Calculate the average rating of recommended movies
            avg_rating = np.mean([movie['Overal_Rating'] for movie in movies])

            for movie in movies:
                movie_idx = movie_id_map[movie['movie_id']]
                # Assign the average rating to the predicted rating matrix
                predicted_ratings[user_idx, movie_idx] = avg_rating

                # Check if the user actually rated the movie and store the actual rating and index
                actual_rating = ground_truth_real_matrix[user_idx, movie_idx]
                if np.any(actual_rating != 0):
                    actual_ratings.append(actual_rating)
                    indices.append((user_idx, movie_idx))

    actual_ratings = np.array(actual_ratings)
    indices = np.array(indices)

    # Split the indices into training and testing sets
    train_indices, test_indices, _, _ = train_test_split(indices, actual_ratings, test_size=0.3, random_state=42)

    # Extract corresponding values from the predicted ratings matrix
    actual_train = ground_truth_real_matrix[train_indices[:, 0], train_indices[:, 1]]
    actual_test = ground_truth_real_matrix[test_indices[:, 0], test_indices[:, 1]]
    predicted_train = predicted_ratings[train_indices[:, 0], train_indices[:, 1]]
    predicted_test = predicted_ratings[test_indices[:, 0], test_indices[:, 1]]

    # Consider Overal_Rating > 8 as positive ratings
    positive_test = actual_test > 8
    positive_predicted = predicted_test > 8

    # Calculate true positives based on the new condition
    true_positives = np.sum(np.logical_and(positive_test, positive_predicted))

    # Calculate false negatives
    false_negatives = np.sum(np.logical_and(positive_test, ~positive_predicted))
    
    # Calculate precision, recall, and F1 based on the modified true positives
    precision = precision_score(positive_test, positive_predicted, average='macro')
    recall = recall_score(positive_test, positive_predicted, average='macro')
    f1 = f1_score(positive_test, positive_predicted, average='macro')

    # Calculate F2 score
    beta = 2  # You can adjust the value of beta according to your preference
    f2 = fbeta_score(positive_test, positive_predicted, beta=beta, average='macro')

    # Calculate Mean Average Precision (MAP)
    map_scores = []

    for i in range(len(test_indices)):
        user_idx, movie_idx = test_indices[i]
        user_actual = ground_truth_real_matrix[user_idx, movie_idx]
        user_predicted = predicted_test[i]

        if np.any(user_actual > 8):  # Use np.any() instead of if user_actual > 8
            # If the user actually rated the movie higher than 8, calculate AP for this user
            ap_score = calculate_map(user_actual > 8, user_predicted)
            map_scores.append(ap_score)

    if len(map_scores) > 0:
        map_score = np.mean(map_scores)
    else:
        map_score = 0.0  # or any other appropriate value

    # Flatten actual_test and predicted_test
    actual_test_flat = actual_test.flatten()
    predicted_test_flat = predicted_test.flatten()

    # Calculate Mean Reciprocal Rank (MRR)
    mrr_score = reciprocal_rank(actual_test_flat, predicted_test_flat)

    # Calculate Average Precision (AP)
    ap_score = average_precision_score(positive_test, positive_predicted, average='macro')

    # Print the results
    results_table = [
        ["True Positives (tp)", true_positives],
        ["False Negatives (fn)", false_negatives],
        ["Precision", precision],
        ["Recall", recall],
        ["F1", f1],
        ["F2", f2],
        ["MAP", map_score],
        ["MRR", mrr_score],
        ["AP", ap_score],
    ]

    print(tabulate(results_table, headers=["Library Metrics ", "Score"], tablefmt="pretty"))

    return precision, recall, f1, f2, map_score, mrr_score, ap_score

# ---------------------------------------------------------------------------------------
# Main Function ---------------------------
# ---------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    # Define the file path and criteria
    # file_path = '/home/z5318340/MoviesDatasetYahoo.xlsx'
    file_path = 'C://Yahoo//Movies.xlsx'
    criteria = ['C1', 'C2', 'C3', 'C4']

    # Call the read_data function to get user_id_map and movie_id_map
    user_id_map, movie_id_map, base_ground_truth_ratings = read_data(file_path, criteria)

    # Call other functions
    create_bipartite_graph(file_path, criteria)
    print("**************************")
    create_subgraphs(file_path, criteria)

    # Read data from the Excel file and create ID mappings
    user_ids = list(user_id_map.keys())
    movie_ids = list(movie_id_map.keys())

    # Call the function to create and normalize adjacency matrices
    result = create_and_normalize_adjacency_matrices(file_path, criteria, user_ids, movie_ids)

    # Print or use the 'result' variable as needed
    # print(result)
    
    matrices, _, _ = L_BGNN(file_path, criteria, user_ids, movie_ids)
    matrix1, matrix2, matrix3, matrix4 = matrices
    
    matrices, user_id_to_index, user_index_to_id = L_BGNN(file_path, criteria, user_ids, movie_ids)
    resized_matrices = resize_matrices(matrices)
                
    # Combine user_ids and movie_ids into a single list to build a unique mapping
    combined_ids = np.concatenate((user_ids, movie_ids))

    # Create a mapping of unique IDs to unique integer values
    unique_ids = np.unique(combined_ids)
    id_to_int = {id_: i for i, id_ in enumerate(unique_ids)}

    # Convert user_ids and movie_ids to integers using the mapping
    user_ids_int = np.array([id_to_int[user_id] for user_id in user_ids])
    movie_ids_int = np.array([id_to_int[movie_id] for movie_id in movie_ids])
    
    # Convert user_ids_int and movie_ids_int to PyTorch tensors
    user_ids_tensor = torch.tensor(user_ids_int).clone().detach()
    movie_ids_tensor = torch.tensor(movie_ids_int).clone().detach()
    
    #---Attention Embedding------
    # GAT and Fusion Embeddings
    model = GAT(in_channels=16, out_channels=256)
    result = model.Multi_Embd(resized_matrices, user_ids_tensor, movie_ids_tensor, num_epochs=100, learning_rate=0.01)
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
    num_movies = len(movie_ids)

    # Calculate the total number of features per criterion
    num_features_per_criterion = num_features // num_criteria
    
    # Train the model
    train_model(model, fused_embeddings_tensor, num_epochs=100, learning_rate=0.01, l2_weight=0.01)

    # After training, you can call the methods for prediction
    fused_embeddings, sigmoid_predictions = model.sigmoid_prediction(fused_embeddings_tensor)
    fused_embeddings_hadamard = model.hadamard_product(fused_embeddings, sigmoid_predictions)
    # print("Shape of fused_embeddings_hadamard:", fused_embeddings_hadamard.shape)
    
    # Call the create_ground_truth_ratings function
    ground_truth_ratings = create_ground_truth_ratings(file_path, criteria)

    # Create a DataFrame with user and movie identifiers as MultiIndex
    df_users_movies = pd.DataFrame(index=pd.MultiIndex.from_tuples([(user_id, movie_id) for user_id in user_id_map.keys() for movie_id in movie_id_map.keys()]))
    
    # normalized_Embeddings vectors of summed_embeddings
    normalized_H_F_embeddings = normalize_hadamard_embeddings(fused_embeddings_hadamard)
    print("Normalize:",normalized_H_F_embeddings)
    
    # Call the create_real_ratings function
    data, ground_truth_real_matrix = create_ground_truth_ratings(file_path, criteria)
    # Call the P_Recommendation_item function
    # recommendations_items = P_Recommendation_item(normalized_embeddings, similarity_threshold, top_k_Pre, file_path, criteria)
    recommendations_items = Generate_Recommendation(normalized_H_F_embeddings, file_path, criteria, threshold=0.9)
        
    # Call this function after calling evaluate_recommendations_Prediction
    mae, rmse, mae_normalized, rmse_normalized = evaluate_Recommendations_Prediction(ground_truth_real_matrix, recommendations_items, user_id_map, movie_id_map, data)
    precision, recall, f1, f2 = Evaluate_RS_ManualMetrics(ground_truth_real_matrix, recommendations_items, user_id_map, movie_id_map)
    precision, recall, f1, f2, map_score, mrr_score, ap_score = Evaluate_RS_LibraryMetrics(ground_truth_real_matrix, recommendations_items, user_id_map, movie_id_map)




