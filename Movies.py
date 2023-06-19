#!/usr/bin/env -S python3 -u

#PBS -N Saman
#PBS -l select=1:ncpus=32:mem=1024gb
#PBS -l walltime=200:00:00
#PBS -j oe
#PBS -k oed
#PBS -M s.forouzandeh@unsw.edu.au
#PBS -m ae


import os
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_recall_fscore_support
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv


# Real Rating for Evaluation and Cosine Users nodes
# Keep track of Users and Movies that have already been added to the graph
Users = set()
Movies = set()

# Define the file path
file_path = '/home/z5318340/Movies_with_missing_values.xlsx'

# Read the Excel file
data = pd.read_excel(file_path)

# Extract the required columns
user_id = data['User_ID']
movie_id = data['Movies_ID']
criteria = ['C1', 'C2', 'C3', 'C4']

# Create the graph
G = nx.MultiGraph()

# Add user IDs as nodes
for uid in user_id:
    G.add_node(uid, bipartite=0)
    Users.add(uid)

# Add movie IDs as nodes
for mid in movie_id:
    G.add_node(mid, bipartite=1)
    Movies.add(mid)

# Add edges between user IDs and movie IDs for each criterion
for criterion in criteria:
    for i in range(len(data)):
        uid = user_id[i]
        mid = movie_id[i]
        rating = data[criterion][i]
        if rating > 0:
            G.add_edge(uid, mid, criterion=criterion, rating=rating)
# Create a dictionary to store the subgraphs
subgraphs = {}

# Create subgraphs based on each criterion
for criterion in criteria:
    # Filter the edges based on the criterion
    edges = [(u, v, data) for u, v, data in G.edges(data=True) if data['criterion'] == criterion and data['rating'] > 0]

    # Create a new subgraph
    subgraph = nx.MultiGraph()
    subgraph.add_edges_from(edges)

    # Store the subgraph in the dictionary
    subgraphs[criterion] = subgraph

    # Print the number of nodes and edges in the subgraph
    # print(f"Subgraph for criterion {criterion}:")
    # print("Number of nodes:", len(subgraph.nodes()))
    # print("Number of edges:", len(subgraph.edges()))
    # print()

# #**********************BGNN Adjacency Matrices based Criteria ************************
# Create node index dictionary
node_idx_dicts = []
adj_matrices = []

# Iterate over each criterion
for criterion in criteria:
    # Get the subgraph for the current criterion
    subgraph = subgraphs[criterion]

    # Create node index dictionary
    node_idx_dict = {}
    for i, node in enumerate(subgraph.nodes()):
        node_idx_dict[node] = i
    node_idx_dicts.append(node_idx_dict)

    n_nodes = len(subgraph.nodes())
    adj_matrix = np.zeros((n_nodes, n_nodes))
    adj_matrices.append(adj_matrix)

    for uid, mid, data in subgraph.edges(data=True):
        uid_idx = node_idx_dict[uid]
        mid_idx = node_idx_dict[mid]
        rating = data['rating']
        adj_matrix[uid_idx][mid_idx] = rating
        adj_matrix[mid_idx][uid_idx] = rating

# Extracted adjacency matrices in BGNN format
A_C_matrices = []

# Iterate over each pair of adjacency matrices
for idx in range(len(adj_matrices)):
    BC_uv = adj_matrices[idx]
    if idx < len(adj_matrices) - 1:
        BC_vu = adj_matrices[idx + 1]
    else:
        BC_vu = adj_matrices[0]  # Wrap around to the first adjacency matrix

    # Calculate the degree matrix of B_uv and B_vu
    DC_uv = np.diag(np.sum(BC_uv, axis=1))
    DC_vu = np.diag(np.sum(BC_vu, axis=1))

    # Normalize B_uv and B_vu
    BC_uv_norm = np.linalg.inv(DC_uv) @ BC_uv
    BC_vu_norm = np.linalg.inv(DC_vu) @ BC_vu

    # Update max_rows and max_cols if necessary
    max_rows = max(BC_uv_norm.shape[0], BC_vu_norm.shape[0])
    max_cols = max(BC_uv_norm.shape[1], BC_vu_norm.shape[1])

    # Ensure that B_uv_norm and B_vu_norm have the same number of rows and columns
    BC_uv_norm = np.pad(BC_uv_norm, ((0, max_rows - BC_uv_norm.shape[0]), (0, max_cols - BC_uv_norm.shape[1])), mode='constant')
    BC_vu_norm = np.pad(BC_vu_norm, ((0, max_rows - BC_vu_norm.shape[0]), (0, max_cols - BC_vu_norm.shape[1])), mode='constant')

    # Construct BGNN adjacency matrix
    A_C = np.block([[np.zeros_like(BC_vu_norm), BC_vu_norm],
                    [BC_uv_norm, np.zeros_like(BC_uv_norm)]])

    # Append to the list of A_C matrices
    A_C_matrices.append(A_C)

# Get the maximum number of rows and columns in the concatenated matrices
max_rows = max(A_C.shape[0] for A_C in A_C_matrices)
max_cols = max(A_C.shape[1] for A_C in A_C_matrices)

# Pad each matrix to have the same number of rows and columns
A_C_matrices_padded = [np.pad(A_C, ((0, max_rows - A_C.shape[0]), (0, max_cols - A_C.shape[1])) , mode='constant') for A_C in A_C_matrices]

# Concatenate all adjacency matrices in A_C_matrices_padded
A_C_concatenated = np.block(A_C_matrices_padded)
# Number of matrices in A_C_matrices
num_matrices = len(A_C_matrices)

zero_indices = np.where(A_C_concatenated == 0)
num_zeros = len(zero_indices[0])
print(f"Number of cells with zero values: {num_zeros}")
    

#**********************Graph Attention Network (GAT) ************************
# Define the GAT model
class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.conv_layers = nn.ModuleList([
            GATConv(in_channels, self.head_dim, heads=1) for _ in range(num_heads)
        ])
        self.fc = nn.Linear(out_channels, out_channels * num_heads)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        head_outs = []
        for conv in self.conv_layers:
            head_out = conv(x, edge_index)
            head_outs.append(head_out)
        x = torch.cat(head_outs, dim=1)
        x = self.leakyrelu(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

# Define the similarity loss function
def similarity_loss(embeddings_list):
    similarity_loss_value = torch.tensor(0.0)
    num_views = len(embeddings_list)

    for i in range(num_views):
        for j in range(i + 1, num_views):
            view_i_embeddings = embeddings_list[i]
            view_j_embeddings = embeddings_list[j]

            # Align tensors to the same size
            min_size = min(view_i_embeddings.size(0), view_j_embeddings.size(0))
            view_i_embeddings = view_i_embeddings[:min_size, :]
            view_j_embeddings = view_j_embeddings[:min_size, :]
            similarity_loss_value += torch.mean(torch.abs(view_i_embeddings - view_j_embeddings))
    return similarity_loss_value

def binary_cross_entropy_loss(inputs, reconstructions, embeddings_list, l2_weight=0.01):
    bce_loss = F.binary_cross_entropy_with_logits(reconstructions, inputs)
    similarity_loss_value = similarity_loss(embeddings_list)
    total_loss = bce_loss + l2_weight * similarity_loss_value
    return total_loss

def train(model, optimizer, data, target, embeddings_list):
    model.train()
    optimizer.zero_grad()
    outputs = model(data.x, data.edge_index)
    embeddings, reconstructions, other_val1, other_val2 = outputs[:4]
    loss = binary_cross_entropy_loss(embeddings, reconstructions, embeddings_list)
    loss.backward()
    optimizer.step()
    return loss.item()

# Set up the dataset
dataset_list = []
for A_C in A_C_matrices:
    edges = torch.tensor(np.array(np.where(A_C)).T, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(A_C[edges[0], edges[1]], dtype=torch.float)
    x = torch.randn(A_C.shape[0], 16)
    y = torch.tensor([1], dtype=torch.float)
    dataset = Data(x=x, edge_index=edges, edge_attr=edge_attr, y=y)
    dataset_list.append(dataset)

# Set up the model and optimizer
model = GAT(16, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the GAT model for each graph
embeddings_list = []
for i, dataset in enumerate(dataset_list):
    print(f'Training GAT graph A_C{i + 1}')
    for epoch in range(1000):
        loss = train(model, optimizer, dataset, dataset.y, embeddings_list)  # Pass embeddings_list argument
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss {loss:.4f}')

    # Generate embeddings for the current graph
    with torch.no_grad():
        embeddings = model(dataset.x, dataset.edge_index)

    embeddings_list.append(embeddings)  # Append embeddings outside the epoch loop

# Confirm the embeddings_list contains embeddings from different graphs
print(f"Number of graphs: {len(embeddings_list)}")

# Fuse the embeddings of all graphs together
min_size = min([embedding.size(0) for embedding in embeddings_list])
embeddings_list = [embedding[:min_size, :] for embedding in embeddings_list]
fused_embeddings = torch.cat(embeddings_list, dim=1)
zero_count = (fused_embeddings == 0).sum().item()
print(f"Number of zero values in fused_embeddings: {zero_count}")

#----------------------------

# Define the file path
file_path = '/home/z5318340/Movies_with_missing_values.xlsx'
# Read the Excel file
data = pd.read_excel(file_path)

# Extract the required columns
user_id = data['User_ID']
movie_id = data['Movies_ID']
criteria = ['C1', 'C2', 'C3', 'C4']

# Create a mapping from user/movie IDs to IDs
user_id_map = {uid: i for i, uid in enumerate(user_id.unique())}
movie_id_map = {mid: i for i, mid in enumerate(movie_id.unique())}

# Define the ground_truth_ratings matrix
num_users = len(user_id_map)
num_movies = len(movie_id_map)
num_criteria = len(criteria)
ground_truth_ratings = np.zeros((num_users, num_movies, num_criteria), dtype=np.int32)

for i, row in data.iterrows():
    uid = row['User_ID']
    mid = row['Movies_ID']
    criterion_ratings = [row[criterion] for criterion in criteria]
    if uid in user_id_map and mid in movie_id_map:
        user_idx = user_id_map[uid]
        movie_idx = movie_id_map[mid]
        ground_truth_ratings[user_idx, movie_idx] = criterion_ratings

zero_count = np.count_nonzero(ground_truth_ratings == 0)
print(f"Number of cells with a value of zero: {zero_count}")


# Convert fused_embeddings to a NumPy array
fused_embeddings = fused_embeddings.numpy()
original_sparsity_ratio = np.count_nonzero(ground_truth_ratings == 0) / ground_truth_ratings.size

# Calculate sparsity ratio
embedding_sparsity_ratio = np.count_nonzero(fused_embeddings == 0) / fused_embeddings.size
print("Original sparsity ratio:", original_sparsity_ratio)
print("Embedding sparsity ratio:", embedding_sparsity_ratio)

# # ----------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class MultiCriteriaRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MultiCriteriaRecommender, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)  # Apply sigmoid activation
        return x


def predict_missing_with_sigmoid(model, fused_embeddings, ground_truth_ratings):
    similarity_threshold = 0.5
    num_users = ground_truth_ratings.shape[0]
    num_movies = ground_truth_ratings.shape[1] - 2
    num_criteria = 4

    new_ratings = ground_truth_ratings.copy()
    sigmoid_output = model(torch.tensor(fused_embeddings, dtype=torch.float32))
    fused_embeddings_sigmoid = fused_embeddings[:, :sigmoid_output.shape[1]] * sigmoid_output[:, :fused_embeddings.shape[1]].numpy()

    for user_id in range(num_users):
        for movie_id in range(num_movies):
            if (new_ratings[user_id, movie_id] == 0).all():
                fused_emb_rating = fused_embeddings_sigmoid[user_id]
                similarities = custom_cosine_similarity(fused_emb_rating, fused_embeddings_sigmoid)
                similar_movies = similarities.topk(k=len(similarities) - 1, largest=True).indices
                prediction = np.zeros(num_criteria)
                total_weight = 0.0

                for i in range(1, len(similar_movies)):
                    similar_movie_id = similar_movies[i]
                    if similar_movie_id >= num_movies:
                        continue
                    if (new_ratings[user_id, similar_movie_id] != 0).any():
                        similarity_weight = similarities[similar_movie_id]
                        prediction = np.add(prediction, similarity_weight * new_ratings[user_id, similar_movie_id, :num_criteria])
                        total_weight += similarity_weight
                        if total_weight >= similarity_threshold:
                            break

                if total_weight > 0:
                    prediction /= total_weight
                    rounded_prediction = torch.round(torch.tensor(prediction))  # Round the prediction to the nearest integer
                    sigmoid_prediction = torch.sigmoid(rounded_prediction)  # Apply sigmoid activation to the rounded prediction
                    new_ratings[user_id, :num_criteria] = sigmoid_prediction.int()

    return new_ratings[:, :-2]


def custom_cosine_similarity(x1, x2, eps=1e-8):
    x1_tensor = torch.tensor(x1, dtype=torch.float32)
    x2_tensor = torch.tensor(x2, dtype=torch.float32)

    dot_product = torch.mm(x1_tensor.unsqueeze(0), x2_tensor.t())
    x1_norm = torch.norm(x1_tensor, p=2)
    x2_norm = torch.norm(x2_tensor, p=2)
    similarity = dot_product / (x1_norm * x2_norm + eps)
    return similarity

def cross_entropy_loss(inputs, reconstructions):
        cross_entropy = nn.CrossEntropyLoss()
        return cross_entropy(reconstructions, inputs)

def cross_entropy_with_l2_loss(inputs, reconstructions, model, l2_weight=0.01):
    cross_entropy = nn.CrossEntropyLoss()
    loss = cross_entropy(reconstructions, inputs)  # Compute the cross-entropy loss
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)  # Computing L2 norm of model parameters
    total_loss = loss + l2_weight * l2_reg  # Add the L2 regularization term to the total loss
    return total_loss

# Split the data into train and test sets
num_samples = min(fused_embeddings.shape[0], ground_truth_ratings.shape[0])
fused_embeddings = fused_embeddings[:num_samples]
ground_truth_ratings = ground_truth_ratings[:num_samples]
train_data, test_data, train_target, test_target = train_test_split(
    fused_embeddings, ground_truth_ratings[:, :-2], test_size=0.3, random_state=42)

# Define the model, loss function, optimizer, and number of epochs
train_target_reshaped = torch.tensor(train_target.reshape(train_data.shape[0], -1), dtype=torch.float32)
output_dim = train_target_reshaped.shape[1]
model = MultiCriteriaRecommender(fused_embeddings.shape[1], 32, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    outputs = model(train_data_tensor)
    train_target_reshaped = torch.tensor(train_target.reshape(train_data.shape[0], -1), dtype=torch.float32)
    loss = cross_entropy_with_l2_loss(outputs, train_target_reshaped, model)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        if len(test_data) > 0:
            test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
            test_outputs = model(test_data_tensor)
            test_outputs_reshaped = test_outputs.view(-1, train_target_reshaped.shape[1])
            test_target_reshaped = torch.tensor(test_target.reshape(test_outputs_reshaped.shape[0], -1),
                                                dtype=torch.float32)
            test_loss = cross_entropy_with_l2_loss(test_outputs_reshaped, test_target_reshaped, model)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} Train Loss: {loss.item():.4f} Test Loss: {test_loss.item():.4f}")
        else:
            print("No test data available.")

# Predict missing values
model.eval()
with torch.no_grad():
    predicted_ratings = predict_missing_with_sigmoid(model, fused_embeddings, ground_truth_ratings)

# Recommendation based on predicted ratings
recommendations = []
for user_id in range(predicted_ratings.shape[0]):
    unrated_movies = np.where(ground_truth_ratings[user_id] == 0)[0]
    sorted_movies = np.argsort(predicted_ratings[user_id])[::-1]
    common_indices = np.where(np.isin(sorted_movies, unrated_movies))
    recommended_movies = sorted_movies[common_indices][:5]
    recommendations.append(recommended_movies)

# Print the recommendations for each user
for user_id, user_recommendations in enumerate(recommendations):
    print(f"User {user_id+1} recommendations: {user_recommendations}")
  
#----------------------------------
# Compute evaluation metrics
num_samples = min(len(test_target), len(predicted_ratings))
test_target = test_target[:num_samples]
test_predictions = predicted_ratings[:num_samples].flatten()

# Flatten test_target and test_predictions if necessary
if len(test_target.shape) > 1 and test_target.shape[1] > 1:
    test_target = test_target.flatten()
if len(test_predictions.shape) > 1 and test_predictions.shape[1] > 1:
    test_predictions = test_predictions.flatten()

# To compute precision, recall, and F1-score, make sure that test_target and binary_predictions have the same length
threshold = 0.5
binary_predictions = (test_predictions > threshold).astype(int)

# Rest of the evaluation code remains the same
mae = mean_absolute_error(test_target, test_predictions)
rmse = mean_squared_error(test_target, test_predictions, squared=False)

# Compute precision, recall, and F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(test_target, binary_predictions, average='weighted', zero_division=1)
accuracy = accuracy_score(test_target, binary_predictions)

# Compute F2-score
beta = 2  # set the beta parameter
f2_score = fbeta_score(test_target, binary_predictions, beta=beta, average='weighted')

# Compute FCP
def compute_fcp(test_target, test_predictions):
    n = len(test_target)
    num_pairs = 0
    num_correct_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            if test_target[i] != test_target[j]:
                num_pairs += 1
                if test_predictions[i] > test_predictions[j]:
                    num_correct_pairs += 1
    if num_pairs == 0:
        return 0.0
    else:
        return num_correct_pairs / num_pairs

# Compute MAP
def compute_map(test_target, test_predictions):
    order = np.argsort(-test_predictions)
    relevant = test_target[order]
    precisions = []
    for i in range(len(test_target)):
        if relevant[i]:
            precisions.append(np.mean(relevant[:i+1]))
    if len(precisions) == 0:
        return 0.0
    else:
        return np.mean(precisions)

# Compute MRR
def compute_mrr(test_target, test_predictions):
    order = np.argsort(-test_predictions)
    ranks = np.arange(1, len(test_target)+1)[order]
    relevant_ranks = ranks[test_target[order].nonzero()[0]]
    if len(relevant_ranks) == 0:
        return 0.0
    else:
        return np.mean(1 / relevant_ranks)

# Compute FCP, MAP, and MRR
fcp = compute_fcp(test_target, test_predictions)
map = compute_map(test_target, test_predictions)
mrr = compute_mrr(test_target, test_predictions)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")
print(f"F2-score: {f2_score:.4f}")
print(f"FCP: {fcp:.4f}")
print(f"MAP: {map:.4f}")
print(f"MRR: {mrr:.4f}")
