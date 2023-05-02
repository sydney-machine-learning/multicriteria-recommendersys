import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import torch_geometric
import torch_scatter
matplotlib.use('TkAgg')  # Specify the backend
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import torch
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
import torch.optim as optim
import copy
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, mean_squared_error, f1_score, fbeta_score

# Define the criteria for ratings
criteria = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Check in / front desk', 'Service', 'Business service']

# Create the graph
G = nx.MultiGraph()

# Keep track of authors and files that have already been added to the graph
authors = []
files = []

# Add the file names as nodes
for (root, dirs, file_list) in os.walk("C://Text//"):
    for file_name in file_list:
        if file_name.endswith('.dat') and file_name not in files:
            G.add_node(file_name, bipartite=0)
            files.append(file_name)

# Add authors and edges between files and authors
for file_name in files:
    with open(os.path.join(root, file_name), encoding='utf-8') as f:
        author_ratings = dict()
        criterion_added = {criterion: False for criterion in criteria}

        for line in f:
            # print(line.strip())
            for criterion in criteria:
                if f"<{criterion}>" in line and not criterion_added[criterion]:
                    rating = int(line.split(f"<{criterion}>")[1].split("</")[0].strip())
                    # print(f"{criterion}: {rating}")
                    author_ratings[criterion] = rating
                    criterion_added[criterion] = True

            if "<Author>" in line:
                author = line.split("<Author>")[1].split("</")[0].strip()
                if author != "":
                    if author not in authors:
                        G.add_node(author, bipartite=1)
                        authors.append(author)

                    # Add edges between file and author for each criterion where rating > 0
                    for criterion, rating in author_ratings.items():
                        if rating > 0:
                            G.add_edge(file_name, author, criterion=criterion, rating=rating)

                    # Reset author_ratings and criterion_added for the next file
                    author_ratings = dict()
                    criterion_added = {criterion: False for criterion in criteria}

# Print the number of nodes and edges
print("Number of nodes:", len(G.nodes()))
print("Number of edges:", len(G.edges()))

# Create a dictionary to store the ratings for each file and author
ratings_dict = {}
for file_name, author, data in G.edges(data=True):
    if isinstance(author, str):
        criterion = data['criterion']
        rating = data['rating']
        if file_name in ratings_dict:
            if author in ratings_dict[file_name]:
                ratings_dict[file_name][author][criterion] = rating
            else:
                ratings_dict[file_name][author] = {criterion: rating}
        else:
            ratings_dict[file_name] = {author: {criterion: rating}}
#
# # Create the ground_truth_ratings matrix
# ground_truth_ratings = []
# for author in authors :
#     author_ratings = []
#     for file_name in files:
#         if file_name in ratings_dict and author in ratings_dict[file_name]:
#             criterion_ratings = ratings_dict[file_name][author]
#             criterion_ratings_list = [1 if criterion_ratings.get(criterion, 0) > 0 else 0 for criterion in criteria]
#             author_ratings += criterion_ratings_list
#         else:
#             author_ratings += [0] * len(criteria)
#     ground_truth_ratings.append(author_ratings)


# Create a list of valid authors with at least one rating
valid_authors = []
for author in authors:
    rated_criteria = 0
    for file_name in files:
        if file_name in ratings_dict and author in ratings_dict[file_name]:
            rated_criteria += len(ratings_dict[file_name][author])
    if rated_criteria >= 1:
        valid_authors.append(author)

# Create the ground_truth_ratings matrix for valid authors only
ground_truth_ratings = []
for author in valid_authors:
    author_ratings = []
    for file_name in files:
        if file_name in ratings_dict and author in ratings_dict[file_name]:
            criterion_ratings = ratings_dict[file_name][author]
            criterion_ratings_list = [1 if criterion_ratings.get(criterion, 0) != 0 else 0 for criterion in criteria]
            author_ratings += criterion_ratings_list
        else:
            author_ratings += [0] * len(criteria)
    ground_truth_ratings.append(author_ratings)

# -------------- Define views-----------------
# Create new graphs for the Value, Rooms, Location, Cleanliness, Check in / front desk, Service, and Business service views
G_value = nx.Graph()
G_rooms = nx.Graph()
G_location = nx.Graph()
G_cleanliness = nx.Graph()
G_checkin = nx.Graph()
G_service = nx.Graph()
G_business = nx.Graph()

# Add the file names as nodes to each graph
for file_name in files:
    G_value.add_node(file_name)
    G_rooms.add_node(file_name)
    G_location.add_node(file_name)
    G_cleanliness.add_node(file_name)
    G_checkin.add_node(file_name)
    G_service.add_node(file_name)
    G_business.add_node(file_name)

# Add edges between file and author for each criterion where rating > 0
for file_name in files:
    with open(os.path.join(root, file_name), encoding='utf-8') as f:
        current_ratings = {}
        criterion_added = {criterion: False for criterion in criteria}

        for line in f:
            for criterion in criteria:
                if f"<{criterion}>" in line and not criterion_added[criterion]:
                    rating = int(line.split(f"<{criterion}>")[1].split("</")[0].strip())
                    current_ratings[criterion] = rating
                    criterion_added[criterion] = True

            if "<Author>" in line:
                author = line.split("<Author>")[1].split("</")[0].strip()
                if author:
                    for criterion, rating in current_ratings.items():
                        if rating > 0:
                            # Add weighted edges to the Value graph based on the Value criterion
                            if criterion == 'Value':
                                G_value.add_edge(file_name, author, weight=rating)
                            # Add weighted edges to the Rooms graph based on the Rooms criterion
                            elif criterion == 'Rooms':
                                G_rooms.add_edge(file_name, author, weight=rating)
                            # Add weighted edges to the Location graph based on the Location criterion
                            elif criterion == 'Location':
                                G_location.add_edge(file_name, author, weight=rating)
                            # Add weighted edges to the Cleanliness graph based on the Cleanliness criterion
                            elif criterion == 'Cleanliness':
                                G_cleanliness.add_edge(file_name, author, weight=rating)
                            # Add weighted edges to the Check in / front desk graph based on the Check in / front desk criterion
                            elif criterion == 'Check in / front desk':
                                G_checkin.add_edge(file_name, author, weight=rating)
                            # Add weighted edges to the Service graph based on the Service criterion
                            elif criterion == 'Service':
                                G_service.add_edge(file_name, author, weight=rating)
                            # Add weighted edges to the Business service graph based on the Business service criterion
                            elif criterion == 'Business service':
                                G_business.add_edge(file_name, author, weight=rating)
                    current_ratings = {}
                    criterion_added = {criterion: False for criterion in criteria}


# # Print the number of nodes and edges in each graph
# print("Number of nodes in Value view:", len(G_value.nodes()))
# print("Number of edges in Value view:", len(G_value.edges()))
# print("Number of nodes in Rooms view:", len(G_rooms.nodes()))
# print("Number of edges in Rooms view:", len(G_rooms.edges()))
# print("Number of nodes in Location view:", len(G_location.nodes()))
# print("Number of edges in Location view:", len(G_location.edges()))
# print("Number of nodes in Cleanliness view:", len(G_cleanliness.nodes()))
# print("Number of edges in Cleanliness view:", len(G_cleanliness.edges()))
# print("Number of nodes in Checkin view:", len(G_checkin.nodes()))
# print("Number of edges in Checkin view:", len(G_checkin.edges()))
# print("Number of nodes in Service view:", len(G_service.nodes()))
# print("Number of edges in Service view:", len(G_service.edges()))
# print("Number of nodes in Business view:", len(G_business.nodes()))
# print("Number of edges in Business view:", len(G_business.edges()))
#
# # Print edges in the Value view
# print("Edges in Value view:")
# for edge in G_value.edges(data=True):
#     print(edge)
#
# # Print edges in the Rooms view
# print("Edges in Rooms view:")
# for edge in G_rooms.edges(data=True):
#     print(edge)
#
# # Print edges in the Location view
# print("Edges in Location view:")
# for edge in G_location.edges(data=True):
#     print(edge)
#
# # Print edges in the Cleanliness view
# print("Edges in Cleanliness view:")
# for edge in G_cleanliness.edges(data=True):
#     print(edge)
#
# # Print edges in the Checkin view
# print("Edges in Checkin view:")
# for edge in G_checkin.edges(data=True):
#     print(edge)
#
# # Print edges in the Service view
# print("Edges in Service view:")
# for edge in G_service.edges(data=True):
#     print(edge)
#
# # Print edges in the Business view
# print("Edges in Business view:")
# for edge in G_business.edges(data=True):
#     print(edge)


#**********************BGNN Adjacency Matrices based Criteria ************************
# Create node index dictionary
node_idx_dicts = []
adj_matrices = []
graph_list = [G_value, G_rooms, G_location, G_cleanliness, G_checkin, G_service, G_business]

for graph in graph_list:
    node_idx_dict = {}
    for i, node in enumerate(graph.nodes()):
        node_idx_dict[node] = i
    node_idx_dicts.append(node_idx_dict)

    n_nodes = len(graph.nodes())
    adj_matrix = np.zeros((n_nodes, n_nodes))
    adj_matrices.append(adj_matrix)

    for file_node, author_node, weight in graph.edges(data='weight'):
        file_idx = node_idx_dict[file_node]
        author_idx = node_idx_dict[author_node]
        adj_matrix[author_idx][file_idx] = weight
        adj_matrix[file_idx][author_idx] = weight

# Extract author and file nodes and create separate matrices for each graph
A_C_matrices = []
max_rows = 0  # Initialize max_rows to keep track of the largest matrix size

for idx, graph in enumerate(graph_list):
    author_nodes = [node for node in graph.nodes() if node in authors]
    file_nodes = [node for node in graph.nodes() if node not in authors]
    BC_uv = adj_matrices[idx][[node_idx_dicts[idx][author_node] for author_node in author_nodes]]
    BC_vu = adj_matrices[idx][[node_idx_dicts[idx][file_node] for file_node in file_nodes]]

    # Calculate the degree matrix of B_uv and B_vu
    DC_uv = np.diag(np.sum(BC_uv, axis=1))
    DC_vu = np.diag(np.sum(BC_vu, axis=1))

    # Normalize B_uv and B_vu
    BC_uv_norm = np.linalg.inv(DC_uv) @ BC_uv
    BC_vu_norm = np.linalg.inv(DC_vu) @ BC_vu

    # Update max_rows if necessary
    max_rows = max(max_rows, BC_uv_norm.shape[0], BC_vu_norm.shape[0])

    # Ensure that B_uv_norm and B_vu_norm have the same number of rows
    BC_uv_norm = np.pad(BC_uv_norm, ((0, max_rows - BC_uv_norm.shape[0]), (0, 0)), mode='constant')
    BC_vu_norm = np.pad(BC_vu_norm, ((0, max_rows - BC_vu_norm.shape[0]), (0, 0)), mode='constant')

    # Concatenate B_uv_norm and B_vu_norm along axis 1
    A_C = np.block([[np.zeros((max_rows, BC_uv_norm.shape[1])), BC_uv_norm],
                    [BC_vu_norm, np.zeros((max_rows, BC_vu_norm.shape[1]))]])
    A_C_matrices.append(A_C)

    # # print each adjacency matrix
    # print(f"A_C{idx + 1}:")
    # print(A_C)
    # print("\n")

# # Loop through A_C_matrices and save each matrix as a text file
# for idx, A_C in enumerate(A_C_matrices):
#     filename = f'A_C_{idx+1}.txt'  # Set a unique filename for each matrix
#     np.savetxt(filename, A_C, fmt='%f', delimiter=',')  # Save matrix as text file
#     print(f'Saved {filename}')

#**********************Graph Attention Network (GAT) based reconstruction_loss ************************
class GAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=4)
        self.conv2 = GATConv(out_channels * 4, out_channels, heads=4)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def reconstruction_loss(inputs, reconstructions):
    mse_loss = F.mse_loss(reconstructions, inputs)
    cosine_sim = F.cosine_similarity(reconstructions.unsqueeze(-1), inputs.unsqueeze(-1), dim=1)
    cosine_loss = torch.mean(torch.pow(cosine_sim - 1, 2))
    total_loss = mse_loss + cosine_loss
    return total_loss

# def cosine_loss(inputs, reconstructions):
#     cosine_sim = F.cosine_similarity(reconstructions.unsqueeze(-1), inputs.unsqueeze(-1), dim=1)
#     cosine_loss = torch.mean(torch.pow(cosine_sim - 1, 2))
#     return cosine_loss

def train(model, optimizer, data, target):
    model.train()
    optimizer.zero_grad()
    outputs = model(data.x, data.edge_index)
    embeddings, reconstructions, other_val1, other_val2 = outputs[:4]
    loss = reconstruction_loss(embeddings, reconstructions)
    # loss = cosine_loss(embeddings, reconstructions)
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
    dataset = torch_geometric.data.Data(x=x, edge_index=edges, edge_attr=edge_attr, y=y)
    dataset_list.append(dataset)

# Set up the model and optimizer
model = GAT(16, 64)  # Update out_channels to 64
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model for each graph
embeddings_list = []
for i, dataset in enumerate(dataset_list):
    print(f'Training GAT graph A_C{i + 1}')
    for epoch in range(100):
        loss = train(model, optimizer, dataset, dataset.y)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss {loss:.4f}')

    # Generate embeddings for the current graph
    with torch.no_grad():
        embeddings = model(dataset.x, dataset.edge_index)
        embeddings_list.append(embeddings)

# Print the embeddings for all graphs
for i, embeddings in enumerate(embeddings_list):
    print(f'Embeddings for graph A_C{i + 1}:')
    print(embeddings)

#----------------- FUSION based Element-wise multiplication------------------------------

# Initialize an empty list to store the embeddings for all graphs
all_embeddings = []
# Train the model for each graph
for i, dataset in enumerate(dataset_list):
    print(f'Training Fusion graph A_C{i + 1}')
    for epoch in range(100):
        loss = train(model, optimizer, dataset, dataset.y)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Fusion Loss {loss:.4f}')
        # Obtain the embedding vectors for the current graph
    embeddings = model(dataset.x, dataset.edge_index)
    # Append the embeddings to the list of all embeddings
    all_embeddings.append(embeddings)

# Find the minimum size along the first dimension
min_size = min([embedding.size(0) for embedding in all_embeddings])

# Resize the embeddings to have the same size along the first dimension
all_embeddings = [embedding[:min_size, :] for embedding in all_embeddings]

# Perform element-wise multiplication to fuse the embeddings
fused_embeddings = torch.ones_like(all_embeddings[0])  # Initialize with ones of the same size as the embeddings
for embeddings in all_embeddings:
    fused_embeddings *= embeddings
    # fused_embeddings += embeddings

# The fused_embeddings tensor now contains the element-wise multiplied embedding vectors for all graphs in the dataset_list
print("Fused Embedding Vectors:")
print(fused_embeddings)
# Check the dimensions of fused_embeddings
print("Dimensions of fused_embeddings:", fused_embeddings.shape)

# ------------------------------- Prediction ------------------------------------
# Set the custom threshold for binarizing the predicted values
threshold = 0.5

# Find the indices of the missing values in ground_truth_ratings
missing_indices = np.where(ground_truth_ratings == 0)

# Initialize a new matrix to store the predicted values
predicted_ratings = np.copy(ground_truth_ratings)

# Initialize a new matrix to store the combined ratings
combined_ratings = np.copy(ground_truth_ratings)

# Loop over each missing value in the matrix
for i in range(len(missing_indices[0])):
    row = missing_indices[0][i]
    col = missing_indices[1][i]

    # Find the indices of the non-zero values in the current row and column
    row_non_zero_indices = np.where(ground_truth_ratings[row] != 0)[0]
    col_non_zero_indices = np.where(ground_truth_ratings[:, col] != 0)[0]

    # Exclude any missing values from row_non_zero_indices and col_non_zero_indices
    row_non_zero_indices = np.setdiff1d(row_non_zero_indices, missing_indices[1])
    col_non_zero_indices = np.setdiff1d(col_non_zero_indices, missing_indices[0])

    # Extract the fused embeddings for the non-zero values in the current row and column
    row_embeddings = fused_embeddings[row_non_zero_indices]
    col_embeddings = fused_embeddings[col_non_zero_indices]

    # Compute the cosine similarity vector between the fused embeddings
    similarity_vector = cosine_similarity(row_embeddings.detach().numpy(), col_embeddings.detach().numpy())

    # Use the maximum value in the similarity vector as the predicted value for the missing cell
    predicted_value = np.max(similarity_vector)

    # Binarize the predicted value based on the custom threshold
    if predicted_value <= threshold:
        predicted_value = 0
    else:
        predicted_value = 1

    # Update the predicted_ratings matrix with the binarized value
    predicted_ratings[row, col] = predicted_value

    # Extract the corresponding predicted value from predicted_ratings
    y_pred = predicted_ratings[row, col]

    # Extract the corresponding ground truth value from ground_truth_ratings
    y_true = ground_truth_ratings[row, col]

    # Replace the missing value in combined_ratings with predicted value
    combined_ratings[row, col] = predicted_value

# Split the data into training and validation sets
train_data, val_data = train_test_split(combined_ratings, test_size=0.2, random_state=42)

def evaluate_model(y_true, y_pred):
    # Flatten the arrays to 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Exclude the missing values from the evaluation
    non_missing_indices = np.where(y_true != 0)[0]
    y_true = y_true[non_missing_indices]
    y_pred = y_pred[non_missing_indices]

    # Compute the precision and recall scores
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')

    # Compute the mean squared error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Compute the root mean squared error (RMSE)
    rmse = np.sqrt(mse)

    # Compute the F1 score
    f1 = f1_score(y_true, y_pred, average='binary')

    # Compute the F2 score
    f2 = fbeta_score(y_true, y_pred, beta=2, average='binary')

    return precision, recall, mse, rmse, f1, f2

# Evaluate the model on the validation set
precision, recall, mse, rmse, f1, f2 = evaluate_model(val_data, combined_ratings)


print("Precision:", precision)
print("Recall:", recall)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("F1 score: ", f1)
print("F2 score: ", f2)


# Print the predicted ratings matrix
# print("Predicted Ratings:")
# print(predicted_ratings)
# print("Combined Ratings :")
# print(combined_ratings)
# print("fused_embeddings:")
# print(fused_embeddings)
# print("ground_truth_ratings:")
# print(ground_truth_ratings)
# np.savetxt('predicted_ratings.txt', predicted_ratings)
# np.savetxt('combined_ratings.txt', combined_ratings)





