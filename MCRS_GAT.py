


import os
import sys
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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso


def read_data(file_path, criteria):
    
    data = pd.read_excel(file_path)
    user_id = data['User_ID']
    item_id = data['Items_ID']

    user_id_map = {uid: i for i, uid in enumerate(user_id.unique())}
    num_users = len(user_id_map)
    item_id_map = {mid: i + num_users for i, mid in enumerate(item_id.unique())}
    num_items = len(item_id_map)

    num_criteria = len(criteria)
    base_ground_truth_ratings = np.zeros((num_users, num_items, num_criteria), dtype=np.int32)

    for i, row in data.iterrows():
        uid = row['User_ID']
        mid = row['Items_ID']
        criterion_ratings = [row[criterion] for criterion in criteria]
        if uid in user_id_map and mid in item_id_map:
            user_idx = user_id_map[uid]
            item_idx = item_id_map[mid] - num_users
            base_ground_truth_ratings[user_idx, item_idx] = criterion_ratings

    return user_id_map, item_id_map, base_ground_truth_ratings

def L_BGNN(file_path, criteria, user_id_map, item_id_map):
    graph_data = pd.read_excel(file_path)
    matrices = []  # Initialize a list to store the normalized matrices for each criterion
    n_nodes = len(user_id_map) + len(item_id_map)

    for criterion in criteria:
        # TODO: Check if this should be a sparse matrix.
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int32)

        for i in range(len(graph_data)):
            uid = user_id_map[graph_data['User_ID'][i]]
            mid = item_id_map[graph_data['Items_ID'][i]]
            rating = graph_data[criterion][i]

            adj_matrix[uid][mid] = rating
            adj_matrix[mid][uid] = rating

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

    return tuple(matrices)

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

    def fusion_embeddings_vectors(self, embeddings_list):  # Add self parameter
        max_size = max([embedding.size(0) for embedding in embeddings_list])
        
        # Pad embeddings to the maximum size
        padded_embeddings = [F.pad(embedding, (0, 0, 0, max_size - embedding.size(0))) for embedding in embeddings_list]
        
        # Concatenate the padded embeddings along the second dimension (axis 1)
        fused_embeddings = torch.cat(padded_embeddings, dim=1)
        
        return fused_embeddings

    def Multi_Embd(self, matrices, num_epochs=100, learning_rate=0.01):
        dataset_list = []

        for normalized_matrix in matrices:
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

        fused_embeddings = self.fusion_embeddings_vectors(embeddings_list)
        
        # Print fused embeddings
        print("Fused Embeddings:")
        print(fused_embeddings)
        
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
    
def check_embeddings_consistency(fused_embeddings):
    # Check for zero rows
    zero_rows = torch.sum(torch.abs(fused_embeddings), dim=1) == 0
    if torch.any(zero_rows):
        print("Warning: There are zero rows in fused_embeddings.")
    
    # Check for non-zero entries
    non_zero_entries = torch.sum(torch.abs(fused_embeddings), dim=0) > 0
    if not torch.all(non_zero_entries):
        print("Warning: There are rows with no non-zero entries in fused_embeddings.")
    
    # Return True if embeddings are consistent, False otherwise
    return not torch.any(zero_rows) and torch.all(non_zero_entries)

# -------------Recommendation Section -------------------------

def create_ground_truth_ratings(file_path, criteria):  
    # data = pd.read_excel(file_path)

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

# def Recommendation_items_Top_k(fused_embeddings, user_id_map, data, criteria, threshold_func=None, top_k=10):
#     recommendations_f_items = {}

#     # Convert fused embeddings to numpy array
#     fused_embeddings_np = fused_embeddings.cpu().detach().numpy()

#     # Compute similarities between embeddings
#     similarities = cosine_similarity(fused_embeddings_np)

#     # Iterate over all users
#     for user_idx, user_id in enumerate(user_id_map.keys()):
#         # Determine threshold value using threshold_func
#         if threshold_func is not None:
#             threshold_A = threshold_func(fused_embeddings[user_idx]).item()
#         else:
#             raise ValueError("A threshold function (threshold_func) must be provided.")

#         # Find similar users based on cosine similarity and dynamic threshold
#         similar_users_idx = np.where(similarities[user_idx] >= threshold_A)[0]

#         # Check if there are similar users for the current user
#         if len(similar_users_idx) > 0:
#             # Sort similar users by similarity score and select top_k users
#             similar_users_sorted_idx = similar_users_idx[np.argsort(similarities[user_idx][similar_users_idx])[::-1][:top_k]]

#             # Initialize recommended items list for the current user
#             recommended_items = []

#             # Retrieve the current user's rating from the data
#             user_data = data[data['User_ID'] == user_id]
#             if len(user_data) > 0:  # Check if there are ratings for this user
#                 current_user_rating = user_data['Overall_Rating'].values[0]

#                 # Get recommended items for the user
#                 for user_idx_2 in similar_users_sorted_idx:
#                     if user_idx_2 >= len(user_id_map.keys()):
#                         continue  # Skip if index is out of range
#                     user_id_2 = list(user_id_map.keys())[user_idx_2]
#                     for _, row in data[data['User_ID'] == user_id_2].iterrows():
#                         item_id = row['Items_ID']
#                         overall_rating = row['Overall_Rating']

#                         # Check if overall rating is similar to the current user's rating
#                         if abs(overall_rating - current_user_rating) <= threshold_A:
#                             recommended_items.append({'item_id': item_id, 'Overall_Rating': overall_rating})

#                 # Filter out items already rated by the current user
#                 recommended_items = [item for item in recommended_items if
#                                      item['item_id'] not in user_data['Items_ID'].values]

#                 # Sort recommended items by overall rating
#                 recommended_items = sorted(recommended_items, key=lambda x: x['Overall_Rating'], reverse=True)[:top_k]

#                 # Add recommended items for the current user to the dictionary
#                 recommendations_f_items[user_id] = recommended_items
#             else:
#                 recommendations_f_items[user_id] = None
#         else:
#             # No similar users found, pass the user's embedding
#             recommendations_f_items[user_id] = None

#     return recommendations_f_items

def Recommendation_items_Fixed_TopK(fused_embeddings, file_path, criteria, threshold_A=0.5, top_k=10):
    data, ground_truth_ratings_matrix, user_id_map, item_id_map = create_ground_truth_ratings(file_path, criteria)
    recommendations_f_items = {}

    num_users_actual, _ = fused_embeddings.shape
    fused_embeddings_2d = fused_embeddings.reshape((num_users_actual, -1))
    similarities = cosine_similarity(fused_embeddings_2d)

    # Iterate over all users
    for i in range(num_users_actual):
        # Find similar users based on cosine similarity and threshold_A
        similar_users_indices = [idx for idx, sim in enumerate(similarities[i]) if sim >= threshold_A]

        # Sort similar users by similarity score and select top_k users
        similar_users_indices = sorted(similar_users_indices, key=lambda idx: similarities[i][idx], reverse=True)[:top_k]

        # Initialize recommended items list for the current user
        recommended_items = []

        # Get recommended items for the user
        for user_idx in similar_users_indices:
            user_id = data.iloc[user_idx]['User_ID']
            item_id = data.iloc[user_idx]['Items_ID']
            overall_rating = data.iloc[user_idx]['Overall_Rating']
            # Check if item_id is the same as the current user's item_id
            if item_id != data.iloc[i]['Items_ID']:
                recommended_items.append({'item_id': item_id, 'Overall_Rating': overall_rating})
        
        # Add recommended items for the current user to the dictionary
        recommendations_f_items[data.iloc[i]['User_ID']] = recommended_items
            
    return recommendations_f_items


def split_and_save_data(file_path, criteria, test_size=0.3, random_state=42):
    # Call the read_data function to obtain the data structures
    user_id_map, item_id_map, base_ground_truth_ratings = read_data(file_path, criteria)

    # Read the data from the file
    data = pd.read_excel(file_path)

    # Convert User_ID and Items_ID columns to string type
    data['User_ID'] = data['User_ID'].astype(str)
    data['Items_ID'] = data['Items_ID'].astype(str)

    # Select only the required columns: 'User_ID', 'Items_ID', and 'Overall_Rating'
    data_subset = data[['User_ID', 'Items_ID', 'Overall_Rating']]

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data_subset, test_size=test_size, random_state=random_state)

    # Define file paths for train and test data
    train_file_path = os.path.join(os.path.dirname(file_path), 'train_data.xlsx')
    test_file_path = os.path.join(os.path.dirname(file_path), 'test_data.xlsx')

    # Save the train and test sets into separate files
    train_data.to_excel(train_file_path, index=False)
    test_data.to_excel(test_file_path, index=False)

    return train_data, test_data

def evaluate_RS_Model(fused_embeddings, file_path, criteria, test_size=0.3, random_state=42):
    # Split and save the data into train and test sets
    train_data, test_data = split_and_save_data(file_path, criteria, test_size=test_size, random_state=random_state)
    
    # Prepare training data
    train_X = fused_embeddings.cpu().detach().numpy()[train_data['User_ID'].astype('category').cat.codes]
    train_y = train_data['Overall_Rating'].values

    # Instantiate and train the SVR model
    svr_model = SVR()
    svr_model.fit(train_X, train_y)
    
    # Call the function with the defined threshold function to get training recommendations
    train_recommendations = Recommendation_items_Fixed_TopK(fused_embeddings, file_path, criteria, threshold_A=0.5, top_k=10)

    # Extract features for training recommendations
    train_recommendation_features = []
    train_recommendation_ratings = []
    for user_id, recommendations in train_recommendations.items():
        if recommendations is not None:
            for recommendation in recommendations:
                train_recommendation_features.append(fused_embeddings[user_id_map[user_id]].cpu().detach().numpy())
                train_recommendation_ratings.append(recommendation['Overall_Rating'])
                
    train_recommendation_features = np.array(train_recommendation_features)
    train_recommendation_ratings = np.array(train_recommendation_ratings)

    # Concatenate Overall_Rating with recommendation features
    train_recommendation_features_with_rating = np.column_stack((train_recommendation_features, train_recommendation_ratings))

    # Train SVR model on training recommendations
    svr_model.fit(train_recommendation_features_with_rating, train_recommendation_ratings)

    # Prepare test data
    test_user_ids = test_data['User_ID'].values.astype(str)
    test_item_ids = test_data['Items_ID'].values.astype(str)

    # Filter test user IDs to only include those present in user_id_map
    valid_test_user_ids = [user_id for user_id in test_user_ids if user_id in user_id_map]
    
    if not valid_test_user_ids:
        print("No valid test user IDs found.")
        return None, None

    valid_test_user_indices = [user_id_map[user_id] for user_id in valid_test_user_ids]

    # Index fused_embeddings with valid test user indices
    test_X = fused_embeddings[valid_test_user_indices]

    # Prepare test data with Overall_Rating as an additional feature
    test_X_with_rating = np.column_stack((test_X, test_data['Overall_Rating'].values))

    # Make predictions for test data using the SVR model
    test_predictions = svr_model.predict(test_X_with_rating)
      
    # Calculate MAE and RMSE for test data
    test_mae = mean_absolute_error(test_data['Overall_Rating'], test_predictions)
    test_rmse = mean_squared_error(test_data['Overall_Rating'], test_predictions, squared=False)
    print("MAE for test data (SVR):", test_mae)
    print("RMSE for test data (SVR):", test_rmse)

    return test_mae, test_rmse



# --------------------------------------------------------------------------------------
# Main Function ---------------------------
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Define your file paths for different datasets in Katana Server
    # file_paths = {
    #     'Movies_Original': '/home/z5318340/MCRS4/MoviesDatasetYahoo.xlsx',
    #     'Movies_Modified': '/home/z5318340/MCRS4/Movies_Modified_Rating_Scores.xlsx',
    #     'BeerAdvocate': '/home/z5318340/MCRS4/BeerAdvocate.xlsx',
    #     'TripAdvisor': '/home/z5318340/MCRS4/new_Trip_filtered_dataset.xlsx'
    # }
    
    # # Define your file paths for different datasets in local Server
    file_paths = {
        'Movies_Original': 'C://Yahoo//Global//Movies.xlsx',
        'Movies_Modified': 'C://Yahoo//Global//Movies_Modified.xlsx',
        'BeerAdvocate': 'C://Yahoo//Global//BeerAdvocate.xlsx',
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
    dataset_to_run = 'Movies_Modified'

    # Read data for the selected dataset
    
    file_path = file_paths[dataset_to_run]
    criteria = criteria_mapping[dataset_to_run]
    user_id_map, item_id_map, base_ground_truth_ratings = read_data(file_path, criteria)
    num_users = len(user_id_map)
    num_items = len(item_id_map)
    num_criteria = len(criteria)
    data = pd.read_excel(file_path)

    # Read data from the Excel file and create ID mappings  
    matrices = L_BGNN(file_path, criteria, user_id_map, item_id_map)

    #---Attention Embedding------
    model = GAT(in_channels=16, out_channels=256)
    fused_embeddings = model.Multi_Embd(matrices, num_epochs=100, learning_rate=0.01)
        
    # Recommendation section
    num_samples = fused_embeddings.shape[0]

    # Create an instance of the MultiCriteriaRecommender class
    output_dim = fused_embeddings.shape[1]  # Set output_dim to the number of criteria

    # Convert fused_embeddings to a torch tensor
    fused_embeddings_tensor = fused_embeddings.clone().detach().to(torch.float32)

    # Reshape fused_embeddings_tensor to match the expected shape
    num_samples, num_features = fused_embeddings_tensor.shape

    # Calculate the total number of features per criterion
    num_features_per_criterion = num_features // num_criteria
        
    # Create a DataFrame with user and item identifiers as MultiIndex
    df_users_items = pd.DataFrame(index=pd.MultiIndex.from_tuples([(user_id, item_id) for user_id in user_id_map.keys() for item_id in item_id_map.keys()]))
    
    # Call the create_real_ratings function
    data, ground_truth_ratings_matrix, user_id_map, item_id_map = create_ground_truth_ratings(file_path, criteria)

    
    # Call the function with the defined threshold function
    recommendations_f_items = Recommendation_items_Fixed_TopK(fused_embeddings, file_path, criteria, threshold_A=0.5, top_k=10)

    # # # Print recommendations for each user
    # for user_id, recommended_items in recommendations.items():
    #     print(f"Recommendations for User {user_id}:".encode('utf-8'))  
    #     for idx, item in enumerate(recommended_items, start=1):
    #         print(f"{idx}. Item ID: {item['item_id']}, Overall Rating: {item['Overall_Rating']}".encode('utf-8')) 
    #     print()
   
    train_data, test_data = split_and_save_data(file_path, criteria)   
    test_mae, test_rmse=evaluate_RS_Model(fused_embeddings, file_path, criteria, test_size=0.3, random_state=42)
    
    # Call this function after getting fused embeddings
    embeddings_consistent = check_embeddings_consistency(fused_embeddings)
    
    # Check the result
    if embeddings_consistent:
        print("Embeddings are consistent.")
    else:
        print("Embeddings are not consistent. Please check.")


    
    
    
   