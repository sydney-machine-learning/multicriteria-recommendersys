
#!/usr/bin/env -S python3 -u
#PBS -N Saman
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l walltime=300:00:00
#PBS -v OMP_NUM_THREADS=2
#PBS -j oe
#PBS -k oed
#PBS -M s.forouzandeh@unsw.edu.au
#PBS -m ae


import os
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
import time
from sklearn.model_selection import LeaveOneOut



def read_data(file_path):
    data = pd.read_excel(file_path)
    data['User_ID'] = data['User_ID'].astype(str)
    data['Items_ID'] = data['Items_ID'].astype(str)

    user_id_map = {uid: i for i, uid in enumerate(data['User_ID'].unique())}
    num_users = len(user_id_map)
    item_id_map = {mid: i + num_users for i, mid in enumerate(data['Items_ID'].unique())}

    return data, user_id_map, item_id_map

def L_BGNN(data, criteria, user_id_map, item_id_map):
    matrices = []  # Initialize a list to store the normalized matrices for each criterion
    n_nodes = len(user_id_map) + len(item_id_map)

    for criterion in criteria:
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int32)

        for i in range(len(data)):
            uid = user_id_map[data['User_ID'][i]]
            mid = item_id_map[data['Items_ID'][i]]
            rating = data[criterion][i]

            adj_matrix[uid][mid] = rating
            adj_matrix[mid][uid] = rating

        # For the following, note that adj_matrix is symmetric.

        # Calculate vector of degrees
        margins = np.maximum(np.sum(adj_matrix, axis=0), 1.0)

        # Divide the matrix by the harmonic mean of its margins.
        normalized_matrix = (adj_matrix / margins[:, None] + adj_matrix / margins[None, :]) / 2

        matrices.append(normalized_matrix)

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
        self.attention_weights = nn.Parameter(torch.ones(num_heads))
        self.att_pool = GlobalAttention(gate_nn=nn.Linear(out_channels, 1))

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Local attention computation
        head_outs = [conv(x, edge_index, edge_attr=edge_attr) for conv in self.conv_layers]
        x_local = torch.cat(head_outs, dim=-1)
        self_attention = F.leaky_relu(self.fc(x_local))
        self_attention = F.softmax(self_attention, dim=-1)
        x_local = x_local * self_attention
        x_local = self.leakyrelu(x_local)
        x_local = self.fc(x_local)
        x_local = self.layer_norm(x_local)
        x_local = F.normalize(x_local, p=2, dim=1)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x_local.size(0), dtype=torch.long, device=x_local.device)
        x_global = self.att_pool(x_local, batch)
        
        # Global attention computation
        global_attention = F.relu(self.global_fc(x_global))
        global_attention = F.softmax(global_attention, dim=-1)
        
        # Apply global attention to local embeddings
        x = x_local * global_attention
        return x

    def fusion_embeddings_vectors(self, embeddings_list):
        max_size = max(embedding.size(0) for embedding in embeddings_list)
        padded_embeddings = [F.pad(embedding, (0, 0, 0, max_size - embedding.size(0))) for embedding in embeddings_list]
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
                loss = self.train_GAT(optimizer, dataset)
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss {loss:.4f}')

            with torch.no_grad():
                embeddings = self(dataset.x, dataset.edge_index, dataset.edge_attr)

            embeddings_list.append(embeddings)

        fused_embeddings = self.fusion_embeddings_vectors(embeddings_list)
        
        # Print fused embeddings
        print("Fused Embeddings:")
        print(fused_embeddings)
        
        return fused_embeddings

    def local_contrastive_loss(self, embeddings_list, adjacency_matrices, dissimilarity_threshold=0.3, temperature=0.1, batch_size=64):
        device = embeddings_list[0].device
        total_loss = torch.tensor(0.0, device=device)
        num_views = len(embeddings_list)
        num_nodes = embeddings_list[0].size(0)

        for i in range(num_views):
            similarities = []
            for node in range(num_nodes):
                node_embedding = embeddings_list[i][node]
                neighbors = torch.nonzero(adjacency_matrices[i][node]).squeeze()
                if len(neighbors.shape) == 0:
                    neighbors = neighbors.unsqueeze(0)
                neighbor_embeddings = embeddings_list[i][neighbors]
                similarity = F.cosine_similarity(node_embedding.unsqueeze(0), neighbor_embeddings, dim=1).mean()
                similarities.append((similarity, node))
            
            _, anchor_node = max(similarities, key=lambda x: x[0])

            for j in range(num_views):
                if i != j:
                    anchor = embeddings_list[i][anchor_node]
                    positive = embeddings_list[j][anchor_node]

                    all_similarities = F.cosine_similarity(anchor.unsqueeze(0), embeddings_list[j], dim=1)
                    negative_node = torch.argmax(all_similarities).item()

                    if negative_node == anchor_node or all_similarities[negative_node] >= dissimilarity_threshold:
                        sorted_similarities, sorted_indices = torch.sort(all_similarities, descending=True)
                        for idx in sorted_indices:
                            if idx != anchor_node and all_similarities[idx] < dissimilarity_threshold:
                                negative_node = idx
                                break

                    negative = embeddings_list[j][negative_node]

                    pos_similarity = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0), dim=-1) / temperature
                    neg_similarity = F.cosine_similarity(anchor.unsqueeze(0), negative.unsqueeze(0), dim=-1) / temperature

                    logits = torch.cat([pos_similarity, neg_similarity], dim=0)
                    labels = torch.tensor([1, 0], device=logits.device).unsqueeze(0)

                    info_nce_loss = F.cross_entropy(logits.unsqueeze(0), labels)
                    total_loss += info_nce_loss

        return total_loss / (num_views * (num_views - 1))

    def global_contrastive_loss(self, embeddings_list, temperature=0.1):
        device = embeddings_list[0].device
        global_loss = torch.tensor(0.0, device=device)
        num_views = len(embeddings_list)
        
        # Compute global embeddings
        global_embeddings = [torch.mean(emb, dim=0) for emb in embeddings_list]

        for i in range(num_views):
            for j in range(num_views):
                if i != j:
                    # Compute positive similarity
                    pos_similarity = F.cosine_similarity(global_embeddings[i].unsqueeze(0), global_embeddings[j].unsqueeze(0), dim=-1) / temperature
                    
                    # Compute negative similarities in a single pass
                    neg_similarities = [
                        F.cosine_similarity(global_embeddings[i].unsqueeze(0), global_embeddings[k].unsqueeze(0), dim=-1) / temperature
                        for k in range(num_views) if k != i and k != j
                    ]
                    neg_similarities = torch.cat(neg_similarities, dim=0)
                    
                    # Combine positive and negative similarities
                    logits = torch.cat([pos_similarity, neg_similarities], dim=0)
                    
                    # Create labels for cross-entropy loss
                    labels = torch.tensor([1] + [0] * (num_views - 2), device=device).unsqueeze(0)
                    
                    # Compute InfoNCE loss
                    info_nce_loss = F.cross_entropy(logits.unsqueeze(0), labels)
                    global_loss += info_nce_loss

        return global_loss / (num_views * (num_views - 1))

    def l2_regularization(self, l2_weight=0.1):
        l2_reg = torch.norm(torch.stack([torch.norm(param, p=2) for param in self.parameters()]), p=2)
        return l2_weight * l2_reg

    def train_GAT(self, optimizer, data, embeddings_list=None, adjacency_matrices=None, alpha=0.5, beta=0.5, gamma=0.1):
        self.train()
        optimizer.zero_grad()
        outputs = self(data.x, data.edge_index, data.edge_attr)
        embeddings = outputs
        local_contrastive_loss_value = self.local_contrastive_loss(embeddings_list, adjacency_matrices) if embeddings_list and adjacency_matrices else 0
        global_contrastive_loss_value = self.global_contrastive_loss(embeddings_list) if embeddings_list else 0
        l2_reg = self.l2_regularization()
        total_loss = local_contrastive_loss_value + gamma * l2_reg
        total_loss.backward()
        optimizer.step()
        return total_loss

# -------------Recommendation Section -------------------------

def Recommendation_items_Top_k(fused_embeddings, user_id_map, data, threshold_func=None, top_k=1):
    recommendations_f_items = {}
    num_users = len(user_id_map)

    # Convert fused embeddings to numpy array, focusing on users
    fused_embeddings_np = fused_embeddings.cpu().detach().numpy()[:num_users]

    # Compute similarities between embeddings
    similarities = cosine_similarity(fused_embeddings_np)

    # Group the data by 'User_ID'
    grouped = data.groupby('User_ID')

    # Sort users by their index (from user_id_map)
    uids = sorted(user_id_map.items(), key=lambda x: x[1])
    
    for user_id, user_idx in uids:
        # Determine threshold value using threshold_func
        if threshold_func is not None:
            threshold_A = threshold_func(fused_embeddings[user_idx]).item()
        else:
            # Set a default threshold value if threshold_func is not provided
            threshold_A = 0.1

        # Find similar users based on cosine similarity and dynamic threshold
        similar_users_idx = np.where(similarities[user_idx] >= threshold_A)[0]

        # Check if there are similar users for the current user
        if len(similar_users_idx) > 0:
            # Sort similar users by similarity score and select top_k users
            similar_users_sorted_idx = similar_users_idx[np.argsort(similarities[user_idx][similar_users_idx])[::-1][:top_k]]

            # Initialize recommended items list for the current user
            recommended_items = []

            # Retrieve the current user's ratings from the data
            if user_id in grouped.groups:
                user_data = grouped.get_group(user_id)
                if len(user_data) > 0:  # Check if there are ratings for this user
                    current_user_rating = user_data['Overall_Rating'].values[0]

                    # Get recommended items for the user
                    for user_idx_2 in similar_users_sorted_idx:
                        user_id_2 = uids[user_idx_2][0]

                        # Check if the similar user's group exists
                        if user_id_2 in grouped.groups:
                            for _, row in grouped.get_group(user_id_2).iterrows():
                                item_id = row['Items_ID']
                                overall_rating = row['Overall_Rating']

                                # Check if the overall rating is similar to the current user's rating
                                # and filter out items already rated by the current user
                                if item_id not in user_data['Items_ID'].values and abs(overall_rating - current_user_rating) <= threshold_A:
                                    recommended_items.append({'item_id': item_id, 'Overall_Rating': overall_rating})

                    # Sort recommended items by overall rating
                    recommended_items = sorted(recommended_items, key=lambda x: x['Overall_Rating'], reverse=True)[:top_k]

                    # Add recommended items for the current user to the dictionary
                    recommendations_f_items[user_id] = recommended_items
                else:
                    recommendations_f_items[user_id] = None
            else:
                recommendations_f_items[user_id] = None
        else:
            # No similar users found, pass the user's embedding
            recommendations_f_items[user_id] = None

    return recommendations_f_items

def split_and_save_data(data, output_path=None, test_size=0.2, random_state=42):
    # Convert User_ID and Items_ID columns to string type
    data['User_ID'] = data['User_ID'].astype(str)
    data['Items_ID'] = data['Items_ID'].astype(str)

    # Select only the required columns: 'User_ID', 'Items_ID', and 'Overall_Rating'
    data_subset = data[['User_ID', 'Items_ID', 'Overall_Rating']]

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data_subset, test_size=test_size, random_state=random_state)

    if output_path:
        # Define file paths for train and test data
        train_file_path = os.path.join(output_path, 'train_data.xlsx')
        test_file_path = os.path.join(output_path, 'test_data.xlsx')

    # Save the train and test sets into separate files
    train_data.to_excel(train_file_path, index=False)
    test_data.to_excel(test_file_path, index=False)

    return train_data, test_data

def evaluate_RS_Model(fused_embeddings, user_id_map, item_id_map, data, output_path, test_size=0.2, random_state=42, test_data=None):
    # If test_data is provided, use it; otherwise, split the data
    if test_data is None:
        train_data, test_data = split_and_save_data(data, output_path, test_size=test_size, random_state=random_state)
    else:
        train_data = data  # Use all data for training when test_data is provided

    # Prepare training data
    train_X = fused_embeddings.cpu().detach().numpy()[train_data['User_ID'].astype('category').cat.codes]
    train_y = train_data['Overall_Rating'].values

    # Instantiate and train the SVR model with sigmoid kernel
    svr_model = SVR(kernel='sigmoid')
    svr_model.fit(train_X, train_y)
    
    # Define the threshold function
    def threshold_function(embedding):
        return torch.tensor(0.1)  # Example threshold

    # Get training recommendations
    train_recommendations = Recommendation_items_Top_k(fused_embeddings, user_id_map, data, threshold_func=threshold_function, top_k=1)
    
    # Extract features and ratings for training recommendations
    train_recommendation_features = []
    train_recommendation_ratings = []
    for user_id, recommendations in train_recommendations.items():
        if recommendations is not None:
            for recommendation in recommendations:
                item_id = recommendation['item_id']
                if user_id_map[user_id] < len(fused_embeddings) and item_id in item_id_map:
                    recommendation_features = fused_embeddings[user_id_map[user_id]].cpu().detach().numpy()
                    recommendation_rating = recommendation['Overall_Rating']
                    train_recommendation_features.append(recommendation_features)
                    train_recommendation_ratings.append(recommendation_rating)

    # Convert lists to NumPy arrays
    train_recommendation_features = np.array(train_recommendation_features)
    train_recommendation_ratings = np.array(train_recommendation_ratings)

    # If there are recommendations, incorporate them into training data
    if len(train_recommendation_features) > 0:
        enhanced_train_X = np.concatenate((train_X, train_recommendation_features), axis=0)
        enhanced_train_y = np.concatenate((train_y, train_recommendation_ratings), axis=0)
        svr_model.fit(enhanced_train_X, enhanced_train_y)
    
    # Prepare test data
    test_user_ids = test_data['User_ID'].values.astype(str)

    for user_id in test_user_ids:
        if user_id not in user_id_map:
            user_id_map[user_id] = len(user_id_map)

    # Convert test user IDs to indices
    test_user_indices = [user_id_map.get(user_id, -1) for user_id in test_user_ids]

    # Filter out invalid indices
    valid_test_user_indices = [index for index in test_user_indices if 0 <= index < len(fused_embeddings)]

    # Index fused_embeddings with valid test user indices
    test_X = fused_embeddings[valid_test_user_indices]

    # Make predictions for test data
    test_predictions = svr_model.predict(test_X)

    # Prepare ground truth ratings for test data
    test_ground_truth_ratings = test_data['Overall_Rating'].values
    trimmed_test_ground_truth_ratings = test_ground_truth_ratings[:len(valid_test_user_indices)]

    # Calculate MAE and RMSE for test data
    test_mae = mean_absolute_error(trimmed_test_ground_truth_ratings, test_predictions)
    test_rmse = mean_squared_error(trimmed_test_ground_truth_ratings, test_predictions)
    print("MAE for test data (SVR):", test_mae)
    print("RMSE for test data (SVR):", test_rmse)

    return test_mae, test_rmse

def evaluate_RS_Model_multiple_runs(fused_embeddings, user_id_map, item_id_map, data, output_path, train_sizes=[0.4, 0.6, 0.8, 1.0], num_runs=30):
    # Lists to store MAE and RMSE values for each training size
    results = {}

    # Perform evaluations for each specified training size
    for train_size in train_sizes:
        mae_values = []
        rmse_values = []

        print(f"Evaluating for training size: {train_size}")

        # Use Leave-One-Out Cross-Validation for train_size 1.0
        if train_size == 1.0:
            loo = LeaveOneOut()
            for i, (train_index, test_index) in enumerate(loo.split(data)):
                if i >= num_runs:  # Limit to num_runs
                    break
                print("Run", i + 1)
                train_data = data.iloc[train_index]
                test_data = data.iloc[test_index]  # Define test_data for LOO
                mae, rmse = evaluate_RS_Model(fused_embeddings, user_id_map, item_id_map, train_data, output_path, test_data=test_data)
                mae_values.append(mae)
                rmse_values.append(rmse)
        else:
            # Perform specified number of runs for other training sizes
            for i in range(num_runs):
                print("Run", i + 1)
                # Split data for training and testing based on the train_size
                train_data, test_data = split_and_save_data(data, output_path, test_size=1-train_size)
                mae, rmse = evaluate_RS_Model(fused_embeddings, user_id_map, item_id_map, train_data, output_path, test_data=test_data)

                mae_values.append(mae)
                rmse_values.append(rmse)

        # Calculate mean and standard deviation
        mean_mae = np.mean(mae_values)
        mae_std = np.std(mae_values)
        mean_rmse = np.mean(rmse_values)
        rmse_std = np.std(rmse_values)

        # Store results for the current training size
        results[train_size] = {
            "mean_mae": mean_mae,
            "mae_std": mae_std,
            "mean_rmse": mean_rmse,
            "rmse_std": rmse_std,
        }

        # Print the results
        print(f"Results for training size {train_size}:")
        print(f"Mean MAE: {mean_mae}, Standard Deviation of MAE: {mae_std}")
        print(f"Mean RMSE: {mean_rmse}, Standard Deviation of RMSE: {rmse_std}")

    return results

# ---------------------Main Function ---------------------------

def main(file_path, criteria, save_embeddings=False):
    # Read data for the selected dataset
    print("Reading data...")
    start_time = time.time()
    data, user_id_map, item_id_map = read_data(file_path)
    print("Reading data finished. Time taken: {:.2f} seconds".format(time.time() - start_time))

    if save_embeddings and not isinstance(save_embeddings, str):
        save_embeddings = file_path + '.embed.pt'
    
    if save_embeddings and os.path.isfile(save_embeddings):
        embeddings_loaded = True
        print("Loading embeddings...")
        start_time = time.time()
        fused_embeddings = torch.load(save_embeddings)
        print("Loading embeddings finished. Time taken: {:.2f} seconds".format(time.time() - start_time))
    else:
        embeddings_loaded = False
        print("Constructing sociomatrices...")
        start_time = time.time()
        matrices = L_BGNN(data, criteria, user_id_map, item_id_map)
        print("Constructing sociomatrices finished. Time taken: {:.2f} seconds".format(time.time() - start_time))

        #---Attention Embedding------
        print("Constructing model...")
        start_time = time.time()
        model = GAT(in_channels=16, out_channels=256)
        print("Constructing model finished. Time taken: {:.2f} seconds".format(time.time() - start_time))

        print("Generating embeddings...")
        start_time = time.time()
        fused_embeddings = model.Multi_Embd(matrices, num_epochs=100, learning_rate=0.01)
        print("Generating embeddings finished. Time taken: {:.2f} seconds".format(time.time() - start_time))

    if save_embeddings and not embeddings_loaded: 
        print("Saving embeddings...")
        start_time = time.time()
        torch.save(fused_embeddings, save_embeddings)
        print("Saving embeddings finished. Time taken: {:.2f} seconds".format(time.time() - start_time))

    # Call the function with the defined threshold function
    print("Evaluating...")
    start_time = time.time()
    Recommendation_items_Top_k(fused_embeddings, user_id_map, data, threshold_func=None, top_k=1)
    print("Recommendation items Top-k evaluation finished. Time taken: {:.2f} seconds".format(time.time() - start_time))

    # Define various training sizes
    training_sizes = [0.4, 0.6, 0.8, 1.0]
    
    # Call evaluate_RS_Model_multiple_runs for each training size
    print("Evaluating multiple runs...")
    start_time = time.time()
    evaluate_RS_Model_multiple_runs(fused_embeddings, user_id_map, item_id_map, data, os.path.dirname(file_path), train_sizes=training_sizes, num_runs=30)
    print("Evaluate RS Model multiple runs finished. Time taken: {:.2f} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    # Define your file paths for different datasets in local Server
    file_paths = {
            'Movies_Yahoo': 'C://MCRS//Movies_Yahoo.xlsx',
            'BeerAdvocate': 'C://MCRS//BeerAdvocate.xlsx',
            'TripAdvisor': 'C://MCRS//TripAdvisor.xlsx'
        }
    
    # Define criteria for different datasets
    criteria_mapping = {
        'Movies_Yahoo': ['C1', 'C2', 'C3', 'C4'],
        'BeerAdvocate': ['C1', 'C2', 'C3', 'C4'],
        'TripAdvisor': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    }

    # Define the dataset to run
    DATASET_TO_RUN = 'TripAdvisor'

    main(file_paths[DATASET_TO_RUN], criteria_mapping[DATASET_TO_RUN], True)
