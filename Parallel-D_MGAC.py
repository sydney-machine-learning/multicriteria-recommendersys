#!/usr/bin/env -S python3 -u
#PBS -N Saman
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -l walltime=72:00:00
#PBS -v OMP_NUM_THREADS=32
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
import concurrent.futures
import logging
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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

        mse_loss_val = self.mse_loss(embeddings, embeddings_list, attention_weights)
        global_similarity_loss_value = self.global_similarity_loss(embeddings_list)
        l2_reg = self.l2_regularization()

        # Combine MSE loss, Similarity loss, and L2 regularization
        total_loss = (alpha * mse_loss_val + beta * global_similarity_loss_value) + gamma * l2_reg

        # Update attention weights based on MSE loss and Similarity loss
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

    # Iterate over all users in order of index
    grouped = data.groupby('User_ID')
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

            # Retrieve the current user's rating from the data
            user_data = grouped.get_group(user_id)
            if len(user_data) > 0:  # Check if there are ratings for this user
                current_user_rating = user_data['Overall_Rating'].values[0]

                # Get recommended items for the user
                for user_idx_2 in similar_users_sorted_idx:
                    user_id_2 = uids[user_idx_2][0]
                    for _, row in grouped.get_group(user_id_2).iterrows():
                        item_id = row['Items_ID']
                        overall_rating = row['Overall_Rating']

                        # Check if overall rating is similar to the
                        # current user's rating and filter out items
                        # already rated by the current user
                        if item_id not in user_data['Items_ID'].values and abs(overall_rating - current_user_rating) <= threshold_A:
                            recommended_items.append({'item_id': item_id, 'Overall_Rating': overall_rating})

                # Sort recommended items by overall rating
                recommended_items = sorted(recommended_items, key=lambda x: x['Overall_Rating'], reverse=True)[:top_k]

                # Add recommended items for the current user to the dictionary
                recommendations_f_items[user_id] = recommended_items
            else:
                recommendations_f_items[user_id] = None
        else:
            # No similar users found, pass the user's embedding
            recommendations_f_items[user_id] = None

    return recommendations_f_items


def split_data(data, test_size=0.2, random_state=42):
    """ Split the data into train and test sets. """
    data['User_ID'] = data['User_ID'].astype(str)
    data['Items_ID'] = data['Items_ID'].astype(str)
    
    data_subset = data[['User_ID', 'Items_ID', 'Overall_Rating']]
    train_data, test_data = train_test_split(data_subset, test_size=test_size, random_state=random_state)
    
    return train_data, test_data

def save_data(train_data, test_data, output_path):
    """ Save train and test sets to Excel files. """
    train_file_path = os.path.join(output_path, 'train_data.xlsx')
    test_file_path = os.path.join(output_path, 'test_data.xlsx')

    train_data.to_excel(train_file_path, index=False)
    test_data.to_excel(test_file_path, index=False)

def train_svr_model(train_data, fused_embeddings, user_id_map):
    """ Train SVR model using training data and embeddings. """
    train_X = fused_embeddings.cpu().detach().numpy()[train_data['User_ID'].astype('category').cat.codes]
    train_y = train_data['Overall_Rating'].values

    svr_model = SVR()
    svr_model.fit(train_X, train_y)

    # Compute train predictions and metrics
    train_predictions = svr_model.predict(train_X)
    train_mae = mean_absolute_error(train_y, train_predictions)
    train_rmse = mean_squared_error(train_y, train_predictions) ** 0.5
    
    return svr_model, train_mae, train_rmse

def create_subsets(train_data, subset_sizes):
    """ Create subsets from the training data based on the specified sizes. """
    subsets = {}
    total_size = len(train_data)

    for size in subset_sizes:
        if size > total_size:
            raise ValueError("Subset size cannot be greater than the total training data size.")
        
        subset = train_data.sample(frac=size / 100.0, random_state=42)
        subsets[f'{size}%'] = subset
    
    return subsets

def evaluate_subsets(subsets, svr_model, fused_embeddings, user_id_map):
    """ Evaluate the SVR model on subsets of training data. """
    subset_mae_values = []
    subset_rmse_values = []

    for label, subset in subsets.items():
        subset_X = fused_embeddings.cpu().detach().numpy()[subset['User_ID'].astype('category').cat.codes]
        subset_y = subset['Overall_Rating'].values
        subset_predictions = svr_model.predict(subset_X)

        subset_mae = mean_absolute_error(subset_y, subset_predictions)
        subset_rmse = mean_squared_error(subset_y, subset_predictions) ** 0.5
        print(f"MAE for {label} subset:", subset_mae)
        print(f"RMSE for {label} subset:", subset_rmse)

        subset_mae_values.append(subset_mae)
        subset_rmse_values.append(subset_rmse)

    # Calculate mean and std for MAE and RMSE
    subset_mae_mean = np.mean(subset_mae_values)
    subset_mae_std = np.std(subset_mae_values)
    subset_rmse_mean = np.mean(subset_rmse_values)
    subset_rmse_std = np.std(subset_rmse_values)

    print(f"\nOverall MAE for subsets: {subset_mae_mean:.4f} ± {subset_mae_std:.4f}")
    print(f"Overall RMSE for subsets: {subset_rmse_mean:.4f} ± {subset_rmse_std:.4f}")


def evaluate_RS_Model(fused_embeddings, user_id_map, item_id_map, data, output_path, test_size=0.2, random_state=42):
    # Split and save the data into train and test sets
    train_data, test_data = split_data(data, test_size=test_size, random_state=random_state)
    save_data(train_data, test_data, output_path)
    
    # Prepare training data
    train_X = fused_embeddings.cpu().detach().numpy()[train_data['User_ID'].astype('category').cat.codes]
    train_y = train_data['Overall_Rating'].values

    # Instantiate and train the SVR model
    svr_model = SVR()
    svr_model.fit(train_X, train_y)

    # Compute predictions for train data
    train_predictions = svr_model.predict(train_X)

    # Calculate MAE and RMSE for train data
    train_mae = mean_absolute_error(train_y, train_predictions)
    train_rmse = mean_squared_error(train_y, train_predictions) ** 0.5
    print("MAE for train data:", train_mae)
    print("RMSE for train data:", train_rmse)
    
    # Define the threshold function
    def threshold_function(embedding):
        return torch.tensor(0.1)

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

    train_recommendation_features = np.array(train_recommendation_features)
    train_recommendation_ratings = np.array(train_recommendation_ratings)

    # If there are recommendations, incorporate them into training data
    if len(train_recommendation_features) > 0:
        enhanced_train_X = np.concatenate((train_X, train_recommendation_features), axis=0)
        enhanced_train_y = np.concatenate((train_y, train_recommendation_ratings), axis=0)
        
        # Retrain the SVR model using enhanced training data
        svr_model.fit(enhanced_train_X, enhanced_train_y)
    
    # Prepare test data
    test_user_ids = test_data['User_ID'].values.astype(str)
    test_user_indices = [user_id_map.get(user_id, -1) for user_id in test_user_ids]
    valid_test_user_indices = [index for index in test_user_indices if 0 <= index < len(fused_embeddings)]
    test_X = fused_embeddings[valid_test_user_indices]

    # Make predictions for test data
    test_predictions = svr_model.predict(test_X)

    # Prepare ground truth ratings for test data
    test_ground_truth_ratings = test_data['Overall_Rating'].values
    trimmed_test_ground_truth_ratings = test_ground_truth_ratings[:len(valid_test_user_indices)]

    # Calculate MAE and RMSE for test data
    test_mae = mean_absolute_error(trimmed_test_ground_truth_ratings, test_predictions)
    test_rmse = mean_squared_error(trimmed_test_ground_truth_ratings, test_predictions) ** 0.5
    print("MAE for test data:", test_mae)
    print("RMSE for test data:", test_rmse)

    # Evaluate subsets
    subset_sizes = [40, 60, 80, 100]  # Define the subset sizes to evaluate
    subsets = create_subsets(train_data, subset_sizes)

    # Lists to hold MAE and RMSE values for each subset
    subset_mae_values = []
    subset_rmse_values = []

    for label, subset in subsets.items():
        subset_X = fused_embeddings.cpu().detach().numpy()[subset['User_ID'].astype('category').cat.codes]
        subset_y = subset['Overall_Rating'].values
        subset_predictions = svr_model.predict(subset_X)

        subset_mae = mean_absolute_error(subset_y, subset_predictions)
        subset_rmse = mean_squared_error(subset_y, subset_predictions) ** 0.5
        print(f"MAE for {label} subset:", subset_mae)
        print(f"RMSE for {label} subset:", subset_rmse)

        # Store values for standard deviation calculation
        subset_mae_values.append(subset_mae)
        subset_rmse_values.append(subset_rmse)

    # Calculate and print the mean and std for the subsets
    subset_mae_mean = np.mean(subset_mae_values)
    subset_mae_std = np.std(subset_mae_values)
    subset_rmse_mean = np.mean(subset_rmse_values)
    subset_rmse_std = np.std(subset_rmse_values)

    print(f"\nOverall MAE for subsets: {subset_mae_mean:.4f} ± {subset_mae_std:.4f}")
    print(f"Overall RMSE for subsets: {subset_rmse_mean:.4f} ± {subset_rmse_std:.4f}")

    return train_mae, train_rmse, test_mae, test_rmse

def evaluate_RS_Model_multiple_runs(fused_embeddings, user_id_map, item_id_map, data, output_path, test_size=0.2, run_counts=[30]):
    results = {}

    for num_runs in run_counts:
        print(f"Evaluating for {num_runs} runs")
        
        # Lists to store MAE and RMSE values for train and test from each run
        train_mae_values = []
        train_rmse_values = []
        test_mae_values = []
        test_rmse_values = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_RS_Model, fused_embeddings, user_id_map, item_id_map, data, output_path, test_size, i) for i in range(num_runs)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    train_mae, train_rmse, test_mae, test_rmse = future.result()
                    train_mae_values.append(train_mae)
                    train_rmse_values.append(train_rmse)
                    test_mae_values.append(test_mae)
                    test_rmse_values.append(test_rmse)
                except Exception as exc:
                    print(f"Run generated an exception: {exc}")

        # Calculate the mean and standard deviation for train and test MAE and RMSE
        train_mae_mean = np.mean(train_mae_values)
        train_mae_std = np.std(train_mae_values)
        train_rmse_mean = np.mean(train_rmse_values)
        train_rmse_std = np.std(train_rmse_values)

        test_mae_mean = np.mean(test_mae_values)
        test_mae_std = np.std(test_mae_values)
        test_rmse_mean = np.mean(test_rmse_values)
        test_rmse_std = np.std(test_rmse_values)

        # Store results in the dictionary
        results[num_runs] = {
            'train_mae_mean': train_mae_mean,
            'train_mae_std': train_mae_std,
            'train_rmse_mean': train_rmse_mean,
            'train_rmse_std': train_rmse_std,
            'test_mae_mean': test_mae_mean,
            'test_mae_std': test_mae_std,
            'test_rmse_mean': test_rmse_mean,
            'test_rmse_std': test_rmse_std
        }

        # Print summary of results for the run
        print(f"\nSummary for {num_runs} runs:")
        print(f"Train MAE: {train_mae_mean:.4f} ± {train_mae_std:.4f}")
        print(f"Train RMSE: {train_rmse_mean:.4f} ± {train_rmse_std:.4f}")
        print(f"Test MAE: {test_mae_mean:.4f} ± {test_mae_std:.4f}")
        print(f"Test RMSE: {test_rmse_mean:.4f} ± {test_rmse_std:.4f}")
    
    return results

# ---------------------Main Function ---------------------------

def main(file_path, criteria, save_embeddings=False):
    # Read data for the selected dataset
    logging.info("Reading data...")
    start_time = time.time()
    data, user_id_map, item_id_map = read_data(file_path)
    logging.info(f"Reading data finished. Time taken: {time.time() - start_time:.2f} seconds")

    # Determine save path for embeddings
    if save_embeddings and not isinstance(save_embeddings, str):
        save_embeddings = file_path + '.embed.pt'

    # Check if embeddings exist; if so, load them
    if save_embeddings and os.path.isfile(save_embeddings):
        embeddings_loaded = True
        logging.info("Loading embeddings...")
        start_time = time.time()
        fused_embeddings = torch.load(save_embeddings, weights_only=True)
        logging.info(f"Loading embeddings finished. Time taken: {time.time() - start_time:.2f} seconds")
    else:
        embeddings_loaded = False
        logging.info("Constructing sociomatrices...")
        start_time = time.time()
        matrices = L_BGNN(data, criteria, user_id_map, item_id_map)
        logging.info(f"Constructing sociomatrices finished. Time taken: {time.time() - start_time:.2f} seconds")

        # Constructing the model
        logging.info("Constructing model...")
        start_time = time.time()
        model = GAT(in_channels=16, out_channels=256)
        logging.info(f"Constructing model finished. Time taken: {time.time() - start_time:.2f} seconds")

        # Generating embeddings
        logging.info("Generating embeddings...")
        start_time = time.time()
        fused_embeddings = model.Multi_Embd(matrices, num_epochs=100, learning_rate=0.01)
        logging.info(f"Generating embeddings finished. Time taken: {time.time() - start_time:.2f} seconds")

    # Save embeddings if they were generated
    if save_embeddings and not embeddings_loaded: 
        logging.info("Saving embeddings...")
        start_time = time.time()
        torch.save(fused_embeddings, save_embeddings)
        logging.info(f"Embeddings saved to {save_embeddings}. Time taken: {time.time() - start_time:.2f} seconds")
    
    # Running the evaluation for multiple runs
    output_path = f"{file_path}.csv"
    results = evaluate_RS_Model_multiple_runs(fused_embeddings, user_id_map, item_id_map, data, output_path, run_counts=[30])

    # Print results
    for run, metrics in results.items():
        print(f"Results for {run} runs:")
        print(f"Train MAE: {metrics['train_mae_mean']} ± {metrics['train_mae_std']}")
        print(f"Train RMSE: {metrics['train_rmse_mean']} ± {metrics['train_rmse_std']}")
        print(f"Test MAE: {metrics['test_mae_mean']} ± {metrics['test_mae_std']}")
        print(f"Test RMSE: {metrics['test_rmse_mean']} ± {metrics['test_rmse_std']}")

if __name__ == "__main__":
    # Define file paths for different datasets on the local server
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

    # Run the main function with the specified dataset and criteria
    main(file_paths[DATASET_TO_RUN], criteria_mapping[DATASET_TO_RUN], True)
