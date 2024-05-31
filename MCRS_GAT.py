import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GATConv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR


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

        # For the following, note that adj_matrix is symmetric.

        # Calculate vector of degrees
        margins = np.maximum(np.sum(adj_matrix, axis=0), 1.0)

        # Divide the matrix by the harmonic mean of its margins.
        normalized_matrix = (adj_matrix / margins[:, None] + adj_matrix / margins[None, :]) / 2

        matrices.append(normalized_matrix)
        
        # # Print the normalized matrix
        # print(f"\nNormalized Matrix for criterion '{criterion}':")
        # print(normalized_matrix)

    return tuple(matrices)

# ------------------------ Define the GAT model
class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, dropout=0.2):
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
        self.dropout = nn.Dropout(dropout)  # Add dropout layer

    def forward(self, x, edge_index):
        x = self.dropout(x)  # Apply dropout to input features

        # Local Attention
        head_outs = [F.dropout(conv(x, edge_index), p=0.2, training=self.training) for conv in self.conv_layers]
        x_local = torch.cat(head_outs, dim=-1)
        # Self-Attention within each head
        self_attention = F.leaky_relu(self.fc(x_local))
        self_attention = F.softmax(self_attention, dim=-1)

        # Multiply each element in x_local by the corresponding element in self_attention
        x_local = x_local * self_attention

        # Apply LeakyReLU activation
        x_local = self.leakyrelu(x_local)

        # Apply Fully Connected Layer
        x_local = F.dropout(self.fc(x_local), p=0.2, training=self.training)

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

        # Calculate cosine similarity between embeddings
        similarities = cosine_similarity(fused_embeddings.cpu().numpy())

        # Print maximum and minimum cosine similarity
        max_similarity = similarities.max()
        min_similarity = similarities.min()
        print(f"Maximum cosine similarity: {max_similarity}")
        print(f"Minimum cosine similarity: {min_similarity}")

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

def create_ground_truth_ratings(data, criteria):
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

def Recommendation_items_Top_k(fused_embeddings, user_id_map, data, threshold_func=None, top_k=1):
    recommendations_f_items = {}

    # Convert fused embeddings to numpy array
    fused_embeddings_np = fused_embeddings.cpu().detach().numpy()

    # Compute similarities between embeddings
    similarities = cosine_similarity(fused_embeddings_np)

    # Iterate over all users
    for user_idx, user_id in enumerate(user_id_map.keys()):
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
            user_data = data[data['User_ID'] == user_id]
            if len(user_data) > 0:  # Check if there are ratings for this user
                current_user_rating = user_data['Overall_Rating'].values[0]

                # Get recommended items for the user
                for user_idx_2 in similar_users_sorted_idx:
                    if user_idx_2 >= len(user_id_map.keys()):
                        continue  # Skip if index is out of range
                    user_id_2 = list(user_id_map.keys())[user_idx_2]
                    for _, row in data[data['User_ID'] == user_id_2].iterrows():
                        item_id = row['Items_ID']
                        overall_rating = row['Overall_Rating']

                        # Check if overall rating is similar to the current user's rating
                        if abs(overall_rating - current_user_rating) <= threshold_A:
                            recommended_items.append({'item_id': item_id, 'Overall_Rating': overall_rating})

                # Filter out items already rated by the current user
                recommended_items = [item for item in recommended_items if
                                     item['item_id'] not in user_data['Items_ID'].values]

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

def split_and_save_data(file_path, criteria, test_size=0.2, random_state=42):
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

def evaluate_RS_Model(fused_embeddings, user_id_map, item_id_map, data, file_path, criteria, test_size=0.2, random_state=42):
    # Split and save the data into train and test sets
    train_data, test_data = split_and_save_data(file_path, criteria, test_size=test_size, random_state=random_state)
    
    # Prepare training data
    train_X = fused_embeddings.cpu().detach().numpy()[train_data['User_ID'].astype('category').cat.codes]
    train_y = train_data['Overall_Rating'].values

    # Instantiate and train the SVR model with sigmoid kernel
    svr_model = SVR()
    svr_model.fit(train_X, train_y)
    
    # Define the threshold function
    def threshold_function(embedding):
        # Define your threshold calculation logic here
        return torch.tensor(0.1)

    # Call the function with the defined threshold function to get training recommendations
    train_recommendations = Recommendation_items_Top_k(fused_embeddings, user_id_map, data, threshold_func=threshold_function, top_k=1)
    
    # Extract features and ratings for training recommendations
    train_recommendation_features = []
    train_recommendation_ratings = []
    for user_id, recommendations in train_recommendations.items():
        if recommendations is not None:
            # Iterate over each recommendation for the user
            for recommendation in recommendations:
                # Extract the item ID for the recommendation
                item_id = recommendation['item_id']
                # Extract features for the recommended item
                if user_id_map[user_id] < len(fused_embeddings) and item_id in item_id_map:
                    recommendation_features = fused_embeddings[user_id_map[user_id]].cpu().detach().numpy()
                    # Extract the rating for the recommendation
                    recommendation_rating = recommendation['Overall_Rating']
                    # Append the features and rating to the respective lists
                    train_recommendation_features.append(recommendation_features)
                    train_recommendation_ratings.append(recommendation_rating)

    # Convert lists to NumPy arrays
    train_recommendation_features = np.array(train_recommendation_features)
    train_recommendation_ratings = np.array(train_recommendation_ratings)

    # If there are recommendations, incorporate them into training data
    if len(train_recommendation_features) > 0:
        # Concatenate original training data with recommendation features and ratings
        enhanced_train_X = np.concatenate((train_X, train_recommendation_features), axis=0)
        enhanced_train_y = np.concatenate((train_y, train_recommendation_ratings), axis=0)
        
        # Retrain the SVR model using enhanced training data
        svr_model.fit(enhanced_train_X, enhanced_train_y)
    
    # Prepare test data
    test_user_ids = test_data['User_ID'].values.astype(str)
    test_item_ids = test_data['Items_ID'].values.astype(str)

    # Ensure all user IDs in test data are present in user_id_map
    for user_id in test_user_ids:
        if user_id not in user_id_map:
            user_id_map[user_id] = len(user_id_map)

    # Convert test user IDs to indices, ensuring they are within bounds
    test_user_indices = [user_id_map.get(user_id, -1) for user_id in test_user_ids]

    # Filter out invalid indices (-1) and those exceeding the length of fused_embeddings
    valid_test_user_indices = [index for index in test_user_indices if 0 <= index < len(fused_embeddings)]

    # Index fused_embeddings with test user indices
    test_X = fused_embeddings[valid_test_user_indices]

    # Make predictions for test data using the SVR model
    test_predictions = svr_model.predict(test_X)

    # Prepare ground truth ratings for test data
    test_ground_truth_ratings = test_data['Overall_Rating'].values

    # Trim ground truth ratings to match the length of valid_test_user_indices
    trimmed_test_ground_truth_ratings = test_ground_truth_ratings[:len(valid_test_user_indices)]

    # Calculate MAE and RMSE for test data
    test_mae = mean_absolute_error(trimmed_test_ground_truth_ratings, test_predictions)
    test_rmse = mean_squared_error(trimmed_test_ground_truth_ratings, test_predictions, squared=False)
    
    
    print("MAE for test data (SVR):", test_mae)
    print("RMSE for test data (SVR):", test_rmse)

    return test_mae, test_rmse

def evaluate_RS_Model_multiple_runs(fused_embeddings, user_id_map, item_id_map, data, file_path, criteria, test_size=0.2, num_runs=30):
    # Lists to store MAE and RMSE values from each run
    mae_values = []
    rmse_values = []

    # Perform specified number of runs of the function and collect MAE and RMSE values
    for i in range(num_runs):
        print("Run", i+1)
        mae, rmse = evaluate_RS_Model(fused_embeddings, user_id_map, item_id_map, data, file_path, criteria, test_size=test_size, random_state=i)
        mae_values.append(mae)
        rmse_values.append(rmse)

    # Calculate the standard deviation
    mae_std = np.std(mae_values)
    rmse_std = np.std(rmse_values)

    # Calculate the mean of standard deviations
    mean_mae_std = np.mean(mae_values)
    mean_rmse_std = np.mean(rmse_values)

    # Print the standard deviation
    print("Standard deviation of MAE over {} runs:".format(num_runs), mae_std)
    print("Standard deviation of RMSE over {} runs:".format(num_runs), rmse_std)

    # Print the mean of standard deviations
    print("Mean of MAE over {} runs:".format(num_runs), mean_mae_std)
    print("Mean of RMSE over {} runs:".format(num_runs), mean_rmse_std)

    # Return the standard deviations
    return mae_std, rmse_std

# ---------------------Main Function ---------------------------

def main(file_path, criteria):
    # Read data for the selected dataset
    user_id_map, item_id_map, base_ground_truth_ratings = read_data(file_path, criteria)
    num_users = len(user_id_map)
    num_items = len(item_id_map)
    num_criteria = len(criteria)
    print("Reading data...")
    data = pd.read_excel(file_path)
    print("Reading data finished.")

    # Read data from the Excel file and create ID mappings  
    print("Constructing sociomatrices...")
    matrices = L_BGNN(file_path, criteria, user_id_map, item_id_map)
    print("Constructing sociomatrices finished.")

    #---Attention Embedding------
    print("Constructing model...")
    model = GAT(in_channels=16, out_channels=256)
    print("Constructing model finished.")
    print("Generating embeddings...")
    fused_embeddings = model.Multi_Embd(matrices, num_epochs=100, learning_rate=0.01)
    print("Generating embeddings finished.")
    torch.save(fused_embeddings, file_path + '.embed.pt')

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
    data, ground_truth_ratings_matrix, user_id_map, item_id_map = create_ground_truth_ratings(data, criteria)

    # Define the threshold function
    def threshold_function(embedding):
        return torch.tensor(0.1)

    # Call the function with the defined threshold function
    print("Evaluating...")
    recommendations = Recommendation_items_Top_k(fused_embeddings, user_id_map, data, threshold_func=None, top_k=1)
    train_data, test_data = split_and_save_data(file_path, criteria)   
    test_mae, test_rmse=evaluate_RS_Model(fused_embeddings, user_id_map, item_id_map, data, file_path, criteria, test_size=0.2, random_state=42)
    mae_std, rmse_std=evaluate_RS_Model_multiple_runs(fused_embeddings, user_id_map, item_id_map, data, file_path, criteria, test_size=0.2, num_runs=30)

if __name__ == "__main__":

    # Define your file paths for different datasets in Katana Server
    # file_paths = {
    #     'Movies_Yahoo': '/home/z5318340/MCRS4/Movies_Original_Second.xlsx',
    #     'BeerAdvocate': '/home/z5318340/MCRS4/BeerAdvocate.xlsx',
    #     'TripAdvisor': '/home/z5318340/MCRS4/Custmoze_Tripadvisor2.xlsx'
    # }

    # Define your file paths for different datasets in local Server
    file_paths = {
        'Movies_Yahoo': 'C://Yahoo//Global//Movies_Yahoo.xlsx',
        'BeerAdvocate': 'C://Yahoo//Global//BeerAdvocate.xlsx',
        'TripAdvisor': 'C://Yahoo//Global//TripAdvisor.xlsx'
    }

    # Define criteria for different datasets
    criteria_mapping = {
        'Movies_Yahoo': ['C1', 'C2', 'C3', 'C4'],
        'BeerAdvocate': ['C1', 'C2', 'C3', 'C4'],
        'TripAdvisor': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    }

    # Define the dataset to run
    dataset_to_run = 'BeerAdvocate'

    main(file_paths[dataset_to_run], criteria_mapping[dataset_to_run])
