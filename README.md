1. Methodology

In this research, we propose a novel framework for Multi-Criteria Recommender Systems (MCRS) that utilizes Multiview and Dual Graph Attention and Contrastive Learning (D-MGAC). We define a bipartite graph structure with multi-edges between users and items, where the number of edges corresponds to the number of criteria. Each criterion is represented as a separate view, and for each view, we generate subgraphs that capture the relationships between users and items based on the specific criterion. To generate embeddings, we apply an attention mechanism that includes both local and global attention. Additionally, we employ contrastive learning with a loss function that includes local and global terms. Positive and negative pairs are defined locally for each view and globally across the entire graph.

2. Datasets

2.1 BeerAdvocate

The BeerAdvocate dataset (www.beeradvocate.com) allows users to rate four attributes of beer: Aroma, Appearance, Palate, and Taste. After filtering out inactive users, the preprocessed dataset consists of 8,831 users, 2,698 items, and 3,880,359 ratings. Both overall and multi-criteria ratings are provided on a 1–5 scale.

2.2 Yahoo!Movies

The Yahoo!Movies dataset (www.movies.yahoo.com) contains multi-criteria movie ratings. Users rate movies based on an overall score as well as four specific criteria: Story, Acting, Direction, and Visuals. The dataset comprises 6,078 users, 976 films, and a total of 758,405 ratings, with scores ranging from 1 to 5.

2.3 TripAdvisor

We curated the TripAdvisor dataset (www.tripadvisor.com) and published the extracted data instance on Zenodo (https://sandbox.zenodo.org/records/109408). This dataset includes 15,000 users and 1,325 hotels, with ratings based on 7 criteria. In total, the dataset contains 504,598 ratings.

3. Code Structure:
   The code is organized into several functions and classes, introduced as follows:

1. `def read_data(file_path)`:  
   This function reads the dataset from the specified file path.

2. `def L_BGNN(data, criteria, user_id_map, item_id_map)`:  
   This function defines adjacency matrices in the L_BGNN format for bipartite graphs, which represent user-item relationships based on the given criteria.

3. `class GAT(nn.Module)`:  
   This class implements the Graph Attention Network (GAT) model, with dual attention mechanisms (local and global). It includes the following methods:
   
   - `def fusion_embeddings_vectors`: Combines embeddings from different views.
   - `def Multi_Embd`: Generates multiple embeddings for different criteria or views.
   - `def local_contrastive_loss`: Computes the contrastive loss at the local level for individual views.
   - `def global_contrastive_loss`: Computes the contrastive loss at the global level across all views.
   - `def l2_regularization`: Implements L2 regularization for model parameters.
   - `def train_GAT`: Trains the GAT model using the defined dual attention mechanisms.

4. `def Recommendation_items_Top_k`:  
   Generates top-k recommendations for users based on embedding similarities.

5. `def split_and_save_data`:  
   Prepares the dataset for model evaluation, splitting it into training (80%) and testing (20%) sets.

6. `def evaluate_RS_Model`:  
   Evaluates the recommender system using performance metrics like Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

7. `def evaluate_RS_Model_multiple_runs`:  
   Implements parallel processing to run the evaluation multiple times standard deviation (Std=30) to assess model robustness and stability.

