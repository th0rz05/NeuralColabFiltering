import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

# Import our custom modules
from model import NCFModel, Trainer, bpr_loss
from recommender import Recommender

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

##############################
# 1. Data Loading & Preprocessing
##############################

# Load data from CSV
data = pd.read_csv("data.csv")
print("Data shape:", data.shape)

# Map original player_id and game_id to consecutive indices
player_ids = data['player_id'].unique().tolist()
game_ids = data['game_id'].unique().tolist()

player2idx = {player: idx for idx, player in enumerate(player_ids)}
game2idx = {game: idx for idx, game in enumerate(game_ids)}

data['player_idx'] = data['player_id'].map(player2idx)
data['game_idx'] = data['game_id'].map(game2idx)

num_players = len(player2idx)
num_games = len(game2idx)
print(f"Number of players: {num_players}, Number of games: {num_games}")

# Create a mapping of user interactions: user_positive
user_positive = data.groupby('player_idx')['game_idx'].apply(set).to_dict()

##############################
# 2. Define Dataset for Pairwise Ranking
##############################

class SlotPairwiseDataset(Dataset):
    def __init__(self, dataframe, num_games):
        self.data = dataframe
        self.num_games = num_games
        # Build a mapping from user to the set of positive game interactions.
        self.user_positive = self.data.groupby('player_idx')['game_idx'].apply(set).to_dict()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user = row['player_idx']
        pos_game = row['game_idx']
        positive_set = self.user_positive[user]
        # Randomly sample a negative game that the user has not interacted with.
        neg_game = np.random.randint(0, self.num_games)
        while neg_game in positive_set:
            neg_game = np.random.randint(0, self.num_games)
        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_game, dtype=torch.long),
                torch.tensor(neg_game, dtype=torch.long))

dataset = SlotPairwiseDataset(data, num_games)

##############################
# 3. Create Training and Validation Sets
##############################

# Split dataset indices (80% train, 20% validation)
all_indices = np.arange(len(dataset))
train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=SEED)

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=256, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=2)

print("Training samples:", len(train_subset))
print("Validation samples:", len(val_subset))

##############################
# 4. Initialize and Train the Model
##############################

embedding_dim = 32
model = NCFModel(num_players, num_games, embedding_dim=embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create Trainer instance
trainer = Trainer(model, optimizer, bpr_loss, device="cpu")  # Change to "cuda" if GPU is available
trained_model = trainer.train(train_loader, val_loader, num_epochs=40, patience=5)

##############################
# 5. Extract Game Embeddings and Build Recommender
##############################

# Extract game embeddings from the trained model.
trained_model.eval()
with torch.no_grad():
    game_embeddings = trained_model.game_embedding.weight.detach().cpu().numpy()

# Create an instance of the Recommender class.
recommender = Recommender(trained_model, game_embeddings, user_positive, num_games, embedding_dim=embedding_dim)

##############################
# 6. Generate Recommendations
##############################

# Example usage: for user with index 0, assuming they recently played slot with index 0.
sample_user_id = 0
sample_recent_slot_id = 0

# Offline recommendation using the model only.
offline_recs, offline_scores = recommender.recommend_slots_for_user(sample_user_id, k=5)
print("Offline Recommendations (Model-Only):")
print("Slot Indices:", offline_recs)
print("Predicted Scores:", offline_scores)

# Offline hybrid recommendation combining model predictions and similarity to a recent slot.
hybrid_recs, hybrid_scores = recommender.recommend_slots_for_user_based_on_recent_slot(sample_user_id, sample_recent_slot_id, k=5, alpha=0.5)
print("\nOffline Hybrid Recommendations (User based on recent slot):")
print("Slot Indices:", hybrid_recs)
print("Combined Scores:", hybrid_scores)

# Real-time recommendations using the FAISS index.
realtime_recs, realtime_dists = recommender.realtime_recommendation_for_recent_slot(sample_user_id, sample_recent_slot_id, k=5, search_k=10)
print("\nReal-Time Recommendations (using FAISS):")
print("Slot Indices:", realtime_recs)
print("L2 Distances:", realtime_dists)
