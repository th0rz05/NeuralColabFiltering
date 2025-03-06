import torch
import torch.nn as nn
import torch.optim as optim

def bpr_loss(pos_scores, neg_scores):
    """
    Calcula a BPR loss para um par (positivo, negativo).
    Loss = -log(sigmoid(pos_score - neg_score))
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

class NCFModel(nn.Module):
    def __init__(self, num_players, num_games, embedding_dim=32):
        """
        Modelo Neural Collaborative Filtering (NCF) com embeddings para jogadores e slots.
        """
        super(NCFModel, self).__init__()
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        self.game_embedding = nn.Embedding(num_games, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, user, game):
        # user e game devem ter formato (batch, 1)
        user_emb = self.player_embedding(user).squeeze(1)  # (batch, embedding_dim)
        game_emb = self.game_embedding(game).squeeze(1)      # (batch, embedding_dim)
        x = torch.cat([user_emb, game_emb], dim=1)           # (batch, embedding_dim*2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)

class Trainer:
    def __init__(self, model, optimizer, criterion, device="cpu"):
        """
        Classe para treinar o modelo com um conjunto de dados de treinamento e validação.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def train(self, train_loader, val_loader, num_epochs=40, patience=5):
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for user, pos_game, neg_game in train_loader:
                user = user.to(self.device)
                pos_game = pos_game.to(self.device)
                neg_game = neg_game.to(self.device)
                
                self.optimizer.zero_grad()
                pos_score = self.model(user, pos_game).squeeze()
                neg_score = self.model(user, neg_game).squeeze()
                loss = self.criterion(pos_score, neg_score)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * user.size(0)
            train_loss /= len(train_loader.dataset)
            
            # Validação
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for user, pos_game, neg_game in val_loader:
                    user = user.to(self.device)
                    pos_game = pos_game.to(self.device)
                    neg_game = neg_game.to(self.device)
                    
                    pos_score = self.model(user, pos_game).squeeze()
                    neg_score = self.model(user, neg_game).squeeze()
                    loss = self.criterion(pos_score, neg_score)
                    val_loss += loss.item() * user.size(0)
            val_loss /= len(val_loader.dataset)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print("Early stopping triggered!")
                break
        
        # Carrega o melhor modelo
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self.model
