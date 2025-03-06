import numpy as np
import torch
import faiss

class Recommender:
    def __init__(self, model, game_embeddings, user_positive, num_games, embedding_dim=32):
        """
        Classe para gerar recomendações utilizando o modelo treinado.
        
        Args:
            model: Modelo NCF treinado.
            game_embeddings: Numpy array com os embeddings dos slots (jogos).
            user_positive: Dicionário que mapeia cada usuário (player_idx) para um conjunto de slots já interagidos.
            num_games: Número total de jogos.
            embedding_dim: Dimensão dos embeddings.
        """
        self.model = model
        self.game_embeddings = game_embeddings.astype('float32')
        self.user_positive = user_positive
        self.num_games = num_games
        self.embedding_dim = embedding_dim
        self.faiss_index = self.build_faiss_index()
    
    def build_faiss_index(self):
        """
        Cria e retorna um índice FAISS para buscas rápidas (usando distância L2).
        """
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(self.game_embeddings)
        return index
    
    def predict_user_scores(self, user_id, candidate_games):
        """
        Prediz as pontuações para um usuário em um conjunto de jogos candidatos usando o modelo NCF.
        """
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id] * len(candidate_games), dtype=torch.long).unsqueeze(1)
            game_tensor = torch.tensor(candidate_games, dtype=torch.long).unsqueeze(1)
            predictions = self.model(user_tensor, game_tensor).squeeze().cpu().numpy()
        return predictions
    
    def recommend_slots_for_user(self, user_id, k=5):
        """
        Gera recomendações offline (model-only) para um usuário, excluindo os jogos já interagidos.
        """
        interacted = self.user_positive.get(user_id, set())
        candidate_games = [g for g in range(self.num_games) if g not in interacted]
        if not candidate_games:
            return np.array([]), np.array([])
        scores = self.predict_user_scores(user_id, candidate_games)
        top_indices = np.argsort(scores)[-k:][::-1]
        return np.array(candidate_games)[top_indices], scores[top_indices]
    
    def cosine_similarity_matrix(self, query_embedding, candidate_embeddings):
        """
        Calcula a similaridade cosseno entre um embedding de consulta e um conjunto de embeddings candidatos.
        """
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        cos_sim = np.dot(candidate_norms, query_norm)
        return (cos_sim + 1) / 2  # Escala de [-1, 1] para [0, 1]
    
    def recommend_slots_for_user_based_on_recent_slot(self, user_id, recent_slot_id, k=5, alpha=0.5):
        """
        Gera recomendações offline híbridas combinando a predição do modelo com similaridade
        ao slot que o usuário jogou recentemente.
        """
        interacted = self.user_positive.get(user_id, set())
        candidate_games = [g for g in range(self.num_games) if g not in interacted]
        if not candidate_games:
            return np.array([]), np.array([])
        predicted_scores = self.predict_user_scores(user_id, candidate_games)
        recent_embedding = self.game_embeddings[recent_slot_id]
        candidate_embeddings = self.game_embeddings[candidate_games]
        similarity_scores = self.cosine_similarity_matrix(recent_embedding, candidate_embeddings)
        combined_scores = alpha * predicted_scores + (1 - alpha) * similarity_scores
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        return np.array(candidate_games)[top_indices], combined_scores[top_indices]
    
    def realtime_recommendation_for_recent_slot(self, user_id, recent_slot_id, k=5, search_k=10):
        """
        Gera recomendações em tempo real utilizando o índice FAISS para buscar slots semelhantes
        ao slot recentemente jogado, filtrando os já interagidos.
        """
        query_vector = np.expand_dims(self.game_embeddings[recent_slot_id], axis=0)
        distances, indices = self.faiss_index.search(query_vector, search_k)
        candidate_games = indices[0]
        candidate_distances = distances[0]
        interacted = self.user_positive.get(user_id, set())
        filtered = []
        for game, dist in zip(candidate_games, candidate_distances):
            if game not in interacted and game != recent_slot_id:
                filtered.append((game, dist))
            if len(filtered) == k:
                break
        if not filtered:
            return np.array([]), np.array([])
        recs = np.array([x[0] for x in filtered])
        dists = np.array([x[1] for x in filtered])
        return recs, dists
