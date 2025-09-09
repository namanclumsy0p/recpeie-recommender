import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from app.models.data_processor import DataProcessor
from app.config import config

class HybridRecommender:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.user_movie_matrix = None
        self.svd_model = None
        self.user_factors = None
        self.movie_factors = None
        
    def initialize(self) -> None:
        """Initialize the recommender system"""
        print("Initializing recommender system...")
        
        # Load data and embeddings
        self.data_processor.load_data()
        self.data_processor.load_embeddings()
        
        # Prepare collaborative filtering
        self.user_movie_matrix = self.data_processor.get_user_movie_matrix()
        self._train_collaborative_filtering()
        
        print("Recommender system initialized successfully")
    
    def _train_collaborative_filtering(self) -> None:
        """Train collaborative filtering model using SVD"""
        print("Training collaborative filtering model...")
        
        # Use SVD for matrix factorization
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        self.user_factors = self.svd_model.fit_transform(self.user_movie_matrix)
        self.movie_factors = self.svd_model.components_.T
        
        print("Collaborative filtering model trained")
    
    def get_content_recommendations(self, movie_id: int, top_k: int = 10) -> List[Dict]:
        """Get content-based recommendations using embeddings"""
        if movie_id not in self.data_processor.movie_id_to_index:
            return []
        
        movie_idx = self.data_processor.movie_id_to_index[movie_id]
        
        # Find similar movies
        similar_movies = self.data_processor.find_similar_movies(movie_idx, top_k)
        
        recommendations = []
        for idx, similarity in similar_movies:
            movie_data = self.data_processor.movies_df.iloc[idx]
            recommendations.append({
                'movieId': int(movie_data['movieId']),
                'title': movie_data['title'],
                'genres': movie_data['genres'],
                'similarity': float(similarity),
                'method': 'content_based'
            })
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict]:
        """Get collaborative filtering recommendations"""
        if user_id not in self.user_movie_matrix.index:
            return self._get_popular_movies(top_k)
        
        user_idx = list(self.user_movie_matrix.index).index(user_id)
        user_vector = self.user_factors[user_idx:user_idx+1]
        
        # Predict ratings for all movies
        predicted_ratings = np.dot(user_vector, self.movie_factors.T)[0]
        
        # Get movies the user hasn't rated
        user_rated_movies = set(
            self.user_movie_matrix.loc[user_id][
                self.user_movie_matrix.loc[user_id] > 0
            ].index
        )
        
        # Create recommendations
        movie_ids = self.user_movie_matrix.columns
        recommendations = []
        
        for movie_id, predicted_rating in zip(movie_ids, predicted_ratings):
            if movie_id not in user_rated_movies:
                movie_data = self.data_processor.movies_df[
                    self.data_processor.movies_df['movieId'] == movie_id
                ].iloc[0]
                
                recommendations.append({
                    'movieId': int(movie_id),
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'predicted_rating': float(predicted_rating),
                    'method': 'collaborative_filtering'
                })
        
        # Sort by predicted rating and return top k
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:top_k]
    
    def get_hybrid_recommendations(
        self, 
        user_id: int = None, 
        movie_id: int = None, 
        top_k: int = 10,
        content_weight: float = 0.6,
        collaborative_weight: float = 0.4
    ) -> List[Dict]:
        """Get hybrid recommendations combining content and collaborative filtering"""
        
        recommendations = []
        
        # Get content-based recommendations if movie_id is provided
        if movie_id:
            content_recs = self.get_content_recommendations(movie_id, top_k)
            for rec in content_recs:
                rec['score'] = rec['similarity'] * content_weight
                recommendations.append(rec)
        
        # Get collaborative filtering recommendations if user_id is provided
        if user_id:
            collab_recs = self.get_collaborative_recommendations(user_id, top_k)
            for rec in collab_recs:
                rec['score'] = rec['predicted_rating'] * collaborative_weight
                recommendations.append(rec)
        
        # If we have both types, merge and re-score
        if movie_id and user_id:
            movie_scores = {rec['movieId']: rec for rec in recommendations}
            
            # Combine scores for movies that appear in both methods
            final_recommendations = []
            processed_movies = set()
            
            for rec in recommendations:
                movie_id_rec = rec['movieId']
                if movie_id_rec not in processed_movies:
                    # Check if this movie appears in both methods
                    combined_score = rec['score']
                    method = rec['method']
                    
                    # Look for the same movie in other recommendations
                    for other_rec in recommendations:
                        if (other_rec['movieId'] == movie_id_rec and 
                            other_rec['method'] != rec['method']):
                            combined_score += other_rec['score']
                            method = 'hybrid'
                            break
                    
                    rec['score'] = combined_score
                    rec['method'] = method
                    final_recommendations.append(rec)
                    processed_movies.add(movie_id_rec)
            
            recommendations = final_recommendations
        
        # Sort by score and return top k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]
    
    def _get_popular_movies(self, top_k: int = 10) -> List[Dict]:
        """Get popular movies as fallback"""
        popular_movies = (
            self.data_processor.ratings_df
            .groupby('movieId')
            .agg({'rating': ['mean', 'count']})
            .reset_index()
        )
        popular_movies.columns = ['movieId', 'avg_rating', 'rating_count']
        
        # Filter movies with at least 5 ratings
        popular_movies = popular_movies[popular_movies['rating_count'] >= 5]
        popular_movies = popular_movies.sort_values('avg_rating', ascending=False)
        
        recommendations = []
        for _, movie in popular_movies.head(top_k).iterrows():
            movie_data = self.data_processor.movies_df[
                self.data_processor.movies_df['movieId'] == movie['movieId']
            ].iloc[0]
            
            recommendations.append({
                'movieId': int(movie['movieId']),
                'title': movie_data['title'],
                'genres': movie_data['genres'],
                'avg_rating': float(movie['avg_rating']),
                'method': 'popular'
            })
        
        return recommendations
    
    def search_movies(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search movies using text similarity"""
        if self.data_processor.use_bert and self.data_processor.sentence_model:
            # Use BERT embeddings
            query_embedding = self.data_processor.sentence_model.encode([query])
            import faiss
            faiss.normalize_L2(query_embedding)
            
            if self.data_processor.faiss_index:
                similarities, indices = self.data_processor.faiss_index.search(
                    query_embedding.astype('float32'), top_k
                )
                results = []
                for similarity, idx in zip(similarities[0], indices[0]):
                    movie_data = self.data_processor.movies_df.iloc[idx]
                    results.append({
                        'movieId': int(movie_data['movieId']),
                        'title': movie_data['title'],
                        'genres': movie_data['genres'],
                        'similarity': float(similarity)
                    })
                return results
        
        # Fallback to TF-IDF search
        if self.data_processor.tfidf_model:
            query_vector = self.data_processor.tfidf_model.transform([query])
            similarities = cosine_similarity(query_vector, self.data_processor.movie_embeddings)[0]
            
            # Get top k similar movies
            similar_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in similar_indices:
                movie_data = self.data_processor.movies_df.iloc[idx]
                results.append({
                    'movieId': int(movie_data['movieId']),
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'similarity': float(similarities[idx])
                })
            return results
        
        return []