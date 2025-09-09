import pandas as pd
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import Dict, List, Tuple
from app.config import config

class DataProcessor:
    def __init__(self):
        self.sentence_model = None
        self.tfidf_model = None
        self.use_bert = False
        self.movies_df = None
        self.ratings_df = None
        self.movie_embeddings = None
        self.faiss_index = None
        self.movie_id_to_index = {}
        
        # Try to initialize BERT model, fallback to TF-IDF
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
                self.use_bert = True
                print("Using BERT embeddings for content analysis")
            except Exception as e:
                print(f"Could not load BERT model: {e}")
                print("Falling back to TF-IDF for content analysis")
                self.tfidf_model = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            print("SentenceTransformers not available, using TF-IDF for content analysis")
            self.tfidf_model = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def load_data(self) -> None:
        """Load movie and ratings data"""
        try:
            self.movies_df = pd.read_csv(config.MOVIES_FILE)
            self.ratings_df = pd.read_csv(config.RATINGS_FILE)
            print(f"Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")
        except FileNotFoundError:
            # Create sample data if files don't exist
            self._create_sample_data()
    
    def _create_sample_data(self) -> None:
        """Create sample movie and ratings data for demonstration"""
        # Sample movies data
        movies_data = {
            'movieId': range(1, 101),
            'title': [f'Movie {i}' for i in range(1, 101)],
            'genres': ['Action|Adventure', 'Comedy', 'Drama', 'Horror', 'Romance'] * 20,
            'overview': [f'This is an exciting movie {i} with great plot and characters.' for i in range(1, 101)]
        }
        self.movies_df = pd.DataFrame(movies_data)
        
        # Sample ratings data
        np.random.seed(42)
        ratings_data = []
        for user_id in range(1, 51):  # 50 users
            num_ratings = np.random.randint(10, 30)  # Each user rates 10-30 movies
            movie_ids = np.random.choice(range(1, 101), num_ratings, replace=False)
            for movie_id in movie_ids:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
                ratings_data.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': rating
                })
        
        self.ratings_df = pd.DataFrame(ratings_data)
        
        # Save sample data
        os.makedirs(config.DATA_DIR, exist_ok=True)
        self.movies_df.to_csv(config.MOVIES_FILE, index=False)
        self.ratings_df.to_csv(config.RATINGS_FILE, index=False)
        print("Created sample data")
    
    def process_movies(self) -> None:
        """Process movie data and create embeddings"""
        if self.movies_df is None:
            self.load_data()
        
        # Create movie features for embedding
        self.movies_df['combined_features'] = (
            self.movies_df['title'].fillna('') + ' ' + 
            self.movies_df['genres'].fillna('') + ' ' + 
            self.movies_df['overview'].fillna('')
        )
        
        # Create embeddings
        if self.use_bert and self.sentence_model:
            print("Creating BERT movie embeddings...")
            self.movie_embeddings = self.sentence_model.encode(
                self.movies_df['combined_features'].tolist(),
                show_progress_bar=True
            )
        else:
            print("Creating TF-IDF movie embeddings...")
            # Use TF-IDF as fallback
            tfidf_matrix = self.tfidf_model.fit_transform(self.movies_df['combined_features'])
            self.movie_embeddings = tfidf_matrix.toarray().astype('float32')
        
        # Create FAISS index if available, otherwise use regular similarity search
        if FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexFlatIP(self.movie_embeddings.shape[1])
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.movie_embeddings)
            self.faiss_index.add(self.movie_embeddings.astype('float32'))
            print("Created FAISS index for fast similarity search")
        else:
            print("FAISS not available, will use sklearn cosine similarity")
        
        # Create movie ID to index mapping
        self.movie_id_to_index = {
            movie_id: idx for idx, movie_id in enumerate(self.movies_df['movieId'])
        }
        
        # Save embeddings and index
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        np.save(config.EMBEDDINGS_FILE, self.movie_embeddings)
        if FAISS_AVAILABLE and self.faiss_index:
            faiss.write_index(self.faiss_index, config.FAISS_INDEX_FILE)
        
        print(f"Processed {len(self.movies_df)} movies and created embeddings")
    
    def load_embeddings(self) -> None:
        """Load pre-computed embeddings and FAISS index"""
        if os.path.exists(config.EMBEDDINGS_FILE):
            self.movie_embeddings = np.load(config.EMBEDDINGS_FILE)
            
            # Create movie ID to index mapping
            if self.movies_df is not None:
                self.movie_id_to_index = {
                    movie_id: idx for idx, movie_id in enumerate(self.movies_df['movieId'])
                }
                
                # Initialize TF-IDF model if not using BERT
                if not self.use_bert and self.tfidf_model is not None:
                    self.movies_df['combined_features'] = (
                        self.movies_df['title'].fillna('') + ' ' + 
                        self.movies_df['genres'].fillna('') + ' ' + 
                        self.movies_df['overview'].fillna('')
                    )
                    self.tfidf_model.fit(self.movies_df['combined_features'])
            
            if FAISS_AVAILABLE and os.path.exists(config.FAISS_INDEX_FILE):
                self.faiss_index = faiss.read_index(config.FAISS_INDEX_FILE)
                print("Loaded pre-computed embeddings and FAISS index")
            else:
                print("Loaded pre-computed embeddings (no FAISS index)")
        else:
            self.process_movies()
    
    def find_similar_movies(self, movie_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find similar movies using either FAISS or cosine similarity"""
        if self.faiss_index:
            # Use FAISS
            query_embedding = self.movie_embeddings[movie_idx:movie_idx+1]
            similarities, indices = self.faiss_index.search(
                query_embedding.astype('float32'), top_k + 1
            )
            # Convert to list of (index, similarity) tuples, excluding the query movie itself
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx != movie_idx:
                    results.append((idx, float(similarity)))
            return results[:top_k]
        else:
            # Use sklearn cosine similarity
            query_embedding = self.movie_embeddings[movie_idx:movie_idx+1]
            similarities = cosine_similarity(query_embedding, self.movie_embeddings)[0]
            # Get indices sorted by similarity (excluding the query movie itself)
            similar_indices = np.argsort(similarities)[::-1]
            results = []
            for idx in similar_indices:
                if idx != movie_idx and len(results) < top_k:
                    results.append((idx, float(similarities[idx])))
            return results
    
    def get_user_movie_matrix(self) -> pd.DataFrame:
        """Create user-item matrix for collaborative filtering"""
        if self.ratings_df is None:
            self.load_data()
        
        user_movie_matrix = self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        return user_movie_matrix