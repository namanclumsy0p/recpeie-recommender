import os
from typing import List

class Config:
    # Data paths
    DATA_DIR = "data"
    MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")
    RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    
    # Model paths
    MODELS_DIR = "models"
    EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "movie_embeddings.npy")
    FAISS_INDEX_FILE = os.path.join(MODELS_DIR, "movie_index.faiss")
    
    # Model settings
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Recommendation settings
    TOP_K_RECOMMENDATIONS = 10
    SIMILARITY_THRESHOLD = 0.5
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Streamlit settings
    STREAMLIT_HOST = "0.0.0.0"
    STREAMLIT_PORT = 8501

config = Config()