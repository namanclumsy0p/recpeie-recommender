# 🎬 AI-Based Smart Movie Recommendation System

A sophisticated movie recommendation system that combines multiple machine learning approaches to provide accurate and personalized movie recommendations using BERT embeddings and collaborative filtering.

## 🌟 Features

- **Hybrid Recommendations**: Combines content-based filtering using BERT embeddings with collaborative filtering using SVD
- **Real-time Search**: Semantic search capabilities using sentence transformers
- **Fast Similarity Search**: Powered by FAISS for efficient vector similarity search
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **RESTful API**: Complete FastAPI backend with comprehensive endpoints
- **Scalable Architecture**: Containerized with Docker for easy deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Streamlit      │    │   FastAPI       │    │  ML Models      │
│  Frontend       │◄───┤   Backend       │◄───┤  & FAISS Index  │
│  (Port 8501)    │    │  (Port 8000)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components

1. **Content-Based Filtering**: Uses BERT embeddings to understand movie semantics
2. **Collaborative Filtering**: SVD matrix factorization for user preference learning
3. **Hybrid Scoring**: Weighted combination of both approaches
4. **FAISS Index**: Efficient similarity search for real-time recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/namanclumsy0p/recpeie-recommender.git
   cd recpeie-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run with Docker (Recommended)**
   ```bash
   docker-compose up --build
   ```

4. **Or run manually**
   
   Start the API server:
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   Start the Streamlit frontend:
   ```bash
   streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port 8501
   ```

### Access the Application

- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

## 📁 Project Structure

```
movie-recommender/
├── app/                        # FastAPI backend
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration settings
│   ├── models/                 # ML models and data processing
│   │   ├── __init__.py
│   │   ├── data_processor.py   # Data processing utilities
│   │   └── recommender.py      # Recommendation engine
│   └── api/                    # API routes
│       ├── __init__.py
│       └── routes.py           # API endpoints
├── streamlit_app/              # Frontend application
│   ├── __init__.py
│   ├── app.py                  # Main Streamlit app
│   └── utils.py                # UI utilities
├── data/                       # Data directory
│   ├── movies.csv              # Movie dataset (generated)
│   ├── ratings.csv             # User ratings (generated)
│   └── processed/              # Processed data
├── models/                     # Saved models and embeddings
│   ├── movie_embeddings.npy    # BERT embeddings (generated)
│   └── movie_index.faiss       # FAISS index (generated)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── setup.py                    # Package configuration
└── README.md                   # This file
```

## 🔧 Configuration

The system can be configured through `app/config.py`:

- **Model Settings**: BERT model selection, embedding dimensions
- **Recommendation Parameters**: Number of recommendations, similarity thresholds
- **API Settings**: Host, port, and CORS configuration
- **Data Paths**: Locations for data files and model storage

## 📊 API Endpoints

### Core Endpoints

- `POST /api/v1/recommend` - Get hybrid recommendations
- `POST /api/v1/search` - Search movies by text query
- `GET /api/v1/movies/{movie_id}/similar` - Get similar movies
- `GET /api/v1/users/{user_id}/recommendations` - Get user recommendations
- `GET /api/v1/movies` - List movies
- `GET /api/v1/health` - Health check

### Example API Usage

```python
import requests

# Get hybrid recommendations
response = requests.post("http://localhost:8000/api/v1/recommend", json={
    "user_id": 1,
    "movie_id": 5,
    "top_k": 10,
    "content_weight": 0.6,
    "collaborative_weight": 0.4
})

# Search movies
response = requests.post("http://localhost:8000/api/v1/search", json={
    "query": "action adventure superhero",
    "top_k": 10
})
```

## 🧠 How It Works

### 1. Data Processing
- Sample movie and rating data is generated if not provided
- Movies are processed to create combined feature text (title + genres + overview)
- BERT embeddings are computed for semantic understanding

### 2. Content-Based Filtering
- Uses `all-MiniLM-L6-v2` sentence transformer model
- Creates 384-dimensional embeddings for movies
- FAISS index enables fast cosine similarity search

### 3. Collaborative Filtering
- Uses Truncated SVD for matrix factorization
- Learns latent factors for users and movies
- Predicts ratings for unrated movies

### 4. Hybrid Approach
- Combines content and collaborative scores with configurable weights
- Handles cold start problems for new users/movies
- Provides fallback to popular movies when needed

## 🎨 Web Interface

The Streamlit frontend provides:

- **Home Page**: System overview and statistics
- **Get Recommendations**: Interactive recommendation interface
- **Search Movies**: Semantic movie search
- **Movie Analysis**: Genre distribution and insights
- **About**: Detailed system documentation

## 🧪 Development

### Running Tests
```bash
# Install development dependencies
pip install pytest pytest-asyncio httpx

# Run tests (to be implemented)
pytest tests/
```

### Code Structure
- Follow PEP 8 style guidelines
- Use type hints throughout the codebase
- Modular design with clear separation of concerns

## 📈 Performance

- **Embedding Generation**: ~2-3 seconds for 100 movies
- **Similarity Search**: Sub-second response with FAISS
- **API Response Time**: Typically < 500ms for recommendations
- **Memory Usage**: ~200MB for 1000 movies with embeddings

## 🐳 Docker Deployment

The application is containerized for easy deployment:

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in production mode
docker-compose -f docker-compose.yml up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentence Transformers**: For BERT embeddings
- **FAISS**: For efficient similarity search
- **FastAPI**: For the robust API framework
- **Streamlit**: For the interactive frontend
- **Scikit-learn**: For machine learning utilities

## 🚨 Troubleshooting

### Common Issues

1. **API not responding**: Ensure the FastAPI server is running on port 8000
2. **Frontend errors**: Check that the backend is accessible at `localhost:8000`
3. **Memory issues**: Reduce the number of movies or embedding dimensions
4. **Slow performance**: Ensure FAISS index is properly built and loaded

### Getting Help

- Check the API documentation at `/docs`
- Review logs in Docker containers
- Open an issue on GitHub for bugs or feature requests

---

**Built with ❤️ using Python, FastAPI, and Streamlit**