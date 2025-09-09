import streamlit as st
import requests
import pandas as pd
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go

def check_api_connection(api_base_url: str) -> bool:
    """Check if the API is responding"""
    try:
        response = requests.get(f"{api_base_url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def fetch_movies(api_base_url: str, skip: int = 0, limit: int = 50) -> Optional[List[Dict]]:
    """Fetch movies from the API"""
    try:
        response = requests.get(
            f"{api_base_url}/movies",
            params={"skip": skip, "limit": limit},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["movies"]
        return None
    except requests.RequestException:
        return None

def get_recommendations(
    api_base_url: str,
    user_id: Optional[int] = None,
    movie_id: Optional[int] = None,
    top_k: int = 10,
    content_weight: float = 0.6,
    collaborative_weight: float = 0.4
) -> Optional[Dict]:
    """Get recommendations from the API"""
    try:
        request_data = {
            "user_id": user_id,
            "movie_id": movie_id,
            "top_k": top_k,
            "content_weight": content_weight,
            "collaborative_weight": collaborative_weight
        }
        
        response = requests.post(
            f"{api_base_url}/recommend",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None

def search_movies(api_base_url: str, query: str, top_k: int = 10) -> Optional[Dict]:
    """Search movies using the API"""
    try:
        response = requests.post(
            f"{api_base_url}/search",
            json={"query": query, "top_k": top_k},
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None

def display_movie_card(movie: Dict, index: int):
    """Display a movie card with information"""
    with st.expander(f"{index}. {movie['title']}", expanded=index <= 3):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**Genres:** {movie.get('genres', 'N/A')}")
            
            # Display different metrics based on what's available
            if 'score' in movie:
                st.write(f"**Score:** {movie['score']:.3f}")
            if 'similarity' in movie:
                st.write(f"**Similarity:** {movie['similarity']:.3f}")
            if 'predicted_rating' in movie:
                st.write(f"**Predicted Rating:** {movie['predicted_rating']:.2f}")
            if 'avg_rating' in movie:
                st.write(f"**Average Rating:** {movie['avg_rating']:.2f}")
            if 'method' in movie:
                st.write(f"**Method:** {movie['method']}")
        
        with col2:
            st.metric("Movie ID", movie['movieId'])

def create_score_visualization(recommendations: List[Dict], score_column: str = 'score'):
    """Create a bar chart visualization for recommendation scores"""
    if len(recommendations) <= 1:
        return None
    
    df = pd.DataFrame(recommendations)
    
    # Determine which score column to use
    available_columns = [col for col in [score_column, 'similarity', 'predicted_rating', 'avg_rating'] 
                        if col in df.columns]
    
    if not available_columns:
        return None
    
    score_col = available_columns[0]
    
    fig = px.bar(
        df,
        x='title',
        y=score_col,
        color='method' if 'method' in df.columns else None,
        title=f"Recommendation {score_col.replace('_', ' ').title()}s",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        xaxis_title="Movie Title",
        yaxis_title=score_col.replace('_', ' ').title()
    )
    
    return fig

def analyze_genres(movies_data: List[Dict]) -> Dict:
    """Analyze genre distribution from movies data"""
    if not movies_data:
        return {}
    
    all_genres = []
    for movie in movies_data:
        genres_str = movie.get('genres', '')
        if isinstance(genres_str, str) and genres_str:
            all_genres.extend(genres_str.split('|'))
    
    if not all_genres:
        return {}
    
    genre_counts = pd.Series(all_genres).value_counts()
    
    return {
        'genre_counts': genre_counts,
        'total_genres': len(genre_counts),
        'most_common': genre_counts.index[0] if len(genre_counts) > 0 else None
    }

def create_genre_pie_chart(genre_analysis: Dict):
    """Create a pie chart for genre distribution"""
    if not genre_analysis or 'genre_counts' not in genre_analysis:
        return None
    
    genre_counts = genre_analysis['genre_counts']
    
    fig = px.pie(
        values=genre_counts.values,
        names=genre_counts.index,
        title="Movie Genres Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig

def create_genre_bar_chart(genre_analysis: Dict):
    """Create a bar chart for genre distribution"""
    if not genre_analysis or 'genre_counts' not in genre_analysis:
        return None
    
    genre_counts = genre_analysis['genre_counts']
    
    fig = px.bar(
        x=genre_counts.index[:10],  # Top 10 genres
        y=genre_counts.values[:10],
        title="Top 10 Movie Genres",
        color=genre_counts.values[:10],
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title="Number of Movies",
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig

def display_error_message(error_type: str, details: str = None):
    """Display formatted error messages"""
    error_messages = {
        'api_connection': "❌ Could not connect to the recommendation API. Please make sure the backend is running.",
        'no_recommendations': "⚠️ No recommendations found. Try different parameters.",
        'no_search_results': "⚠️ No movies found matching your search.",
        'invalid_input': "⚠️ Please check your input parameters.",
        'server_error': "❌ Server error occurred. Please try again later."
    }
    
    message = error_messages.get(error_type, "❌ An unexpected error occurred.")
    
    if details:
        message += f"\n\nDetails: {details}"
    
    st.error(message)

def display_success_message(message_type: str, count: int = None):
    """Display formatted success messages"""
    if message_type == 'recommendations_found':
        st.success(f"✅ Found {count} recommendations!")
    elif message_type == 'search_results_found':
        st.success(f"✅ Found {count} movies matching your search!")
    elif message_type == 'data_loaded':
        st.success(f"✅ Successfully loaded {count} movies!")

def format_movie_info(movie: Dict) -> str:
    """Format movie information for display"""
    info_parts = []
    
    if 'title' in movie:
        info_parts.append(f"**{movie['title']}**")
    
    if 'genres' in movie:
        info_parts.append(f"Genres: {movie['genres']}")
    
    if 'overview' in movie:
        overview = movie['overview']
        if len(overview) > 200:
            overview = overview[:200] + "..."
        info_parts.append(f"Overview: {overview}")
    
    return "\n\n".join(info_parts)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_api_call(endpoint: str, params: Dict = None):
    """Cache API calls to reduce server load"""
    # This is a placeholder for caching API calls
    # In a real implementation, you might use this to cache frequent requests
    pass