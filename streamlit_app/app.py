import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import json

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🎬 AI-Based Smart Movie Recommendation System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Get Recommendations", "Search Movies", "Movie Analysis", "About"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Get Recommendations":
        show_recommendations_page()
    elif page == "Search Movies":
        show_search_page()
    elif page == "Movie Analysis":
        show_analysis_page()
    elif page == "About":
        show_about_page()

def show_home_page():
    st.header("Welcome to the Movie Recommendation System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Features")
        st.markdown("""
        - **Hybrid Recommendations**: Combines content-based and collaborative filtering
        - **BERT Embeddings**: Uses advanced NLP for understanding movie content
        - **Real-time Search**: Find movies instantly with semantic search
        - **Personalized**: Get recommendations tailored to your preferences
        - **Fast & Scalable**: Powered by FAISS for efficient similarity search
        """)
    
    with col2:
        st.subheader("🚀 How it Works")
        st.markdown("""
        1. **Content-Based**: Analyzes movie descriptions, genres, and metadata
        2. **Collaborative Filtering**: Learns from user rating patterns
        3. **BERT Embeddings**: Understands semantic meaning of movie content
        4. **FAISS Index**: Enables fast similarity search across thousands of movies
        5. **Hybrid Scoring**: Combines multiple recommendation strategies
        """)
    
    # Quick stats
    try:
        response = requests.get(f"{API_BASE_URL}/movies")
        if response.status_code == 200:
            data = response.json()
            total_movies = len(data["movies"])
            
            st.subheader("📊 System Stats")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Movies", total_movies)
            
            with col2:
                st.metric("Recommendation Methods", "3")
            
            with col3:
                st.metric("ML Models", "2")
    
    except requests.RequestException:
        st.warning("⚠️ API is not responding. Please make sure the backend is running.")

def show_recommendations_page():
    st.header("🎯 Get Movie Recommendations")
    
    # Input form
    with st.form("recommendation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.number_input(
                "User ID (for personalized recommendations)",
                min_value=0,
                max_value=1000,
                value=0,
                help="Enter 0 for no user-specific recommendations"
            )
            
            movie_id = st.number_input(
                "Movie ID (for similar movie recommendations)",
                min_value=0,
                max_value=1000,
                value=0,
                help="Enter 0 for no content-based recommendations"
            )
        
        with col2:
            top_k = st.slider("Number of recommendations", 5, 20, 10)
            
            st.subheader("Algorithm Weights")
            content_weight = st.slider("Content-based weight", 0.0, 1.0, 0.6, 0.1)
            collaborative_weight = st.slider("Collaborative filtering weight", 0.0, 1.0, 0.4, 0.1)
        
        submitted = st.form_submit_button("Get Recommendations")
    
    if submitted:
        if user_id == 0 and movie_id == 0:
            st.warning("Please enter either a User ID or Movie ID (or both)")
            return
        
        # Prepare request
        request_data = {
            "user_id": user_id if user_id > 0 else None,
            "movie_id": movie_id if movie_id > 0 else None,
            "top_k": top_k,
            "content_weight": content_weight,
            "collaborative_weight": collaborative_weight
        }
        
        try:
            with st.spinner("Getting recommendations..."):
                response = requests.post(f"{API_BASE_URL}/recommend", json=request_data)
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data["recommendations"]
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommendations!")
                    
                    # Display recommendations
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"{i}. {rec['title']}", expanded=i <= 3):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write(f"**Genres:** {rec['genres']}")
                                st.write(f"**Method:** {rec['method']}")
                                
                                if 'score' in rec:
                                    st.write(f"**Score:** {rec['score']:.3f}")
                                if 'similarity' in rec:
                                    st.write(f"**Similarity:** {rec['similarity']:.3f}")
                                if 'predicted_rating' in rec:
                                    st.write(f"**Predicted Rating:** {rec['predicted_rating']:.2f}")
                            
                            with col2:
                                st.metric("Movie ID", rec['movieId'])
                    
                    # Visualization
                    if len(recommendations) > 1:
                        st.subheader("📈 Recommendation Scores")
                        
                        df = pd.DataFrame(recommendations)
                        score_col = 'score' if 'score' in df.columns else 'similarity'
                        
                        if score_col in df.columns:
                            fig = px.bar(
                                df,
                                x='title',
                                y=score_col,
                                color='method',
                                title=f"Recommendation {score_col.title()}s"
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("No recommendations found. Try different parameters.")
            
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        
        except requests.RequestException:
            st.error("❌ Could not connect to the recommendation API")

def show_search_page():
    st.header("🔍 Search Movies")
    
    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Search for movies",
                placeholder="Enter movie title, genre, or description..."
            )
        
        with col2:
            top_k = st.number_input("Max results", 5, 20, 10)
        
        submitted = st.form_submit_button("Search")
    
    if submitted and query:
        try:
            with st.spinner("Searching movies..."):
                response = requests.post(
                    f"{API_BASE_URL}/search",
                    json={"query": query, "top_k": top_k}
                )
            
            if response.status_code == 200:
                data = response.json()
                results = data["results"]
                
                if results:
                    st.success(f"Found {len(results)} movies matching your search!")
                    
                    # Display results
                    for i, movie in enumerate(results, 1):
                        with st.expander(f"{i}. {movie['title']}", expanded=i <= 5):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write(f"**Genres:** {movie['genres']}")
                                st.write(f"**Similarity:** {movie['similarity']:.3f}")
                            
                            with col2:
                                st.metric("Movie ID", movie['movieId'])
                    
                    # Visualization
                    if len(results) > 1:
                        st.subheader("📊 Search Results Similarity")
                        
                        df = pd.DataFrame(results)
                        fig = px.bar(
                            df,
                            x='title',
                            y='similarity',
                            title="Movie Similarity Scores",
                            color='similarity',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("No movies found matching your search.")
            
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        
        except requests.RequestException:
            st.error("❌ Could not connect to the search API")

def show_analysis_page():
    st.header("📊 Movie Analysis")
    
    try:
        # Get sample movies for analysis
        response = requests.get(f"{API_BASE_URL}/movies", params={"limit": 100})
        if response.status_code == 200:
            data = response.json()
            movies_data = data["movies"]
            
            if movies_data:
                df = pd.DataFrame(movies_data)
                
                st.subheader("Genre Distribution")
                # Extract and analyze genres
                all_genres = []
                for genres_str in df['genres']:
                    if isinstance(genres_str, str):
                        all_genres.extend(genres_str.split('|'))
                
                genre_counts = pd.Series(all_genres).value_counts()
                
                fig = px.pie(
                    values=genre_counts.values,
                    names=genre_counts.index,
                    title="Movie Genres Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Movies by Genre")
                fig_bar = px.bar(
                    x=genre_counts.index,
                    y=genre_counts.values,
                    title="Number of Movies by Genre"
                )
                fig_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                st.subheader("Sample Movies")
                st.dataframe(df.head(20), use_container_width=True)
            
            else:
                st.warning("No movie data available for analysis.")
        
        else:
            st.error("Could not fetch movie data for analysis.")
    
    except requests.RequestException:
        st.error("❌ Could not connect to the API for analysis data")

def show_about_page():
    st.header("📖 About This System")
    
    st.markdown("""
    ## AI-Based Smart Movie Recommendation System
    
    This is a sophisticated movie recommendation system that combines multiple machine learning approaches to provide accurate and personalized movie recommendations.
    
    ### 🧠 Technology Stack
    
    **Backend:**
    - **FastAPI**: Modern, fast web framework for building APIs
    - **BERT Embeddings**: Using Sentence Transformers for semantic understanding
    - **FAISS**: Facebook's library for efficient similarity search
    - **Scikit-learn**: For collaborative filtering using SVD
    - **Pandas & NumPy**: Data processing and manipulation
    
    **Frontend:**
    - **Streamlit**: Interactive web application framework
    - **Plotly**: Interactive visualizations
    - **Requests**: API communication
    
    ### 🎯 Recommendation Methods
    
    1. **Content-Based Filtering**
       - Uses BERT embeddings to understand movie content
       - Analyzes titles, genres, and descriptions
       - Finds movies with similar semantic content
    
    2. **Collaborative Filtering**
       - Uses Singular Value Decomposition (SVD)
       - Learns from user rating patterns
       - Predicts ratings for unseen movies
    
    3. **Hybrid Approach**
       - Combines content and collaborative methods
       - Weighted scoring system
       - Handles cold start problems
    
    ### 🚀 Features
    
    - **Real-time Recommendations**: Get instant suggestions
    - **Semantic Search**: Find movies using natural language
    - **Personalized Results**: Tailored to user preferences  
    - **Scalable Architecture**: Handles large movie catalogs
    - **Interactive UI**: Easy-to-use web interface
    
    ### 📊 Performance
    
    - **Fast Search**: FAISS enables sub-second similarity search
    - **Memory Efficient**: Optimized embeddings storage
    - **Scalable**: Can handle thousands of movies and users
    
    ### 🔧 Configuration
    
    The system is highly configurable with adjustable weights for different recommendation methods, customizable similarity thresholds, and flexible API parameters.
    """)
    
    st.subheader("🏗️ System Architecture")
    
    # Create a simple architecture diagram using plotly
    fig = go.Figure()
    
    # Add rectangles for different components
    fig.add_shape(type="rect", x0=0, y0=0, x1=2, y1=1, 
                  fillcolor="lightblue", line=dict(color="blue"))
    fig.add_annotation(x=1, y=0.5, text="Streamlit<br>Frontend", showarrow=False)
    
    fig.add_shape(type="rect", x0=3, y0=0, x1=5, y1=1,
                  fillcolor="lightgreen", line=dict(color="green"))
    fig.add_annotation(x=4, y=0.5, text="FastAPI<br>Backend", showarrow=False)
    
    fig.add_shape(type="rect", x0=6, y0=0, x1=8, y1=1,
                  fillcolor="lightyellow", line=dict(color="orange"))
    fig.add_annotation(x=7, y=0.5, text="ML Models<br>& FAISS", showarrow=False)
    
    # Add arrows
    fig.add_annotation(x=2.5, y=0.5, ax=2, ay=0.5, 
                       arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black")
    fig.add_annotation(x=5.5, y=0.5, ax=5, ay=0.5,
                       arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black")
    
    fig.update_layout(
        title="System Architecture",
        xaxis=dict(range=[-0.5, 8.5], showticklabels=False),
        yaxis=dict(range=[-0.5, 1.5], showticklabels=False),
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()