from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.models.recommender import HybridRecommender

router = APIRouter()

# Global recommender instance
recommender = HybridRecommender()

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    movie_id: Optional[int] = None
    top_k: int = 10
    content_weight: float = 0.6
    collaborative_weight: float = 0.4

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

@router.on_event("startup")
async def startup_event():
    """Initialize the recommender system on startup"""
    recommender.initialize()

@router.get("/")
async def root():
    return {"message": "Movie Recommendation API", "status": "running"}

@router.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations"""
    try:
        recommendations = recommender.get_hybrid_recommendations(
            user_id=request.user_id,
            movie_id=request.movie_id,
            top_k=request.top_k,
            content_weight=request.content_weight,
            collaborative_weight=request.collaborative_weight
        )
        
        return {
            "recommendations": recommendations,
            "total": len(recommendations),
            "user_id": request.user_id,
            "movie_id": request.movie_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_movies(request: SearchRequest):
    """Search movies by text query"""
    try:
        results = recommender.search_movies(request.query, request.top_k)
        
        return {
            "results": results,
            "total": len(results),
            "query": request.query
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/movies/{movie_id}/similar")
async def get_similar_movies(
    movie_id: int,
    top_k: int = Query(10, ge=1, le=50)
):
    """Get movies similar to a specific movie"""
    try:
        recommendations = recommender.get_content_recommendations(movie_id, top_k)
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        return {
            "similar_movies": recommendations,
            "total": len(recommendations),
            "movie_id": movie_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/recommendations")
async def get_user_recommendations(
    user_id: int,
    top_k: int = Query(10, ge=1, le=50)
):
    """Get personalized recommendations for a user"""
    try:
        recommendations = recommender.get_collaborative_recommendations(user_id, top_k)
        
        return {
            "recommendations": recommendations,
            "total": len(recommendations),
            "user_id": user_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/movies")
async def get_movies(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """Get list of movies"""
    try:
        movies = recommender.data_processor.movies_df.iloc[skip:skip+limit]
        
        movies_list = []
        for _, movie in movies.iterrows():
            movies_list.append({
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres']
            })
        
        return {
            "movies": movies_list,
            "total": len(movies_list),
            "skip": skip,
            "limit": limit
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "recommender_initialized": recommender.data_processor.movies_df is not None
    }