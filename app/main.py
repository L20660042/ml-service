from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from app.services.learning_analytics import LearningAnalytics
from app.services.sentiment_analysis import SentimentAnalyzer
from app.services.recommendation_engine import RecommendationEngine

app = FastAPI(
    title="Metricampus ML Service",
    description="Servicio de análisis de aprendizaje con IA",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicios
learning_analytics = LearningAnalytics()
sentiment_analyzer = SentimentAnalyzer()
recommendation_engine = RecommendationEngine()

class StudentData(BaseModel):
    grades: List[float]
    attendance: List[float]
    tutoring_sessions: int
    evaluation_scores: List[float]

class FeedbackData(BaseModel):
    text: str
    context: str = "general"

class RecommendationRequest(BaseModel):
    student_id: str
    historical_data: List[Dict[str, Any]]
    current_performance: Dict[str, float]

@app.get("/")
async def root():
    return {"message": "Metricampus ML Service", "status": "running"}

@app.post("/api/analyze/student-risk")
async def analyze_student_risk(student_data: StudentData):
    """Analiza el riesgo académico de un estudiante"""
    try:
        risk_level, confidence, factors = learning_analytics.predict_risk(
            student_data.grades,
            student_data.attendance,
            student_data.tutoring_sessions,
            student_data.evaluation_scores
        )
        
        return {
            "risk_level": risk_level,
            "confidence": confidence,
            "risk_factors": factors,
            "recommendations": learning_analytics.get_recommendations(risk_level, factors)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/feedback-sentiment")
async def analyze_feedback_sentiment(feedback: FeedbackData):
    """Analiza el sentimiento en feedback de estudiantes"""
    try:
        sentiment = sentiment_analyzer.analyze(feedback.text)
        return {
            "sentiment": sentiment.sentiment,
            "confidence": sentiment.confidence,
            "key_phrases": sentiment.key_phrases,
            "urgency_level": sentiment.urgency_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend/interventions")
async def recommend_interventions(request: RecommendationRequest):
    """Recomienda intervenciones personalizadas"""
    try:
        recommendations = recommendation_engine.generate_recommendations(
            request.student_id,
            request.historical_data,
            request.current_performance
        )
        return {
            "student_id": request.student_id,
            "recommendations": recommendations,
            "priority": "high" if any(r["priority"] == "high" for r in recommendations) else "medium"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "ml-analytics"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)