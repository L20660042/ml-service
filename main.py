from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from app.services.simple_analytics import SimpleAnalytics

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

# Inicializar servicio
analytics = SimpleAnalytics()

class StudentData(BaseModel):
    grades: List[float]
    attendance: List[float]
    tutoring_sessions: int
    evaluation_scores: List[float]

class FeedbackData(BaseModel):
    text: str
    context: str = "general"

@app.get("/")
async def root():
    return {"message": "Metricampus ML Service", "status": "running"}

@app.post("/api/analyze/student-risk")
async def analyze_student_risk(student_data: StudentData):
    """Analiza el riesgo académico de un estudiante"""
    try:
        result = analytics.predict_risk(
            student_data.grades,
            student_data.attendance,
            student_data.tutoring_sessions,
            student_data.evaluation_scores
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/feedback-sentiment")
async def analyze_feedback_sentiment(feedback: FeedbackData):
    """Analiza el sentimiento en feedback de estudiantes"""
    try:
        result = analytics.analyze_sentiment(feedback.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "ml-analytics"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)