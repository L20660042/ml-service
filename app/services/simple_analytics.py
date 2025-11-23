import numpy as np
from typing import List, Dict, Any

class SimpleAnalytics:
    def __init__(self):
        pass
    
    def predict_risk(self, grades: List[float], attendance: List[float], 
                    tutoring_sessions: int, evaluation_scores: List[float]) -> Dict[str, Any]:
        """Predicción de riesgo simplificada sin ML pesado"""
        try:
            # Métricas básicas
            avg_grade = np.mean(grades) if grades else 0
            avg_attendance = np.mean(attendance) if attendance else 0
            avg_evaluation = np.mean(evaluation_scores) if evaluation_scores else 0
            
            # Lógica basada en reglas simples
            risk_score = 0
            
            if avg_grade < 60: risk_score += 2
            elif avg_grade < 70: risk_score += 1
            
            if avg_attendance < 70: risk_score += 2
            elif avg_attendance < 80: risk_score += 1
            
            if tutoring_sessions > 5: risk_score += 1
            if avg_evaluation < 2.5: risk_score += 1
            
            # Determinar nivel de riesgo
            if risk_score >= 4:
                risk_level = "alto"
                confidence = 0.85
                factors = ["calificaciones_bajas", "asistencia_baja", "múltiples_tutorías"]
            elif risk_score >= 2:
                risk_level = "medio" 
                confidence = 0.75
                factors = ["algunas_dificultades"]
            else:
                risk_level = "bajo"
                confidence = 0.9
                factors = []
            
            return {
                "risk_level": risk_level,
                "confidence": confidence,
                "risk_factors": factors,
                "recommendations": self._get_recommendations(risk_level, factors)
            }
            
        except Exception as e:
            # Fallback absoluto
            return {
                "risk_level": "bajo",
                "confidence": 0.5,
                "risk_factors": ["error_analisis"],
                "recommendations": ["Revisar datos del estudiante"]
            }
    
    def _get_recommendations(self, risk_level: str, factors: List[str]) -> List[str]:
        """Recomendaciones basadas en nivel de riesgo"""
        recommendations = {
            "bajo": [
                "Continuar con el buen desempeño actual",
                "Mantener participación activa en clase"
            ],
            "medio": [
                "Reforzar temas con dificultad identificada",
                "Incrementar horas de estudio en áreas débiles",
                "Participar en tutorías preventivas"
            ],
            "alto": [
                "Plan de intervención académica inmediata",
                "Tutorías intensivas 3 veces por semana", 
                "Reunión con coordinador académico",
                "Acompañamiento personalizado continuo",
                "Evaluación psicopedagógica"
            ]
        }
        
        return recommendations.get(risk_level, ["Seguimiento general"])
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Análisis de sentimiento simplificado"""
        if not text:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "key_phrases": [],
                "urgency_level": "low"
            }
        
        text_lower = text.lower()
        
        # Palabras positivas
        positive_words = ['excelente', 'bueno', 'genial', 'perfecto', 'claro', 'entendí', 'gracias']
        # Palabras negativas  
        negative_words = ['problema', 'difícil', 'confuso', 'malo', 'horrible', 'terrible', 'no entiendo']
        # Palabras urgentes
        urgent_words = ['urgente', 'emergencia', 'ayuda', 'desesperado', 'frustrado']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        urgent_count = sum(1 for word in urgent_words if word in text_lower)
        
        # Determinar sentimiento
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(positive_count / len(positive_words) * 0.8 + 0.2, 0.95)
        elif negative_count > positive_count:
            sentiment = "negative" 
            confidence = min(negative_count / len(negative_words) * 0.8 + 0.2, 0.95)
        else:
            sentiment = "neutral"
            confidence = 0.7
        
        # Determinar urgencia
        if urgent_count > 0:
            urgency = "high"
        elif negative_count > 2:
            urgency = "medium" 
        else:
            urgency = "low"
        
        # Frases clave simples (primeras 100 caracteres)
        key_phrases = [text[:100] + "..." if len(text) > 100 else text]
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "key_phrases": key_phrases,
            "urgency_level": urgency
        }