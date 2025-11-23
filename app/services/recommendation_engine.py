from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

class RecommendationEngine:
    def __init__(self):
        # Modelo para embeddings de texto
        try:
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except:
            self.model = None
        
        # Base de conocimiento de intervenciones
        self.interventions_db = self._load_interventions_db()
    
    def _load_interventions_db(self):
        """Carga la base de conocimiento de intervenciones"""
        return {
            "academic_support": [
                {
                    "type": "tutoring",
                    "title": "Tutorías personalizadas",
                    "description": "Sesiones uno a uno con tutor especializado",
                    "priority": "high",
                    "subjects": ["matemáticas", "ciencias", "lenguaje"]
                },
                {
                    "type": "workshop",
                    "title": "Taller de técnicas de estudio",
                    "description": "Aprende métodos efectivos de estudio",
                    "priority": "medium",
                    "audience": "general"
                }
            ],
            "emotional_support": [
                {
                    "type": "counseling",
                    "title": "Orientación psicológica",
                    "description": "Apoyo emocional y manejo de estrés",
                    "priority": "high",
                    "triggers": ["ansiedad", "frustración", "baja autoestima"]
                }
            ],
            "teaching_improvement": [
                {
                    "type": "feedback",
                    "title": "Retroalimentación específica al docente",
                    "description": "Sugerencias para mejorar la claridad en explicaciones",
                    "priority": "medium",
                    "context": "evaluaciones_bajas"
                }
            ]
        }
    
    def generate_recommendations(self, student_id: str, historical_data: List[Dict], current_performance: Dict):
        """Genera recomendaciones personalizadas"""
        recommendations = []
        
        # Análisis de desempeño actual
        if current_performance.get('average_grade', 0) < 70:
            recommendations.extend(self._get_academic_recommendations(current_performance))
        
        # Análisis de tendencias históricas
        if historical_data:
            trend_recommendations = self._analyze_trends(historical_data)
            recommendations.extend(trend_recommendations)
        
        # Recomendaciones basadas en asistencia
        if current_performance.get('attendance_rate', 0) < 75:
            recommendations.append({
                "type": "attendance_plan",
                "title": "Plan de mejora de asistencia",
                "description": "Estrategias para incrementar la participación en clase",
                "priority": "high",
                "actions": ["Registro diario", "Comunicación con familia", "Seguimiento semanal"]
            })
        
        # Eliminar duplicados y limitar a 5 recomendaciones
        unique_recommendations = []
        seen_titles = set()
        
        for rec in recommendations:
            if rec['title'] not in seen_titles:
                unique_recommendations.append(rec)
                seen_titles.add(rec['title'])
        
        return unique_recommendations[:5]
    
    def _get_academic_recommendations(self, performance: Dict):
        """Recomendaciones para bajo rendimiento académico"""
        recommendations = []
        
        # Tutoría intensiva si las calificaciones son muy bajas
        if performance.get('average_grade', 0) < 60:
            recommendations.append({
                "type": "intensive_tutoring",
                "title": "Tutoría académica intensiva",
                "description": "Sesiones frecuentes de reforzamiento",
                "priority": "high",
                "frequency": "3 veces por semana",
                "duration": "4 semanas"
            })
        
        # Talleres de habilidades específicas
        recommendations.append({
            "type": "skill_workshop",
            "title": "Taller de comprensión lectora",
            "description": "Mejora tu capacidad de análisis y comprensión",
            "priority": "medium",
            "schedule": "Sábados 9:00-11:00"
        })
        
        return recommendations
    
    def _analyze_trends(self, historical_data: List[Dict]):
        """Analiza tendencias en datos históricos"""
        if len(historical_data) < 2:
            return []
        
        # Calcular tendencia de calificaciones
        grades = [data.get('average_grade', 0) for data in historical_data]
        trend = self._calculate_trend(grades)
        
        recommendations = []
        
        if trend < -0.5:  # Tendencia negativa fuerte
            recommendations.append({
                "type": "trend_intervention",
                "title": "Intervención por tendencia negativa",
                "description": "Desempeño en declive requiere atención inmediata",
                "priority": "high",
                "actions": ["Análisis de causas", "Plan correctivo", "Seguimiento cercano"]
            })
        
        return recommendations
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calcula la tendencia de una serie de valores"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalizar la pendiente
        max_val = max(values) if max(values) > 0 else 1
        return slope / max_val