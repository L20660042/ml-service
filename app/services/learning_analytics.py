import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class LearningAnalytics:
    def __init__(self):
        self.model = self._load_or_train_model()
        self.scaler = StandardScaler()
        
    def _load_or_train_model(self):
        """Carga o entrena un modelo simple para riesgo académico"""
        model_path = "app/models/risk_model.joblib"
        
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            # Modelo simple basado en datos sintéticos
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Datos de ejemplo para entrenamiento inicial
            # [promedio_calificaciones, asistencia, sesiones_tutoria, evaluaciones_docente]
            X_train = np.array([
                [85, 90, 2, 4.5],
                [45, 60, 5, 2.0],
                [70, 80, 1, 3.5],
                [35, 50, 8, 1.5],
                [90, 95, 0, 4.8],
                [60, 70, 3, 3.0],
                [25, 40, 10, 1.0],
                [80, 85, 1, 4.2]
            ])
            
            # 0: bajo, 1: medio, 2: alto
            y_train = np.array([0, 2, 1, 2, 0, 1, 2, 0])
            
            model.fit(X_train, y_train)
            
            # Guardar modelo
            os.makedirs("app/models", exist_ok=True)
            joblib.dump(model, model_path)
            
            return model
    
    def predict_risk(self, grades, attendance, tutoring_sessions, evaluation_scores):
        """Predice el nivel de riesgo académico"""
        try:
            # Calcular métricas
            avg_grade = np.mean(grades) if grades else 0
            avg_attendance = np.mean(attendance) if attendance else 0
            avg_evaluation = np.mean(evaluation_scores) if evaluation_scores else 0
            
            # Preparar features
            features = np.array([[avg_grade, avg_attendance, tutoring_sessions, avg_evaluation]])
            
            # Escalar features
            features_scaled = self.scaler.fit_transform(features)
            
            # Predecir
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            risk_levels = {0: "bajo", 1: "medio", 2: "alto"}
            confidence = float(np.max(probabilities))
            
            # Identificar factores de riesgo
            factors = self._identify_risk_factors(avg_grade, avg_attendance, tutoring_sessions, avg_evaluation)
            
            return risk_levels[prediction], confidence, factors
            
        except Exception as e:
            # Fallback a lógica basada en reglas
            return self._rule_based_risk(grades, attendance, tutoring_sessions, evaluation_scores)
    
    def _identify_risk_factors(self, avg_grade, attendance, tutoring_sessions, evaluation):
        factors = []
        
        if avg_grade < 70:
            factors.append("calificaciones_bajas")
        if attendance < 75:
            factors.append("asistencia_baja")
        if tutoring_sessions > 5:
            factors.append("multiple_tutorias")
        if evaluation < 2.5:
            factors.append("evaluacion_docente_baja")
            
        return factors
    
    def _rule_based_risk(self, grades, attendance, tutoring_sessions, evaluation_scores):
        avg_grade = np.mean(grades) if grades else 0
        avg_attendance = np.mean(attendance) if attendance else 0
        avg_evaluation = np.mean(evaluation_scores) if evaluation_scores else 0
        
        risk_score = 0
        
        if avg_grade < 60: risk_score += 2
        elif avg_grade < 70: risk_score += 1
        
        if avg_attendance < 70: risk_score += 2
        elif avg_attendance < 80: risk_score += 1
        
        if tutoring_sessions > 5: risk_score += 1
        
        if avg_evaluation < 2.0: risk_score += 1
        
        if risk_score >= 4: return "alto", 0.8, ["multiple_factores"]
        elif risk_score >= 2: return "medio", 0.7, ["algunos_factores"]
        else: return "bajo", 0.6, []
    
    def get_recommendations(self, risk_level, factors):
        recommendations = {
            "bajo": [
                "Continuar con el buen desempeño actual",
                "Mantener participación en clase"
            ],
            "medio": [
                "Reforzar temas con dificultad",
                "Incrementar horas de estudio",
                "Asistir a tutorías preventivas"
            ],
            "alto": [
                "Plan de intervención inmediata",
                "Tutorías intensivas",
                "Reunión con coordinador académico",
                "Acompañamiento personalizado"
            ]
        }
        
        base_recommendations = recommendations.get(risk_level, [])
        
        # Recomendaciones específicas por factor
        factor_recommendations = {
            "calificaciones_bajas": ["Talleres de reforzamiento", "Ejercicios adicionales"],
            "asistencia_baja": ["Plan de asistencia", "Seguimiento de presencia"],
            "multiple_tutorias": ["Análisis de causas raíz", "Estrategias de aprendizaje"],
            "evaluacion_docente_baja": ["Retroalimentación específica", "Rubricas claras"]
        }
        
        for factor in factors:
            if factor in factor_recommendations:
                base_recommendations.extend(factor_recommendations[factor])
        
        return list(set(base_recommendations))  # Remover duplicados