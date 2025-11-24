from textblob import TextBlob
from transformers import pipeline
import nltk
from dataclasses import dataclass
from typing import List, Optional
import re

# Descargar datos de NLTK (solo primera vez)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

@dataclass
class SentimentResult:
    sentiment: str  # positive, negative, neutral
    confidence: float
    key_phrases: List[str]
    urgency_level: str  # low, medium, high
    emotions: List[str]  # alegría, tristeza, enojo, etc.

class SentimentAnalyzer:
    def __init__(self):
        self.spanish_stopwords = set(stopwords.words('spanish'))
        
        # Intentar cargar modelo transformer más avanzado
        try:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            self.use_transformer = True
        except Exception as e:
            print(f"Transformer model not available, using TextBlob: {e}")
            self.analyzer = None
            self.use_transformer = False
    
    def analyze(self, text: str, context: str = "general") -> SentimentResult:
        """Analiza sentimiento del texto con múltiples métodos"""
        if not text or len(text.strip()) < 3:
            return SentimentResult("neutral", 0.0, [], "low", [])
        
        # Limpiar texto
        cleaned_text = self._clean_text(text)
        
        # Análisis con múltiples métodos
        transformer_result = self._analyze_with_transformer(cleaned_text) if self.use_transformer else None
        textblob_result = self._analyze_with_textblob(cleaned_text)
        rule_based_result = self._analyze_with_rules(cleaned_text, context)
        
        # Combinar resultados
        final_sentiment = self._combine_analyses(
            transformer_result, textblob_result, rule_based_result
        )
        
        # Extraer frases clave
        key_phrases = self._extract_key_phrases(cleaned_text, context)
        
        # Detectar emociones específicas
        emotions = self._detect_emotions(cleaned_text)
        
        # Determinar urgencia
        urgency = self._determine_urgency(cleaned_text, final_sentiment.sentiment, emotions)
        
        return SentimentResult(
            sentiment=final_sentiment.sentiment,
            confidence=final_sentiment.confidence,
            key_phrases=key_phrases,
            urgency_level=urgency,
            emotions=emotions
        )
    
    def _clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto"""
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover caracteres especiales pero mantener acentos españoles
        text = re.sub(r'[^\w\sáéíóúñü]', ' ', text)
        
        # Remover espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _analyze_with_transformer(self, text: str) -> Optional[SentimentResult]:
        """Análisis usando modelo transformer (más preciso)"""
        try:
            if len(text) < 10:  # Texto muy corto para transformer
                return None
                
            result = self.analyzer(text[:512])[0]  # Limitar longitud
            label = result['label']
            score = result['score']
            
            # Mapear etiquetas del modelo a nuestro formato
            sentiment_map = {
                '1 star': 'negative', '2 stars': 'negative',
                '3 stars': 'neutral',
                '4 stars': 'positive', '5 stars': 'positive'
            }
            
            sentiment = sentiment_map.get(label, 'neutral')
            
            return SentimentResult(
                sentiment=sentiment,
                confidence=score,
                key_phrases=[],
                urgency_level="low",
                emotions=[]
            )
        except Exception as e:
            print(f"Transformer analysis failed: {e}")
            return None
    
    def _analyze_with_textblob(self, text: str) -> SentimentResult:
        """Análisis usando TextBlob (fallback confiable)"""
        try:
            # TextBlob funciona mejor con inglés, pero podemos adaptar
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determinar sentimiento base
            if polarity > 0.1:
                sentiment = "positive"
                confidence = min(abs(polarity) + 0.3, 0.9)
            elif polarity < -0.1:
                sentiment = "negative"
                confidence = min(abs(polarity) + 0.3, 0.9)
            else:
                sentiment = "neutral"
                confidence = 0.7
            
            # Ajustar confianza basado en subjetividad
            if subjectivity < 0.3:  # Texto muy objetivo
                confidence *= 0.8
            else:  # Texto subjetivo (mejor para análisis de sentimiento)
                confidence *= 1.1
                confidence = min(confidence, 0.95)
            
            return SentimentResult(
                sentiment=sentiment,
                confidence=round(confidence, 2),
                key_phrases=[],
                urgency_level="low",
                emotions=[]
            )
        except Exception as e:
            print(f"TextBlob analysis failed: {e}")
            return SentimentResult("neutral", 0.5, [], "low", [])
    
    def _analyze_with_rules(self, text: str, context: str) -> SentimentResult:
        """Análisis basado en reglas y diccionarios en español"""
        # Diccionarios de palabras en español
        positive_words_es = {
            'excelente', 'bueno', 'buena', 'genial', 'perfecto', 'perfecta',
            'claro', 'clara', 'entendí', 'comprendí', 'fácil', 'sencillo',
            'útil', 'práctico', 'interesante', 'motivador', 'agradable',
            'amable', 'paciente', 'explica bien', 'se preocupa', 'ayuda',
            'gracias', 'felicidades', 'apoyo', 'comprensivo'
        }
        
        negative_words_es = {
            'problema', 'problemas', 'difícil', 'complicado', 'confuso',
            'confusa', 'mal', 'malo', 'mala', 'horrible', 'terrible',
            'pésimo', 'aburrido', 'lento', 'rápido', 'no entiendo',
            'no comprendo', 'no explica', 'no ayuda', 'frustrante',
            'desesperante', 'injusto', 'injusta', 'molesto', 'molesta',
            'enojado', 'enojada', 'preocupado', 'preocupada'
        }
        
        strong_negative_es = {
            'odio', 'detesto', 'horrible', 'terrible', 'pésimo', 'asco',
            'inútil', 'desastre', 'catástrofe', 'fracaso', 'burla'
        }
        
        # Contar ocurrencias
        words = set(text.split())
        positive_count = len(words.intersection(positive_words_es))
        negative_count = len(words.intersection(negative_words_es))
        strong_negative_count = len(words.intersection(strong_negative_es))
        
        # Determinar sentimiento basado en conteos
        total_relevant = positive_count + negative_count
        
        if total_relevant == 0:
            return SentimentResult("neutral", 0.6, [], "low", [])
        
        positive_ratio = positive_count / total_relevant
        negative_ratio = negative_count / total_relevant
        
        if strong_negative_count > 0:
            sentiment = "negative"
            confidence = 0.9
        elif positive_ratio > 0.6:
            sentiment = "positive"
            confidence = min(0.7 + (positive_ratio - 0.6) * 0.5, 0.9)
        elif negative_ratio > 0.6:
            sentiment = "negative"
            confidence = min(0.7 + (negative_ratio - 0.6) * 0.5, 0.9)
        else:
            sentiment = "neutral"
            confidence = 0.7
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=round(confidence, 2),
            key_phrases=[],
            urgency_level="low",
            emotions=[]
        )
    
    def _combine_analyses(self, transformer_result: Optional[SentimentResult], 
                         textblob_result: SentimentResult, 
                         rule_based_result: SentimentResult) -> SentimentResult:
        """Combina resultados de múltiples métodos de análisis"""
        results = []
        weights = []
        
        # Priorizar transformer si está disponible y confiable
        if transformer_result and transformer_result.confidence > 0.7:
            results.append(transformer_result)
            weights.append(0.6)
        
        # Incluir textblob
        results.append(textblob_result)
        weights.append(0.25)
        
        # Incluir análisis por reglas
        results.append(rule_based_result)
        weights.append(0.15)
        
        # Normalizar pesos
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calcular sentimiento ponderado
        sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}
        total_confidence = 0
        
        for result, weight in zip(results, weights):
            sentiment_scores[result.sentiment] += weight * result.confidence
            total_confidence += result.confidence * weight
        
        # Determinar sentimiento final
        final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        final_confidence = sentiment_scores[final_sentiment]
        
        return SentimentResult(
            sentiment=final_sentiment,
            confidence=round(final_confidence, 2),
            key_phrases=[],
            urgency_level="low",
            emotions=[]
        )
    
    def _extract_key_phrases(self, text: str, context: str) -> List[str]:
        """Extrae frases clave relevantes del contexto educativo"""
        sentences = sent_tokenize(text)
        key_phrases = []
        
        # Patrones específicos por contexto
        context_patterns = {
            "evaluation": [
                r'.*explica.*', r'.*clase.*', r'.*material.*', 
                r'.*evaluaci[óo]n.*', r'.*calificaci[óo]n.*'
            ],
            "tutoring": [
                r'.*problema.*', r'.*dificultad.*', r'.*entender.*',
                r'.*ayuda.*', r'.*apoyo.*'
            ],
            "general": [
                r'.*no\s+.*', r'.*me\s+gusta.*', r'.*deber[ií]a.*',
                r'.*ser[ií]a.*', r'.*podr[ií]a.*'
            ]
        }
        
        patterns = context_patterns.get(context, context_patterns["general"])
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Verificar si contiene palabras de alta importancia
            high_importance_words = ['no entiendo', 'no puedo', 'urgente', 'problema']
            if any(word in sentence_lower for word in high_importance_words):
                key_phrases.append(self._trim_phrase(sentence))
                continue
            
            # Verificar patrones específicos del contexto
            for pattern in patterns:
                if re.search(pattern, sentence_lower, re.IGNORECASE):
                    key_phrases.append(self._trim_phrase(sentence))
                    break
            
            # Limitar número de frases
            if len(key_phrases) >= 3:
                break
        
        return key_phrases[:3]  # Máximo 3 frases
    
    def _trim_phrase(self, phrase: str, max_length: int = 100) -> str:
        """Recorta la frase a longitud máxima"""
        if len(phrase) <= max_length:
            return phrase
        return phrase[:max_length].rsplit(' ', 1)[0] + "..."
    
    def _detect_emotions(self, text: str) -> List[str]:
        """Detecta emociones específicas en el texto"""
        emotions = []
        text_lower = text.lower()
        
        # Diccionario de emociones y sus indicadores
        emotion_indicators = {
            "frustración": ['frustrado', 'frustrada', 'no puedo', 'no sirve', 'difícil'],
            "confusión": ['confuso', 'confusa', 'no entiendo', 'no comprendo', 'perdido'],
            "enojo": ['enojado', 'enojada', 'molesto', 'molesta', 'ira', 'rabia'],
            "preocupación": ['preocupado', 'preocupada', 'nervioso', 'nerviosa', 'ansiedad'],
            "alegría": ['feliz', 'contento', 'contenta', 'alegre', 'emocionado'],
            "satisfacción": ['satisfecho', 'satisfecha', 'conforme', 'agradecido']
        }
        
        for emotion, indicators in emotion_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                emotions.append(emotion)
        
        return emotions[:2]  # Máximo 2 emociones principales
    
    def _determine_urgency(self, text: str, sentiment: str, emotions: List[str]) -> str:
        """Determina el nivel de urgencia basado en múltiples factores"""
        text_lower = text.lower()
        urgency_score = 0
        
        # Palabras que indican alta urgencia
        high_urgency_indicators = [
            'urgente', 'emergencia', 'desesperado', 'desesperada',
            'no aguanto', 'no puedo más', 'ayuda inmediata'
        ]
        
        # Palabras que indican urgencia media
        medium_urgency_indicators = [
            'problema', 'dificultad', 'confuso', 'ayuda',
            'no puedo', 'no sé', 'complicado'
        ]
        
        # Factores de puntuación
        if any(word in text_lower for word in high_urgency_indicators):
            urgency_score += 3
        
        if any(word in text_lower for word in medium_urgency_indicators):
            urgency_score += 2
        
        if sentiment == "negative":
            urgency_score += 1
        
        if "frustración" in emotions or "enojo" in emotions:
            urgency_score += 2
        
        if "preocupación" in emotions:
            urgency_score += 1
        
        # Determinar nivel final
        if urgency_score >= 4:
            return "high"
        elif urgency_score >= 2:
            return "medium"
        else:
            return "low"

# Ejemplo de uso
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Ejemplos de prueba
    test_texts = [
        "El profesor explica muy bien pero los ejercicios son demasiado difíciles",
        "No entiendo nada de la clase, necesito ayuda urgente",
        "Excelente metodología, me encanta como enseña",
        "La clase fue normal, ni buena ni mala"
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text, "evaluation")
        print(f"Texto: {text}")
        print(f"Sentimiento: {result.sentiment} (confianza: {result.confidence})")
        print(f"Urgencia: {result.urgency_level}")
        print(f"Emociones: {result.emotions}")
        print(f"Frases clave: {result.key_phrases}")
        print("-" * 50)