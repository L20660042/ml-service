from textblob import TextBlob
from transformers import pipeline
import nltk
from dataclasses import dataclass
from typing import List

# Descargar datos de NLTK (solo primera vez)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class SentimentResult:
    sentiment: str  # positive, negative, neutral
    confidence: float
    key_phrases: List[str]
    urgency_level: str  # low, medium, high

class SentimentAnalyzer:
    def __init__(self):
        # Usar pipeline de transformers para análisis más preciso
        try:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
            )
        except:
            self.analyzer = None
    
    def analyze(self, text: str) -> SentimentResult:
        """Analiza sentimiento del texto"""
        if not text or len(text.strip()) < 3:
            return SentimentResult("neutral", 0.0, [], "low")
        
        # Análisis con TextBlob (fallback)
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determinar sentimiento base
        if polarity > 0.1:
            base_sentiment = "positive"
            confidence = min(abs(polarity) + 0.3, 0.9)
        elif polarity < -0.1:
            base_sentiment = "negative"
            confidence = min(abs(polarity) + 0.3, 0.9)
        else:
            base_sentiment = "neutral"
            confidence = 0.7
        
        # Extraer frases clave (simplificado)
        key_phrases = self._extract_key_phrases(text)
        
        # Determinar urgencia
        urgency = self._determine_urgency(text, base_sentiment, polarity)
        
        return SentimentResult(
            sentiment=base_sentiment,
            confidence=round(confidence, 2),
            key_phrases=key_phrases,
            urgency_level=urgency
        )
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extrae frases clave del texto"""
        # Implementación simple - puedes mejorar con RAKE o KeyBERT
        sentences = text.split('.')
        key_phrases = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and any(keyword in sentence.lower() for keyword in 
                                         ['problema', 'dificultad', 'excelente', 'mejorar', 
                                          'confuso', 'claro', 'ayuda', 'no entiendo']):
                key_phrases.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)
        
        return key_phrases[:3]  # Máximo 3 frases
    
    def _determine_urgency(self, text: str, sentiment: str, polarity: float) -> str:
        """Determina el nivel de urgencia basado en el contenido"""
        text_lower = text.lower()
        
        # Palabras que indican alta urgencia
        high_urgency_words = ['urgente', 'emergencia', 'no entiendo nada', 'desesperado', 
                             'frustrado', 'nunca', 'siempre', 'horrible', 'terrible']
        
        # Palabras que indican urgencia media
        medium_urgency_words = ['problema', 'dificultad', 'confuso', 'ayuda', 
                               'no puedo', 'no sé', 'complicado']
        
        if any(word in text_lower for word in high_urgency_words):
            return "high"
        elif any(word in text_lower for word in medium_urgency_words):
            return "medium"
        elif sentiment == "negative" and polarity < -0.3:
            return "medium"
        else:
            return "low"