from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import re
import time
import json
import pandas as pd
from typing import List, Dict

class PhysicsMetricsCalculator:
    def __init__(self):
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_content_metrics(self, response: str) -> Dict:
        """Calculate physics content-specific metrics"""
        metrics = {
            'equations_count': len(re.findall(r'[A-Za-z]=.*[0-9+\-*/^()]', response)),
            'units_count': len(re.findall(r'\b(m/s|kg|N|J|W|Pa|Hz|V|Ω|°C|K)\b', response)),
            'steps_count': len(re.findall(r'step|:|\d\)', response.lower())),
            'response_length': len(response.split())
        }
        return metrics
    
    def calculate_relevance_metrics(self, question: str, response: str, context: str) -> Dict:
        """Calculate relevance metrics using embeddings"""
        question_emb = self.embeddings_model.encode([question])
        response_emb = self.embeddings_model.encode([response])
        context_emb = self.embeddings_model.encode([context])
        
        metrics = {
            'question_similarity': float(cosine_similarity(question_emb, response_emb)[0][0]),
            'context_similarity': float(cosine_similarity(context_emb, response_emb)[0][0])
        }
        return metrics