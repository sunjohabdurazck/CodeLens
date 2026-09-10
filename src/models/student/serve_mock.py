"""
CodeLens Mock Server - No internet required.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import random
import math
import re

app = FastAPI(title="CodeLens Mock Server")

class ScoreRequest(BaseModel):
    prompt: str

class ScoreResponse(BaseModel):
    score: float

def analyze_prompt_quality(prompt: str) -> float:
    """Mock quality analysis based on prompt characteristics."""
    if not prompt:
        return 0.0
    
    # Calculate quality score based on multiple factors
    quality_score = 0.5  # Start with base
    
    # Factor 1: Length (ideal: 50-200 chars)
    length = len(prompt)
    if 50 <= length <= 200:
        quality_score += 0.2
    elif length > 200:
        quality_score += 0.1
    else:
        quality_score -= 0.1
    
    # Factor 2: Code-related keywords
    code_keywords = ['function', 'class', 'method', 'api', 'endpoint', 'component', 
                     'module', 'import', 'export', 'react', 'vue', 'angular', 'javascript',
                     'python', 'java', 'c++', 'typescript', 'node', 'express']
    if any(kw in prompt.lower() for kw in code_keywords):
        quality_score += 0.15
    
    # Factor 3: Specificity
    specific_words = ['specific', 'example', 'input', 'output', 'parameter', 'return', 
                     'type', 'string', 'list', 'array', 'object', 'interface']
    specificity = sum(1 for w in specific_words if w in prompt.lower()) / len(specific_words)
    quality_score += specificity * 0.15
    
    # Factor 4: Has question mark (good for clarity)
    if '?' in prompt:
        quality_score += 0.05
    
    # Factor 5: Has code block
    if '```' in prompt:
        quality_score += 0.1
    
    # Clamp and add slight randomness
    quality_score += random.uniform(-0.05, 0.05)
    return max(0.1, min(0.95, quality_score))

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    score = analyze_prompt_quality(req.prompt)
    return ScoreResponse(score=score)

@app.get("/health")
def health():
    return {"status": "ok", "mode": "mock", "message": "CodeLens Mock Server Running"}
