import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import WordComplexityPredictor

# 1. Load environment variables from .env into os.environ
load_dotenv()

app = FastAPI(title="English Word Complexity Classifier API", version="1.0")

bundle_path = os.getenv("BUNDLE_DIR", "./export_bundle")
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    print("⚠️ Warning: GROQ_API_KEY not found in .env. LLM simplification will be disabled.")

class PassageRequest(BaseModel):
    passage: str
    target_comfort_level: int = 1  # 0: Beginner, 1: Intermediate, 2: Advanced

@app.post("/simplify")
async def simplify_passage(req: PassageRequest):
    if not req.passage.strip():
        raise HTTPException(status_code=400, detail="Passage cannot be empty.")
    
    reconstructed, audit_log = predictor.analyze_passage(
        passage=req.passage,
        target_comfort_level=req.target_comfort_level
    )
    
    return {
        "target_comfort_level": req.target_comfort_level,
        "reconstructed_passage": reconstructed,
        "audit_log": audit_log
    }