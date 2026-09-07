from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Union, Any

from .translator import WeightsTranslator
from .generator import DublikatAIGenerator

app = FastAPI(title="Sun1 API", version="0.1.0")
generator = DublikatAIGenerator()

class GenerationRequest(BaseModel):
    weights: Union[Dict[str, float], List[float]] = Field(..., description="Weight map or vector")
    context: Dict[str, Any] = Field(default_factory=dict)
    temperature: float = 0.3

class GenerationResponse(BaseModel):
    status: str
    text: str

@app.post("/v1/generate", response_model=GenerationResponse)
async def generate_text(payload: GenerationRequest):
    try:
        prompt = WeightsTranslator.to_prompt(payload.weights, payload.context)
        result_text = await generator.generate_response(prompt, temperature=payload.temperature)
        return GenerationResponse(status="success", text=result_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
      
