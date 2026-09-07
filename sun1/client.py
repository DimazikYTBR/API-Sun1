import asyncio
from pathlib import Path
from typing import Union, Dict, List, Any
import httpx

from .translator import WeightsTranslator
from .generator import DublikataAIGenerator

class APISun1:
    """Core client for API Sun1 integrations."""

    def __init__(self, dublikata_url: str = None, api_key: str = None):
        self.generator = DublikataAIGenerator(endpoint_url=dublikata_url, api_key=api_key)

    async def to_text_async(
        self, 
        weights: Union[Dict[str, float], List[float]], 
        context: Dict[str, Any] = None,
        temperature: float = 0.3
    ) -> str:
        prompt = WeightsTranslator.to_prompt(weights, context)
        return await self.generator.generate_response(prompt, temperature=temperature)

    def to_text(
        self, 
        weights: Union[Dict[str, float], List[float]], 
        context: Dict[str, Any] = None,
        temperature: float = 0.3
    ) -> str:
        """Synchronous wrapper for weights-to-text generation."""
        return asyncio.run(self.to_text_async(weights, context, temperature))

    def download_weights(self, url: str, destination_path: str):
        """Downloads weights file with clean start and finish status markers."""
        target_path = Path(destination_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[API Sun1] Download started: {target_path.name}")

        with httpx.Client(follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(target_path, "wb") as file:
                    for chunk in response.iter_bytes(chunk_size=1024 * 128):
                        if chunk:
                            file.write(chunk)

        print(f"[API Sun1] Download completed: {target_path.name}")
