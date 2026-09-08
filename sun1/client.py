import asyncio
from pathlib import Path
from typing import Union, Dict, List, Any, Optional, Iterator
import httpx

from .translator import WeightsTranslator
from .generator import DublikataAIGenerator


class APISun1:
    """Core client and inference engine for API Sun1."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        dublikata_url: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0
    ):
        self.generator = DublikataAIGenerator(
            model_path=model_path,
            endpoint_url=dublikata_url,
            api_key=api_key,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers
        )

    async def to_text_async(
        self, 
        weights: Union[Dict[str, float], List[float]], 
        context: Dict[str, Any] = None,
        temperature: float = 0.3,
        max_tokens: int = 512
    ) -> str:
        prompt = WeightsTranslator.to_prompt(weights, context)
        return await self.generator.generate_response(
            prompt=prompt, 
            temperature=temperature,
            max_tokens=max_tokens
        )

    def to_text(
        self, 
        weights: Union[Dict[str, float], List[float]], 
        context: Dict[str, Any] = None,
        temperature: float = 0.3,
        max_tokens: int = 512
    ) -> str:
        """Synchronous text generation based on weights."""
        return asyncio.run(
            self.to_text_async(weights, context, temperature, max_tokens)
        )

    def to_text_stream(
        self, 
        weights: Union[Dict[str, float], List[float]], 
        context: Dict[str, Any] = None,
        temperature: float = 0.3,
        max_tokens: int = 512
    ) -> Iterator[str]:
        """Streaming text generation (token by token)."""
        prompt = WeightsTranslator.to_prompt(weights, context)
        yield from self.generator.stream_response(
            prompt=prompt, 
            temperature=temperature,
            max_tokens=max_tokens
        )

    def download_weights(self, url: str, destination_path: str):
        """Downloading the weights file (.gguf / .bin) with silent markers."""
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