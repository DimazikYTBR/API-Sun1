import os
from typing import Optional, Iterator
import httpx

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class DublikataAIGenerator:
    """Hybrid Engine: runs locally via llama.cpp or connects via Remote API."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0
    ):
        self.model_path = model_path or os.getenv("DUBLIKATA_MODEL_PATH")
        self.endpoint_url = endpoint_url or os.getenv("DUBLIKATA_AI_URL")
        self.api_key = api_key or os.getenv("DUBLIKATA_AI_KEY")
        self.local_engine: Optional[Llama] = None

        if self.model_path:
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError(
                    "llama-cpp-python is required for local inference. "
                    "Install it via: pip install llama-cpp-python"
                )
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")

            print(f"[API Sun1] Loading GGUF weights from: {self.model_path}")
            self.local_engine = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            print("[API Sun1] Local inference engine initialized successfully.")

    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.3
    ) -> str:

        if self.local_engine:
            output = self.local_engine(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "\n\n\n"]
            )
            return output["choices"][0]["text"].strip()

        if self.endpoint_url:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                res = await client.post(
                    self.endpoint_url,
                    json={
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    headers=headers,
                    timeout=60.0
                )
                res.raise_for_status()
                return res.json().get("text", "")

        return "Dublikata AI: Synthesis complete. (Mock mode: specify model_path or endpoint_url)"

    def stream_response(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.3
    ) -> Iterator[str]:
        """Real-time token streaming generation."""
        if self.local_engine:
            stream = self.local_engine(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                yield chunk["choices"][0]["text"]
        else:
            raise NotImplementedError("Streaming is currently supported only for local model_path.")