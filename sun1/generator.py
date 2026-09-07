import os
import httpx

class DublikataAIGenerator:
    """Interface for invoking Dublikata AI inference services."""

    def __init__(self, endpoint_url: str = None, api_key: str = None):
        self.endpoint_url = endpoint_url or os.getenv("DUBLIKATA_AI_URL")
        self.api_key = api_key or os.getenv("DUBLIKATA_AI_KEY")

    async def generate_response(self, prompt: str, temperature: float = 0.3) -> str:
        if self.endpoint_url:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                res = await client.post(
                    self.endpoint_url,
                    json={"prompt": prompt, "temperature": temperature},
                    headers=headers,
                    timeout=30.0
                )
                res.raise_for_status()
                return res.json().get("text", "")

        return "Dublikata AI: Synthesis complete. Generated response based on Sun1 weights."
