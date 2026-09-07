from typing import Union, Dict, List, Any

class WeightsTranslator:
    """Translates numeric weights, scores, or vectors into a structured prompt."""

    @staticmethod
    def to_prompt(
        weights: Union[Dict[str, float], List[float]], 
        context: Dict[str, Any] = None
    ) -> str:
        context = context or {}
        task = context.get("task", "general_synthesis")

        if isinstance(weights, dict):
            weight_lines = [f"- {param}: {val:.4f}" for param, val in weights.items()]
            weights_repr = "Weights & metrics:\n" + "\n".join(weight_lines)
        elif isinstance(weights, list):
            sample = ", ".join([f"{x:.3f}" for x in weights[:8]])
            weights_repr = (
                f"Latent space embedding (dimension: {len(weights)}).\n"
                f"Head slice: [{sample}...]"
            )
        else:
            raise ValueError("'weights' must be provided as a dict or a list of floats.")

        prompt = (
            f"[SYSTEM: Sun1 Engine]\n"
            f"[TASK: {task}]\n"
            f"{weights_repr}\n\n"
            f"[INSTRUCTION FOR DUBLIKATA AI]:\n"
            f"Analyze the provided weights and parameters to synthesize a clear, "
            f"objective, and detailed text summary."
        )
        return prompt
