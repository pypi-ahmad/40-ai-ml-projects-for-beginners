from src.ollama_client import OllamaClient

PROMPT_LENGTHS = {
    "short": "What is machine learning?",
    "medium": (
        "Explain the differences between supervised, unsupervised, and reinforcement learning. "
        "Include examples of each and discuss real-world applications."
    ),
    "long": (
        "Provide a comprehensive explanation of deep learning including: "
        "(1) how neural networks work, (2) the role of backpropagation, "
        "(3) different activation functions and their use cases, "
        "(4) common architectures like CNNs, RNNs, and Transformers, "
        "(5) training techniques including dropout, batch normalization, "
        "and learning rate scheduling, (6) challenges like vanishing gradients "
        "and overfitting, and (7) recent advances in the field."
    ),
}

MODELS = ["qwen3.5:2b", "qwen3.5:4b", "granite4.1:3b", "nemotron-3-nano:4b"]


class BenchmarkRunner:
    def __init__(self, models: list[str] | None = None) -> None:
        self.models = models or MODELS
        self._client = OllamaClient()

    def run_all(self, prompt_lengths: dict[str, str] | None = None) -> dict:
        lengths = prompt_lengths or PROMPT_LENGTHS
        results: dict = {}
        for model in self.models:
            results[model] = {}
            for label, prompt in lengths.items():
                m = self._client.measure_inference_time(model, prompt)
                results[model][label] = m
        return results

    def close(self) -> None:
        self._client.close()
