"""Cache the two frozen contextual encoders used by the final experiments."""

from pathlib import Path

from transformers import AutoModel, AutoTokenizer


MODELS = {
    "MARBERTv2": "UBC-NLP/MARBERTv2",
    "bert-base-uncased": "bert-base-uncased",
}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "models"
    for directory, model_name in MODELS.items():
        destination = root / directory
        print(f"Caching {model_name} in {destination}")
        AutoTokenizer.from_pretrained(model_name).save_pretrained(destination)
        AutoModel.from_pretrained(model_name).save_pretrained(destination)
