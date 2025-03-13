import json
from huggingface_hub import HfApi

FILE_PATH = "models.json"

def get_models():
    """Fetch Hugging Face models and return a JSON list."""
    hf_api = HfApi()
    models_itr = hf_api.list_models(task="text-generation", library="transformers")

    models = [
        {
            "model": x.modelId,
            "is_base_model": "base_model" in " ".join(str(y) for y in x.tags)
        }
        for x in models_itr
    ]

    return models

def update_file():
    """Update models.json with the latest models."""
    models = get_models()
    with open(FILE_PATH, "w") as f:
        json.dump(models, f, indent=2)
    print(f"Updated {FILE_PATH} with {len(models)} models.")

if __name__ == "__main__":
    update_file()
