from huggingface_hub import HfApi

FILE_PATH = "models.txt"

def get_models():
    """Fetch Hugging Face models and return a sorted list."""
    hf_api = HfApi()
    # Get a list of models for text generation with the "transformers" library
    models_itr = hf_api.list_models(task="text-generation", library="transformers")

    # Filter out private models
    public_models = [model for model in models_itr if not model.private]
    
    # Filter models and sort alphabetically
    models = sorted([
        x.modelId for x in public_models if "base_model" not in " ".join(str(y) for y in x.tags)
    ])
    
    return models

def update_file():
    """Update models.txt with the latest sorted models."""
    models = get_models()
    with open(FILE_PATH, "w") as f:
        f.write("\n".join(models))
    print(f"Updated {FILE_PATH} with {len(models)} models.")

if __name__ == "__main__":
    update_file()
