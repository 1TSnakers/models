from huggingface_hub import HfApi

FILE_PATH = "models.txt"

def get_models():
    """Fetch Hugging Face models and return a list."""
    hf_api = HfApi()
    models_itr = hf_api.list_models(task="text-generation", library="transformers")
    return [
        x.modelId for x in models_itr if "base_model" not in " ".join(str(y) for y in x.tags)
    ]

def update_file():
    """Update models.txt with the latest models."""
    models = get_models().sort()
    with open(FILE_PATH, "w") as f:
        f.write("\n".join(models))
    print(f"Updated {FILE_PATH} with {len(models)} models.")

if __name__ == "__main__":
    update_file()
