import json
import datetime
import os
from huggingface_hub import HfApi

# Path to save models.json in the 'outputs' folder
OUTPUT_FOLDER = "outputs"
FILE_PATH = os.path.join(OUTPUT_FOLDER, "models.json")

utc_now = datetime.now(timezone.utc)
utc_string = utc_now.isoformat()
parsed_utc = datetime.fromisoformat(utc_string)

def get_models():
    """Fetch Hugging Face models and return a JSON list."""
    hf_api = HfApi()
    models_itr = hf_api.list_models(task="text-generation", library="transformers")

    models = []
    for x in models_itr:
        model_entry = {
            "model": x.modelId,
            "is_base_model": not "base_model" in " ".join(str(y) for y in x.tags)
        }
        models.append(model_entry)
        if len(models) % 1000 == 0:
            print(f"Current count: {len(models)}")

    print(f"Total models fetched: {len(models)}")
    return models

def update_file():
    """Update models.json with the latest models and timestamp."""
    models = get_models()
    
    # Add timestamp to JSON output
    data = {
        "last_updated": str(parsed_utc),
        "models": models
    }
    
    with open(FILE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated {FILE_PATH} with {len(models)} models.")

if __name__ == "__main__":
    update_file()
