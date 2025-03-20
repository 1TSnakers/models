import json
import datetime
import os
from huggingface_hub import HfApi

# Path to save models.json in the 'outputs' folder
OUTPUT_FOLDER = "outputs"
FILE_PATH = os.path.join(OUTPUT_FOLDER, "models.json")

def get_models():
    """Fetch Hugging Face models and return a JSON list."""
    hf_api = HfApi()
    models_itr = hf_api.list_models(task="text-generation", library="transformers")

    models = []
    for x in models_itr:
        model_entry = {
            "model": x.modelId,
            "is_base_model": "base_model" in " ".join(str(y) for y in x.tags)
        }
        models.append(model_entry)
        if len(models) % 1000 == 0:
            print(f"Current count: {len(models)}")

    print(f"Total models fetched: {len(models)}")
    return models

def get_timestamps():
    """Generate both JS and Python-style timestamps."""
    now_utc = datetime.datetime.utcnow()

    # JS-style: ISO 8601 with Z (still readable for humans)
    js_last_updated = now_utc.isoformat(timespec="seconds") + "Z"

    # Python-style: Clean, human-readable format
    py_last_updated = now_utc.strftime("%Y-%m-%d %I:%M %p UTC")

    return js_last_updated, py_last_updated

def update_file():
    """Update models.json with the latest models and timestamps."""
    models = get_models()

    # Create the 'outputs' folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get timestamps in both formats
    js_last_updated, py_last_updated = get_timestamps()

    # Add timestamps and models to JSON output
    data = {
        "JS_last_updated": js_last_updated,
        "PY_last_updated": py_last_updated,
        "models": models
    }

    with open(FILE_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {FILE_PATH} with {len(models)} models.")

if __name__ == "__main__":
    update_file()
