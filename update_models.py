import json
import os
from datetime import datetime
from huggingface_hub import HfApi

# Load previous models if exists
OUTPUT_FILE = "outputs/models.json"
prev_models = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        prev_models = json.load(f)

# Fetch new models
# Fetch new models
hf_api = HfApi()
modelsITR = hf_api.list_models(
    task="text-generation",
    library="transformers",
)

models = []
for x in modelsITR:
    is_base_model = "base_model" in " ".join(str(y) for y in x.tags)
    models.append({"model": x.modelId, "is_base_model": is_base_model})

print(f"✅ Processed {len(models)} models.")

# Add JS and Python timestamps
now = datetime.utcnow()
models_info = {
    "JS_last_updated": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),  # ISO-8601 format
    "PY_last_updated": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
    "models": models,
}

# Write updated models to file
os.makedirs("outputs", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(models_info, f, indent=2)

print(f"📂 Updated models.json with {len(models)} models.")
