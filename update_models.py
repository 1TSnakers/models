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
hf_api = HfApi()
modelsITR = hf_api.list_models(
    task="text-generation",
    library="transformers",
)

models = []
for x in modelsITR:
    is_base_model = "base_model" in " ".join(str(y) for y in x.tags)
    models.append({"model": x.modelId, "is_base_model": is_base_model})
    print(f"Processing model: {x.modelId}")

# Compare old and new lists
prev_set = {json.dumps(m, sort_keys=True) for m in prev_models}
new_set = {json.dumps(m, sort_keys=True) for m in models}

added_models = [json.loads(m) for m in new_set - prev_set]
removed_models = [json.loads(m) for m in prev_set - new_set]

# Print changes to console
if added_models:
    print(f"✅ Added models: {len(added_models)}")
    for model in added_models:
        print(f"  + {model['model']}")

if removed_models:
    print(f"❌ Removed models: {len(removed_models)}")
    for model in removed_models:
        print(f"  - {model['model']}")

if not added_models and not removed_models:
    print("⚡ No changes detected.")

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
