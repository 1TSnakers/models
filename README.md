[![Update Hugging Face Models](https://github.com/1TSnakers/models/actions/workflows/update_models.yml/badge.svg)](https://github.com/1TSnakers/models/actions/workflows/update_models.yml)

This is a thing for a side project, it's just a thing that filters HuggingFace models so that I don't need to run something that takes a long time.

Because the time syntax could be a little bit confusing, here are some code samples of JavaScript and Python.

JavaScript:
``` js
var time = "2025-03-13T14:23:45Z"
time = new Date(time);
// Of course, you have to change the date input, this is an example.
```

Python:
``` python
from datetime import datetime
time = "2025-03-13 02:23 PM UTC"
time = datetime.strptime(time, "%Y-%m-%d %I:%M %p UTC")
# Of course, you have to change the date input, this is an example.
```

If you want to know what it's filtering for some reason, here:
- Text generation models
- Compatibility with the GGUF, or ollama
- Base models (Not fine-tuned or quantized)
