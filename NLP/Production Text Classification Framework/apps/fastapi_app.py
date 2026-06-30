"""FastAPI app entrypoint."""

from textclf_framework.serving.api import APIConfig, create_app

app = create_app(APIConfig(model_path="artifacts/models/champion", label_names=["class_0", "class_1"]))
