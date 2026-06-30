"""Root FastAPI app module for local serving."""

from peft_platform.api.app import create_app

app = create_app()
