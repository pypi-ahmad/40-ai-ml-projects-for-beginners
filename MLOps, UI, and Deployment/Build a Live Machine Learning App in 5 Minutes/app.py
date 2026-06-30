"""Main Gradio entrypoint for Project #8: live local ML application."""

from __future__ import annotations

import logging
import os
import socket

import gradio as gr
import pandas as pd

from src.config import get_config
from src.translation import SUPPORTED_LANGUAGES
from src.ui_handlers import (
    BENCHMARK_MODELS,
    CHAT_MODELS,
    OCR_MODELS,
    SENTIMENT_MODELS,
    SUMMARY_MODELS,
    AppHandlers,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

HANDLERS = AppHandlers()


def handle_sentiment(text: str, model: str) -> tuple[str, float, str, str]:
    """Forward sentiment callback to shared handler implementation."""

    return HANDLERS.handle_sentiment(text=text, model=model)


def handle_summary(text: str, model: str) -> tuple[str, str, str]:
    """Forward summarization callback to shared handler implementation."""

    return HANDLERS.handle_summary(text=text, model=model)


def handle_translation(
    text: str, source_lang: str, target_lang: str, model: str
) -> tuple[str, str]:
    """Forward translation callback to shared handler implementation."""

    return HANDLERS.handle_translation(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        model=model,
    )


def handle_chat(
    user_message: str,
    history_by_model: dict[str, list[dict[str, str]]] | None,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, list[dict[str, str]], dict[str, list[dict[str, str]]], str]:
    """Forward chat callback to shared handler implementation."""

    return HANDLERS.handle_chat(
        user_message=user_message,
        history_by_model=history_by_model,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def reset_chat(
    model: str,
    history_by_model: dict[str, list[dict[str, str]]] | None,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]], str]:
    """Forward reset callback to shared handler implementation."""

    return HANDLERS.reset_chat(model=model, history_by_model=history_by_model)


def handle_document_analysis(
    file_path: str | None,
    question: str,
    ocr_model: str,
    qa_model: str,
) -> tuple[str, str, str, str]:
    """Forward document callback to shared handler implementation."""

    return HANDLERS.handle_document_analysis(
        file_path=file_path,
        question=question,
        ocr_model=ocr_model,
        qa_model=qa_model,
    )


def run_benchmarks(
    prompt_profile: str, runs: int
) -> tuple[pd.DataFrame, str, str, str, str, str, str]:
    """Forward benchmark callback to shared handler implementation."""

    return HANDLERS.run_benchmarks(prompt_profile=prompt_profile, runs=runs)


def warmup_selected_models(selected_models: list[str]) -> str:
    """Forward warmup callback to shared handler implementation."""

    return HANDLERS.warmup_selected_models(selected_models=selected_models)


def get_system_status() -> str:
    """Forward system-status callback to shared handler implementation."""

    return HANDLERS.get_system_status()


def switch_chat_model(
    history_by_model: dict[str, list[dict[str, str]]] | None,
    model: str,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    """Load per-model chat history when user switches model."""

    return HANDLERS.switch_chat_model(history_by_model=history_by_model, model=model)


def resolve_launch_port(host: str, preferred_port: int, max_attempts: int = 20) -> int:
    """Resolve first available port from preferred port upward."""

    for offset in range(max_attempts):
        candidate_port = preferred_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate_port))
            except OSError:
                continue
        return candidate_port

    end_port = preferred_port + max_attempts - 1
    raise OSError(f"No available port in range {preferred_port}-{end_port}.")


def build_app() -> gr.Blocks:
    """Construct complete multi-tab Gradio application."""

    cfg = get_config()
    translation_model = cfg.translation_model
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="teal", neutral_hue="slate")

    with gr.Blocks(
        title="Live ML App in 5 Minutes - Local LLM Product",
        fill_width=True,
        theme=theme,
    ) as demo:
        gr.Markdown(
            """
# Live Machine Learning App in 5 Minutes (Production-Style)

Build and explore end-to-end AI product patterns: model serving, interactive UI, state management,
OCR pipelines, benchmarking, and deployment readiness using **local Ollama models**.
            """
        )

        with gr.Accordion("System Status + Warmup", open=False):
            status_btn = gr.Button("Refresh Runtime Status")
            status_output = gr.Markdown(
                "Click **Refresh Runtime Status** to load model availability."
            )
            warm_models = gr.CheckboxGroup(
                choices=sorted(
                    set(
                        SENTIMENT_MODELS
                        + SUMMARY_MODELS
                        + CHAT_MODELS
                        + OCR_MODELS
                        + BENCHMARK_MODELS
                        + [translation_model]
                    )
                ),
                label="Select models to warm",
                value=[cfg.chat_model],
            )
            warm_btn = gr.Button("Warm Selected Models")
            warm_output = gr.Markdown()

            status_btn.click(fn=get_system_status, outputs=status_output)
            warm_btn.click(fn=warmup_selected_models, inputs=warm_models, outputs=warm_output)

        with gr.Tabs():
            with gr.TabItem("Sentiment Analysis"):
                gr.Markdown("Analyze sentiment with structured output and confidence score.")
                with gr.Row():
                    sentiment_text = gr.Textbox(
                        label="Input Text",
                        lines=5,
                        placeholder="Paste review, tweet, or customer feedback...",
                    )
                    sentiment_model = gr.Dropdown(
                        SENTIMENT_MODELS,
                        value=cfg.sentiment_model,
                        label="Model",
                    )
                sentiment_run = gr.Button("Analyze Sentiment", variant="primary")
                with gr.Row():
                    sentiment_label = gr.Label(label="Predicted Label", num_top_classes=3)
                    sentiment_conf = gr.Number(label="Confidence", precision=3)
                sentiment_explanation = gr.Markdown(label="Explanation")
                sentiment_debug = gr.Code(language="json", label="Raw Structured Output")

                def _sentiment_ui(
                    text: str, model: str
                ) -> tuple[dict[str, float], float, str, str]:
                    label, confidence, explanation, payload = handle_sentiment(text, model)
                    label_payload = {label: float(confidence)} if label else {"Neutral": 0.0}
                    return label_payload, confidence, explanation, payload

                sentiment_run.click(
                    fn=_sentiment_ui,
                    inputs=[sentiment_text, sentiment_model],
                    outputs=[
                        sentiment_label,
                        sentiment_conf,
                        sentiment_explanation,
                        sentiment_debug,
                    ],
                )

            with gr.TabItem("Text Summarization"):
                gr.Markdown("Summarize long-form text into concise summary + key points.")
                with gr.Row():
                    summary_text = gr.Textbox(
                        label="Long Text",
                        lines=12,
                        placeholder="Paste article, report, or transcript...",
                    )
                    summary_model = gr.Dropdown(
                        SUMMARY_MODELS,
                        value=cfg.summarization_model,
                        label="Model",
                    )
                summary_btn = gr.Button("Generate Summary", variant="primary")
                summary_out = gr.Textbox(label="Summary", lines=6)
                summary_points = gr.Markdown(label="Key Points")
                summary_debug = gr.Code(language="json", label="Raw Structured Output")
                summary_btn.click(
                    fn=handle_summary,
                    inputs=[summary_text, summary_model],
                    outputs=[summary_out, summary_points, summary_debug],
                )

            with gr.TabItem("Translation"):
                gr.Markdown("Translate text using local `translategemma:4b` model.")
                with gr.Row():
                    source_lang = gr.Dropdown(
                        SUPPORTED_LANGUAGES, value="English", label="Source Language"
                    )
                    target_lang = gr.Dropdown(
                        SUPPORTED_LANGUAGES, value="Spanish", label="Target Language"
                    )
                    translation_model_dropdown = gr.Dropdown(
                        [translation_model],
                        value=translation_model,
                        label="Model",
                    )
                translation_text = gr.Textbox(label="Input Text", lines=8)
                translate_btn = gr.Button("Translate", variant="primary")
                translation_out = gr.Textbox(label="Translated Text", lines=8)
                translation_debug = gr.Code(language="json", label="Raw Structured Output")
                translate_btn.click(
                    fn=handle_translation,
                    inputs=[translation_text, source_lang, target_lang, translation_model_dropdown],
                    outputs=[translation_out, translation_debug],
                )

            with gr.TabItem("Local LLM Chat"):
                gr.Markdown(
                    "Multi-turn chat with persistent conversation memory and controllable decoding."
                )
                chat_state = gr.State({})

                with gr.Row():
                    chat_model = gr.Dropdown(CHAT_MODELS, value=cfg.chat_model, label="Model")
                    chat_temp = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="Temperature",
                    )
                    chat_max_tokens = gr.Slider(
                        minimum=128,
                        maximum=1200,
                        value=700,
                        step=32,
                        label="Max Tokens",
                    )

                chatbot = gr.Chatbot(type="messages", allow_tags=False, label="Conversation")
                chat_input = gr.Textbox(label="Your message", lines=3)

                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

                chat_debug = gr.Code(language="json", label="Latest Response Metadata")

                send_btn.click(
                    fn=handle_chat,
                    inputs=[chat_input, chat_state, chat_model, chat_temp, chat_max_tokens],
                    outputs=[chat_input, chatbot, chat_state, chat_debug],
                )
                chat_input.submit(
                    fn=handle_chat,
                    inputs=[chat_input, chat_state, chat_model, chat_temp, chat_max_tokens],
                    outputs=[chat_input, chatbot, chat_state, chat_debug],
                )
                clear_btn.click(
                    fn=reset_chat,
                    inputs=[chat_model, chat_state],
                    outputs=[chatbot, chat_state, chat_debug],
                )
                chat_model.change(
                    fn=switch_chat_model,
                    inputs=[chat_state, chat_model],
                    outputs=[chatbot, chat_state],
                )

            with gr.TabItem("Document Analyzer"):
                gr.Markdown(
                    "Upload PDF/image for extraction, summarization, "
                    "and grounded Q&A using OCR + local LLM."
                )
                with gr.Row():
                    uploaded_file = gr.File(
                        label="Upload PDF or Image",
                        type="filepath",
                        file_types=[
                            ".pdf",
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".bmp",
                            ".webp",
                            ".tif",
                            ".tiff",
                        ],
                    )
                    question = gr.Textbox(label="Question (optional)", lines=3)
                with gr.Row():
                    ocr_model = gr.Dropdown(
                        OCR_MODELS, value=cfg.ocr_primary_model, label="OCR Model"
                    )
                    qa_model = gr.Dropdown(CHAT_MODELS, value=cfg.chat_model, label="Q&A Model")

                doc_btn = gr.Button("Analyze Document", variant="primary")
                extracted_out = gr.Textbox(label="Extracted Text", lines=10)
                summary_out_doc = gr.Textbox(label="Document Summary", lines=6)
                answer_out = gr.Markdown(label="Answer + Metadata")
                doc_debug = gr.Code(language="json", label="Raw Structured Output")

                doc_btn.click(
                    fn=handle_document_analysis,
                    inputs=[uploaded_file, question, ocr_model, qa_model],
                    outputs=[extracted_out, summary_out_doc, answer_out, doc_debug],
                )

            with gr.TabItem("Benchmarking + Visualization"):
                gr.Markdown(
                    "Run local model benchmarks with real latency, throughput, "
                    "memory, and quality scores."
                )
                with gr.Row():
                    prompt_profile = gr.Dropdown(
                        ["short", "medium", "long"],
                        value="medium",
                        label="Primary Prompt Profile",
                    )
                    runs = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=cfg.benchmark_runs,
                        step=1,
                        label="Runs per model",
                    )
                benchmark_btn = gr.Button("Run Benchmark Suite", variant="primary")

                benchmark_df = gr.DataFrame(label="Benchmark Metrics", interactive=False)
                benchmark_table = gr.Markdown(label="Comparison Table")
                with gr.Row():
                    latency_img = gr.Image(type="filepath", label="Latency Chart")
                    throughput_img = gr.Image(type="filepath", label="Throughput Chart")
                with gr.Row():
                    memory_img = gr.Image(type="filepath", label="Memory Chart")
                    prompt_scale_img = gr.Image(type="filepath", label="Prompt Scale Chart")
                radar_img = gr.Image(type="filepath", label="Radar Chart")

                benchmark_btn.click(
                    fn=run_benchmarks,
                    inputs=[prompt_profile, runs],
                    outputs=[
                        benchmark_df,
                        benchmark_table,
                        latency_img,
                        throughput_img,
                        memory_img,
                        prompt_scale_img,
                        radar_img,
                    ],
                )

        gr.Markdown(
            """
### Deployment Notes
- Local run: `uv run python app.py`
- Public share test: set `GRADIO_SHARE=1` then run app.
- For production, move to containerized API + frontend with auth, monitoring, and autoscaling.
            """
        )

    return demo


def main() -> None:
    """Launch Gradio app on all interfaces for LAN access."""

    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}

    app = build_app()
    app.queue(default_concurrency_limit=4)

    launch_port = resolve_launch_port(host=host, preferred_port=preferred_port)
    if launch_port != preferred_port:
        logger.warning(
            "Preferred port %s unavailable. Using fallback port %s.",
            preferred_port,
            launch_port,
        )

    app.launch(
        server_name=host,
        server_port=launch_port,
        share=share,
    )


if __name__ == "__main__":
    main()
