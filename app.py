from typing import cast

import gradio as gr

from src.benchmarking import BenchmarkRunner
from src.chat import ChatEngine
from src.document_analyzer import DocumentAnalyzer
from src.sentiment import SentimentAnalyzer
from src.summarization import Summarizer
from src.translation import Translator
from src.visualization import BenchmarkVisualizer

sentiment = SentimentAnalyzer()
summarizer = Summarizer()
translator = Translator()
chat = ChatEngine()
doc_analyzer = DocumentAnalyzer()
bench = BenchmarkRunner()
vis = BenchmarkVisualizer()


def tab_sentiment(text: str) -> str:
    if not text.strip():
        return "Enter text to analyze."
    try:
        result = sentiment.analyze(text)
        return (
            f"**Label:** {result['label']}\n\n"
            f"**Score:** {result['score']:.2f}\n\n"
            f"**Explanation:** {result['explanation']}"
        )
    except Exception as e:
        return f"**Error:** {e}"


def tab_summarize(text: str) -> str:
    if not text.strip():
        return "Enter text to summarize."
    if len(text.strip()) < 50:
        return "Please enter at least 50 characters."
    try:
        result = summarizer.summarize(text)
        return (
            f"**TL;DR:** {result['tldr']}\n\n"
            f"**Summary:** {result['summary']}\n\n"
            f"**Key Points:**\n" + "\n".join(f"- {k}" for k in result["key_points"])
        )
    except Exception as e:
        return f"**Error:** {e}"


def tab_translate(text: str, lang: str) -> str:
    if not text.strip():
        return "Enter text to translate."
    try:
        result = translator.translate(text, lang)
        return cast(str, result["translated_text"])
    except Exception as e:
        return f"**Error:** {e}"


def tab_chat(message: str, history: list) -> tuple[str, list]:
    if not message.strip():
        return "", history
    try:
        reply = chat.send(message)
        history.append([message, reply])
        return "", history
    except Exception as e:
        return "", history + [[message, f"**Error:** {e}"]]


def tab_chat_reset() -> list:
    chat.reset()
    return []


def tab_doc_analyze(file_path: str | None, question: str) -> str:
    if file_path is None:
        return "Upload an image or PDF first."
    try:
        text = doc_analyzer.extract_text(file_path)
        if not question.strip():
            return f"**Extracted Text:**\n{text}"
        answer = doc_analyzer.answer_question(text, question)
        return (
            f"**Question:** {question}\n\n"
            f"**Answer:** {answer}\n\n"
            f"**Context (extracted):**\n{text[:500]}"
        )
    except Exception as e:
        return f"**Error analyzing document:** {e}"


def tab_benchmark() -> tuple:
    try:
        data = bench.run_all()
    except Exception as e:
        return f"**Benchmark error:** {e}", None, None, None, None
    paths = vis.generate_all(data)
    md = "## Benchmark Results\n\n"
    for model, lengths in data.items():
        md += f"### {model}\n"
        for label, m in lengths.items():
            md += f"- {label}: {m['latency_s']:.2f}s ({m['tokens']} tokens)\n"
        md += "\n"
    return md, paths["latency"], paths["throughput"], paths["prompt_scale"], paths["radar"]


with gr.Blocks(title="Multi-Model LLM Playground") as app:
    gr.Markdown("# Multi-Model LLM Playground")
    gr.Markdown(
        "Powered by **Ollama** with 4 local models optimized per task."
    )

    with gr.Tab("Sentiment"):
        gr.Markdown("### Sentiment Analysis — `qwen3.5:2b`")
        txt_input = gr.Textbox(label="Text", placeholder="Enter text to analyze...", lines=3)
        btn_sent = gr.Button("Analyze")
        sent_out = gr.Markdown()
        btn_sent.click(tab_sentiment, txt_input, sent_out)

    with gr.Tab("Summarization"):
        gr.Markdown("### Text Summarization — `granite4.1:3b`")
        sum_input = gr.Textbox(
            label="Text", placeholder="Enter text to summarize (50+ chars)...", lines=5
        )
        btn_sum = gr.Button("Summarize")
        sum_out = gr.Markdown()
        btn_sum.click(tab_summarize, sum_input, sum_out)

    with gr.Tab("Translation"):
        gr.Markdown("### Translation — `translategemma:4b`")
        with gr.Row():
            trans_input = gr.Textbox(label="Text to translate", lines=3, scale=2)
            lang_dd = gr.Dropdown(
                label="Target Language",
                choices=translator.supported_languages,
                value="Spanish",
                scale=1,
            )
        btn_trans = gr.Button("Translate")
        trans_out = gr.Textbox(label="Translation")
        btn_trans.click(tab_translate, [trans_input, lang_dd], trans_out)

    with gr.Tab("Chat"):
        gr.Markdown("### Chat — `qwen3.5:4b`")
        chatbot = gr.Chatbot(label="Conversation")
        chat_input = gr.Textbox(label="Message", placeholder="Type your message...")
        chat_input.submit(tab_chat, [chat_input, chatbot], [chat_input, chatbot])
        btn_reset = gr.Button("Reset Conversation")
        btn_reset.click(tab_chat_reset, None, chatbot)

    with gr.Tab("Document Analyzer"):
        gr.Markdown("### Document OCR + Q&A — `glm-ocr` + `qwen3.5:4b`")
        file_input = gr.File(
            label="Upload Image or PDF", file_types=[".png", ".jpg", ".jpeg", ".pdf"]
        )
        doc_question = gr.Textbox(
            label="Question (optional)", placeholder="Ask about the document..."
        )
        btn_doc = gr.Button("Analyze")
        doc_out = gr.Markdown()
        btn_doc.click(tab_doc_analyze, [file_input, doc_question], doc_out)

    with gr.Tab("Benchmarking"):
        gr.Markdown("### Benchmark: 4 Models × 3 Prompt Lengths")
        btn_bench = gr.Button("Run Benchmark")
        bench_out = gr.Markdown()
        with gr.Row():
            lat_img = gr.Image(label="Latency")
            tp_img = gr.Image(label="Throughput")
        with gr.Row():
            ps_img = gr.Image(label="Prompt Scaling")
            rad_img = gr.Image(label="Radar")
        btn_bench.click(tab_benchmark, None, [bench_out, lat_img, tp_img, ps_img, rad_img])

    app.load(lambda: None, None, None)  # type: ignore[arg-type]


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())  # nosec — intentional for LAN access
