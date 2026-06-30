"""LangChain SQL generator using SQLDatabase + ChatOllama."""

from __future__ import annotations

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

from ai_sql_assistant.config import AppSettings
from ai_sql_assistant.llm.ollama_client import LangChainOllamaFactory
from ai_sql_assistant.prompts.templates import PERSONA_TEMPLATES
from ai_sql_assistant.types import GenerationCandidate
from ai_sql_assistant.utils.sql_utils import clean_sql_text, format_sql


class LangChainSQLGenerator:
    """Generate SQL with LangChain SQL query chain."""

    def __init__(self, settings: AppSettings, db_uri: str) -> None:
        self.settings = settings
        self.db = SQLDatabase.from_uri(db_uri)
        self.factory = LangChainOllamaFactory(settings)

    def generate(self, question: str, persona: str, model: str) -> GenerationCandidate:
        """Generate SQL candidate via create_sql_query_chain."""
        chat_model = self.factory.chat_model(model)
        chain = create_sql_query_chain(chat_model, self.db)
        persona_instruction = PERSONA_TEMPLATES.get(
            persona,
            PERSONA_TEMPLATES["Business Analyst"],
        ).role_instruction
        final_question = (
            f"Persona objective: {persona_instruction}\n"
            f"User question: {question}\n"
            "Return only SQLite SELECT query."
        )

        result = chain.invoke({"question": final_question})
        sql = format_sql(clean_sql_text(str(result)))

        # LangChain chain does not expose latency directly, measured externally in pipeline.
        return GenerationCandidate(
            sql=sql,
            approach="langchain",
            model=model,
            prompt_name=persona,
            latency_ms=0.0,
            raw_response=str(result),
        )
