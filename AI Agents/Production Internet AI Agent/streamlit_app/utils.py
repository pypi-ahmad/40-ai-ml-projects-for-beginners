"""Shared Streamlit helpers."""

from __future__ import annotations

import asyncio

import streamlit as st

from internet_agent.config import get_settings
from internet_agent.services.agent_service import InternetAgentService


@st.cache_resource
def get_service() -> InternetAgentService:
    return InternetAgentService(settings=get_settings())


def run_async(coro):
    return asyncio.run(coro)
