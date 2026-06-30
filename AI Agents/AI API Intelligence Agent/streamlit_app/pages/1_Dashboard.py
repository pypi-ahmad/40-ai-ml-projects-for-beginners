from __future__ import annotations

import streamlit as st

from api_intel_agent.monitoring import MetricsCollector

st.title('Dashboard')
metrics = MetricsCollector().snapshot()

c1, c2, c3 = st.columns(3)
c1.metric('CPU %', metrics.cpu_percent)
c2.metric('Memory %', metrics.memory_percent)
c3.metric('Cache Hit Rate', metrics.cache_hit_rate)

st.json({'gpu_name': metrics.gpu_name, 'gpu_vram_used_mb': metrics.gpu_vram_used_mb, 'api_latency_ms': metrics.api_latency_ms})
