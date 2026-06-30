from __future__ import annotations

import streamlit as st

from api_intel_agent.monitoring import MetricsCollector

st.title('API Metrics')
collector = MetricsCollector()
snapshot = collector.snapshot()

st.json({
    'cpu_percent': snapshot.cpu_percent,
    'memory_percent': snapshot.memory_percent,
    'gpu_name': snapshot.gpu_name,
    'gpu_vram_used_mb': snapshot.gpu_vram_used_mb,
    'gpu_vram_total_mb': snapshot.gpu_vram_total_mb,
    'cache_hit_rate': snapshot.cache_hit_rate,
    'api_latency_ms': snapshot.api_latency_ms,
})
