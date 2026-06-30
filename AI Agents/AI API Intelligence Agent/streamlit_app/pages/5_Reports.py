from __future__ import annotations

from pathlib import Path

import streamlit as st

st.title('Reports')
report_dir = Path('artifacts/reports')
report_dir.mkdir(parents=True, exist_ok=True)
files = sorted(report_dir.glob('*'))

if not files:
    st.info('No reports available yet.')
else:
    selected = st.selectbox('Select report', files, format_func=lambda p: p.name)
    content = selected.read_text(errors='ignore') if selected.suffix in {'.md', '.html', '.json', '.csv'} else '(binary file)'
    st.code(content[:8000])
    st.download_button('Download', selected.read_bytes(), file_name=selected.name)
