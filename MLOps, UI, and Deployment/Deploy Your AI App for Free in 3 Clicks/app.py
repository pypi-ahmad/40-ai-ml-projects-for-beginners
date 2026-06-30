"""
Root entry point for Streamlit Cloud deployment.

Streamlit Cloud expects app.py at repository root.
This delegates to the real application in streamlit_app/.

Usage:  streamlit run app.py
"""

from streamlit_app.app import main

main()
