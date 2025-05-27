import os


def run_streamlit():
    os.system("streamlit run /aml/streamlit_app/welcome.py --server.port=8080")
