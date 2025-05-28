import os
import threading
from api.api import serve


def run_streamlit() -> None:
    """
    Run the api on a different thread and also run the streamlit app
    """
    api_thread = threading.Thread(target=serve)
    api_thread.start()
    os.system("streamlit run /aml/streamlit_app/streamlit.py --server.port=8080")
