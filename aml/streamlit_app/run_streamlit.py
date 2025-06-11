import os
import threading
from api.api import serve


def run_streamlit() -> None:
    """
    Run the api on a different thread and also run the streamlit app
    """
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights"), "Please ensure the /weights directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"
    assert os.path.exists("/weights/ViT/126.pth"), "Please ensure that you have the latest weights"
    assert os.path.exists("/weights/forest/124"), "Please ensure that you have the latest weights"

    api_thread = threading.Thread(target=serve)
    api_thread.start()
    os.system("streamlit run /aml/streamlit_app/streamlit.py --server.port=8080")
