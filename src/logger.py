import logging

def load_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler("logs.txt"),
        ]
    )
    return logging.getLogger(__name__)

logger = load_logger()