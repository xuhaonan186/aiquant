import logging
from functools import wraps
from datetime import datetime
# from .config import cached_config

today = datetime.today().strftime("%Y%m%d")
# file_pth = str(cached_config.log_url / f"{today}")
logger = logging.getLogger("logger")

# info_handler = logging.FileHandler(filename="{}_3info.log".format(file_pth), encoding="utf-8")
# warning_handler = logging.FileHandler(filename="{}_3warning.log".format(file_pth), encoding="utf-8")
# error_handler = logging.FileHandler(filename="{}_3error.log".format(file_pth), encoding="utf-8")
stream_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
# info_handler.setFormatter(formatter)
# warning_handler.setFormatter(formatter)
# error_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
# info_handler.setLevel(logging.INFO)
# warning_handler.setLevel(logging.WARNING)
# error_handler.setLevel(logging.ERROR)
stream_handler.setLevel(logging.INFO)
#
# logger.addHandler(info_handler)
# logger.addHandler(warning_handler)
# logger.addHandler(error_handler)
logger.addHandler(stream_handler)


def log_method(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{e.__repr__()} at {e.__traceback__.tb_frame}")
            raise
        return result
    return wrapper
