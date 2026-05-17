"""Project logging helpers."""

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Return a module logger with a consistent console formatter.

    Parameters
    ----------
    name : str
        Logger name, normally ``__name__``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
