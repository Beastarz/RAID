"""
Module: logging_utils
Project: AI Image Detector

Shared logging setup for the training pipeline. Entry-point scripts
(train.py, evaluate.py) call `setup_logger()` once to configure output
format/level; library modules (dataset.py, augmentations.py, datamodule.py)
just use `logging.getLogger(__name__)` and inherit that configuration.
"""

import logging

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logger(name: str = "training", level: str = "INFO") -> logging.Logger:
    """Configures and returns a logger, idempotently.

    Safe to call multiple times (e.g. once per test, or if an entry script's
    main() runs more than once in a process) without duplicating handlers or
    log lines.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
