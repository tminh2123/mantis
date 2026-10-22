# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import logging
from colorlog import ColoredFormatter
import random
from torch.backends import cudnn


def logger_config(log_path=None):
    """
    Configure a logger with file and console handlers.
    Allows switching between file-only, console-only, or both.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear old handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler (optional)
    file_handler = None
    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode="a", encoding='UTF-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    color_formatter = ColoredFormatter("%(log_color)s%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S",
                                       log_colors={'DEBUG': 'blue',
                                                   'INFO': 'cyan',
                                                   'WARNING': 'yellow',
                                                   'ERROR': 'red',
                                                   'CRITICAL': 'bold_red'
                                                   })
    console_handler.setFormatter(color_formatter)

    # Default: attach both handlers
    if file_handler:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Utility methods to dynamically switch handlers
    def file():
        """Enable file logging only"""
        if file_handler and file_handler not in logger.handlers:
            logger.addHandler(file_handler)
        if console_handler in logger.handlers:
            logger.removeHandler(console_handler)

    def console():
        """Enable console logging only"""
        if console_handler not in logger.handlers:
            logger.addHandler(console_handler)
        if file_handler and file_handler in logger.handlers:
            logger.removeHandler(file_handler)

    def both():
        """Enable both file and console logging"""
        if file_handler and file_handler not in logger.handlers:
            logger.addHandler(file_handler)
        if console_handler not in logger.handlers:
            logger.addHandler(console_handler)

    # Attach methods to logger
    logger.file = file
    logger.console = console
    logger.both = both
    return logger


def save_checkpoint(state, save_path):
    """
    Save model checkpoint.
    Args:
        state (dict): training state, should contain 'epoch', 'best_model', 'model', etc.
        save_path (str): directory to save the checkpoint
    """
    os.makedirs(save_path, exist_ok=True)

    epoch = state['epoch']
    best_model = state['best_model']  # bool flag
    model = state['model']  # model name/type

    if best_model:
        filename = os.path.join(save_path, f'best_model-{model}.pth.tar')
    else:
        filename = os.path.join(save_path, f'model-{model}-{epoch}.pth.tar')

    torch.save(state, filename)


def seed_torch(seed=922, deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

