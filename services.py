# -*- coding: utf-8 -*-
"""
Services
--------

`Provides`

# Config utility
# Verbose logging, for unit testing and applications

Implementation note
-------------------

Using the following conventions for yielding "singeltone" loggers:
    # "checkcap" - reserved name for package Logger
    # allowing a single handler of each type
"""
import os as _os
import time as _time
import json as _json
import logging as _logging
import tempfile as _tempfile

try:
    import IPython as _IPython
    _IPython.get_ipython().Completer.limit_to__all__
except:
    _logging.info('failed to limit ipython autocomplete')

_package_logger = "checkcap"


def package_logger(level='debug'):
    """Yields a "checkcap" logger, for verbose handlers"""

    if _package_logger not in _logging.Logger.manager.loggerDict.keys():
        logger = _logging.getLogger(_package_logger)
        logger.propagate = False
#        logger.addHandler(_logging.NullHandler())

    set_level(level)


def set_level(level):

    logger = _logging.getLogger(_package_logger)

    if 'debug' == level.lower():
        logger.setLevel(_logging.DEBUG)
    elif 'info' == level.lower():
        logger.setLevel(_logging.INFO)
    elif 'warning' == level.lower() or 'warn' == level.lower():
        logger.setLevel(_logging.WARNING)
    elif 'error' == level.lower():
        logger.setLevel(_logging.ERROR)
    elif 'critical' == level.lower():
        logger.setLevel(_logging.CRITICAL)
    elif 'disable' == level.lower():
        logger.setLevel(_logging.CRITICAL + 1)
    else:
        try:
            logger.setLevel(level)
        except:
            logger.warning('unsuported level "{}"'.format(level))


def stream_log():
    """Attaches a verbose stream handler to "checkcap" Logger"""

    package_logger()
    logger = _logging.getLogger(_package_logger)

    for k in logger.handlers:
        if isinstance(k, _logging.StreamHandler):
            return

    handler = _logging.StreamHandler()
    handler.setLevel(_logging.DEBUG)

    msgfmt = _logging.Formatter(': '.join((
        '%(levelname)s', '%(module)s', 'line %(lineno)d', '%(message)s')))

    handler.setFormatter(msgfmt)

    logger.addHandler(handler)

    logger.info('created verbose stream logger')


def logfile_setup(folder=None):
    """Set up log file handler

    Intented use: production, "main" functions

    Input
    -----

        folder: full directory path

            if None, use temporary directory
    """

    package_logger()
    logger = _logging.getLogger(_package_logger)

    for k in logger.handlers:
        if isinstance(k, _logging.FileHandler):
            return logger

    if folder is not None:
        folder = _os.path.normpath(folder)
    else:
        folder = _tempfile.gettempdir()

    file_handler = _logging.FileHandler(folder + '\\log python {}.txt'.format(
        _time.strftime('%Y %b %d %Hh %Mm %Ssec', _time.gmtime())))
    file_handler.setLevel(_logging.DEBUG)

    file_form = _logging.Formatter(
        '\t'.join((
            '%(levelname)s', '%(asctime)s',
            '%(module)s', 'line %(lineno)d', '%(message)s')))

    file_handler.setFormatter(file_form)

    logger.addHandler(file_handler)

    message = 'log file created in {}'.format(folder)
    logger.info(message)
    print(message, '\n')


def logging_shutdown():
    """free log file resources"""
    _logging.shutdown()

    logger = _logging.getLogger(_package_logger)
    for k_hand in logger.handlers:
        if isinstance(k_hand, _logging.FileHandler):
            k_hand.close()
            logger.removeHandler(k_hand)


def get_config(key):
    """Read params from package config file"""
    try:
        fid = open('../Apps/config.txt')
    except:
        try:
            fid = open('Apps/config.txt')
        except:
            fid = open('config.txt')

    config = _json.load(fid)
    fid.close()

    if key not in config:
        raise Exception(
            'failed to find "{}" params in config file'.format(key))

    return config[key]
