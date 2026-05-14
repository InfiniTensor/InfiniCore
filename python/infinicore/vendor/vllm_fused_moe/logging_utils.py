# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any


class _OnceLogger:
    def __init__(self, log: logging.Logger):
        self._log = log
        self._seen: set[tuple[str, str]] = set()

    def info_once(self, msg: str, *args: Any, scope: str = "global", **kwargs: Any) -> None:
        key = ("info", scope, msg)
        if key in self._seen:
            return
        self._seen.add(key)
        self._log.info(msg, *args, **kwargs)

    def warning_once(self, msg: str, *args: Any, scope: str = "global", **kwargs: Any) -> None:
        key = ("warn", scope, msg)
        if key in self._seen:
            return
        self._seen.add(key)
        self._log.warning(msg, *args, **kwargs)


def wrap_logger(log: logging.Logger) -> _OnceLogger:
    return _OnceLogger(log)


def init_logger(name: str) -> _OnceLogger:
    return wrap_logger(logging.getLogger(name))
