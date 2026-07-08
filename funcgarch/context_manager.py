"""Context manager for hot-reloading modules in Jupyter notebooks."""

import importlib
import logging
import sys

_logger = logging.getLogger(__name__)


class ImportContextManager:
    """Import a module inside a with-block and unload it on exit.

    Useful in notebooks where you want a fresh import without restarting
    the kernel while preserving other global variables.

    Example::

        with ImportContextManager('funcgarch.garch') as garch:
            result = garch.fit(mY, ...)
    """

    def __init__(
        self,
        module_name: str,
        init_fn=None,
        cleanup_fn=None,
    ):
        self.module_name = module_name
        self.module = None
        self.init_fn = init_fn
        self.cleanup_fn = cleanup_fn

    def __enter__(self):
        try:
            _logger.info(f'Importing {self.module_name}')
            self.module = importlib.import_module(self.module_name)
            if self.init_fn:
                _logger.info(f'Running init for {self.module_name}')
                self.init_fn(self.module)
            return self.module
        except ImportError as exc:
            _logger.error(f'Failed to import {self.module_name}: {exc}')
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_fn:
            try:
                _logger.info(f'Running cleanup for {self.module_name}')
                self.cleanup_fn(self.module)
            except Exception as exc:
                _logger.error(f'Cleanup error for {self.module_name}: {exc}')
                raise
        if self.module_name in sys.modules:
            _logger.info(f'Removing {self.module_name} from sys.modules')
            del sys.modules[self.module_name]
        if exc_type:
            _logger.error(f'Exception: {exc_type}, {exc_val}')
