"""Speckle phase retrieval. Submodules are imported on first access."""
from importlib import import_module

__version__ = '0.1'
__all__ = ['metrics', 'plots', 'retrieval', 'supports', 'transforms', 'noise',
           'filter', 'retrievalmath', 'initializers', 'torch', 'utils']


def __getattr__(name):
    if name in __all__:
        module = import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
