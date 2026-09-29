"""Runtime compatibility shim for the Python-3.7 CVCI environment."""

try:
    import torch

    if not hasattr(torch, "compiler"):
        class _CompilerCompat(object):
            @staticmethod
            def disable(*args, **kwargs):
                def decorate(function):
                    return function
                return decorate

        torch.compiler = _CompilerCompat()
except Exception:
    pass
