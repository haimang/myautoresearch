"""Compatibility shim for selector research services."""

from services.research import selector_service as _selector_service

globals().update(
    {
        name: value
        for name, value in _selector_service.__dict__.items()
        if not name.startswith("__")
    }
)
