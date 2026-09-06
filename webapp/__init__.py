"""
webapp/__init__.py
==================
RetinaGuard Standalone Web Application Package.
Includes runtime compatibility shim for modern Starlette/FastAPI environments.
"""

import starlette.routing

# Compatibility shim: Starlette >= 1.0 removed on_startup/on_shutdown from Router.__init__
# but older FastAPI releases still pass them as kwargs.
_orig_router_init = starlette.routing.Router.__init__

def _compat_router_init(self, *args, on_startup=None, on_shutdown=None, **kwargs):
    _orig_router_init(self, *args, **kwargs)

if "on_startup" not in _orig_router_init.__code__.co_varnames:
    starlette.routing.Router.__init__ = _compat_router_init
