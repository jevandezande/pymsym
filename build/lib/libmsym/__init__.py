import os

__all__ = []

_libmsym_install_location = os.path.join('/usr/local/lib','libmsym.0.2.4.dylib')

if len(_libmsym_install_location) == 0 or _libmsym_install_location[0] == '$':
    _libmsym_install_location = None

def export(defn):
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

from . import libmsym

