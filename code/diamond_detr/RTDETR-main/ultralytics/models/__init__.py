# Ultralytics YOLO 🚀, AGPL-3.0 license
# Lazy imports: SAM/YOLO modules have empty subpackages that would fail eagerly.
# Only load what's requested.

_lazy = {}


def __getattr__(name):
    if name == 'RTDETR':
        if 'RTDETR' not in _lazy:
            from .rtdetr import RTDETR
            _lazy['RTDETR'] = RTDETR
        return _lazy['RTDETR']
    if name == 'YOLO':
        if 'YOLO' not in _lazy:
            from .yolo import YOLO
            _lazy['YOLO'] = YOLO
        return _lazy['YOLO']
    if name == 'SAM':
        if 'SAM' not in _lazy:
            from .sam import SAM
            _lazy['SAM'] = SAM
        return _lazy['SAM']
    raise AttributeError(f"module 'ultralytics.models' has no attribute {name!r}")


__all__ = 'YOLO', 'RTDETR', 'SAM'
