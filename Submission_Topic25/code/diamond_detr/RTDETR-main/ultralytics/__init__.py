# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.201'

# Lazy imports to break the circular dependency:
#   tasks.py → extra_modules → ultralytics.__init__ → models → engine → tasks.py
# Models are imported on first access via __getattr__ instead of at module load time.

from ultralytics.utils import SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

_lazy = {}


def __getattr__(name):
    if name in ('YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR'):
        if name not in _lazy:
            # Import each model individually to avoid loading broken subpackages (SAM, etc.)
            from ultralytics.models import RTDETR
            _lazy['RTDETR'] = RTDETR
            try:
                from ultralytics.models import YOLO
                _lazy['YOLO'] = YOLO
            except Exception:
                pass
            try:
                from ultralytics.models import SAM
                _lazy['SAM'] = SAM
            except Exception:
                pass
            try:
                from ultralytics.models.fastsam import FastSAM
                _lazy['FastSAM'] = FastSAM
            except Exception:
                pass
            try:
                from ultralytics.models.nas import NAS
                _lazy['NAS'] = NAS
            except Exception:
                pass
        return _lazy.get(name)
    raise AttributeError(f"module 'ultralytics' has no attribute {name!r}")


__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'settings'
