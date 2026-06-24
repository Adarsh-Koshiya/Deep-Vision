# Ultralytics YOLO 🚀, AGPL-3.0 license
# Lazy imports to break circular dependency:
#   tasks.py → extra_modules → ultralytics.nn (this file) → tasks.py (circular!)
# tasks.py symbols are imported lazily via __getattr__ instead of at module load time.

_tasks_exports = None
_tasks_names = ('attempt_load_one_weight', 'attempt_load_weights', 'parse_model',
                'yaml_model_load', 'guess_model_task', 'guess_model_scale',
                'torch_safe_load', 'DetectionModel', 'SegmentationModel',
                'ClassificationModel', 'BaseModel')


def _load_tasks():
    global _tasks_exports
    if _tasks_exports is None:
        from . import tasks as _tasks
        _tasks_exports = {n: getattr(_tasks, n) for n in _tasks_names if hasattr(_tasks, n)}
    return _tasks_exports


def __getattr__(name):
    if name in _tasks_names:
        return _load_tasks()[name]
    raise AttributeError(f"module 'ultralytics.nn' has no attribute {name!r}")


__all__ = _tasks_names
