# Compatibility shim: stock ultralytics 8.4+ moved NMS to utils/nms.py;
# fork (8.0.201) keeps it in utils/ops.py — re-export from there.
from ultralytics.utils.ops import non_max_suppression  # noqa: F401
