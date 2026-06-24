try:
    import swattention
    from ultralytics.nn.backbone.TransNeXt.TransNext_cuda import *
except ImportError:
    try:
        from ultralytics.nn.backbone.TransNeXt.TransNext_native import *
    except ImportError:
        # TransNeXt native CUDA extension not compiled — TransNeXt backbone unavailable.
        # Diamond-DETR and RT-DETR do not use TransNeXt, so this is safe to skip.
        # Stub None values so parse_model set literals don't raise NameError.
        transnext_micro = transnext_tiny = transnext_small = transnext_base = None
        AggregatedAttention = get_relative_position_cpb = None