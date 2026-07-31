import inspect
try:
    from ultralytics.utils.metrics import bbox_iou
    print(inspect.signature(bbox_iou))
except Exception as e:
    print(e)
