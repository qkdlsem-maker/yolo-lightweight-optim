"""Enforce the prespecified640x640 geometry in Ultralytics8.2.103 prediction.

That version's default pre_transform enables auto letterboxing for equal image
shapes regardless of args.rect. All other inference and NMS behavior is native.
"""
from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect.predict import DetectionPredictor


class FixedSquarePredictor(DetectionPredictor):
    def pre_transform(self, im):
        if tuple(self.imgsz) != (640, 640):
            raise ValueError('Frozen prediction geometry is640x640')
        letterbox = LetterBox(self.imgsz, auto=False, stride=self.model.stride)
        transformed = [letterbox(image=x) for x in im]
        if not all(x.shape[:2] == (640, 640) for x in transformed):
            raise ValueError('Unexpected preprocessed dimensions')
        return transformed
