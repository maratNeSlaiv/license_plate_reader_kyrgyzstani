from torch import no_grad
from typing import Any, Dict, Optional, Union

from ..tools.pipeline import BaseImageLoader
from ..tools.pipeline import Pipeline
from ..tools.pipeline import unzip
from .ocr_and_text_detector import TextDetector
from .ocr_tools import DEFAULT_PRESETS

class NumberPlateTextReading(Pipeline):
    """
    Number Plate Text Reading Pipeline
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 presets: Dict = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 class_detector=TextDetector,
                 option_detector_width=0,
                 option_detector_height=0,
                 off_number_plate_classification=True,
                 **kwargs):
        if presets is None:
            presets = DEFAULT_PRESETS
        super().__init__(task, image_loader, **kwargs)
        self.detector = class_detector(presets, default_label, default_lines_count,
                                       option_detector_width=option_detector_width,
                                       option_detector_height=option_detector_height,
                                       off_number_plate_classification=off_number_plate_classification)

    def sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images, labels, lines, preprocessed_np = unzip(inputs)
        images = [self.image_loader.load(item) for item in images]
        return unzip([images, labels, lines, preprocessed_np])

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        images, labels, lines, preprocessed_np = unzip(inputs)
        preprocessed_np = [zone if pnp is None else pnp for pnp, zone in zip(preprocessed_np, images)]
        model_inputs = self.detector.preprocess(preprocessed_np, labels, lines)
        model_outputs = self.detector.forward(model_inputs)
        model_outputs = self.detector.postprocess(model_outputs)
        return unzip([images, model_outputs, labels])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        images, model_outputs, labels = unzip(inputs)
        return unzip([model_outputs, images])
