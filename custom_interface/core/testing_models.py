"""number plate detection and reading pipeline

Examples:
    >>> from nomeroff_net import pipeline
    >>> from nomeroff_net.tools import unzip
    >>> number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv")
    >>> results = number_plate_detection_and_reading(['./data/examples/oneline_images/example1.jpeg', './data/examples/oneline_images/example2.jpeg'])
    >>> (images, images_bboxs, images_points, images_zones, region_ids,region_names, count_lines, confidences, texts) = unzip(results)
    >>> print(texts)
    (['AC4921CB'], ['RP70012', 'JJF509'])
"""
from typing import Any, Dict, Optional, List, Union

import numpy as np

from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.tools import unzip
#from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector
from nomeroff_net.pipes.number_plate_localizators.yolo_v8_detector import Detector

from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline, CompositePipeline, empty_method
from nomeroff_net.pipelines.number_plate_key_points_detection import NumberPlateKeyPointsDetection
from nomeroff_net.pipelines.number_plate_text_reading import NumberPlateTextReading
from nomeroff_net.pipelines.number_plate_classification import NumberPlateClassification
from nomeroff_net.tools.image_processing import crop_number_plate_zones_from_images, group_by_image_ids
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_multiline_extractors.multiline_np_extractor \
    import convert_multiline_images_to_one_line, convert_multiline_to_one_line
import cv2

import torch
import numpy as np
from typing import List
from nomeroff_net.pipes.number_plate_localizators.yolo_v5_detector import Detector as YoloDetector
from nomeroff_net.tools.mcm import (modelhub, get_device_torch)


class Detector(YoloDetector):
    """

    """
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self, numberplate_classes=None, yolo_model_type='yolov8') -> None:
        self.model = None
        self.numberplate_classes = ["numberplate"]
        if numberplate_classes is not None:
            self.numberplate_classes = numberplate_classes
        self.device = get_device_torch()
        self.yolo_model_type = yolo_model_type

    def load_model(self, weights: str, device: str = '') -> None:
        from ultralytics import YOLO

        device = device or self.device
        # model = torch.hub.load(repo_path, 'custom', path=weights, source="local")
        model = YOLO(weights)
        model.to(device)
        # if device != 'cpu':  # half precision only supported on CUDA
        #     model.half()  # to FP16
        self.model = model
        self.device = device

    def load(self, path_to_model: str = "latest") -> None:
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name(self.yolo_model_type)
            path_to_model = model_info["path"]
            self.numberplate_classes = model_info.get("classes", self.numberplate_classes)
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model, self.get_classname(), "numberplate_options")
            path_to_model = model_info["path"]
        elif path_to_model.startswith("modelhub://"):
            path_to_model = path_to_model.split("modelhub://")[1]
            model_info = modelhub.download_model_by_name(path_to_model)
            self.numberplate_classes = model_info.get("classes", self.numberplate_classes)
            path_to_model = model_info["path"]
        print('Это модель для того чтобы узнать версию YOLO ==========',path_to_model)
        self.load_model(path_to_model)

    def convert_model_outputs_to_array(self, model_outputs):
        return [self.convert_model_output_to_array(model_output) for model_output in model_outputs]

    @staticmethod
    def convert_model_output_to_array(result):
        model_output = []
        for item, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy()):
            model_output.append([item[0], item[1], item[2], item[3], conf, int(cls)])
        return model_output

    @torch.no_grad()
    def predict(self, imgs: List[np.ndarray], min_accuracy: float = 0.4) -> np.ndarray:
        model_outputs = self.model(imgs, conf=min_accuracy, verbose=False, save=False, save_txt=False, show=False)
        result = self.convert_model_outputs_to_array(model_outputs)
        return np.array(result)



class NumberPlateLocalization(Pipeline):
    """
    Number Plate Localization
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model="latest",
                 detector=None,
                 **kwargs):
        super().__init__(task, image_loader, **kwargs)
        if detector is None:
            detector = Detector
        self.detector = detector()
        self.detector.load(path_to_model)

    def sanitize_parameters(self, img_size=None, stride=None, min_accuracy=None, **kwargs):
        parameters = {}
        postprocess_parameters = {}
        if img_size is not None:
            parameters["img_size"] = img_size
        if stride is not None:
            parameters["stride"] = stride
        if min_accuracy is not None:
            postprocess_parameters["min_accuracy"] = min_accuracy
        return {}, parameters, postprocess_parameters

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        return images

    @no_grad()
    def forward(self, images: Any, **forward_parameters: Dict) -> Any:
        model_outputs = self.detector.predict(images)
        return unzip([model_outputs, images])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        return inputs


class Debub:
    def __init__(self) -> None:
        self.number_plate_localization_class = NumberPlateLocalization("number_plate_localization",
            image_loader=None,
            path_to_model=path_to_model,
            detector=number_plate_localization_detector)