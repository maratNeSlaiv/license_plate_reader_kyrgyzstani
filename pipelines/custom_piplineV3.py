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

from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline, CompositePipeline, empty_method
from nomeroff_net.pipelines.number_plate_localization import NumberPlateLocalization as DefaultNumberPlateLocalization
from nomeroff_net.pipelines.number_plate_key_points_detection import NumberPlateKeyPointsDetection
from nomeroff_net.pipelines.number_plate_text_reading import NumberPlateTextReading
from nomeroff_net.pipelines.number_plate_classification import NumberPlateClassification
from nomeroff_net.tools.image_processing import crop_number_plate_zones_from_images, group_by_image_ids
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_multiline_extractors.multiline_np_extractor \
    import convert_multiline_images_to_one_line, convert_multiline_to_one_line
from nomeroff_net.pipelines.multiline_ocr_pipline import QuadraticMultilineReader, abhasian_number_error_checker
import cv2


class AnyNumberPlateDetectionAndReading(Pipeline, CompositePipeline):
    """
    Number Plate Detection And Reading Class
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model: str = "latest",
                 mtl_model_path: str = "latest",
                 refiner_model_path: str = "latest",
                 path_to_classification_model: str = "latest",
                 presets: Dict = None,
                 off_number_plate_classification: bool = False,
                 classification_options: List = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 number_plate_localization_class: Pipeline = DefaultNumberPlateLocalization,
                 number_plate_localization_detector=None,
                 multiline_ocr: QuadraticMultilineReader = QuadraticMultilineReader(),
                 **kwargs):
        """
        init NumberPlateDetectionAndReading Class
        Args:
            image_loader (): image_loader
            path_to_model (): path_to_model
            mtl_model_path (): mtl_model_path
            refiner_model_path (): refiner_model_path
            path_to_classification_model (): path_to_classification_model
            presets (): presets
            off_number_plate_classification (): off_number_plate_classification
            classification_options (): classification_options
            default_label (): default_label
            default_lines_count (): default_lines_count
            number_plate_localization_class (): number_plate_localization_class
            number_plate_localization_detector (): number_plate_localization_detector

        """
        self.default_label = default_label
        self.default_lines_count = default_lines_count
        self.number_plate_localization = number_plate_localization_class(
            "number_plate_localization",
            image_loader=None,
            path_to_model=path_to_model,
            detector=number_plate_localization_detector
        )
        self.number_plate_key_points_detection = NumberPlateKeyPointsDetection(
            "number_plate_key_points_detection",
            image_loader=None,
            mtl_model_path=mtl_model_path,
            refiner_model_path=refiner_model_path)
        self.number_plate_classification = None
        option_detector_width = 0
        option_detector_height = 0
        if not off_number_plate_classification:
            self.number_plate_classification = NumberPlateClassification(
                "number_plate_classification",
                image_loader=None,
                path_to_model=path_to_classification_model,
                options=classification_options)
            option_detector_width = self.number_plate_classification.detector.width
            option_detector_height = self.number_plate_classification.detector.height
        self.number_plate_text_reading = NumberPlateTextReading(
            "number_plate_text_reading",
            image_loader=None,
            presets=presets,
            option_detector_width=option_detector_width,
            option_detector_height=option_detector_height,
            default_label=default_label,
            default_lines_count=default_lines_count,
            off_number_plate_classification=off_number_plate_classification,
        )
        self.pipelines = [
            self.number_plate_localization,
            self.number_plate_key_points_detection,
            self.number_plate_text_reading,
        ]
        if self.number_plate_classification is not None:
            self.pipelines.append(self.number_plate_classification)
        self.multiline_ocr = multiline_ocr
        Pipeline.__init__(self, task, image_loader, **kwargs)
        CompositePipeline.__init__(self, self.pipelines)

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        return images

    def forward_detection_np(self, inputs: Any, **forward_parameters: Dict):
        images_bboxs, images = unzip(self.number_plate_localization(inputs, **forward_parameters))
        images_points, images_mline_boxes = unzip(self.number_plate_key_points_detection(unzip([images, images_bboxs]),
                                                                                         **forward_parameters))
        zones, image_ids = crop_number_plate_zones_from_images(images, images_points)
        if self.number_plate_classification is None or not len(zones):
            region_ids = [-1 for _ in zones]
            region_names = [self.default_label for _ in zones]
            count_lines = [self.default_lines_count for _ in zones]
            confidences = [-1 for _ in zones]
            predicted = [-1 for _ in zones]
            preprocessed_np = [None for _ in zones]
        else:
            (region_ids, region_names, count_lines,
             confidences, predicted, preprocessed_np) = unzip(self.number_plate_classification(zones,
                                                                                               **forward_parameters))
            region_ids, region_names, count_lines, confidences, predicted, preprocessed_np = [region_ids[0]], [region_names[0]], [count_lines[0]], [confidences[0]], [predicted[0]], [preprocessed_np[0]]

        return (region_ids, region_names, count_lines, confidences,
                predicted, zones, image_ids, images_bboxs, images,
                images_points, images_mline_boxes, preprocessed_np)

    def forward_recognition_np(self, region_ids, region_names,
                               count_lines, confidences,
                               zones, image_ids,
                               images_bboxs, images,
                               images_points, preprocessed_np, **forward_parameters):
        if count_lines and count_lines[0] == 2:
            number_plate_text_reading_res = self.multiline_ocr(zones[0])
        if count_lines and count_lines[0] != 2 or not count_lines:
            number_plate_text_reading_res = unzip(
                self.number_plate_text_reading(unzip([zones,
                                                    region_names,
                                                    count_lines, preprocessed_np]), **forward_parameters))
        # print('number_plate_text_reading_res ==== ',number_plate_text_reading_res)
        if len(number_plate_text_reading_res):
            texts, _ = number_plate_text_reading_res
        else:
            texts = []
        region_ids, region_names, count_lines, confidences, texts, zones = region_ids, region_names, count_lines, confidences, texts, zones
        (region_ids, region_names, count_lines, confidences, texts, zones) = \
            group_by_image_ids(image_ids, (region_ids, region_names, count_lines, confidences, texts, zones))
        return unzip([images, images_bboxs,
                      images_points, zones,
                      region_ids, region_names,
                      count_lines, confidences, texts])

    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        """
        TODO: split into two methods so that there is no duplication of code
        """
        (region_ids, region_names,
         count_lines, confidences, predicted,
         zones, image_ids,
         images_bboxs, images,
         images_points, images_mline_boxes, preprocessed_np) = self.forward_detection_np(inputs, **forward_parameters)
        return self.forward_recognition_np(region_ids, region_names,
                                           count_lines, confidences,
                                           zones, image_ids,
                                           images_bboxs, images,
                                           images_points, preprocessed_np, **forward_parameters)

    @empty_method
    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        return inputs


class CustomPipline(AnyNumberPlateDetectionAndReading):

    def custom_transform_image(self, image):
        image = image[..., ::-1]
        return image
    
    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:       
        images = [self.custom_transform_image(item) for item in inputs]
        return images


def transform_image(image: np.array) -> np.array:
    image = image[..., ::-1]
    return image


def get_text_and_region_one(pipline: CustomPipline, frame: np.array) -> tuple[str, str] | None:
    (images, images_bboxs, 
    images_points, images_zones, region_ids, 
    region_names, count_lines, 
    confidences, texts) = unzip(pipline([frame]))

    if abhasian_number_error_checker(texts[0][0]):
        texts[0][0] = texts[0][0][:6]

    if texts[0] and region_names:
        return texts[0][0], region_names[0][0]
    return


def get_text_and_region_one_v2(pipline: CustomPipline, frame: np.array) -> tuple[str, str] | None:
    (images, images_bboxs, 
    images_points, images_zones, region_ids, 
    region_names, count_lines, 
    confidences, texts) = unzip(pipline([frame]))
    print(texts)
    if texts[0] and region_names:
        return texts[0][0], region_names[0][0]


if __name__ == '__main__':
    pipline = CustomPipline('number_plate_detection_and_reading_runtime', image_loader='cv2')
    frames = [cv2.imread('/Users/maratorozaliev/Desktop/trash_of_marat_12.jpg')]
    import time
    total_result = 0.0
    for frame in frames:
        start_time = time.time()
        res = get_text_and_region_one(pipline, frame)
        total_result += time.time() - start_time
        print(res)
    print('Average time for one image: ', total_result / len(frames))