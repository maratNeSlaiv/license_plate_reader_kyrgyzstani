from typing import Any, Dict, Optional, List, Union

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
import cv2
from PIL import Image
import numpy as np
import torch


class custom_pipeline_v2(Pipeline, CompositePipeline):
    
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
                 **kwargs):
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

        self.pipelines = [
            self.number_plate_localization,
            self.number_plate_key_points_detection
        ]

        if self.number_plate_classification is not None:
            self.pipelines.append(self.number_plate_classification)
        Pipeline.__init__(self, task, image_loader, **kwargs)
        CompositePipeline.__init__(self, self.pipelines)

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        return images

    def forward(self, inputs: Any, **forward_parameters: Dict):
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
            
            if(count_lines == 2):
                upper_symbols, lower_symbols = self.quadratic_detection(zones)
                return upper_symbols, lower_symbols
            else:
                region, symbols = self.rectangular_detection(zones)
                return region, symbols

        return "classificator is not specified", "please check"

    def remake_image(self, image: Any):
        height, width = image.shape[:2]
        midpoint = height // 2
        upper_half = image[:midpoint, :]
        lower_half = image[midpoint:, :]
        cv2.imwrite('/Users/adilet/nomeroff-net/media/test_case/trash_of_mine_night_upper_half_1.jpg', upper_half)
        cv2.imwrite('/Users/adilet/nomeroff-net/media/test_case/trash_of_mine_night_lower_half_2.jpg', lower_half)
        region_and_numbers = 0 
        bottom_characters = 0
        return region_and_numbers, bottom_characters


    def rectangular_detection(self, zones: Any, **forward_parameters: Dict):
        
        image = zones[0]

        height, width = image.shape[:2]
        region_and_numberplate_divider_point = width * (70/260)
        left_half = Image.fromarray( image[:, :region_and_numberplate_divider_point] )
        right_half = Image.fromarray( image[:, region_and_numberplate_divider_point:] ) 
        
        # Нужно вынести за функцию чтобы один раз загружать.
        parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
        region = OCR_parseq(parseq, left_half)
        symbols = OCR_parseq(parseq, right_half)
        return region, symbols

    def quadratic_detection(self, zones: Any, **forward_parameters: Dict):
            
        image = zones[0]
        height, width = image.shape[:2]
        midpoint = height // 2
        upper_half = image[:midpoint, :]
        lower_half = image[midpoint:, :] 
        
        cv2.imwrite('trash_of_mine_night_upper_half_1.jpg', upper_half)
        cv2.imwrite('trash_of_mine_night_lower_half_2.jpg', lower_half)
        

        # Нужно вынести за функцию чтобы один раз загружать.
        # parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
        # region_and_numbers = OCR_parseq(parseq, upper_half)
        # bottom_characters = OCR_parseq(parseq, lower_half)
        region_and_numbers = 0 
        bottom_characters = 0
        return region_and_numbers, bottom_characters
            
            # cv2.imwrite('trash_of_mine_night_upper_half.jpg', upper_half)
            # cv2.imwrite('trash_of_mine_night_lower_half.jpg', lower_half)

if __name__ == '__main__':
    
    frame = cv2.imread('/Users/maratorozaliev/Desktop/image_5_.jpg')
    pipline = custom_pipeline_v2(image_loader='cv2')
    upper_symbols, lower_symbols = pipline.forward(frame)
    
    print('''Upper symbols:
          
          ''', upper_symbols,
          '''
          Lower symbols:
          
          ''', lower_symbols)