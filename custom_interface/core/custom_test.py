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
import cv2
from PIL import Image
from nomeroff_net.pipelines.parseq_main_file import OCR_parseq




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

        # cv2.imwrite('/Users/maratorozaliev/Desktop/trash_of_mina_1.jpg' , zones[0])

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
            
            if count_lines[0] == 2:
                
                # SQUARE NUMBER PLATE RECOGNITION

                image = zones[0]
                image = Image.fromarray(image)

                print(image, type(image))

                # Code to divide region vs digits_and_characters

                width, height = image.size
                width_part1 = width // 3
                width_part2 = width - width_part1
                
                # box1 = (0,0, width_part1, height)
                box1 = (0,0, width_part1, height//2)

                box2 = (width_part1, 0, width, height)

                region = image.crop(box1)
                numbers_and_digits = image.crop(box2)



                # Code to divide upper half and lower half
                width, height = numbers_and_digits.size
                midpoint = height // 2
                top_half = numbers_and_digits.crop((0, 0, width, midpoint))
                bottom_half = numbers_and_digits.crop((0, midpoint, width, height))


                # Checking cropped images                
                region.save('trash_1_region.jpg')
                top_half.save('trash_1_top_half.jpg')
                bottom_half.save('trash_1_bottom_half.jpg')


                texts = [[
                    str(OCR_parseq(region)) + " <- region " + str(OCR_parseq(top_half)) + '  |||||   ' + str(OCR_parseq(bottom_half)) + "-> Not my output ->",
                ]]
                
                print('THE WORK OF THE OCR_PARSEQ MODEL')
                
                return [images, images_bboxs,
                      images_points, zones,
                      region_ids, region_names,
                      count_lines, confidences, texts]

            (region_ids, region_names, count_lines,
            confidences, predicted, preprocessed_np) = unzip(self.number_plate_classification(zones,
                                                                                               **forward_parameters))
        print('''ITS THE WORK OF RECTANGULAR MODEL
              
              
              ''')
        return self.rectangular_recognition(region_ids, region_names,
                                           count_lines, confidences,
                                           zones, image_ids,
                                           images_bboxs, images,
                                           images_points, preprocessed_np, **forward_parameters)
    
    def quadratic_recognition(self, image: Any):
        # Put the OCR_parseq recognition code here, when all tests are passed
        return
    

    def rectangular_recognition(self, region_ids, region_names,
                               count_lines, confidences,
                               zones, image_ids,
                               images_bboxs, images,
                               images_points, preprocessed_np, **forward_parameters):

        number_plate_text_reading_res = unzip(
            self.number_plate_text_reading(unzip([zones,
                                                  region_names,
                                                  count_lines, preprocessed_np]), **forward_parameters))

        if len(number_plate_text_reading_res):
            texts, _ = number_plate_text_reading_res
        else:
            texts = []
        (region_ids, region_names, count_lines, confidences, texts, zones) = \
            group_by_image_ids(image_ids, (region_ids, region_names, count_lines, confidences, texts, zones))
        return [images, images_bboxs,
                      images_points, zones,
                      region_ids, region_names,
                      count_lines, confidences, texts]

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

def get_text_and_region(pipline: CustomPipline, frame: np.array) -> tuple[str, str] | None:
    (images, images_bboxs, 
    images_points, images_zones, region_ids, 
    region_names, count_lines, 
    confidences, texts) = pipline([frame])

    print('''WE GOT THERE MAAAN!
          
          ''')

    if texts[0] and region_names:
        return texts[0][0], region_names[0][0], images_zones[0][0]
    return


if __name__ == '__main__':
    pipline = CustomPipline('number_plate_detection_and_reading_runtime', image_loader='cv2')
    frame = cv2.imread('/Users/adilet/nomeroff-net/media/q_images/39GBAM098.jpeg')
    res = get_text_and_region(pipline, frame)

    # cv2.imwrite('/Users/adilet/nomeroff-net/media/2_res.jpg', res[2])
    print(res[0], res[1])

    # print(res)