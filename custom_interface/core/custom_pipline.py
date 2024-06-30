from typing import Any, Dict
import numpy as np

import cv2

from nomeroff_net.pipelines import NumberPlateDetectionAndReading
from nomeroff_net.tools import unzip

class CustomPipline(NumberPlateDetectionAndReading):

    def custom_transform_image(self, image):
        image = image[..., ::-1]
        return image
    
    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:       
        images = [self.custom_transform_image(item) for item in inputs]
        return images


def transform_image(image: np.array) -> np.array:
    image = image[..., ::-1]
    return image


# pipline = CustomPipline('number_plate_detection_and_reading_runtime', image_loader='cv2')

def get_text_and_region_one(pipline: CustomPipline, frame: np.array) -> tuple[str, str] | None:
    (images, images_bboxs, 
    images_points, images_zones, region_ids, 
    region_names, count_lines, 
    confidences, texts) = unzip(pipline([frame]))
    if texts[0] and region_names:
        return texts[0][0], region_names[0][0]
    return


# if __name__ == '__main__':
#     frame = cv2.imread('/Users/adilet/nomeroff-net/media/4.jpg')
#     res = get_text_and_region_one(pipline, frame)
#     # cv2.imwrite('/Users/adilet/nomeroff-net/media/1_res.jpg', res[2])
#     print(res)