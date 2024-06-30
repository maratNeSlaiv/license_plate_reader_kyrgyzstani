# import numpy as np
# import cv2

# from nomeroff_net import pipeline
# from nomeroff_net.tools import unzip
# from custom_interface.core.custom_pipline import pipline as custom_pipline

# number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading",
#                                               image_loader="opencv")


# def record_one(path: str) -> tuple[str | None, str | None] | None:
#     (images, images_bboxs,
#      images_points, images_zones, region_ids,
#      region_names, count_lines,
#      confidences, texts) = unzip(number_plate_detection_and_reading([path]))

#     if texts[0] and region_names:
#         return texts[0][0], region_names[0][0]


# def record_by_array(image: np.array):
#     (images, images_bboxs,
#      images_points, images_zones, region_ids,
#      region_names, count_lines,
#      confidences, texts) = unzip(custom_pipline(image))
#     if texts[0] and region_names:
#         return texts[0][0], region_names[0][0]




# def image_loader(img_path):
#     img = cv2.imread(img_path)
#     img = img[..., ::-1]
#     return img


# if __name__ == '__main__':
#     image = image_loader(f'/Users/adilet/nomeroff-net/media/{1}.jpg')
#     print(record_by_array([image]))
    # print(record_one(f'/Users/adilet/nomeroff-net/media/{1}.jpg'))