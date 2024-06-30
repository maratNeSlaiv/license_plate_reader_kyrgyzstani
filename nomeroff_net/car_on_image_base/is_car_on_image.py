import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class CarDetection():
    def __init__(self):
        self.model = YOLO("yolov8l.pt")
        # model = YOLOv10('./yolo-models/yolov10l.pt')

    def pre_forward(self, image : np.ndarray) -> np.ndarray:
    # Нужно указать 4 переменные. Каждая означает меру выреза изображения.
    # Например если from_left = 100 и переданный флаг by_pads вырежет 100 пикселей слева
        
    # Например если from_left = 1/5 и переданный флаг by_proportions вырежет 1/5 часть изображения слева
        
    # Если переданный флаг by_coordinates то from_left, from_top это координаты левого верха (x1, y1)
    # А from_right, from_bottom, координаты правого низа (x2, y2)
    # Начальная точка координат левый вверх изображения (0, 0).
         
        # cv2.imwrite('the_crop_before.jpg', image)

        from_left = 0
        from_top = 0
        from_right = 0
        from_bottom = 0
        sides = [from_left, from_top, from_right, from_bottom]
        cropped_image = self.cut_the_image(image = image, by_pads = False, 
                                    by_proportions = False, by_coordinates = False, sides = sides)
        
        # cv2.imwrite('the_crop_after.jpg', cropped_image)
        
        return cropped_image
    
    def cut_the_image(self, image: np.ndarray, sides:list, by_pads = False, by_coordinates = False, 
                      by_proportions = False) -> np.ndarray:
        
        print('''Если указано несколько способов вырезки. Будет произведено то что следует раньше. 
                  Путь такой: by_proportions -> by_coordinates -> by_pads''')

        if(by_proportions):
            print('Вырезка по пропорциям')
            cropped_image = self.cut_the_image_by_proportions(image= image, proportions=sides)
            return cropped_image
        
        if(by_coordinates):
            print('Вырезка по координатам')
            cropped_image = self.cut_the_image_by_coordinates(image= image, coordinates=sides)
            return cropped_image
            
        if(by_pads):
            print('Вырезка по отступам')
            cropped_image = self.cut_the_image_by_pads(image= image, pads= sides)
            return cropped_image

        if not (by_pads or by_coordinates or by_proportions):
            print('Не указан способ вырезки.')
        return image
    
    def cut_the_image_by_coordinates(self, image : np.ndarray, coordinates) -> np.ndarray:
        x1, y1, x2, y2 = coordinates
        """
        Вырезка изображения по координатам (x1, y1) верхнего левого угла и
        (x2, y2) правого нижнего угла используя OpenCV.
        """
        # Ensure the coordinates are within the image bounds
        try:
            cropped_image = image[y1:y2, x1:x2]
        except:
            print('Одна или несколько из указанных координат выходит за грани изображения. Изображение не изменено')
            cropped_image = image
        return cropped_image
    
    def cut_the_image_by_proportions(self, image: np.ndarray, proportions: list) -> np.ndarray:
        """
        Вырезает изображение по указанным пропорциям.
        proportions: список пропорций [from_left, from_top, from_right, from_bottom].
        return: вырезанное изображение.
        """
        from_left, from_top, from_right, from_bottom = proportions
        
        height, width = image.shape[:2]
        
        # Вычисляем координаты для вырезки
        left = int(width * from_left)
        top = int(height * from_top)
        right = int(width * (1 - from_right))
        bottom = int(height * (1 - from_bottom))
        
        # Проверяем, что координаты допустимы
        if left < 0 or top < 0 or right > width or bottom > height or left >= right or top >= bottom:
            print('Пропорции вырезают слишком много. Изображение не изменено')
            return image
        
        # Вырезаем изображение
        cropped_image = image[top:bottom, left:right]
        
        return cropped_image
    
    
    def cut_the_image_by_pads(self, image: np.ndarray, pads:list) -> np.ndarray:
        from_left, from_top, from_right, from_bottom = pads
        height, width = image.shape[:2]

        # Вычисляем новые границы изображения
        left = from_left
        top = from_top
        right = width - from_right
        bottom = height - from_bottom

        # Проверяем корректность границ
        if left < 0 or top < 0 or right > width or bottom > height or left >= right or top >= bottom:
            print('Отступы слишком большие. Изображение не изменено.')
            return image

        # Вырезаем изображение по заданным границам
        cropped_image = image[top:bottom, left:right]

        return cropped_image

    def forward(self, image: np.ndarray):
        answer = self.is_car_in_frame(frame= image)
        return answer
    
    def is_car_in_frame(self, frame):
        results = self.model(frame)[0]     
        labels = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        idx = np.argwhere(labels == 2)
            
        if len(idx) == 0:
            return False
        return (confidences[idx] >= 0.5).any()
    
    def post_forward(self, answer) -> bool:
        return answer

    def __call__(self, image: np.ndarray) -> True | False:
        cropped_image = self.pre_forward(image)
        prediction = self.forward(cropped_image)
        answer = self.post_forward(prediction)
        return answer
    
if __name__ == '__main__':
    car_detection = CarDetection()
    url = '/Users/maratorozaliev/Desktop/trash_of_marat_29.jpg'
    frame = cv2.imread(url)
    result = car_detection(frame)
    print(frame)
    print(f'IS THERE ANY CAR IN IMAGE ???: ANSWER IS {result}')
