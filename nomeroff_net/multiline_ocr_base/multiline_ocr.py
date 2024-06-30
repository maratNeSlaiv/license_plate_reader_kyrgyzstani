# Libraries
from typing import Any
import re
import torch
from PIL import Image
import cv2

# Dependencies
from .parseq_main_file_1 import SceneTextDataModule

class EightParts:
    def __init__(self, upper_left: Image, upper_right: Image, lower_left: Image,
                lower_right: Image, upper: Image, lower: Image, left: Image, right: Image) -> None:
        self.upper_left = upper_left
        self.upper_right = upper_right
        self.lower_left = lower_left
        self.lower_right = lower_right
        self.upper = upper
        self.lower = lower
        self.left = left
        self.right = right

        self.all_eight = (self.upper_left, self.upper_right, self.lower_left, 
                          self.lower_right,self.upper, self.lower, 
                          self.left, self.right)
    
    def __iter__(self):
        return iter(self.all_eight)



class QuadraticMultilineReader:
    def __init__(self, temperature=-1) -> None:

        """
        temperature: Greedy decoding
        """

        self.temperature = temperature
        self.parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
        self.img_transformer = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)

    def parsing_image(self, image: Image) -> EightParts:
            # Метод делящий картинку на 8 частей ( ~ так надо )
            width, height = image.size
            midpoint = height // 2
            upper_image = image.crop((0, 0, width, midpoint))
            lower_image = image.crop((0, midpoint, width, height))

                            
            # Левая и правая картинка 
            width_part1 = width // (67/20)
            width_part2 = width - width_part1  
            box1 = (0,0, width_part1, height)
            box2 = (width_part1, 0, width, height)
            left_half: Image = image.crop(box1)
            right_half: Image = image.crop(box2)

            # Лева и право делим еще пополам каждую
            width, height = left_half.size
            midpoint = height // 2
            upper_left: Image = left_half.crop((0, 0, width, midpoint))
            lower_left: Image = left_half.crop((0, midpoint, width, height))

            width, height = right_half.size
            midpoint = height // 2
            upper_right: Image = right_half.crop((0, 0, width, midpoint))
            lower_right: Image = right_half.crop((0, midpoint, width, height))

            ########
            upper_left.save('trash_of_mine_upper_left.jpg')
            upper_right.save('trash_of_mine_upper_right.jpg')
            lower_right.save('trash_of_mine_lower_right.jpg')
            lower_left.save('trash_of_mine_lower_left.jpg')
            upper_image.save('trash_of_mine_upper_image.jpg')
            lower_image.save('trash_of_mine_lower_image.jpg')
            right_half.save('trash_of_mine_right_half.jpg')
            left_half.save('trash_of_mine_left_half.jpg')
            ########

            return EightParts(upper_left, upper_right, lower_left, lower_right,
                                upper_image, lower_image, left_half, right_half)


    def pre_forward(self, image: cv2.typing.MatLike):
        image = Image.fromarray(image)
        eight_parsed = self.parsing_image(image)
        return eight_parsed

    def reader(self, image: Image, upper_left = False, only_digits = False, 
               only_characters = False, armenian = False) -> str | None:
        img_transformed = self.img_transformer(image).unsqueeze(0)
        logits = self.parseq(img_transformed)
        preds = logits.softmax(self.temperature)
        label, confidence = self.parseq.tokenizer.decode(preds)
        print(label)
        
        # Проверка на кыргызский регион с ошибкой (G вместо 0)
        if upper_left :
            if( bool(len(label[0]) == 2) # длина строки - 2 символа
               and (label[0][0] == 'G' or label[0][0] == '0') # первая цифра "G" или "0"
                and bool(re.match(r'\d', label[0][1])) ): # второй символ - цифра
                return ['0' + label[0][1], ], confidence

        # Ну например мы уверены что все прочтенное должно быть числами
        if only_digits : 
            return self.only_digits_check(label, confidence)
        ####

        #Ну скажем мы знаем что только буквы в изображении
        if only_characters:
            return self.only_characters_check(label, confidence)
        ####

        # Помощь при проверке армянских номеров - у них первые 2 символа - цифры
        # Аккуратней с этой функцией - потому что старые кыргызские квадратные номера
        # которые имеют 4 цифры вначале и 2-3 буквы в конце тоже сюда попадают
        if armenian:
            print('Armenian flag worked !', label)
            label, confidence = self.confidence_check(label, confidence)
            print('After armenian confidence check', label)
            label[0] = self.clean_from_trash(label[0])
            label, confidence = self.armenian_first_two_numbers_correction(label, confidence)
            print('After armenian two numbers correction', label)
            return label, confidence
        ####

        return self.confidence_check(label, confidence)

    def forward(self, image: cv2.typing.MatLike) -> str | None:
        eight_parts = self.pre_forward(image)
        result = self.full_read(eight_parts)
        return self.post_forward(result)  
    
    def only_digits_check(self, label, confidence) -> tuple[str, list]:
        # Метод если все символы - цифры
        ocr_mistakes = {
            'G': '0', # Читает G а правильно - 0
            # 'I': '1' # как пример
        }
        label[0] = ''.join(ocr_mistakes.get(char, char) for char in label[0])
        return label, confidence
    
    def only_characters_check(self, label, confidence) -> tuple[str, list]:
        # Метод если все символы - буквы
        ocr_mistakes = {
                '1': 'I', # читает 1 а правильно I
                '0' : 'O',
                #
            }
        label[0] = ''.join(ocr_mistakes.get(char, char) for char in label[0])
        return label, confidence

    def armenian_first_two_numbers_correction(self, label, confidence) -> tuple[str, list]:
        # В армянских номерах первые две цифры и модель часто путает G и 0.
        # Этот метод исправляет эту погрешность
        if(len(label[0]) > 2):
            first_two_characters = label[0][0:2]
            rest_characters = label[0][2:]
            if len(first_two_characters) > 1:
                if first_two_characters[0] == 'G':
                    first_two_characters = '0' + first_two_characters[1]
                    confidence[0][0] = 0.99 
                    
                if first_two_characters[1] == 'G':
                    first_two_characters = first_two_characters[0] + '0'
                    confidence[0][1] = 0.99
            
            label[0] = first_two_characters + rest_characters    
        return label, confidence
    
    def confidence_check(self, label, confidence) -> tuple[str, list]:
        # Фильтр для проверки насколько модель уверена в своих предсказаниях
        # Если модель неуверена в своих предсказаниях (conf < 0.8) буква не считается
        # Метод помогает отсеивать '-' , '.' , ',' и тп
        result_label = []
        for index, i in enumerate(label[0]):
            
            conf = confidence[0][index].item()

            if(conf > 0.75):
                result_label.append(i)
        
        label = [''.join(result_label)]
        return label, confidence

    def stripe_deletion(self, image : Image) -> Image:
        # У кыргызских номеров есть полоска делящая регион с флагом и цифры с буквами
        # метод убирает полоску с фотки  (а то модель может перепутать за однерку)
        width, height = image.size
        left_crop = width // 10  
        right_crop = width - left_crop  
        image = image.crop((left_crop, 0, right_crop, height))

        return image
    
    def clean_from_trash(self, string) -> str:
        # Чистим все кроме символов и цифр
        if string:
            pattern = re.compile(r'[^A-Z0-9]')
            string = pattern.sub('', string)
        return string
    
    def I_1_7_cleaner(self, text) -> str | None:
        looks_like_slash_list = ['I', '1', '7', '|', '/']
        if text:
            if( (len(text) == 4) and (text[0] in looks_like_slash_list)):
                text = text[1::]
        return text

    def full_read(self, number_plate: EightParts) -> str | None:
        (check_upper_left,), _ = self.reader(number_plate.upper_left, upper_left= True)
                # функция чтобы оставить только символы и цифры
        pattern = re.compile(r'[^a-zA-Z0-9]')
        check_upper_left = pattern.sub('', check_upper_left)

        # Проверка на две цифры в левом верхнем углу (кыргызский номер - регион)
        pattern = re.compile(r'^\d{2}$')
        if pattern.match(check_upper_left):
            print('Chitaem kak kyrgyzskiy', check_upper_left)

            # СВЕРХУ
            upper_text = number_plate.upper_right
            # upper_text.save('marchert_42_before.jpg')
            upper_text = self.stripe_deletion(upper_text)
            # upper_text.save('marchert_42_after.jpg')
            (upper_text,), _ = self.reader(upper_text, only_digits = True)

            print(f'Upper text : {upper_text}')

            # СНИЗУ
            lower_text = number_plate.lower_right
            # lower_text = self.stripe_deletion(lower_text)
            (lower_text,), _ = self.reader(lower_text, only_characters = True)
            print(f'Lower text before : {lower_text}')
            lower_text = self.I_1_7_cleaner(lower_text)
            print(f'Lower text after : {lower_text}')


            return f'{check_upper_left}{upper_text if upper_text else ""}{lower_text if lower_text else ""}'

        # Только одна буква (абхазский номер)
        pattern = re.compile(r'^[A-Z]$')
        if pattern.match(check_upper_left):
            print('Chitaem kak Abhazskiy', check_upper_left)
            (upper_text,), _ = self.reader(number_plate.upper)
            (lower_text,), _ = self.reader(number_plate.lower)
            return f'{upper_text if upper_text else ""}{lower_text if lower_text else ""}'

        # Armenian numbers
        # СВЕРХУ
        print('Chitaem kak armyanskiy', check_upper_left)
        upper_text = number_plate.upper
        (upper_text,), _ = self.reader(upper_text, armenian = True)

        # СНИЗУ
        lower_text = number_plate.lower
        (lower_text,), _ = self.reader(lower_text)
        print('Lower text:', lower_text)
        if lower_text:
            lower_text = self.clean_from_trash(lower_text)
            if lower_text[:2].lower() == 'am' and len(lower_text) >= 5:
                lower_text = lower_text[2:]
    
        return f'{upper_text if upper_text else ""}{lower_text if lower_text else ""}'
        #

    def post_forward(self, lable: str):
        lable = self.clean_from_trash(lable)
        return [(lable,), None] if lable else []


    
    def __call__(self, image: cv2.typing.MatLike) -> Any:
        return self.forward(image)

def abhasian_number_error_checker(string: str) -> bool:
    # 1ая 5ая и 6ая цифры и 2-4 буквы а последние два нуля
    pattern = r'^[A-Z][0-9]{3}[A-Z]{2}00$'
    if len(string) >= 8:
        if re.match(pattern, string):
            return True
    return False
#
#
#
#
#
#
#
    
#
#
#
#
#
#
    
#
#
#
#
#
#
# Use case
if __name__ == '__main__':
    imgs = []
    for i in ('/Users/maratorozaliev/Desktop/image_1_.jpg',):
        img = cv2.imread(i)
        if img is not None:
            imgs.append(img)

    parseq = QuadraticMultilineReader()
    for i in imgs:
        print(parseq(i))

#
#
#
#
#
#
        
#
#
#
#
#
#
        
#
#
#
#
#
#
        
# class RectangularParseqReader:
#     def __init__(self, temperature=-1) -> None:

#         """
#         temperature: Greedy decoding
#         """

#         self.temperature = temperature
#         self.parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
#         self.img_transformer = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)

#     def pre_forward(self, image: cv2.typing.MatLike):
#         image = Image.fromarray(image)
#         return image

#     def reader(self, image: Image) -> str | None:
#         img_transformed = self.img_transformer(image).unsqueeze(0)
#         logits = self.parseq(img_transformed)
#         preds = logits.softmax(self.temperature)
#         label, confidence = self.parseq.tokenizer.decode(preds)
#         print(label[0])
#         return label, confidence

#     def forward(self, image: cv2.typing.MatLike) -> str | None:
#         image = self.pre_forward(image)
#         label, confidence = self.reader(image)
#         label, confidence = self.confidence_check(label, confidence)
#         label[0] = self.post_forward(label[0])
#         return label[0]
    
#     def post_forward(self, label: str):
#         if label:
#             label = self.clean_from_trash(label)
#         return label if label else None
    
#     def confidence_check(self, label, confidence) -> tuple[str, list]:
#         # Фильтр для проверки насколько модель уверена в своих предсказаниях
#         # Если модель неуверена в своих предсказаниях (conf < 0.8) буква не считается
#         # Метод помогает отсеивать '-' , '.' , ',' и тп
#         result_label = []
#         for index, i in enumerate(label[0]):
            
#             conf = confidence[0][index].item()

#             if(conf > 0.6):
#                 result_label.append(i)
        
#         label = [''.join(result_label)]
#         return label, confidence

#     # def stripe_deletion(self, image : Image) -> Image:
#     #     # У кыргызских номеров есть полоска делящая регион с флагом и цифры с буквами
#     #     # метод убирает полоску с фотки  (а то модель может перепутать за однерку)
#     #     width, height = image.size
#     #     new_width = width * 16 // 20  
#     #     left = width - new_width

#     #     upper, right, lower = 0, width, height
#     #     image = image.crop((left, upper, right, lower))
#     #     return image
    
#     def clean_from_trash(self, string) -> str:
#         # Чистим все кроме символов и цифр
#         pattern = re.compile(r'[^a-zA-Z0-9]')
#         res = pattern.sub('', string)
#         return res
    
#     def __call__(self, image: cv2.typing.MatLike) -> Any:
#         print('AM I THERE ?')
#         return self.forward(image)
