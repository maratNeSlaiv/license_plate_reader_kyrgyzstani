import csv
import cv2
from datetime import datetime


from interface.core.settinngs import settings
from interface.core.custom_piplineV3 import get_text_and_region_one, CustomPipline
from interface.core.custom_pipline2 import get_text_and_region_one as get_text_and_region_one_1, CustomPipline as CustomPipline1

IMAGE_DATA = 'dataset/images'
TEST_DATASET = 'dataset/dataset'
TEST_RESULT = 'dataset/result'

DATA_SET_FILE = 'dataset_file_4.csv'


def get_dataset_as_dict(csv_file_path: str):
    result = {}
    with open(csv_file_path, 'r') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row, in spamreader:
            if len(row) < 40:
                continue
            file_name, car_number = row.split(',')
            result[file_name.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")] = car_number.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
    return result


def test_new_any_number_pipline(file_name):
    dict_dataset = get_dataset_as_dict(f'{settings.BASE_DIR / TEST_DATASET}/{file_name}')
    pipline = CustomPipline('number_plate_detection_and_reading_runtime', image_loader='cv2')
    result = {}
    for file_name in dict_dataset.keys():
        frame = cv2.imread(f'{settings.BASE_DIR / IMAGE_DATA}/{file_name}.jpg')
        if frame is None:
            print(f'{settings.BASE_DIR / IMAGE_DATA}/{file_name}.jpg')
        pipline_res = get_text_and_region_one(pipline, frame)
        result[file_name] = pipline_res[0] if pipline_res else ''
    return result


def test_old_any_number_pipline(file_name):
    dict_dataset = get_dataset_as_dict(f'{settings.BASE_DIR / TEST_DATASET}/{file_name}')
    pipline = CustomPipline1('number_plate_detection_and_reading_runtime', image_loader='cv2')
    result = {}
    for file_name in dict_dataset.keys():
        frame = cv2.imread(f'{settings.BASE_DIR / IMAGE_DATA}/{file_name}.jpg')
        if frame is None:
            print(f'{settings.BASE_DIR / IMAGE_DATA}/{file_name}.jpg')
        if frame is None:
            print(f'{settings.BASE_DIR / IMAGE_DATA}/{file_name}.jpg')
        pipline_res = get_text_and_region_one(pipline, frame)
        result[file_name] = pipline_res[0] if pipline_res else ''
    return result


def get_result_new_pipline_test():
    result = test_new_any_number_pipline(DATA_SET_FILE)
    with open(f'{settings.BASE_DIR / TEST_RESULT}/{test_new_any_number_pipline.__name__}_result_{datetime.now().strftime("%Y-%m-%d %H:%S")}.csv', 'x') as f:
        writer = csv.writer(f)
        for row in result.items():
            writer.writerow(row)


def get_result_old_pipline_test():
    result = test_old_any_number_pipline(DATA_SET_FILE)
    with open(f'{settings.BASE_DIR / TEST_RESULT}/{test_old_any_number_pipline.__name__}_result_{datetime.now().strftime("%Y-%m-%d %H:%S")}.csv', 'x') as f:
        writer = csv.writer(f)
        if writer is None:
            print()
        for row in result.items():
            writer.writerow(row)


def test_pipline_performens():
    def get_test_dataset_as_dict(csv_file_path: str):
        result = {}
        image_count = 0
        with open(csv_file_path, 'r') as csv_file:
            spamreader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            for row, in spamreader:
                if image_count > 1000:
                    return result
                if len(row) < 40:
                    continue
                car_number, file_name = row.split(',')
                result[file_name.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")] = car_number.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
                image_count += 1
        return result
    
    dataset = get_test_dataset_as_dict(f'{settings.BASE_DIR / TEST_DATASET}/{DATA_SET_FILE}')
    old_pipline = CustomPipline1('number_plate_detection_and_reading_runtime', image_loader='cv2')
    new_pipline = CustomPipline('number_plate_detection_and_reading_runtime', image_loader='cv2')
    import time
    new_pipline_time = 0.0
    for file_name in dataset.keys():
        frame = cv2.imread(f'{settings.BASE_DIR / IMAGE_DATA}/{file_name}')
        start_time = time.time()
        get_text_and_region_one(new_pipline, frame)
        new_pipline_time += time.time() - start_time
    new_pipline_time = new_pipline_time / len(dataset.keys())

    old_pipline_time = 0.0
    for file_name in dataset.keys():
        frame = cv2.imread(f'{settings.BASE_DIR / IMAGE_DATA}/{file_name}')
        start_time = time.time()
        get_text_and_region_one(old_pipline, frame)
        old_pipline_time += time.time() - start_time
    old_pipline_time = old_pipline_time / len(dataset.keys())
    print('среднее время нового пайплайна: ', new_pipline_time, ' кол-во фото: ', len(dataset.keys()))
    print('среднее время старого пайплайна: ', old_pipline_time, ' кол-во фото: ', len(dataset.keys()))


if __name__ == '__main__':
    get_result_old_pipline_test()
