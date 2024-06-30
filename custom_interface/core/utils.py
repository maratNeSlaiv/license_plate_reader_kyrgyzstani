from threading import Thread
from uuid import UUID

import cv2
from numpy import array

from custom_interface.core.settinngs import settings


def save_frame(frame: array, image_name: UUID):
    def save_frame_thread(frame: array, image_name: UUID):
        try:
            file_uri = f'{settings.MEDIA_FOLDER}/{image_name}.jpg'
            print('file_uri: ',file_uri)
            cv2.imwrite(file_uri, frame)
            return file_uri
        except AttributeError:
            pass
    return Thread(target=save_frame_thread, args=(frame, image_name)).start()
