from uuid import UUID
from threading import Thread

import cv2

from custom_interface.camera.exceptions import NotGettingFrame
from custom_interface.core.settinngs import settings


class VideoScreenshot(object):
    def __init__(self, src: str):
        self.capture = cv2.VideoCapture(src)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))
        # Start the thread to read frames from the video stream
        self.should_stop = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while not self.should_stop:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()


    def save_frame(self, image_name: UUID):
        def save_frame_thread(image_name: UUID):
            try:
                file_uri = f'{settings.MEDIA_FOLDER}/{image_name}.jpg'
                cv2.imwrite(file_uri, self.frame)
                return file_uri
            except AttributeError:
                pass
        return Thread(target=save_frame_thread, args=(image_name,)).start()

    def is_available(self) -> bool:
        if hasattr(self, 'status') and self.status:
            return True
        return False


    def get_frame(self):
        if not self.is_available():
            raise NotGettingFrame
        return self.frame
    
    def capture_close(self):
        print('Delete capture')
        self.should_stop = True
        self.thread.join()
        self.capture.release()
        print('===== Deletet capture =====')