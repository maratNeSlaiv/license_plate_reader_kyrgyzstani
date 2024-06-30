from uuid import UUID
from threading import Thread

import cv2

from custom_interface.camera.exceptions import NotGettingFrame
from custom_interface.core.settinngs import settings
from custom_interface.core.exceptions import CameraConnectionError


class VideoScreenshot(object):
    def __init__(self, src: str):
        self.capture = cv2.VideoCapture(src)
        if not self.capture.isOpened():
            raise CameraConnectionError('asdfsdfs')
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
    

    # def get_frame(self) -> np.array:
    #     def get_frame_thread():
    #         try:
    #             return self.frame
    #         except AttributeError:
    #             pass
    #     return Thread(target=get_frame_thread).start()

    def get_frame(self):
        if hasattr(self, 'frame') and self.frame is None:
            raise NotGettingFrame
        return self.frame
    
    def capture_close(self):
        if self.capture.isOpened():
            print('Delete capture')
            self.should_stop = True
            self.thread.join()
            self.capture.release()
            print('===== Deletet capture =====')


if __name__ == '__main__':
    import time
    from datetime import datetime
    counter = 0
    cap = VideoScreenshot('rtsp://admin:am23hKmc@192.168.88.13/Streaming/Channels/1')
    time.sleep(1)
    while True:
        # if hasattr(cap, 'status'):
        #     if not cap.status:
        #         cap = VideoScreenshot('rtsp://admin:am23hKmc@192.168.88.13/Streaming/Channels/1')
        #         time.sleep(1)
        #         print('A new connection')
        #         continue
        # elif not hasattr(cap, 'status'):
        #     cap = VideoScreenshot('rtsp://admin:am23hKmc@192.168.88.13/Streaming/Channels/1')
        #     time.sleep(1)
        #     print('A new connection by not hasattr')
        #     continue 
        # time.sleep(3)
        if cap.status:
            cap.capture_close()
            cap = VideoScreenshot('rtsp://admin:am23hKmc@192.168.88.13/Streaming/Channels/1')
            time.sleep(1)
        print(type(cap.frame), counter, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cap.status)
        counter += 1