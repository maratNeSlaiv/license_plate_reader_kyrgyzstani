import time
import asyncio
from uuid import UUID
from random import choice

import numpy as np
import cv2

from custom_interface.camera.exceptions import NotGettingFrame
from custom_interface.core.exceptions import CameraConnectionError
from custom_interface.core.schemas import Camera
from custom_interface.camera.streaming import VideoScreenshot


rtsp_url = 'rtsp://{}:{}@{}/Streaming/Channels/1'

blockposts = [{'ip_camera': '192.168.88.222', 'username': 'admin', 'password': 'am23hKmc'}]


class CameraService:

    def __init__(self, cam_data: Camera) -> None:
        self.camera = self._connect(**dict(cam_data))

    def _connect(self, ip_address: str, username: str, password: str) -> None:
        while True:
            try:
                return VideoScreenshot(rtsp_url.format(username, password, ip_address))
            except CameraConnectionError as es:
                print('Не удается полключиться к камере: ', ip_address, ' | ошибка: ', es)
                time.sleep()
                continue

    def get_connection(self) -> VideoScreenshot:
        return self.camera

    def get_and_screen_from_camera(self, ip_address: str, username: str, password: str, file_name: UUID) -> str:
        cap = self.get_connection()
        return cap.save_frame(file_name)

    async def get_frame(self, ip_address: str, username: str, password: str) -> np.array:
        cap = self.get_connection()
        while True:
            try:
                return cap.get_frame()
            except NotGettingFrame:
                del self.camera
                self.camera = self._connect(ip_address, username, password)
                await asyncio.sleep(0.5)
                cap = self.get_connection()
                return cap.get_frame()
            except Exception as e:
                print(f'Ошибка получения кадра с видеокамеры -- {e} -- {ip_address}')
                asyncio.sleep(2)
                continue


class ManyCameraService:

    def __init__(self, cameras: list[Camera]) -> None:
        self.cap_dict = {}
        for item in cameras:
            dict_item = dict(item)
            try:
                self.cap_dict[item.ip_address] = self._connect(**dict_item)
            except CameraConnectionError as es:
                print(es)


    def _connect(self, ip_address: str, username: str, password: str) -> None:
        try:
            cap = VideoScreenshot(rtsp_url.format(username, password, ip_address))
            time.sleep(2.5)
            if cap.capture.isOpened():
                return cap
            cap.capture_close()
        except CameraConnectionError as es:
            print('Не удается полключиться к камере: ', ip_address, ' | ошибка: ', es)
            time.sleep(1)

    def get_connection(self, ip_address: str, username: str, password: str) -> VideoScreenshot:
        if not self.cap_dict.get(ip_address):
            self.cap_dict[ip_address] = self._connect(ip_address, username, password)
        return self.cap_dict[ip_address]

    async def reconnecting(self, ip_address, username, password) -> bool | VideoScreenshot:
        print('reconnecting to camera: ', ip_address)
        cap = self.get_connection(ip_address, username, password)
        cap.capture_close()
        del self.cap_dict[ip_address]
        try:
            self.cap_dict[ip_address] = self._connect(ip_address, username, password)
            cap = self.get_connection(ip_address, username, password)
            if cap.capture.isOpened():
                print('Переподключение удалась')
                return cap
            print('Переподключение не удалась')
            cap.capture_close()
            del self.cap_dict[ip_address]
        except Exception as e:
            print('Не удается восстанавить соедение с камерой: ', ip_address, ' | ', e)
        return False

    async def get_frame(self, ip_address: str, username: str, password: str) -> np.array:
        cap = self.get_connection(ip_address, username, password)
        try:
            return cap.get_frame()
        except NotGettingFrame:
            cap = await self.reconnecting(ip_address, username, password)
            if cap:
                return cap.get_frame()
        except AttributeError as e:
            print('Не удается подключиться к камере: ', ip_address, e)

    
    def close_all_cameras(self):
        for cap in self.cap_dict.values():
            if cap is not None:
                cap.capture_close()