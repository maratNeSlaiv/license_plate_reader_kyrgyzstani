from uuid import uuid4, UUID

import aiohttp
import numpy as np

from custom_interface.core.redis import Redis
from custom_interface.core.settinngs import settings
from custom_interface.handle_motion.schemas import CarInFromCamera, MotionDetected, CarOutFromCamera, ResponseFromBackSchema
from custom_interface.handle_motion.enums import CreateCarStatuses
from custom_interface.camera.camera import Camera
from custom_interface.camera.camera import CameraService, ManyCameraService
from custom_interface.core.schemas import Camera
from custom_interface.core.custom_pipline2 import get_text_and_region_one_v2
from custom_interface.core.custom_piplineV3 import CustomPipline
from custom_interface.core.utils import save_frame
from custom_interface.services import adjustment_number_text


class DetectionService:

    def __init__(self, cam_data: Camera):
        self.redis = Redis()
        self.camera_service = CameraService(cam_data)
        self.cam_data = cam_data
        self.pipline = CustomPipline('number_plate_detection_and_reading_runtime', image_loader='cv2')


    async def check_detection_barrier(self, session: aiohttp.ClientSession, motion_detected: MotionDetected):
        barrier_status = await self.redis.check_ready_barrier(motion_detected.pk)
        if not barrier_status:
            print('Barrier is busy: ', self.cam_data.ip_address)
            return
        frame = await self.camera_service.get_frame(motion_detected.ip_address, motion_detected.username, motion_detected.password)
        if frame is None:
            return
        result_of_reading = get_text_and_region_one_v2(self.pipline, frame)
        print(result_of_reading, ' == ', self.cam_data.ip_address)
        if result_of_reading:
            car_number, region = result_of_reading
            if len(car_number) < 5:
                return
            car_number = adjustment_number_text(car_number, region)
            if motion_detected.type_barier == 'for_in':
                await self.request_to_for_in(
                    session,
                    car_number,
                    region,
                    motion_detected,
                    frame,
                )
            elif motion_detected.type_barier == 'for_out':
                await self.request_to_for_out(
                    session,
                    car_number,
                    region,
                    motion_detected,
                    frame,
                )
    
    async def request_to_for_in(
            self,
            session: aiohttp.ClientSession,
            car_number: str,
            region: str,
            motion_detected: MotionDetected,
            frame: np.ndarray,
            ) -> None:
        image_name = uuid4()
        photo_in = f'{image_name}.jpg'
        post_data = CarInFromCamera(
        number_region=region,
        car_number=car_number,
        photo_in=photo_in,
        )
        async with session.post(
                f'{settings.BACK_END_URL}/api/v1/parking/car_in/scaning/{motion_detected.pk}',
                json=dict(post_data),
        ) as back_response:
            if back_response.status == 200:
                save_frame(frame, image_name)
    
    async def request_to_for_out(
            self,
            session: aiohttp.ClientSession,
            car_number: str,
            region: str,
            motion_detected: MotionDetected,
            frame: np.ndarray,
            ) -> None:
        print('Новыая функция')
        image_name = uuid4()
        photo_out = f'{image_name}.jpg'
        post_data = CarOutFromCamera(
        number_region=region,
        car_number=car_number,
        photo_out=photo_out,
        )
        async with session.post(
                f'{settings.BACK_END_URL}/api/v1/parking/car_out/scaning/{motion_detected.pk}',
                json=dict(post_data),
        ) as back_response:
            if back_response.status == 200:
                save_frame(frame, image_name)


class DetectionServiceManyCameras:

    def __init__(self, cameras: list[Camera]):
        self.redis = Redis()
        self.camera_service = ManyCameraService(cameras)
        self.pipline = CustomPipline('number_plate_detection_and_reading_runtime', image_loader='cv2')

    async def check_detection_barrier(self, session: aiohttp.ClientSession, motion_detected: MotionDetected):
        barrier_status = await self.redis.check_ready_barrier(motion_detected.pk)
        if not barrier_status:
            print('Barrier is busy: ', motion_detected.ip_address)
            return
        if motion_detected.type_barier == 'for_in':
            reading_text = await self.reading_text_from_camera(motion_detected)
            if not reading_text:
                return
            result_of_reading, frame = reading_text
            print(result_of_reading, ' == ', motion_detected.ip_address)
            car_number, region = result_of_reading
            await self.request_to_for_in(
                session,
                car_number,
                region,
                motion_detected,
                frame
            )
        elif motion_detected.type_barier == 'for_out':
            frame = await self.camera_service.get_frame(motion_detected.ip_address, motion_detected.username, motion_detected.password)
            if frame is None:
                return
            result_of_reading = get_text_and_region_one_v2(self.pipline, frame)
            print(result_of_reading, ' == ', motion_detected.ip_address)
            if not result_of_reading:
                return
            car_number, region = result_of_reading
            await self.request_to_for_out(
                session,
                car_number,
                region,
                motion_detected,
                frame,
            )
    
    async def request_to_for_in(
            self,
            session: aiohttp.ClientSession,
            car_number: str,
            region: str,
            motion_detected: MotionDetected,
            frame: np.ndarray,
            ) -> None:
        image_name = uuid4()
        photo_in = f'{image_name}.jpg'
        post_data = CarInFromCamera(
        number_region=region,
        car_number=car_number,
        blockpostpoint_id=motion_detected.pk,
        photo_in=photo_in,
        )
        async with session.post(
                f'{settings.BACK_END_URL}/api/v1/parking/car_in/scaning/{motion_detected.pk}',
                json=dict(post_data),
        ) as back_response:
            if back_response.status == 200:
                res = await back_response.json()
                parsed_response = ResponseFromBackSchema(**res)
                if parsed_response.status_code == CreateCarStatuses.car_check_created.value:
                    save_frame(frame, image_name)
    
    async def request_to_for_out(
            self,
            session: aiohttp.ClientSession,
            car_number: str,
            region: str,
            motion_detected: MotionDetected,
            frame: np.ndarray,
            ) -> None:
        image_name = uuid4()
        photo_out = f'{image_name}.jpg'
        post_data = CarOutFromCamera(
        number_region=region,
        car_number=car_number,
        photo_out=photo_out,
        )
        async with session.post(
                f'{settings.BACK_END_URL}/api/v1/parking/car_out/scaning/{motion_detected.pk}',
                json=dict(post_data),
        ) as back_response:
            if back_response.status == 200:
                res = await back_response.json()
                parsed_response = ResponseFromBackSchema(**res)
                if parsed_response.status_code == CreateCarStatuses.car_success_out.value:
                    save_frame(frame, image_name)

    async def reading_text_from_camera(self, motion_detected: MotionDetected) -> tuple[str, np.ndarray] | None:
        total_result = {}
        last_frame = None
        for i in range(4):
            frame = await self.camera_service.get_frame(motion_detected.ip_address, motion_detected.username, motion_detected.password)
            if frame is None:
                return None
            last_frame = frame
            current_result = get_text_and_region_one_v2(self.pipline, frame)
            # print('current_result', current_result)
            if current_result and len(current_result[0]) > 5:
                total_result[current_result] = total_result.setdefault(current_result, 0) + 1
                
        for number, counter in total_result.items():
            if counter >= 3:
                return number, last_frame


if __name__ == '__main__':
    import asyncio
    service = DetectionServiceManyCameras(list)

    asyncio.run(service.check_detection_barrier(list, MotionDetected(motion_detected=True,
                                                         pk='qwer',
                                                         ip_address='192.168.1.1',
                                                         type_barier='for_in',
                                                         username='admin',
                                                         password='admin'
                                                         )))
