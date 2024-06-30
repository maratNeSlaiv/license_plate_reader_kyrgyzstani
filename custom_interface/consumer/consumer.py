import json
import asyncio
import traceback

import aiohttp

from custom_interface.core.settinngs import settings
from custom_interface.core.schemas import Camera
from custom_interface.handle_motion.schemas import MotionDetected
from custom_interface.handle_motion.serivces import DetectionService, DetectionServiceManyCameras

import aio_pika
import aio_pika.abc


class AsynConsumer:
    def __init__(self, cam_data: Camera):
        self.route = cam_data.ip_address
        self.service = DetectionService(cam_data)

    def message_parser(self, message: bytes) -> MotionDetected:
        dict_message = json.loads(message)
        return MotionDetected(**dict_message)

    async def setup(self):
        connection = await aio_pika.connect_robust(
            f"amqp://guest:guest@{settings.RABBIT_HOST}/",
        )
        async with connection:
            queue_name = ""

            # Creating channel
            channel: aio_pika.abc.AbstractChannel = await connection.channel()
            # await channel.declare_exchange('barriers', auto_delete=False)
            # Declaring queue
            queue: aio_pika.abc.AbstractQueue = await channel.declare_queue(
                queue_name,
                auto_delete=True
            )
            await queue.bind(settings.RABBIT_EXCHANGE, routing_key=self.route)
            print('Setup consumer: ', self.route)

            async with aiohttp.ClientSession() as session:
                async with queue.iterator() as queue_iter:
                    # Cancel consuming after __aexit__
                    async for message in queue_iter:
                        async with message.process():
                            detect_message = self.message_parser(message.body)
                            await self.service.check_detection_barrier(session, detect_message)
                            if queue.name in message.body.decode():
                                break


class AsynConsumerV2:
    def __init__(self, cameras: list[Camera]):
        self.cameras = cameras
        self.service = DetectionServiceManyCameras(cameras)

    def message_parser(self, message: bytes) -> MotionDetected:
        dict_message = json.loads(message)
        return MotionDetected(**dict_message)

    async def setup(self):
        connection = await aio_pika.connect_robust(
            f"amqp://guest:guest@{settings.RABBIT_HOST}/",
        )
        async with connection:
            queue_name = ""

            # Creating channel
            channel: aio_pika.abc.AbstractChannel = await connection.channel()
            queue: aio_pika.abc.AbstractQueue = await channel.declare_queue(
                queue_name,
                auto_delete=True
            )
            for camera in self.cameras:
                await queue.bind(settings.RABBIT_EXCHANGE, routing_key=camera.ip_address)
                print('Setup consumer: ', camera.ip_address)

            async with aiohttp.ClientSession() as session:
                async with queue.iterator() as queue_iter:
                    # Cancel consuming after __aexit__
                    async for message in queue_iter:
                        async with message.process():
                            detect_message = self.message_parser(message.body)
                            await self.service.check_detection_barrier(session, detect_message)
                            if queue.name in message.body.decode():
                                break


def run_consumer(cam_data):
    while True:
        try:
            loop = asyncio.get_event_loop()
            consumer = AsynConsumer(cam_data)
            loop.run_until_complete(consumer.setup())
        except Exception as e:
            # Обработка исключения, например, вывод в лог
            print(f"Exception in process for {cam_data.ip_address}: {e}")
            traceback.print_exc()
            continue


def run_consumer_v2(cameras: list[Camera]):
    while True:
        try:
            loop = asyncio.get_event_loop()
            consumer = AsynConsumerV2(cameras)
            loop.run_until_complete(consumer.setup())
        except Exception as e:
            print(f"Exception in process for {cameras}: {e}")
            consumer.service.camera_service.close_all_cameras()
            traceback.print_exc()
            continue


def split_into_subarrays(arr, subarray_length=2):
    return [arr[i:i+subarray_length] for i in range(0, len(arr), subarray_length)]


if __name__ == '__main__':

    import torch.multiprocessing as mp
    from custom_interface.core.db import DataBase

    # db = DataBase()
    # cameras = db.get_cameras()
    # cameras = [Camera(ip_address='192.168.88.13', username='admin', password='am23hKmc')]
    cameras = [
        Camera(ip_address='192.168.96.15', username='admin', password='am23hKmc'),
        Camera(ip_address='192.168.96.12', username='admin', password='am23hKmc'),
        Camera(ip_address='192.168.96.10', username='admin', password='am23hKmc'),
        Camera(ip_address='192.168.96.11', username='admin', password='am23hKmc'),
        Camera(ip_address='192.168.96.16', username='admin', password='am23hKmc'),
        ]

    cameras_for_prosseses = split_into_subarrays(cameras)
    # run_consumer_v2(cameras)
    mp.set_start_method('spawn')
    processes = []
    for cameras_to_prosses in cameras_for_prosseses:
        print('Запускаются камеры: ', cameras_to_prosses)
        process = mp.Process(target=run_consumer_v2, args=(cameras_to_prosses,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

# if __name__ == '__main__':

#     import torch.multiprocessing as mp
#     from custom_interface.core.db import DataBase

#     db = DataBase()
#     cameras = db.get_cameras()
#     # cameras = [Camera(ip_address='192.168.88.13', username='admin', password='am23hKmc'), Camera(ip_address='192.168.88.13', username='admin', password='am23hKmc')]

#     mp.set_start_method('spawn')
#     processes = []
#     for camera in cameras:
#         process = mp.Process(target=run_consumer, args=(camera,))
#         processes.append(process)
#         process.start()

#     for process in processes:
#         process.join()

#     # with Pool(processes=len(cameras)) as pool:
#     #     pool.map(run_consumer, cameras)


# if __name__ == '__main__':

#     from multiprocessing import Pool
#     cameras = [Camera(ip_address='192.168.88.13', username='admin', password='am23hKmc')]
#     run_consumer_v2(cameras)
    # from custom_interface.core.db import DataBase

    # # db = DataBase()
    # # cameras = db.get_cameras()
    # cameras = [Camera(ip_address='192.168.88.13', username='admin', password='am23hKmc')]
    # cameras_for_prosseses = split_into_subarrays(cameras)
    
    # with Pool(len(cameras_for_prosseses)) as p:
    #     p.map(run_consumer_v2, cameras_for_prosseses)